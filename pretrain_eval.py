"""Pretrain quality metrics: bits-per-byte and multiple-choice loglikelihood."""

from __future__ import annotations

import math
from pathlib import Path
from typing import Any, Sequence

import pyarrow.parquet as pq
import torch
import torch.nn.functional as F

from tokenizer_gpt2 import decode_tokens, encode_text
from token_dataset import TokenShard


ROOT = Path(__file__).resolve().parent
BENCH_WEIGHT = 2.0  # score = bpb + BENCH_WEIGHT * (1 - mean_acc)


def bits_per_byte_from_token_nll(
    total_nll_nats: float,
    token_count: int,
    byte_count: int,
) -> float:
    if token_count <= 0 or byte_count <= 0:
        raise ValueError("token_count and byte_count must be positive")
    # BPB = (L_T * mean_token_nll) / (L_B * ln 2)
    return (token_count * (total_nll_nats / token_count)) / (byte_count * math.log(2.0))


@torch.inference_mode()
def evaluate_bpb(
    model: torch.nn.Module,
    shard: TokenShard,
    *,
    seq_len: int,
    batch_size: int,
    max_batches: int,
    seed: int,
    device: torch.device,
) -> dict[str, float]:
    model.eval()
    generator = torch.Generator(device="cpu").manual_seed(seed)
    total_nll = 0.0
    total_tokens = 0
    total_bytes = 0
    for _ in range(max_batches):
        inputs, targets = shard.sample_batch(batch_size, seq_len, generator, device)
        with torch.autocast(device_type=device.type, dtype=torch.float16, enabled=device.type == "cuda"):
            output = model(inputs, targets=targets, return_logits=False, loss_chunk_size=min(256, seq_len))
        if output.loss is None:
            raise RuntimeError("BPB evaluation requires a loss")
        # model loss is mean nll over non-ignored tokens
        n_tokens = int(targets.numel())
        total_nll += float(output.loss.detach().float().item()) * n_tokens
        total_tokens += n_tokens
        # Approximate UTF-8 bytes via detokenized batch (stable enough for ranking).
        flat = targets.detach().cpu().tolist()
        for row in flat:
            text = decode_tokens([int(t) for t in row]).encode("utf-8", errors="replace")
            total_bytes += len(text)
    mean_nll = total_nll / max(total_tokens, 1)
    bpb = bits_per_byte_from_token_nll(total_nll, total_tokens, max(total_bytes, 1))
    return {
        "bpb": bpb,
        "validation_loss": mean_nll,
        "tokens_evaluated": float(total_tokens),
        "bytes_evaluated": float(total_bytes),
    }


def _choice_texts_arc_obqa(choices: dict[str, Any]) -> list[str]:
    return list(choices["text"])


def _load_arc(limit: int) -> list[dict[str, Any]]:
    path = ROOT / "benchmarks" / "arc" / "ARC-Challenge" / "test-00000-of-00001.parquet"
    rows = pq.read_table(path).to_pylist()
    items = []
    for row in rows[:limit]:
        labels = list(row["choices"]["label"])
        texts = _choice_texts_arc_obqa(row["choices"])
        answer = labels.index(row["answerKey"])
        items.append(
            {
                "question": row["question"],
                "choices": texts,
                "answer": answer,
                "task": "arc",
            }
        )
    return items


def _load_openbookqa(limit: int) -> list[dict[str, Any]]:
    path = ROOT / "benchmarks" / "openbookqa" / "main" / "test-00000-of-00001.parquet"
    if not path.exists():
        matches = list((ROOT / "benchmarks" / "openbookqa").rglob("test*.parquet"))
        if not matches:
            raise FileNotFoundError("OpenBookQA test parquet not found")
        path = matches[0]
    rows = pq.read_table(path).to_pylist()
    items = []
    for row in rows[:limit]:
        labels = list(row["choices"]["label"])
        texts = _choice_texts_arc_obqa(row["choices"])
        answer = labels.index(row["answerKey"])
        items.append(
            {
                "question": row["question_stem"],
                "choices": texts,
                "answer": answer,
                "task": "openbookqa",
            }
        )
    return items


def _load_mmlu(limit: int) -> list[dict[str, Any]]:
    # Prefer the combined "all" split when present; otherwise sample subjects.
    all_test = list((ROOT / "benchmarks" / "mmlu" / "all").glob("test*.parquet"))
    rows: list[dict[str, Any]] = []
    if all_test:
        rows = pq.read_table(all_test[0]).to_pylist()
    else:
        subjects = sorted(
            path
            for path in (ROOT / "benchmarks" / "mmlu").iterdir()
            if path.is_dir() and path.name not in {"all", "auxiliary_train", ".cache"}
        )
        for subject in subjects:
            files = list(subject.glob("test*.parquet"))
            if not files:
                continue
            rows.extend(pq.read_table(files[0]).to_pylist())
            if len(rows) >= limit:
                break
    items = []
    for row in rows[:limit]:
        items.append(
            {
                "question": row["question"],
                "choices": list(row["choices"]),
                "answer": int(row["answer"]),
                "task": "mmlu",
            }
        )
    return items


def load_benchmark_items(limit_per_task: int) -> list[dict[str, Any]]:
    return (
        _load_mmlu(limit_per_task)
        + _load_arc(limit_per_task)
        + _load_openbookqa(limit_per_task)
    )


def format_mcq_prompt(question: str, choice: str) -> tuple[list[int], list[int]]:
    """Return (context_tokens, full_tokens) for loglikelihood of the choice."""
    prefix = f"Question: {question.strip()}\nAnswer:"
    completion = f" {choice.strip()}"
    context = encode_text(prefix)
    full = encode_text(prefix + completion)
    if len(full) <= len(context):
        # Rare tokenizer edge case: force at least one completion token.
        full = context + encode_text(completion)
    return context, full


@torch.inference_mode()
def loglikelihood(
    model: torch.nn.Module,
    context: Sequence[int],
    full: Sequence[int],
    device: torch.device,
) -> float:
    if len(full) <= len(context):
        raise ValueError("completion must add tokens")
    max_seq = model.config.max_seq_len
    if len(full) > max_seq:
        overflow = len(full) - max_seq
        context = context[overflow:]
        full = full[overflow:]
        if len(full) <= len(context):
            return float("-inf")
    tokens = torch.tensor([full], dtype=torch.long, device=device)
    with torch.autocast(device_type=device.type, dtype=torch.float16, enabled=device.type == "cuda"):
        output = model(tokens[:, :-1], return_logits=True)
    if output.logits is None:
        raise RuntimeError("loglikelihood requires logits")
    logits = output.logits[0]
    cont = full[len(context) :]
    start = len(context) - 1
    nll = 0.0
    for offset, token in enumerate(cont):
        position = start + offset
        if position < 0 or position >= logits.size(0):
            return float("-inf")
        log_probs = F.log_softmax(logits[position].float(), dim=-1)
        nll += float(-log_probs[token].item())
    return -nll


@torch.inference_mode()
def evaluate_mcq_accuracy(
    model: torch.nn.Module,
    items: list[dict[str, Any]],
    device: torch.device,
) -> dict[str, Any]:
    model.eval()
    per_task: dict[str, list[bool]] = {}
    for item in items:
        scores = []
        for choice in item["choices"]:
            context, full = format_mcq_prompt(item["question"], choice)
            scores.append(loglikelihood(model, context, full, device))
        predicted = int(max(range(len(scores)), key=lambda index: scores[index]))
        correct = predicted == int(item["answer"])
        per_task.setdefault(item["task"], []).append(correct)

    task_accuracy = {
        task: float(sum(values) / max(len(values), 1)) for task, values in per_task.items()
    }
    all_correct = [value for values in per_task.values() for value in values]
    mean_accuracy = float(sum(all_correct) / max(len(all_correct), 1))
    return {
        "mean_benchmark_acc": mean_accuracy,
        "task_accuracy": task_accuracy,
        "examples_evaluated": len(all_correct),
    }


def overall_score(bpb: float, mean_benchmark_acc: float, parameter_count: int = 0) -> float:
    """Lower is better: BPB plus penalty for wrong benchmark answers."""
    return float(bpb + BENCH_WEIGHT * (1.0 - mean_benchmark_acc) + 1e-9 * parameter_count)
