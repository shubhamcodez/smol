"""Model scoreboard: our runs + published reference numbers.

Append-only ledger. Reference rows are seeded from public tech reports; our
protocol is truncated log-likelihood MCQ and is NOT directly comparable to
published few-shot setups — both are stored with an explicit `protocol` field.
"""

from __future__ import annotations

import argparse
import csv
import json
from datetime import datetime, timezone
from pathlib import Path
from typing import Any


ROOT = Path(__file__).resolve().parent
DEFAULT_LOG = ROOT / "scores.tsv"

COLUMNS = [
    "timestamp_utc",
    "model_id",
    "family",
    "params_b",
    "source",
    "protocol",
    "mmlu",
    "arc_challenge",
    "openbookqa",
    "mean_acc",
    "bpb",
    "score",
    "run_id",
    "notes",
    "citation",
]

# Published base-model numbers (percent accuracy). Missing metrics left blank.
# Qwen2.5 blog / tech report: https://qwenlm.github.io/blog/qwen2.5-llm/
# https://arxiv.org/abs/2412.15115
REFERENCE_ROWS: list[dict[str, str]] = [
    {
        "model_id": "Qwen2.5-0.5B",
        "family": "qwen2.5",
        "params_b": "0.5",
        "mmlu": "47.5",
        "arc_challenge": "35.6",
        "openbookqa": "",
        "notes": "published base; MMLU 5-shot, ARC-C 25-shot",
        "citation": "https://qwenlm.github.io/blog/qwen2.5-llm/",
    },
    {
        "model_id": "Qwen2.5-1.5B",
        "family": "qwen2.5",
        "params_b": "1.5",
        "mmlu": "60.9",
        "arc_challenge": "54.7",
        "openbookqa": "",
        "notes": "published base; MMLU 5-shot, ARC-C 25-shot",
        "citation": "https://qwenlm.github.io/blog/qwen2.5-llm/",
    },
    {
        "model_id": "Qwen2.5-3B",
        "family": "qwen2.5",
        "params_b": "3.0",
        "mmlu": "65.6",
        "arc_challenge": "56.5",
        "openbookqa": "",
        "notes": "published base; MMLU 5-shot, ARC-C 25-shot",
        "citation": "https://qwenlm.github.io/blog/qwen2.5-llm/",
    },
    {
        "model_id": "Qwen2.5-7B",
        "family": "qwen2.5",
        "params_b": "7.0",
        "mmlu": "74.2",
        "arc_challenge": "63.7",
        "openbookqa": "",
        "notes": "published base; MMLU 5-shot, ARC-C 25-shot",
        "citation": "https://arxiv.org/abs/2412.15115",
    },
    {
        "model_id": "Qwen2-0.5B",
        "family": "qwen2",
        "params_b": "0.5",
        "mmlu": "44.3",
        "arc_challenge": "31.0",
        "openbookqa": "",
        "notes": "published base",
        "citation": "https://qwenlm.github.io/blog/qwen2.5-llm/",
    },
    {
        "model_id": "Qwen2-1.5B",
        "family": "qwen2",
        "params_b": "1.5",
        "mmlu": "55.9",
        "arc_challenge": "43.7",
        "openbookqa": "",
        "notes": "published base",
        "citation": "https://qwenlm.github.io/blog/qwen2.5-llm/",
    },
    {
        "model_id": "Gemma2-2.6B",
        "family": "gemma2",
        "params_b": "2.6",
        "mmlu": "52.2",
        "arc_challenge": "55.7",
        "openbookqa": "",
        "notes": "as reported alongside Qwen2.5 small models",
        "citation": "https://qwenlm.github.io/blog/qwen2.5-llm/",
    },
    {
        "model_id": "Llama-3-8B",
        "family": "llama3",
        "params_b": "8.0",
        "mmlu": "66.6",
        "arc_challenge": "59.3",
        "openbookqa": "",
        "notes": "as reported in Qwen2.5 7B comparison table",
        "citation": "https://arxiv.org/abs/2412.15115",
    },
    {
        "model_id": "Mistral-7B",
        "family": "mistral",
        "params_b": "7.0",
        "mmlu": "64.2",
        "arc_challenge": "60.0",
        "openbookqa": "",
        "notes": "as reported in Qwen2.5 7B comparison table",
        "citation": "https://arxiv.org/abs/2412.15115",
    },
    {
        "model_id": "GPT-2-124M",
        "family": "gpt2",
        "params_b": "0.124",
        "mmlu": "",
        "arc_challenge": "",
        "openbookqa": "",
        "notes": "classic LM; no standard MMLU/ARC suite in original paper",
        "citation": "https://cdn.openai.com/better-language-models/language_models_are_unsupervised_multitask_learners.pdf",
    },
    {
        "model_id": "chance-4way",
        "family": "baseline",
        "params_b": "0",
        "mmlu": "25.0",
        "arc_challenge": "25.0",
        "openbookqa": "25.0",
        "notes": "uniform random among 4 choices",
        "citation": "analytic",
    },
]


def utc_now() -> str:
    return datetime.now(timezone.utc).isoformat()


def load_rows(path: Path) -> list[dict[str, str]]:
    if not path.exists() or path.stat().st_size == 0:
        return []
    with path.open("r", encoding="utf-8", newline="") as handle:
        return list(csv.DictReader(handle, delimiter="\t"))


def append_row(path: Path, row: dict[str, Any]) -> None:
    exists = path.exists() and path.stat().st_size > 0
    with path.open("a", encoding="utf-8", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=COLUMNS, delimiter="\t")
        if not exists:
            writer.writeheader()
        writer.writerow({key: str(row.get(key, "") or "") for key in COLUMNS})


def _mean_available(*values: str) -> str:
    nums = []
    for value in values:
        if value is None or value == "":
            continue
        nums.append(float(value))
    if not nums:
        return ""
    return f"{sum(nums) / len(nums):.4f}"


def seed_references(path: Path, force: bool = False) -> int:
    existing = {(row.get("model_id"), row.get("source"), row.get("protocol")) for row in load_rows(path)}
    added = 0
    for ref in REFERENCE_ROWS:
        key = (ref["model_id"], "reference", "published")
        if key in existing and not force:
            continue
        mean_acc = _mean_available(ref.get("mmlu", ""), ref.get("arc_challenge", ""), ref.get("openbookqa", ""))
        append_row(
            path,
            {
                "timestamp_utc": utc_now(),
                "model_id": ref["model_id"],
                "family": ref["family"],
                "params_b": ref["params_b"],
                "source": "reference",
                "protocol": "published",
                "mmlu": ref.get("mmlu", ""),
                "arc_challenge": ref.get("arc_challenge", ""),
                "openbookqa": ref.get("openbookqa", ""),
                "mean_acc": mean_acc,
                "bpb": "",
                "score": "",
                "run_id": "",
                "notes": ref.get("notes", ""),
                "citation": ref.get("citation", ""),
            },
        )
        existing.add(key)
        added += 1
    return added


def record_our_run(path: Path, metrics: dict[str, Any], status: str) -> None:
    task = metrics.get("task_accuracy") or {}
    mmlu = task.get("mmlu")
    arc = task.get("arc")
    obqa = task.get("openbookqa")
    params = metrics.get("parameter_count")
    params_b = f"{params / 1e9:.4f}" if isinstance(params, (int, float)) else ""
    candidate = metrics.get("candidate") or {}
    model_id = (
        f"smol-{candidate.get('n_layer', '?')}L-"
        f"{candidate.get('n_embd', '?')}d-"
        f"{metrics.get('run_id', 'run')}"
    )
    append_row(
        path,
        {
            "timestamp_utc": utc_now(),
            "model_id": model_id,
            "family": "smol",
            "params_b": params_b,
            "source": "ours",
            "protocol": "smol-truncated-ll",
            "mmlu": "" if mmlu is None else f"{100.0 * float(mmlu):.4f}",
            "arc_challenge": "" if arc is None else f"{100.0 * float(arc):.4f}",
            "openbookqa": "" if obqa is None else f"{100.0 * float(obqa):.4f}",
            "mean_acc": f"{100.0 * float(metrics.get('mean_benchmark_acc', 0.0)):.4f}",
            "bpb": f"{float(metrics.get('bpb', 0.0)):.6f}",
            "score": f"{float(metrics.get('score', 0.0)):.6f}",
            "run_id": metrics.get("run_id", ""),
            "notes": status,
            "citation": "local",
        },
    )


def cmd_seed(args: argparse.Namespace) -> None:
    added = seed_references(args.log, force=args.force)
    print(f"SEEDED {added} reference rows into {args.log}")


def cmd_add(args: argparse.Namespace) -> None:
    mean_acc = args.mean_acc
    if not mean_acc:
        mean_acc = _mean_available(args.mmlu, args.arc_challenge, args.openbookqa)
    append_row(
        args.log,
        {
            "timestamp_utc": utc_now(),
            "model_id": args.model_id,
            "family": args.family,
            "params_b": args.params_b,
            "source": args.source,
            "protocol": args.protocol,
            "mmlu": args.mmlu,
            "arc_challenge": args.arc_challenge,
            "openbookqa": args.openbookqa,
            "mean_acc": mean_acc,
            "bpb": args.bpb,
            "score": args.score,
            "run_id": args.run_id,
            "notes": args.notes,
            "citation": args.citation,
        },
    )
    print(f"ADDED {args.model_id}")


def cmd_list(args: argparse.Namespace) -> None:
    rows = load_rows(args.log)
    if args.source:
        rows = [row for row in rows if row.get("source") == args.source]
    if args.family:
        rows = [row for row in rows if row.get("family") == args.family]
    header = (
        f"{'source':8} {'model_id':28} {'params':6} {'mmlu':7} {'arc':7} "
        f"{'obqa':7} {'mean':7} {'bpb':8} {'score':8}"
    )
    print(header)
    print("-" * len(header))
    for row in rows:
        print(
            f"{row.get('source',''):8} {row.get('model_id',''):28} {row.get('params_b',''):6} "
            f"{row.get('mmlu',''):7} {row.get('arc_challenge',''):7} {row.get('openbookqa',''):7} "
            f"{row.get('mean_acc',''):7} {row.get('bpb',''):8} {row.get('score',''):8}"
        )
    print(f"COUNT={len(rows)}")


def cmd_compare(args: argparse.Namespace) -> None:
    """Show best ours vs nearest-size references by params."""
    rows = load_rows(args.log)
    ours = [row for row in rows if row.get("source") == "ours" and row.get("score")]
    refs = [row for row in rows if row.get("source") == "reference"]
    if not ours:
        print("No local scored runs yet.")
    else:
        best = min(ours, key=lambda row: float(row["score"]))
        print("BEST_OURS")
        print(json.dumps(best, indent=2))
        print()
    print("REFERENCES (published %; not identical protocol)")
    for row in sorted(refs, key=lambda item: float(item["params_b"] or 0)):
        print(
            f"{row['model_id']:20} params={row['params_b']:>5} "
            f"MMLU={row.get('mmlu') or '-':>5} ARC={row.get('arc_challenge') or '-':>5} "
            f"OBQA={row.get('openbookqa') or '-':>5} mean={row.get('mean_acc') or '-':>7}"
        )


def main() -> None:
    parser = argparse.ArgumentParser(description="Model scoreboard ledger")
    parser.add_argument("--log", type=Path, default=DEFAULT_LOG)
    sub = parser.add_subparsers(dest="command", required=True)

    seed = sub.add_parser("seed", help="add published reference rows if missing")
    seed.add_argument("--force", action="store_true")
    seed.set_defaults(func=cmd_seed)

    add = sub.add_parser("add", help="manually append a score row")
    add.add_argument("--model-id", required=True)
    add.add_argument("--family", default="other")
    add.add_argument("--params-b", default="")
    add.add_argument("--source", default="reference", choices=["ours", "reference"])
    add.add_argument("--protocol", default="published")
    add.add_argument("--mmlu", default="")
    add.add_argument("--arc-challenge", default="")
    add.add_argument("--openbookqa", default="")
    add.add_argument("--mean-acc", default="")
    add.add_argument("--bpb", default="")
    add.add_argument("--score", default="")
    add.add_argument("--run-id", default="")
    add.add_argument("--notes", default="")
    add.add_argument("--citation", default="")
    add.set_defaults(func=cmd_add)

    listing = sub.add_parser("list", help="list scoreboard rows")
    listing.add_argument("--source", choices=["ours", "reference"], default=None)
    listing.add_argument("--family", default=None)
    listing.set_defaults(func=cmd_list)

    compare = sub.add_parser("compare", help="best ours vs published references")
    compare.set_defaults(func=cmd_compare)

    args = parser.parse_args()
    args.func(args)


if __name__ == "__main__":
    main()
