"""Forward/backward CUDA smoke test for the modernized model.py model."""

from __future__ import annotations

import argparse
import json
import sys
import time
from pathlib import Path

import torch


sys.path.insert(0, str(Path(__file__).resolve().parent.parent))
from model import ModelConfig, TransformerLM  # noqa: E402


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--full", action="store_true", help="test the 203.8M default model")
    parser.add_argument("--sequence-length", type=int, default=None)
    args = parser.parse_args()

    if not torch.cuda.is_available():
        raise SystemExit("CUDA is unavailable; install a CUDA-enabled PyTorch build on the NVIDIA PC")
    device = torch.device("cuda")
    torch.manual_seed(20260715)
    if args.full:
        config = ModelConfig()
        sequence_length = args.sequence_length or config.max_seq_len
        batch_size = 1
        loss_chunk_size = 256
    else:
        config = ModelConfig(
            vocab_size=1024,
            max_seq_len=128,
            n_layer=4,
            n_embd=256,
            n_head=8,
            n_kv_head=2,
            intermediate_size=768,
        )
        sequence_length = args.sequence_length or config.max_seq_len
        batch_size = 4
        loss_chunk_size = 32
    if not 1 <= sequence_length <= config.max_seq_len:
        raise SystemExit(f"sequence length must be between 1 and {config.max_seq_len}")

    model = TransformerLM(config).to(device)
    model.gradient_checkpointing_enable()
    optimizer = model.configure_optimizer(
        weight_decay=0.1,
        learning_rate=3e-4,
        device_type="cuda",
    )
    scaler = torch.amp.GradScaler("cuda")
    tokens = torch.randint(
        config.vocab_size,
        (batch_size, sequence_length),
        device=device,
    )
    targets = torch.roll(tokens, shifts=-1, dims=1)

    torch.cuda.reset_peak_memory_stats()
    torch.cuda.synchronize()
    start = time.perf_counter()
    with torch.autocast(device_type="cuda", dtype=torch.float16):
        result = model(
            tokens,
            targets=targets,
            return_logits=False,
            loss_chunk_size=loss_chunk_size,
        )
    if result.loss is None:
        raise RuntimeError("model did not return a training loss")
    scaler.scale(result.loss).backward()
    scaler.step(optimizer)
    scaler.update()
    torch.cuda.synchronize()
    elapsed = time.perf_counter() - start

    model.eval()
    with torch.inference_mode(), torch.autocast(device_type="cuda", dtype=torch.float16):
        inference = model(tokens[:, :16], last_token_only=True)
    if inference.logits is None:
        raise RuntimeError("model did not return inference logits")
    torch.cuda.synchronize()
    proof = {
        "backend": "cuda",
        "device": torch.cuda.get_device_name(0),
        "torch_version": torch.__version__,
        "cuda_version": torch.version.cuda,
        "model_parameters": model.get_num_params(),
        "sequence_length": sequence_length,
        "batch_size": batch_size,
        "loss": float(result.loss.detach().float().item()),
        "forward_backward_seconds": elapsed,
        "peak_allocated_gib": torch.cuda.max_memory_allocated() / 2**30,
        "peak_reserved_gib": torch.cuda.max_memory_reserved() / 2**30,
        "chunked_training_logits_omitted": result.logits is None,
        "inference_logits_on_cuda": inference.logits.is_cuda,
        "gradient_checkpointing": model.gradient_checkpointing,
    }
    print(json.dumps(proof, indent=2))


if __name__ == "__main__":
    main()
