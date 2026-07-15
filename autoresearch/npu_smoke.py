"""Reproducible, standalone proof that a graph executes on the Hexagon NPU."""

from __future__ import annotations

import argparse
import json
from pathlib import Path

import numpy as np

from backend import detect_runtime
from npu_runtime import STATIC_BATCH, benchmark_qnn_npu, export_quantized_mlp


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--artifacts", type=Path, default=Path(__file__).parent / "artifacts" / "npu_smoke")
    parser.add_argument("--iterations", type=int, default=100)
    args = parser.parse_args()

    rng = np.random.default_rng(20260715)
    input_dim, hidden_dim, output_dim = 64, 96, 32
    weights = {
        "w1": rng.normal(0.0, 0.08, (input_dim, hidden_dim)).astype(np.float32),
        "b1": np.zeros(hidden_dim, dtype=np.float32),
        "w2": rng.normal(0.0, 0.08, (hidden_dim, output_dim)).astype(np.float32),
        "b2": np.zeros(output_dim, dtype=np.float32),
    }
    calibration = rng.normal(size=(STATIC_BATCH * 4, input_dim)).astype(np.float32)
    sample = calibration[:STATIC_BATCH]
    _, quantized_path = export_quantized_mlp(weights, calibration, args.artifacts)
    proof = benchmark_qnn_npu(quantized_path, sample, iterations=args.iterations)
    proof["runtime_inventory"] = detect_runtime().to_dict()
    proof["model"] = str(quantized_path.resolve())

    args.artifacts.mkdir(parents=True, exist_ok=True)
    proof_path = args.artifacts / "npu_proof.json"
    proof_path.write_text(json.dumps(proof, indent=2) + "\n", encoding="utf-8")
    print(json.dumps(proof, indent=2))
    print(f"NPU_PROOF={proof_path.resolve()}")


if __name__ == "__main__":
    main()
