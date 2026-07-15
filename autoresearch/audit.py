"""Audit saved evidence from a completed autoresearch run."""

from __future__ import annotations

import argparse
import csv
import json
from pathlib import Path
from typing import Any


ROOT = Path(__file__).resolve().parent


def require(condition: bool, message: str) -> None:
    if not condition:
        raise AssertionError(message)


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--ledger", type=Path, default=ROOT / "results.tsv")
    parser.add_argument("--best", type=Path, default=ROOT / "best.json")
    parser.add_argument("--proof", type=Path, default=ROOT / "artifacts" / "npu_smoke" / "npu_proof.json")
    parser.add_argument("--expected-deployment", default="qnn-npu")
    args = parser.parse_args()

    proof = json.loads(args.proof.read_text(encoding="utf-8"))
    require(proof["backend"] == "qnn-npu", "proof did not select qnn-npu")
    require(proof["device"]["execution_provider"] == "QNNExecutionProvider", "wrong EP in proof")
    require(proof["device"]["hardware_type"] == "NPU", "QNN device was not typed as NPU")
    require(proof["cpu_ep_fallback_disabled"] is True, "CPU EP fallback was not disabled")
    require(Path(proof["provider_options"]["backend_path"]).name.lower() == "qnnhtp.dll", "HTP backend was not loaded")
    require(proof["iterations"] > 0 and proof["latency_ms"] > 0.0, "NPU timing is missing")

    with args.ledger.open("r", encoding="utf-8", newline="") as handle:
        rows = list(csv.DictReader(handle, delimiter="\t"))
    require(bool(rows), "experiment ledger is empty")
    require(rows[0]["status"] == "baseline", "first ledger row is not a baseline")

    best_score = float(rows[0]["score"])
    last_kept = json.loads(rows[0]["candidate_json"])
    kept = 0
    discarded = 0
    for row in rows:
        if row["status"] == "crash":
            continue
        require(row["deployment_backend"] == args.expected_deployment, f"{row['run_id']} used the wrong deployment backend")
        score = float(row["score"])
        if row["status"] == "keep":
            require(score < best_score, f"kept run {row['run_id']} did not improve")
            best_score = score
            last_kept = json.loads(row["candidate_json"])
            kept += 1
        elif row["status"] == "discard":
            discarded += 1

    best = json.loads(args.best.read_text(encoding="utf-8"))
    require(best == last_kept, "best.json does not match the last kept candidate")
    report: dict[str, Any] = {
        "passed": True,
        "npu_proof": {
            "execution_provider": proof["device"]["execution_provider"],
            "hardware_type": proof["device"]["hardware_type"],
            "backend_library": proof["provider_options"]["backend_path"],
            "cpu_ep_fallback_disabled": proof["cpu_ep_fallback_disabled"],
            "latency_ms": proof["latency_ms"],
        },
        "experiments": len(rows),
        "kept": kept,
        "discarded": discarded,
        "best_score": best_score,
        "best_candidate": best,
    }
    report_path = ROOT / "artifacts" / "audit_report.json"
    report_path.write_text(json.dumps(report, indent=2) + "\n", encoding="utf-8")
    print(json.dumps(report, indent=2))
    print(f"AUDIT_REPORT={report_path.resolve()}")


if __name__ == "__main__":
    main()
