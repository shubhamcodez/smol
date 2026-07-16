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
    parser.add_argument("--expected-deployment", default="cuda", choices=["cuda", "cpu", "torch-cuda", "numpy-cpu"])
    args = parser.parse_args()

    expected = {
        "cuda": "torch-cuda",
        "torch-cuda": "torch-cuda",
        "cpu": "numpy-cpu",
        "numpy-cpu": "numpy-cpu",
    }[args.expected_deployment]

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
        require(
            row["deployment_backend"] == expected,
            f"{row['run_id']} used the wrong deployment backend "
            f"(got {row['deployment_backend']!r}, expected {expected!r})",
        )
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
        "expected_deployment": expected,
        "experiments": len(rows),
        "kept": kept,
        "discarded": discarded,
        "best_score": best_score,
        "best_candidate": best,
    }
    report_path = ROOT / "artifacts" / "audit_report.json"
    report_path.parent.mkdir(parents=True, exist_ok=True)
    report_path.write_text(json.dumps(report, indent=2) + "\n", encoding="utf-8")
    print(json.dumps(report, indent=2))
    print(f"AUDIT_REPORT={report_path.resolve()}")


if __name__ == "__main__":
    main()
