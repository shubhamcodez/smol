"""A bounded hill-climbing research loop with an append-only experiment ledger."""

from __future__ import annotations

import argparse
import csv
import hashlib
import json
import subprocess
import sys
import time
from copy import deepcopy
from datetime import datetime, timezone
from pathlib import Path
from typing import Any


ROOT = Path(__file__).resolve().parent
MIN_IMPROVEMENT = 1e-3
LEDGER_COLUMNS = [
    "timestamp_utc",
    "run_id",
    "status",
    "score",
    "validation_loss",
    "latency_ms",
    "parameter_count",
    "training_backend",
    "deployment_backend",
    "mutation",
    "candidate_json",
]


def canonical(candidate: dict[str, Any]) -> str:
    return json.dumps(candidate, sort_keys=True, separators=(",", ":"))


def short_hash(candidate: dict[str, Any]) -> str:
    return hashlib.sha256(canonical(candidate).encode("utf-8")).hexdigest()[:10]


def write_json(path: Path, value: dict[str, Any]) -> None:
    temporary = path.with_suffix(path.suffix + ".tmp")
    temporary.write_text(json.dumps(value, indent=2) + "\n", encoding="utf-8")
    temporary.replace(path)


def append_ledger(path: Path, metrics: dict[str, Any], status: str, mutation: str) -> None:
    exists = path.exists() and path.stat().st_size > 0
    deployment = metrics["deployment"]
    row = {
        "timestamp_utc": datetime.now(timezone.utc).isoformat(),
        "run_id": metrics["run_id"],
        "status": status,
        "score": f'{metrics["score"]:.9f}',
        "validation_loss": f'{metrics["validation_loss"]:.9f}',
        "latency_ms": f'{deployment["latency_ms"]:.6f}',
        "parameter_count": metrics["parameter_count"],
        "training_backend": metrics["training_backend"],
        "deployment_backend": deployment["backend"],
        "mutation": mutation,
        "candidate_json": canonical(metrics["candidate"]),
    }
    with path.open("a", encoding="utf-8", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=LEDGER_COLUMNS, delimiter="\t")
        if not exists:
            writer.writeheader()
        writer.writerow(row)


def append_crash(
    path: Path,
    run_id: str,
    mutation: str,
    candidate: dict[str, Any],
    error: Exception,
) -> None:
    exists = path.exists() and path.stat().st_size > 0
    row = {
        "timestamp_utc": datetime.now(timezone.utc).isoformat(),
        "run_id": run_id,
        "status": "crash",
        "score": "",
        "validation_loss": "",
        "latency_ms": "",
        "parameter_count": "",
        "training_backend": "",
        "deployment_backend": "",
        "mutation": f"{mutation}; {type(error).__name__}: {str(error)[:300]}",
        "candidate_json": canonical(candidate),
    }
    with path.open("a", encoding="utf-8", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=LEDGER_COLUMNS, delimiter="\t")
        if not exists:
            writer.writeheader()
        writer.writerow(row)


def evaluate(
    candidate_path: Path,
    candidate: dict[str, Any],
    run_id: str,
    artifacts: Path,
    budget_seconds: float,
    training_backend: str,
    deployment: str,
) -> dict[str, Any]:
    write_json(candidate_path, candidate)
    command = [
        sys.executable,
        str(ROOT / "fixed_benchmark.py"),
        "--candidate",
        str(candidate_path),
        "--run-id",
        run_id,
        "--artifacts",
        str(artifacts),
        "--budget-seconds",
        str(budget_seconds),
        "--training-backend",
        training_backend,
        "--deployment",
        deployment,
    ]
    completed = subprocess.run(
        command,
        cwd=ROOT.parent,
        text=True,
        capture_output=True,
        timeout=max(180.0, budget_seconds + 150.0),
        check=False,
    )
    for line in reversed(completed.stdout.splitlines()):
        if line.startswith("AUTORESEARCH_METRICS "):
            return json.loads(line.removeprefix("AUTORESEARCH_METRICS "))
    details = (completed.stdout + "\n" + completed.stderr)[-8000:]
    raise RuntimeError(f"evaluator failed with exit code {completed.returncode}:\n{details}")


def propose(best: dict[str, Any], index: int) -> tuple[dict[str, Any], str]:
    proposal = deepcopy(best)
    move = index % 6
    if move == 0:
        old = float(proposal["learning_rate"])
        proposal["learning_rate"] = min(0.05, round(old * 2.0, 8))
        description = f"learning_rate {old:g} -> {proposal['learning_rate']:g}"
    elif move == 1:
        choices = [32, 48, 64, 96, 128, 160, 192, 256]
        old = int(proposal["hidden_dim"])
        proposal["hidden_dim"] = choices[min(choices.index(old) + 1, len(choices) - 1)]
        description = f"hidden_dim {old} -> {proposal['hidden_dim']}"
    elif move == 2:
        old = float(proposal["learning_rate"])
        proposal["learning_rate"] = max(1e-5, round(old * 0.7, 8))
        description = f"learning_rate {old:g} -> {proposal['learning_rate']:g}"
    elif move == 3:
        old = float(proposal["weight_decay"])
        proposal["weight_decay"] = round(old / 3.0, 8)
        description = f"weight_decay {old:g} -> {proposal['weight_decay']:g}"
    elif move == 4:
        choices = [32, 64, 128, 256]
        old = int(proposal["batch_size"])
        proposal["batch_size"] = choices[min(choices.index(old) + 1, len(choices) - 1)]
        description = f"batch_size {old} -> {proposal['batch_size']}"
    else:
        old = float(proposal["beta2"])
        proposal["beta2"] = 0.99 if old > 0.99 else 0.999
        description = f"beta2 {old:g} -> {proposal['beta2']:g}"
    return proposal, description


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--candidate", type=Path, default=ROOT / "candidate.json")
    parser.add_argument("--best", type=Path, default=ROOT / "best.json")
    parser.add_argument("--ledger", type=Path, default=ROOT / "results.tsv")
    parser.add_argument("--artifacts", type=Path, default=ROOT / "artifacts")
    parser.add_argument("--iterations", type=int, default=4)
    parser.add_argument("--budget-seconds", type=float, default=1.0)
    parser.add_argument("--training-backend", choices=["auto", "cpu", "cuda"], default="auto")
    parser.add_argument("--deployment", choices=["auto", "cuda", "cpu"], default="auto")
    parser.add_argument("--reset", action="store_true")
    args = parser.parse_args()

    if args.iterations < 0:
        raise ValueError("iterations must be non-negative")
    if args.budget_seconds <= 0:
        raise ValueError("budget-seconds must be positive")
    args.artifacts.mkdir(parents=True, exist_ok=True)
    if args.reset:
        archive_stamp = datetime.now(timezone.utc).strftime("%Y%m%dT%H%M%S%fZ")
        if args.ledger.exists() and args.ledger.stat().st_size > 0:
            archive = args.ledger.with_name(f"{args.ledger.stem}.{archive_stamp}{args.ledger.suffix}")
            args.ledger.replace(archive)
        if args.best.exists():
            archive = args.best.with_name(f"{args.best.stem}.{archive_stamp}{args.best.suffix}")
            args.best.replace(archive)

    source = args.best if args.best.exists() else args.candidate
    best = json.loads(source.read_text(encoding="utf-8"))
    session = datetime.now(timezone.utc).strftime("%Y%m%dT%H%M%SZ")
    baseline_id = f"{session}-baseline-{short_hash(best)}"
    print(f"Evaluating baseline {baseline_id}", flush=True)
    best_metrics = evaluate(
        args.candidate,
        best,
        baseline_id,
        args.artifacts,
        args.budget_seconds,
        args.training_backend,
        args.deployment,
    )
    append_ledger(args.ledger, best_metrics, "baseline", "none")
    write_json(args.best, best)
    print(
        f"baseline score={best_metrics['score']:.6f} "
        f"loss={best_metrics['validation_loss']:.6f} "
        f"deployment={best_metrics['deployment']['backend']} ",
        flush=True,
    )

    accepted = 0
    for index in range(args.iterations):
        proposal, mutation = propose(best, index)
        run_id = f"{session}-trial{index + 1:02d}-{short_hash(proposal)}"
        print(f"Evaluating {run_id}: {mutation}", flush=True)
        try:
            metrics = evaluate(
                args.candidate,
                proposal,
                run_id,
                args.artifacts,
                args.budget_seconds,
                args.training_backend,
                args.deployment,
            )
        except Exception as error:
            write_json(args.candidate, best)
            append_crash(args.ledger, run_id, mutation, proposal, error)
            print(f"discard {run_id}: evaluator error: {error}", flush=True)
            continue

        improved = metrics["score"] < best_metrics["score"] - MIN_IMPROVEMENT
        status = "keep" if improved else "discard"
        append_ledger(args.ledger, metrics, status, mutation)
        print(
            f"{status} score={metrics['score']:.6f} "
            f"loss={metrics['validation_loss']:.6f} "
            f"latency_ms={metrics['deployment']['latency_ms']:.4f}",
            flush=True,
        )
        if improved:
            best = proposal
            best_metrics = metrics
            accepted += 1
            write_json(args.best, best)
        write_json(args.candidate, best)

    summary = {
        "accepted_trials": accepted,
        "attempted_trials": args.iterations,
        "best_score": best_metrics["score"],
        "best_validation_loss": best_metrics["validation_loss"],
        "best_deployment": best_metrics["deployment"],
        "best_candidate": best,
        "ledger": str(args.ledger.resolve()),
    }
    summary_path = args.artifacts / f"{session}-summary.json"
    summary_path.write_text(json.dumps(summary, indent=2) + "\n", encoding="utf-8")
    print("AUTORESEARCH_SUMMARY " + json.dumps(summary, separators=(",", ":")))


if __name__ == "__main__":
    main()
