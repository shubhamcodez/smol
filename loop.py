"""A bounded hill-climbing research loop with an append-only experiment ledger."""

from __future__ import annotations

import argparse
import csv
import hashlib
import json
import shutil
import subprocess
import sys
from copy import deepcopy
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

from checkpointing import checkpoint_paths
from papers import attempt_summaries, normalize_arxiv_id
from scores import record_our_run, seed_references


ROOT = Path(__file__).resolve().parent
MIN_IMPROVEMENT = 1e-3
LEDGER_COLUMNS = [
    "timestamp_utc",
    "run_id",
    "status",
    "score",
    "bpb",
    "validation_loss",
    "mean_benchmark_acc",
    "parameter_count",
    "training_backend",
    "paper_ids",
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
    paper_ids = metrics.get("paper_ids") or []
    row = {
        "timestamp_utc": datetime.now(timezone.utc).isoformat(),
        "run_id": metrics["run_id"],
        "status": status,
        "score": f'{metrics["score"]:.9f}',
        "bpb": f'{metrics["bpb"]:.9f}',
        "validation_loss": f'{metrics["validation_loss"]:.9f}',
        "mean_benchmark_acc": f'{metrics["mean_benchmark_acc"]:.9f}',
        "parameter_count": metrics["parameter_count"],
        "training_backend": metrics["training_backend"],
        "paper_ids": ",".join(paper_ids),
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
    paper_ids: list[str],
) -> None:
    exists = path.exists() and path.stat().st_size > 0
    row = {
        "timestamp_utc": datetime.now(timezone.utc).isoformat(),
        "run_id": run_id,
        "status": "crash",
        "score": "",
        "bpb": "",
        "validation_loss": "",
        "mean_benchmark_acc": "",
        "parameter_count": "",
        "training_backend": "",
        "paper_ids": ",".join(paper_ids),
        "mutation": f"{mutation}; {type(error).__name__}: {str(error)[:300]}",
        "candidate_json": canonical(candidate),
    }
    with path.open("a", encoding="utf-8", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=LEDGER_COLUMNS, delimiter="\t")
        if not exists:
            writer.writeheader()
        writer.writerow(row)


def promote_checkpoint(run_dir: Path, best_root: Path) -> None:
    paths = checkpoint_paths(best_root)
    paths["dir"].mkdir(parents=True, exist_ok=True)
    for name in ("model.pt", "candidate.json", "metrics.json", "meta.json"):
        source = run_dir / name
        if source.exists():
            shutil.copy2(source, paths["dir"] / name)


def evaluate(
    candidate_path: Path,
    candidate: dict[str, Any],
    run_id: str,
    artifacts: Path,
    budget_seconds: float,
    training_backend: str,
    benchmark_limit: int,
    bpb_batches: int,
    init_checkpoint: Path | None,
    paper_ids: list[str],
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
        "--benchmark-limit",
        str(benchmark_limit),
        "--bpb-batches",
        str(bpb_batches),
        "--paper-ids",
        ",".join(paper_ids),
    ]
    if init_checkpoint is not None:
        command.extend(["--init-checkpoint", str(init_checkpoint)])
    completed = subprocess.run(
        command,
        cwd=ROOT,
        text=True,
        capture_output=True,
        timeout=max(600.0, budget_seconds + 480.0),
        check=False,
    )
    for line in reversed(completed.stdout.splitlines()):
        if line.startswith("AUTORESEARCH_METRICS "):
            return json.loads(line.removeprefix("AUTORESEARCH_METRICS "))
    details = (completed.stdout + "\n" + completed.stderr)[-8000:]
    raise RuntimeError(f"evaluator failed with exit code {completed.returncode}:\n{details}")


def propose(best: dict[str, Any], index: int) -> tuple[dict[str, Any], str]:
    proposal = deepcopy(best)
    move = index % 8
    if move == 0:
        old = float(proposal["learning_rate"])
        proposal["learning_rate"] = min(3e-3, round(old * 1.6, 8))
        description = f"learning_rate {old:g} -> {proposal['learning_rate']:g}"
    elif move == 1:
        old = float(proposal["learning_rate"])
        proposal["learning_rate"] = max(1e-5, round(old * 0.7, 8))
        description = f"learning_rate {old:g} -> {proposal['learning_rate']:g}"
    elif move == 2:
        choices = [128, 256, 384, 512, 768]
        old = int(proposal["n_embd"])
        proposal["n_embd"] = choices[min(choices.index(old) + 1, len(choices) - 1)]
        if proposal["n_embd"] % int(proposal["n_head"]) != 0:
            proposal["n_head"] = 8 if proposal["n_embd"] % 8 == 0 else 4
        proposal["intermediate_size"] = int(proposal["n_embd"] * 3)
        description = f"n_embd {old} -> {proposal['n_embd']}"
    elif move == 3:
        choices = [2, 4, 6, 8, 12]
        old = int(proposal["n_layer"])
        proposal["n_layer"] = choices[min(choices.index(old) + 1, len(choices) - 1)]
        description = f"n_layer {old} -> {proposal['n_layer']}"
    elif move == 4:
        old = float(proposal["weight_decay"])
        proposal["weight_decay"] = round(min(0.2, max(0.0, old / 2.0 if old > 0 else 0.01)), 8)
        description = f"weight_decay {old:g} -> {proposal['weight_decay']:g}"
    elif move == 5:
        choices = [1, 2, 4, 8]
        old = int(proposal["batch_size"])
        proposal["batch_size"] = choices[min(choices.index(old) + 1, len(choices) - 1)]
        description = f"batch_size {old} -> {proposal['batch_size']}"
    elif move == 6:
        choices = [128, 256, 512, 1024]
        old = int(proposal["max_seq_len"])
        proposal["max_seq_len"] = choices[min(choices.index(old) + 1, len(choices) - 1)]
        description = f"max_seq_len {old} -> {proposal['max_seq_len']}"
    else:
        old = float(proposal["beta2"])
        proposal["beta2"] = 0.99 if old > 0.99 else 0.95
        description = f"beta2 {old:g} -> {proposal['beta2']:g}"
    return proposal, description


def paper_attempt_line(paper_ids: list[str]) -> str:
    summaries = attempt_summaries(paper_ids)
    if not summaries:
        return ""
    return " | ".join(summaries)


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--candidate", type=Path, default=ROOT / "candidate.json")
    parser.add_argument("--best", type=Path, default=ROOT / "best.json")
    parser.add_argument("--ledger", type=Path, default=ROOT / "results.tsv")
    parser.add_argument("--artifacts", type=Path, default=ROOT / "artifacts")
    parser.add_argument("--checkpoint-dir", type=Path, default=ROOT / "checkpoints" / "best")
    parser.add_argument("--resume-checkpoint", type=Path, default=None,
                        help="Warm-start from this checkpoint dir (default: checkpoints/best if present with --resume)")
    parser.add_argument("--resume", action="store_true",
                        help="Resume from --checkpoint-dir (candidate + weights)")
    parser.add_argument("--paper-id", action="append", default=[],
                        help="arXiv id informing this session; repeatable")
    parser.add_argument("--scores", type=Path, default=ROOT / "scores.tsv")
    parser.add_argument("--iterations", type=int, default=4)
    parser.add_argument("--budget-seconds", type=float, default=30.0)
    parser.add_argument("--training-backend", choices=["auto", "cpu", "cuda"], default="auto")
    parser.add_argument("--benchmark-limit", type=int, default=32)
    parser.add_argument("--bpb-batches", type=int, default=8)
    parser.add_argument("--reset", action="store_true")
    args = parser.parse_args()

    if args.iterations < 0:
        raise ValueError("iterations must be non-negative")
    if args.budget_seconds <= 0:
        raise ValueError("budget-seconds must be positive")
    args.artifacts.mkdir(parents=True, exist_ok=True)

    paper_ids = [normalize_arxiv_id(value) for value in args.paper_id]
    seed_references(args.scores)

    if args.reset:
        archive_stamp = datetime.now(timezone.utc).strftime("%Y%m%dT%H%M%S%fZ")
        if args.ledger.exists() and args.ledger.stat().st_size > 0:
            archive = args.ledger.with_name(f"{args.ledger.stem}.{archive_stamp}{args.ledger.suffix}")
            args.ledger.replace(archive)
        if args.best.exists():
            archive = args.best.with_name(f"{args.best.stem}.{archive_stamp}{args.best.suffix}")
            args.best.replace(archive)

    init_checkpoint: Path | None = None
    if args.resume or args.resume_checkpoint is not None:
        init_checkpoint = args.resume_checkpoint or args.checkpoint_dir
        ckpt_candidate = init_checkpoint / "candidate.json"
        if not (init_checkpoint / "model.pt").exists():
            raise SystemExit(f"cannot resume; missing {init_checkpoint / 'model.pt'}")
        if ckpt_candidate.exists():
            best = json.loads(ckpt_candidate.read_text(encoding="utf-8"))
            write_json(args.best, best)
            print(f"Resuming candidate from {ckpt_candidate}", flush=True)
        else:
            best = json.loads((args.best if args.best.exists() else args.candidate).read_text(encoding="utf-8"))
    else:
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
        args.benchmark_limit,
        args.bpb_batches,
        init_checkpoint,
        paper_ids,
    )
    append_ledger(args.ledger, best_metrics, "baseline", "none")
    record_our_run(args.scores, best_metrics, "baseline")
    write_json(args.best, best)
    promote_checkpoint(args.artifacts / baseline_id, args.checkpoint_dir)
    print(
        f"baseline score={best_metrics['score']:.6f} "
        f"bpb={best_metrics['bpb']:.6f} "
        f"acc={best_metrics['mean_benchmark_acc']:.4f} "
        f"resumed={best_metrics.get('resumed_from_checkpoint')}",
        flush=True,
    )

    accepted = 0
    for index in range(args.iterations):
        proposal, mutation = propose(best, index)
        run_id = f"{session}-trial{index + 1:02d}-{short_hash(proposal)}"
        paper_line = paper_attempt_line(paper_ids)
        print(f"Evaluating {run_id}: {mutation}", flush=True)
        if paper_line:
            print(f"  paper {paper_line}", flush=True)
            mutation = f"{mutation} || {paper_line}"
        try:
            # Warm-start from current best weights when architecture is unchanged.
            metrics = evaluate(
                args.candidate,
                proposal,
                run_id,
                args.artifacts,
                args.budget_seconds,
                args.training_backend,
                args.benchmark_limit,
                args.bpb_batches,
                args.checkpoint_dir,
                paper_ids,
            )
        except Exception as error:
            write_json(args.candidate, best)
            append_crash(args.ledger, run_id, mutation, proposal, error, paper_ids)
            print(f"discard {run_id}: evaluator error: {error}", flush=True)
            continue

        improved = metrics["score"] < best_metrics["score"] - MIN_IMPROVEMENT
        status = "keep" if improved else "discard"
        append_ledger(args.ledger, metrics, status, mutation)
        record_our_run(args.scores, metrics, status)
        print(
            f"{status} score={metrics['score']:.6f} "
            f"bpb={metrics['bpb']:.6f} "
            f"acc={metrics['mean_benchmark_acc']:.4f}",
            flush=True,
        )
        if improved:
            best = proposal
            best_metrics = metrics
            accepted += 1
            write_json(args.best, best)
            promote_checkpoint(args.artifacts / run_id, args.checkpoint_dir)
            if paper_ids:
                for paper_id in paper_ids:
                    subprocess.run(
                        [
                            sys.executable,
                            str(ROOT / "papers.py"),
                            "mark",
                            paper_id,
                            "--status",
                            "applied",
                            "--run-id",
                            run_id,
                        ],
                        cwd=ROOT,
                        check=False,
                    )
        write_json(args.candidate, best)

    summary = {
        "accepted_trials": accepted,
        "attempted_trials": args.iterations,
        "best_score": best_metrics["score"],
        "best_bpb": best_metrics["bpb"],
        "best_mean_benchmark_acc": best_metrics["mean_benchmark_acc"],
        "best_validation_loss": best_metrics["validation_loss"],
        "best_candidate": best,
        "paper_ids": paper_ids,
        "checkpoint": str(args.checkpoint_dir.resolve()),
        "ledger": str(args.ledger.resolve()),
    }
    summary_path = args.artifacts / f"{session}-summary.json"
    summary_path.write_text(json.dumps(summary, indent=2) + "\n", encoding="utf-8")
    print("AUTORESEARCH_SUMMARY " + json.dumps(summary, separators=(",", ":")))


if __name__ == "__main__":
    main()
