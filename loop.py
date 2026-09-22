"""A bounded hill-climbing research loop with an append-only experiment ledger."""

from __future__ import annotations

import argparse
import csv
import hashlib
import json
import shutil
import subprocess
import sys
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

from checkpointing import checkpoint_paths
from grok_propose import find_method_paper, load_api_key, propose_architecture
from papers import append_row, known_ids, utc_now
from scores import record_our_run, seed_references


ROOT = Path(__file__).resolve().parent
MODEL_PATH = ROOT / "model.py"
MODEL_KEEP = ROOT / "checkpoints" / "model_keep.py"
MODEL_BASELINE = ROOT / "checkpoints" / "model_baseline.py"
MIN_IMPROVEMENT = 1e-2
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
    benchmark_seed: int,
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
        "--benchmark-seed",
        str(benchmark_seed),
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


def snapshot_architecture(resume: bool) -> None:
    MODEL_KEEP.parent.mkdir(parents=True, exist_ok=True)
    if resume and MODEL_KEEP.exists():
        shutil.copy2(MODEL_KEEP, MODEL_PATH)
        print(f"Resuming architecture from {MODEL_KEEP}", flush=True)
    else:
        shutil.copy2(MODEL_PATH, MODEL_KEEP)
    if not MODEL_BASELINE.exists():
        shutil.copy2(MODEL_PATH, MODEL_BASELINE)


def restore_architecture() -> None:
    if MODEL_KEEP.exists():
        shutil.copy2(MODEL_KEEP, MODEL_PATH)


def accept_architecture() -> None:
    MODEL_KEEP.parent.mkdir(parents=True, exist_ok=True)
    shutil.copy2(MODEL_PATH, MODEL_KEEP)


def architecture_imports() -> None:
    completed = subprocess.run(
        [
            sys.executable,
            "-c",
            "from model import ModelConfig, TransformerLM; "
            "TransformerLM(ModelConfig(n_layer=2, n_embd=128, n_head=4, n_kv_head=2, "
            "intermediate_size=256, max_seq_len=128))",
        ],
        cwd=ROOT,
        text=True,
        capture_output=True,
        timeout=180,
        check=False,
    )
    if completed.returncode != 0:
        detail = (completed.stderr or completed.stdout)[-1500:]
        raise RuntimeError(detail)


def tail_text(path: Path, limit: int = 15) -> str:
    if not path.exists():
        return ""
    return "\n".join(path.read_text(encoding="utf-8").splitlines()[-limit:])


def remember_paper(paper_id: str, title: str, idea: str) -> None:
    if not paper_id or paper_id in known_ids(ROOT / "papers.tsv"):
        return
    append_row(
        ROOT / "papers.tsv",
        {
            "timestamp_utc": utc_now(),
            "arxiv_id": paper_id,
            "title": title,
            "venue": "",
            "year": "",
            "status": "read",
            "idea": idea,
            "run_id": "",
            "notes": "proposed by grok",
            "url": f"https://huggingface.co/papers/{paper_id}",
        },
    )


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
    parser.add_argument("--scores", type=Path, default=ROOT / "scores.tsv")
    parser.add_argument("--iterations", type=int, default=4)
    parser.add_argument("--budget-seconds", type=float, default=30.0)
    parser.add_argument("--training-backend", choices=["auto", "cpu", "cuda"], default="auto")
    parser.add_argument("--benchmark-limit", type=int, default=64)
    parser.add_argument("--bpb-batches", type=int, default=32)
    parser.add_argument("--reset", action="store_true",
                        help="Delete best.json and results.tsv before starting")
    args = parser.parse_args()

    if args.iterations < 0:
        raise ValueError("iterations must be non-negative")
    if args.budget_seconds <= 0:
        raise ValueError("budget-seconds must be positive")
    args.artifacts.mkdir(parents=True, exist_ok=True)

    seed_references(args.scores)
    api_key = load_api_key()
    if not api_key:
        raise SystemExit("XAI_API_KEY is missing from .env")
    snapshot_architecture(args.resume)
    print(
        "Each trial searches for a method, then rewrites model.py from that paper.",
        flush=True,
    )

    if args.reset:
        if args.ledger.exists():
            args.ledger.unlink()
        if args.best.exists():
            args.best.unlink()

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
    benchmark_seed = int(hashlib.sha256(session.encode("utf-8")).hexdigest()[:8], 16)
    # Every arm in this session starts from the same weights and the same 64-per-task draw.
    session_init: Path | None = None
    if init_checkpoint is not None:
        session_init = args.artifacts / f"{session}-init"
        if session_init.exists():
            shutil.rmtree(session_init)
        shutil.copytree(init_checkpoint, session_init)
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
        session_init,
        [],
        benchmark_seed,
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
        trial_papers: list[str] = []
        try:
            found = find_method_paper(
                api_key,
                results_tail=tail_text(args.ledger),
                papers_tail=tail_text(ROOT / "papers.tsv"),
                known=known_ids(ROOT / "papers.tsv"),
            )
            print(
                f"search {found['query']!r} -> {found['arxiv_id']} {found['title']}",
                flush=True,
            )
            remember_paper(found["arxiv_id"], found["title"], found.get("summary") or found["query"])
            arch = propose_architecture(
                api_key,
                MODEL_PATH.read_text(encoding="utf-8"),
                found,
                results_tail=tail_text(args.ledger),
                candidate_json=json.dumps(best, indent=2),
                best_score=float(best_metrics["score"]),
            )
            MODEL_PATH.write_text(arch.source, encoding="utf-8")
            architecture_imports()
        except Exception as error:
            restore_architecture()
            run_id = f"{session}-trial{index + 1:02d}-search"
            append_crash(args.ledger, run_id, "method search", best, error, [])
            print(f"discard {run_id}: {error}", flush=True)
            continue
        proposal = best
        mutation = arch.idea
        paper_line = arch.paper_line
        trial_papers = [arch.paper_id]
        remember_paper(arch.paper_id, arch.paper_title, arch.summary)
        run_id = f"{session}-trial{index + 1:02d}-{short_hash(proposal)}"
        print(f"Evaluating {run_id}: {mutation}", flush=True)
        if paper_line:
            print(f"  paper {paper_line}", flush=True)
            mutation = f"{mutation} || {paper_line}"
        try:
            metrics = evaluate(
                args.candidate,
                proposal,
                run_id,
                args.artifacts,
                args.budget_seconds,
                args.training_backend,
                args.benchmark_limit,
                args.bpb_batches,
                session_init,
                trial_papers,
                benchmark_seed,
            )
        except Exception as error:
            write_json(args.candidate, best)
            restore_architecture()
            append_crash(args.ledger, run_id, mutation, proposal, error, trial_papers)
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
            accept_architecture()
            for paper_id in trial_papers:
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
        else:
            restore_architecture()
            for paper_id in trial_papers:
                subprocess.run(
                    [
                        sys.executable,
                        str(ROOT / "papers.py"),
                        "mark",
                        paper_id,
                        "--status",
                        "rejected",
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
        "checkpoint": str(args.checkpoint_dir.resolve()),
        "ledger": str(args.ledger.resolve()),
    }
    summary_path = args.artifacts / f"{session}-summary.json"
    summary_path.write_text(json.dumps(summary, indent=2) + "\n", encoding="utf-8")
    print("AUTORESEARCH_SUMMARY " + json.dumps(summary, separators=(",", ":")))


if __name__ == "__main__":
    main()
