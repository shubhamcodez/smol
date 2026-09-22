"""Best-model checkpoint helpers for resumable autoresearch sessions."""

from __future__ import annotations

import json
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

import torch


ROOT = Path(__file__).resolve().parent
DEFAULT_BEST = ROOT / "checkpoints" / "best"


def checkpoint_paths(root: Path | None = None) -> dict[str, Path]:
    base = root or DEFAULT_BEST
    return {
        "dir": base,
        "model": base / "model.pt",
        "candidate": base / "candidate.json",
        "metrics": base / "metrics.json",
        "meta": base / "meta.json",
    }


def save_best_checkpoint(
    model: torch.nn.Module,
    candidate: dict[str, Any],
    metrics: dict[str, Any],
    *,
    root: Path | None = None,
    paper_ids: list[str] | None = None,
) -> Path:
    paths = checkpoint_paths(root)
    paths["dir"].mkdir(parents=True, exist_ok=True)
    payload = {
        "model": model.state_dict(),
        "config": model.config.__dict__,
        "candidate": candidate,
    }
    torch.save(payload, paths["model"])
    paths["candidate"].write_text(json.dumps(candidate, indent=2) + "\n", encoding="utf-8")
    paths["metrics"].write_text(json.dumps(metrics, indent=2) + "\n", encoding="utf-8")
    meta = {
        "saved_at_utc": datetime.now(timezone.utc).isoformat(),
        "run_id": metrics.get("run_id"),
        "score": metrics.get("score"),
        "bpb": metrics.get("bpb"),
        "mean_benchmark_acc": metrics.get("mean_benchmark_acc"),
        "paper_ids": paper_ids or metrics.get("paper_ids") or [],
        "parameter_count": metrics.get("parameter_count"),
    }
    paths["meta"].write_text(json.dumps(meta, indent=2) + "\n", encoding="utf-8")
    return paths["dir"]


def load_checkpoint_bundle(root: Path | None = None) -> dict[str, Any]:
    paths = checkpoint_paths(root)
    if not paths["model"].exists():
        raise FileNotFoundError(f"missing checkpoint at {paths['model']}")
    bundle = torch.load(paths["model"], map_location="cpu", weights_only=False)
    meta = {}
    if paths["meta"].exists():
        meta = json.loads(paths["meta"].read_text(encoding="utf-8"))
    candidate = bundle.get("candidate")
    if candidate is None and paths["candidate"].exists():
        candidate = json.loads(paths["candidate"].read_text(encoding="utf-8"))
    return {
        "state_dict": bundle["model"],
        "config": bundle.get("config"),
        "candidate": candidate,
        "meta": meta,
        "paths": paths,
    }
