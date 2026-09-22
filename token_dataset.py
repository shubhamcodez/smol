"""Memory-mapped FineWeb token shards for pretraining."""

from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path

import numpy as np
import torch


@dataclass(frozen=True)
class ShardPaths:
    train: Path
    val: Path
    meta: Path


def default_shard_paths(root: Path | None = None) -> ShardPaths:
    base = (root or Path(__file__).resolve().parent) / "data" / "shards"
    return ShardPaths(
        train=base / "train.bin",
        val=base / "val.bin",
        meta=base / "meta.json",
    )


class TokenShard:
    """Random contiguous windows from a uint16 token memmap."""

    def __init__(self, path: Path) -> None:
        if not path.exists():
            raise FileNotFoundError(
                f"missing token shard {path}; run prepare_data.py first"
            )
        self.path = path
        self.data = np.memmap(path, dtype=np.uint16, mode="r")

    def __len__(self) -> int:
        return int(self.data.shape[0])

    def sample_batch(
        self,
        batch_size: int,
        seq_len: int,
        generator: torch.Generator,
        device: torch.device,
    ) -> tuple[torch.Tensor, torch.Tensor]:
        if len(self) < seq_len + 1:
            raise ValueError(f"shard {self.path} is shorter than seq_len+1")
        high = len(self) - seq_len
        starts = torch.randint(0, high, (batch_size,), generator=generator)
        sequences = []
        for start in starts.tolist():
            window = np.asarray(self.data[start : start + seq_len + 1], dtype=np.int64)
            sequences.append(torch.from_numpy(window))
        batch = torch.stack(sequences, dim=0).to(device=device, non_blocking=True)
        return batch[:, :-1].contiguous(), batch[:, 1:].contiguous()
