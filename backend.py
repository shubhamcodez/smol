"""Runtime discovery shared by the autoresearch tools."""

from __future__ import annotations

from dataclasses import dataclass


@dataclass(frozen=True)
class RuntimeInventory:
    cuda: bool


def detect_runtime() -> RuntimeInventory:
    cuda = False
    try:
        import torch

        cuda = bool(torch.cuda.is_available())
    except (ImportError, OSError):
        pass
    return RuntimeInventory(cuda=cuda)
