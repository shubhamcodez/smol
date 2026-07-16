"""Runtime discovery shared by the autoresearch tools."""

from __future__ import annotations

from dataclasses import asdict, dataclass
from typing import Any


@dataclass(frozen=True)
class RuntimeInventory:
    python_machine: str
    cuda: bool
    cuda_device: str | None
    torch_version: str | None

    def to_dict(self) -> dict[str, Any]:
        return asdict(self)


def detect_runtime() -> RuntimeInventory:
    import platform

    cuda = False
    cuda_name = None
    torch_version = None
    try:
        import torch

        torch_version = torch.__version__
        cuda = torch.cuda.is_available()
        if cuda:
            cuda_name = torch.cuda.get_device_name(0)
    except (ImportError, OSError):
        pass

    return RuntimeInventory(
        python_machine=platform.machine(),
        cuda=cuda,
        cuda_device=cuda_name,
        torch_version=torch_version,
    )
