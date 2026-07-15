"""Runtime discovery shared by the autoresearch tools.

The important distinction is intentional: Qualcomm QNN/HTP is an inference
backend, while CUDA can be both a training and inference backend.
"""

from __future__ import annotations

from dataclasses import asdict, dataclass
from typing import Any


_QNN_REGISTERED = False


@dataclass(frozen=True)
class RuntimeInventory:
    python_machine: str
    qnn_npu: bool
    qnn_gpu: bool
    cuda: bool
    cuda_device: str | None
    torch_version: str | None
    onnxruntime_version: str | None
    qnn_plugin_version: str | None

    def to_dict(self) -> dict[str, Any]:
        return asdict(self)


def register_qnn():
    """Register the dynamically installed QNN execution provider plugin."""
    global _QNN_REGISTERED

    import onnxruntime as ort
    import onnxruntime_qnn as qnn

    if not _QNN_REGISTERED:
        ort.register_execution_provider_library(
            "QNNExecutionProvider", qnn.get_library_path()
        )
        _QNN_REGISTERED = True
    return ort, qnn


def qnn_devices() -> list[Any]:
    try:
        ort, _ = register_qnn()
    except (ImportError, OSError, RuntimeError):
        return []
    return [device for device in ort.get_ep_devices() if device.ep_name == "QNNExecutionProvider"]


def qnn_npu_devices() -> list[Any]:
    try:
        ort, _ = register_qnn()
    except (ImportError, OSError, RuntimeError):
        return []
    return [
        device
        for device in ort.get_ep_devices()
        if device.ep_name == "QNNExecutionProvider"
        and device.device.type == ort.OrtHardwareDeviceType.NPU
    ]


def detect_runtime() -> RuntimeInventory:
    import platform

    qnn_npu = False
    qnn_gpu = False
    ort_version = None
    qnn_version = None
    try:
        ort, qnn = register_qnn()
        ort_version = ort.__version__
        qnn_version = getattr(qnn, "__version__", None)
        for ep_device in ort.get_ep_devices():
            if ep_device.ep_name != "QNNExecutionProvider":
                continue
            qnn_npu |= ep_device.device.type == ort.OrtHardwareDeviceType.NPU
            qnn_gpu |= ep_device.device.type == ort.OrtHardwareDeviceType.GPU
    except (ImportError, OSError, RuntimeError):
        try:
            import onnxruntime as ort

            ort_version = ort.__version__
        except ImportError:
            pass

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
        qnn_npu=qnn_npu,
        qnn_gpu=qnn_gpu,
        cuda=cuda,
        cuda_device=cuda_name,
        torch_version=torch_version,
        onnxruntime_version=ort_version,
        qnn_plugin_version=qnn_version,
    )
