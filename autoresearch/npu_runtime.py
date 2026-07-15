"""Export a small trained MLP and benchmark it on Qualcomm's Hexagon NPU."""

from __future__ import annotations

import time
from pathlib import Path
from typing import Any, Mapping

import numpy as np

from backend import qnn_npu_devices, register_qnn


INPUT_NAME = "features"
OUTPUT_NAME = "logits"
STATIC_BATCH = 32


class ArrayCalibrationReader:
    def __init__(self, batches: list[np.ndarray]) -> None:
        self._batches = iter({INPUT_NAME: batch.astype(np.float32)} for batch in batches)

    def get_next(self) -> dict[str, np.ndarray] | None:
        return next(self._batches, None)


def export_quantized_mlp(
    weights: Mapping[str, np.ndarray],
    calibration: np.ndarray,
    output_dir: Path,
) -> tuple[Path, Path]:
    """Write float and QDQ uint8/int8 ONNX models with a static NPU shape."""
    import onnx
    from onnx import TensorProto, helper, numpy_helper
    from onnxruntime.quantization import QuantFormat, QuantType, quantize_static

    output_dir.mkdir(parents=True, exist_ok=True)
    float_path = output_dir / "candidate_float.onnx"
    quantized_path = output_dir / "candidate_qdq.onnx"

    input_dim, hidden_dim = weights["w1"].shape
    _, output_dim = weights["w2"].shape
    graph = helper.make_graph(
        [
            helper.make_node("MatMul", [INPUT_NAME, "w1"], ["hidden_linear"]),
            helper.make_node("Add", ["hidden_linear", "b1"], ["hidden_bias"]),
            helper.make_node("Relu", ["hidden_bias"], ["hidden"]),
            helper.make_node("MatMul", ["hidden", "w2"], ["output_linear"]),
            helper.make_node("Add", ["output_linear", "b2"], [OUTPUT_NAME]),
        ],
        "autoresearch_candidate_mlp",
        [helper.make_tensor_value_info(INPUT_NAME, TensorProto.FLOAT, [STATIC_BATCH, input_dim])],
        [helper.make_tensor_value_info(OUTPUT_NAME, TensorProto.FLOAT, [STATIC_BATCH, output_dim])],
        [
            numpy_helper.from_array(weights["w1"].astype(np.float32), "w1"),
            numpy_helper.from_array(weights["b1"].astype(np.float32), "b1"),
            numpy_helper.from_array(weights["w2"].astype(np.float32), "w2"),
            numpy_helper.from_array(weights["b2"].astype(np.float32), "b2"),
        ],
    )
    model = helper.make_model(
        graph,
        producer_name="agi-autoresearch",
        opset_imports=[helper.make_opsetid("", 17)],
    )
    onnx.checker.check_model(model)
    onnx.save(model, float_path)

    rows = calibration[: STATIC_BATCH * 4]
    if len(rows) < STATIC_BATCH:
        raise ValueError(f"calibration needs at least {STATIC_BATCH} rows")
    batches = [
        rows[start : start + STATIC_BATCH]
        for start in range(0, len(rows) - STATIC_BATCH + 1, STATIC_BATCH)
    ]
    quantize_static(
        str(float_path),
        str(quantized_path),
        ArrayCalibrationReader(batches),
        quant_format=QuantFormat.QDQ,
        activation_type=QuantType.QUInt8,
        weight_type=QuantType.QInt8,
        extra_options={"WeightSymmetric": True},
    )
    onnx.checker.check_model(onnx.load(quantized_path))
    return float_path, quantized_path


def _device_description(ep_device: Any) -> dict[str, Any]:
    hardware = ep_device.device
    kind = getattr(hardware.type, "name", str(hardware.type).split(".")[-1])
    vendor = str(getattr(hardware, "vendor", "")).strip() or "Qualcomm"
    return {
        "execution_provider": ep_device.ep_name,
        "hardware_type": kind,
        "vendor": vendor,
        "device_id": str(getattr(hardware, "device_id", "")),
    }


def benchmark_qnn_npu(
    model_path: Path,
    sample: np.ndarray,
    warmup: int = 10,
    iterations: int = 100,
) -> dict[str, Any]:
    """Run with an NPU EP device and forbid CPU execution-provider fallback."""
    ort, qnn = register_qnn()
    devices = qnn_npu_devices()
    if not devices:
        raise RuntimeError("QNNExecutionProvider did not expose an NPU device")

    options = {
        "backend_path": qnn.get_qnn_htp_path(),
        "htp_performance_mode": "burst",
    }
    session_options = ort.SessionOptions()
    session_options.log_severity_level = 3
    session_options.add_session_config_entry("session.disable_cpu_ep_fallback", "1")
    session_options.add_provider_for_devices(devices, options)
    compile_start = time.perf_counter()
    session = ort.InferenceSession(str(model_path), sess_options=session_options)
    compile_seconds = time.perf_counter() - compile_start

    feed = {INPUT_NAME: sample.astype(np.float32, copy=False)}
    for _ in range(warmup):
        output = session.run([OUTPUT_NAME], feed)[0]
    start = time.perf_counter()
    for _ in range(iterations):
        output = session.run([OUTPUT_NAME], feed)[0]
    elapsed = time.perf_counter() - start

    return {
        "backend": "qnn-npu",
        "device": _device_description(devices[0]),
        "cpu_ep_fallback_disabled": True,
        "provider_options": options,
        "compile_seconds": compile_seconds,
        "latency_ms": elapsed * 1000.0 / iterations,
        "iterations": iterations,
        "output_shape": list(output.shape),
        "output_checksum": float(output.sum()),
        "onnxruntime_version": ort.__version__,
        "qnn_plugin_version": getattr(qnn, "__version__", None),
    }


def benchmark_ort_cpu(
    model_path: Path,
    sample: np.ndarray,
    warmup: int = 10,
    iterations: int = 100,
) -> dict[str, Any]:
    import onnxruntime as ort

    session = ort.InferenceSession(str(model_path), providers=["CPUExecutionProvider"])
    feed = {INPUT_NAME: sample.astype(np.float32, copy=False)}
    for _ in range(warmup):
        output = session.run([OUTPUT_NAME], feed)[0]
    start = time.perf_counter()
    for _ in range(iterations):
        output = session.run([OUTPUT_NAME], feed)[0]
    elapsed = time.perf_counter() - start
    return {
        "backend": "onnx-cpu",
        "device": {"execution_provider": "CPUExecutionProvider", "hardware_type": "CPU"},
        "cpu_ep_fallback_disabled": False,
        "latency_ms": elapsed * 1000.0 / iterations,
        "iterations": iterations,
        "output_shape": list(output.shape),
        "output_checksum": float(output.sum()),
        "onnxruntime_version": ort.__version__,
    }
