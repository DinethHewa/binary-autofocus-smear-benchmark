from __future__ import annotations

import time
from typing import Tuple

import numpy as np

from focus_binary.utils.logging import get_logger

logger = get_logger(__name__)

try:
    import tensorflow as tf
except Exception:  # pragma: no cover
    tf = None


def count_params(model) -> int:
    if model is None:
        return 0
    if hasattr(model, "count_params"):
        return int(model.count_params())
    return 0


def hardware_string() -> str:
    if tf is None:
        return "tensorflow_not_installed"
    devices = tf.config.list_physical_devices()
    if not devices:
        return "cpu"
    parts = []
    for dev in devices:
        dtype = getattr(dev, "device_type", "UNKNOWN")
        name = getattr(dev, "name", "")
        parts.append(f"{dtype}:{name}")
    return ", ".join(parts)


def _parse_input_size(input_size: int | Tuple[int, int]) -> Tuple[int, int]:
    if isinstance(input_size, int):
        return (input_size, input_size)
    if isinstance(input_size, (list, tuple)) and len(input_size) == 2:
        return (int(input_size[0]), int(input_size[1]))
    raise ValueError("input_size must be int or (h, w)")


def measure_latency(
    model,
    input_size: int | Tuple[int, int],
    batch_size: int = 1,
    warmup: int = 20,
    iters: int = 100,
) -> Tuple[float, float]:
    if tf is None:
        raise ImportError("TensorFlow is required for latency measurement.")

    height, width = _parse_input_size(input_size)
    dummy = tf.zeros((batch_size, height, width, 3), dtype=tf.float32)

    @tf.function
    def _infer(x):
        return model(x, training=False)

    logger.info(
        "measuring latency",
        extra={
            "batch_size": batch_size,
            "input_size": f"{height}x{width}",
            "device": hardware_string(),
        },
    )

    for _ in range(warmup):
        out = _infer(dummy)
        _ = out.numpy() if hasattr(out, "numpy") else out

    times = []
    for _ in range(iters):
        start = time.perf_counter()
        out = _infer(dummy)
        _ = out.numpy() if hasattr(out, "numpy") else out
        times.append(time.perf_counter() - start)

    times = np.asarray(times, dtype=float)
    mean_ms = float(times.mean() * 1000.0)
    p95_ms = float(np.percentile(times, 95) * 1000.0)
    return mean_ms, p95_ms
