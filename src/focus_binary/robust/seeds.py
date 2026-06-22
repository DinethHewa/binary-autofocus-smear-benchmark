from __future__ import annotations

import os
import random

import numpy as np

from focus_binary.utils.logging import get_logger
from focus_binary.utils.seed import set_global_seed

logger = get_logger(__name__)

try:
    import tensorflow as tf
except Exception:  # pragma: no cover
    tf = None


def set_global_determinism(seed: int) -> None:
    """Enable as much determinism as possible across Python, NumPy, and TensorFlow."""

    os.environ["PYTHONHASHSEED"] = str(seed)
    random.seed(seed)
    np.random.seed(seed)

    os.environ["TF_DETERMINISTIC_OPS"] = "1"
    os.environ["TF_CUDNN_DETERMINISTIC"] = "1"

    if tf is None:
        logger.warning("TensorFlow not installed; only Python/NumPy determinism applied.")
        return

    try:
        tf.keras.utils.set_random_seed(seed)
    except Exception:
        tf.random.set_seed(seed)

    enable_det = getattr(tf.config.experimental, "enable_op_determinism", None)
    if enable_det is not None:
        try:
            enable_det()
        except Exception as exc:
            logger.warning("Could not enable deterministic ops", extra={"error": str(exc)})
    else:
        logger.warning("TensorFlow does not support enable_op_determinism in this version.")

    if tf.config.list_physical_devices("GPU"):
        logger.info("Determinism requested; some GPU kernels may still be nondeterministic.")


def set_seeds(seed: int, deterministic: bool = False) -> None:
    """Backward-compatible seed setter."""
    set_global_seed(seed, deterministic=deterministic)
