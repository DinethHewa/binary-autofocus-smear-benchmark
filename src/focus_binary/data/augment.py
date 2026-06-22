from __future__ import annotations

from typing import Callable, Optional

from focus_binary.utils.logging import get_logger

logger = get_logger(__name__)

try:
    import tensorflow as tf
except Exception:  # pragma: no cover
    tf = None


def _identity(x):
    return x


def build_augmenter(enabled: bool = True, seed: Optional[int] = None) -> Callable:
    """Return a lightweight augmentation function suitable for tf.data."""

    if tf is None:
        logger.warning("TensorFlow not installed; returning identity augmentation")
        return _identity

    if not enabled:
        return _identity

    def _fn(img):
        img = tf.image.random_flip_left_right(img, seed=seed)
        # Keep rotation minimal: 0/90/180/270 degrees.
        k = tf.random.uniform([], minval=0, maxval=4, dtype=tf.int32, seed=seed)
        img = tf.image.rot90(img, k)
        img = tf.image.random_contrast(img, lower=0.9, upper=1.1, seed=seed)
        img = tf.clip_by_value(img, 0.0, 1.0)
        return img

    return _fn


def basic_augment() -> Callable:
    """Backward-compatible alias for the default augmenter."""
    return build_augmenter(enabled=True)
