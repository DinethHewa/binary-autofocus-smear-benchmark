from __future__ import annotations

from typing import List

import tensorflow as tf

from focus_binary.features import focus_measures


_MEASURE_MAP = {
    "lapvar": focus_measures.variance_of_laplacian,
    "tenengrad": focus_measures.tenengrad,
    "brenner": focus_measures.brenner,
    "sml": focus_measures.sml,
}


def _resolve_measures(enabled_measures: List[str]) -> List[str]:
    measures = [m.strip().lower() for m in enabled_measures if m.strip()]
    if not measures:
        raise ValueError("enabled_measures must be non-empty")
    for name in measures:
        if name not in _MEASURE_MAP:
            raise KeyError(f"Unknown focus measure '{name}'")
    return measures


def compute_focus_vector(img: tf.Tensor, enabled_measures: List[str]) -> tf.Tensor:
    measures = _resolve_measures(enabled_measures)
    values = [tf.reshape(_MEASURE_MAP[name](img), []) for name in measures]
    return tf.stack(values, axis=0)


def batch_compute_focus_vectors(images_batch: tf.Tensor, enabled_measures: List[str]) -> tf.Tensor:
    measures = _resolve_measures(enabled_measures)
    img = tf.convert_to_tensor(images_batch, dtype=tf.float32)

    def _single(image: tf.Tensor) -> tf.Tensor:
        values = [tf.reshape(_MEASURE_MAP[name](image), []) for name in measures]
        return tf.stack(values, axis=0)

    return tf.map_fn(_single, img, fn_output_signature=tf.float32)
