from __future__ import annotations

from typing import Iterable, List, Optional, Sequence

import numpy as np

try:
    import tensorflow as tf
except Exception:  # pragma: no cover
    tf = None


def _require_tf():
    if tf is None:
        raise ImportError("TensorFlow is required for ViT attention rollout.")


if tf is not None:
    class RecordingMultiHeadAttention(tf.keras.layers.MultiHeadAttention):
        """MultiHeadAttention that stores attention scores."""

        def __init__(self, *args, **kwargs):
            super().__init__(*args, **kwargs)
            self.last_attention_scores: Optional[tf.Tensor] = None

        def call(self, query, value, key=None, attention_mask=None, training=None, use_causal_mask=False):
            output, scores = super().call(
                query,
                value,
                key=key,
                attention_mask=attention_mask,
                return_attention_scores=True,
                training=training,
                use_causal_mask=use_causal_mask,
            )
            self.last_attention_scores = scores
            return output


    def _clone_layer(layer):
        if isinstance(layer, tf.keras.layers.MultiHeadAttention):
            return RecordingMultiHeadAttention.from_config(layer.get_config())
        return layer.__class__.from_config(layer.get_config())


    def _build_attention_wrapper(model) -> tf.keras.Model:
        try:
            wrapper = tf.keras.models.clone_model(model, clone_function=_clone_layer)
            wrapper.set_weights(model.get_weights())
        except Exception as exc:  # pragma: no cover
            raise RuntimeError(f"Failed to build attention wrapper: {exc}") from exc
        return wrapper
else:  # pragma: no cover
    RecordingMultiHeadAttention = None
    _clone_layer = None
    _build_attention_wrapper = None


def extract_attention_matrices(model, image_tensor: "tf.Tensor") -> List[np.ndarray]:
    """Extract attention matrices from a model using RecordingMultiHeadAttention."""
    _require_tf()

    image_tensor = tf.convert_to_tensor(image_tensor, dtype=tf.float32)
    if image_tensor.ndim == 3:
        image_tensor = tf.expand_dims(image_tensor, axis=0)

    layers = list(model.submodules) if hasattr(model, "submodules") else model.layers
    attn_layers = [layer for layer in layers if hasattr(layer, "last_attention_scores")]
    if not attn_layers:
        if _build_attention_wrapper is None:
            raise RuntimeError("Cannot build attention wrapper without TensorFlow.")
        model = _build_attention_wrapper(model)
        layers = list(model.submodules) if hasattr(model, "submodules") else model.layers
        attn_layers = [layer for layer in layers if hasattr(layer, "last_attention_scores")]

    _ = model(image_tensor, training=False)

    mats = []
    for layer in attn_layers:
        scores = getattr(layer, "last_attention_scores", None)
        if scores is not None:
            mats.append(scores.numpy())

    if not mats:
        raise ValueError("No attention scores found; ensure model uses MultiHeadAttention layers.")
    return mats


def _sqrt_int(val: int) -> Optional[int]:
    root = int(np.sqrt(val))
    if root * root == val:
        return root
    return None

def attention_rollout(
    attention_mats: Iterable[np.ndarray],
    head_fusion: str = "mean",
    discard_ratio: float = 0.0,
    start_layer: int = 0,
) -> np.ndarray:
    """Compute attention rollout from a list of attention matrices.

    Each matrix is expected to have shape: (batch, heads, tokens, tokens).
    """
    mats = list(attention_mats)
    if not mats:
        raise ValueError("attention_mats must be a non-empty list.")

    fused_layers: List[np.ndarray] = []
    for mat in mats[start_layer:]:
        if head_fusion == "mean":
            fused = mat.mean(axis=1)
        elif head_fusion == "max":
            fused = mat.max(axis=1)
        else:
            raise ValueError("head_fusion must be 'mean' or 'max'")

        if discard_ratio > 0:
            flat = fused.reshape(fused.shape[0], -1)
            k = int(flat.shape[1] * discard_ratio)
            if k > 0:
                idx = np.argpartition(flat, k, axis=1)[:, :k]
                flat[np.arange(flat.shape[0])[:, None], idx] = 0.0
                fused = flat.reshape(fused.shape)

        eye = np.eye(fused.shape[-1], dtype=fused.dtype)
        fused = fused + eye[None, ...]
        fused = fused / fused.sum(axis=-1, keepdims=True)
        fused_layers.append(fused)

    rollout = fused_layers[0]
    for mat in fused_layers[1:]:
        rollout = mat @ rollout
    tokens = rollout.shape[-1]
    cls_tokens = tokens - 1
    grid = _sqrt_int(cls_tokens)
    if grid is not None:
        maps = rollout[:, 0, 1:]
    else:
        grid = _sqrt_int(tokens)
        if grid is None:
            raise ValueError("Token count is not a perfect square; cannot reshape to grid.")
        maps = rollout.mean(axis=1)
    return maps.reshape(maps.shape[0], grid, grid)


def cls_attention(rollout: np.ndarray, cls_index: int = 0) -> np.ndarray:
    """Return attention from the CLS token to all tokens."""
    if rollout.ndim != 3:
        raise ValueError("rollout must have shape (batch, tokens, tokens)")
    return rollout[:, cls_index, :]


def upscale_to_image(attn_map: np.ndarray, target_size: Sequence[int]) -> np.ndarray:
    """Upscale a patch grid attention map to image size."""
    _require_tf()
    attn = np.asarray(attn_map).astype(np.float32)
    if attn.ndim == 2:
        attn = attn[None, ..., None]
    elif attn.ndim == 3:
        attn = attn[..., None]
    resized = tf.image.resize(attn, target_size, method="bilinear")
    resized = tf.squeeze(resized, axis=-1)
    max_val = tf.reduce_max(resized, axis=(1, 2), keepdims=True)
    resized = tf.where(max_val > 0, resized / max_val, resized)
    return resized.numpy()[0]
