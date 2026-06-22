from __future__ import annotations

from typing import Tuple

from focus_binary.utils.logging import get_logger

logger = get_logger(__name__)

try:
    import tensorflow as tf
except Exception:  # pragma: no cover
    tf = None


def _transformer_block(x, embed_dim: int, num_heads: int, mlp_dim: int, dropout: float):
    y = tf.keras.layers.LayerNormalization(epsilon=1e-6)(x)
    y = tf.keras.layers.MultiHeadAttention(
        num_heads=num_heads,
        key_dim=embed_dim // num_heads,
        dropout=dropout,
    )(y, y)
    if dropout and dropout > 0:
        y = tf.keras.layers.Dropout(dropout)(y)
    x = tf.keras.layers.Add()([x, y])

    y = tf.keras.layers.LayerNormalization(epsilon=1e-6)(x)
    y = tf.keras.layers.Dense(mlp_dim, activation=tf.nn.gelu)(y)
    if dropout and dropout > 0:
        y = tf.keras.layers.Dropout(dropout)(y)
    y = tf.keras.layers.Dense(embed_dim, activation=tf.nn.gelu)(y)
    return tf.keras.layers.Add()([x, y])


def build_hybrid_vit(
    input_shape: Tuple[int, int, int] = (224, 224, 3),
    num_classes: int = 1,
    stem_type: str = "cnn",  # "cnn" or "mobilenetv2"
    embed_dim: int = 96,
    transformer_depth: int = 2,
    num_heads: int = 4,
    mlp_dim: int = 192,
    dropout: float = 0.1,
    patch_size: int = 4,
    stem_blocks: int = 2,
    weights: str | None = "imagenet",
):
    """Efficient CNN stem + shallow ViT encoder."""

    if tf is None:
        logger.warning("TensorFlow not installed; returning placeholder model spec for hybrid_vit")
        return {
            "model": "hybrid_vit",
            "stem_type": stem_type,
            "transformer_depth": transformer_depth,
            "embed_dim": embed_dim,
        }

    if embed_dim < 64 or embed_dim > 192:
        raise ValueError("embed_dim must be in [64, 192]")
    if transformer_depth < 1 or transformer_depth > 4:
        raise ValueError("transformer_depth must be in [1, 4]")
    if stem_type not in {"cnn", "mobilenetv2"}:
        raise ValueError("stem_type must be 'cnn' or 'mobilenetv2'")
    if embed_dim % num_heads != 0:
        raise ValueError("embed_dim must be divisible by num_heads")
    if stem_type == "cnn" and stem_blocks not in (2, 3):
        raise ValueError("stem_blocks must be 2 or 3 for cnn stem")

    inputs = tf.keras.Input(shape=input_shape)
    x = inputs

    if stem_type == "mobilenetv2":
        base = tf.keras.applications.MobileNetV2(include_top=False, weights=weights, input_shape=input_shape)
        base.trainable = False
        x = base(x, training=False)
        x = tf.keras.layers.Conv2D(embed_dim, 1, padding="same")(x)
    else:
        filters = [32, 64, embed_dim]
        for idx in range(stem_blocks):
            x = tf.keras.layers.Conv2D(filters[min(idx, len(filters) - 1)], 3, strides=2, padding="same", use_bias=False)(x)
            x = tf.keras.layers.BatchNormalization()(x)
            x = tf.keras.layers.ReLU()(x)
        x = tf.keras.layers.Conv2D(embed_dim, patch_size, strides=patch_size, padding="valid")(x)

    h, w, c = x.shape[1], x.shape[2], x.shape[3]
    x = tf.keras.layers.Reshape((h * w, c))(x)

    for _ in range(transformer_depth):
        x = _transformer_block(x, embed_dim=embed_dim, num_heads=num_heads, mlp_dim=mlp_dim, dropout=dropout)

    x = tf.keras.layers.LayerNormalization(epsilon=1e-6)(x)
    x = tf.keras.layers.GlobalAveragePooling1D()(x)
    if dropout and dropout > 0:
        x = tf.keras.layers.Dropout(dropout)(x)
    activation = "sigmoid" if num_classes == 1 else "softmax"
    outputs = tf.keras.layers.Dense(num_classes, activation=activation)(x)
    model = tf.keras.Model(inputs, outputs, name="hybrid_vit")
    model.compile(
        optimizer=tf.keras.optimizers.Adam(1e-4),
        loss="binary_crossentropy",
        metrics=[tf.keras.metrics.BinaryAccuracy(name="binary_accuracy"), tf.keras.metrics.AUC(name="auc")],
    )
    return model


def build_hybrid(
    input_shape: Tuple[int, int, int] = (224, 224, 3),
    num_classes: int = 1,
    projection_dim: int = 96,
    transformer_blocks: int = 1,
    stem: str = "cnn",
    dropout: float = 0.1,
):
    """Backward-compatible wrapper for the hybrid ViT model."""
    return build_hybrid_vit(
        input_shape=input_shape,
        num_classes=num_classes,
        stem_type=stem,
        embed_dim=projection_dim,
        transformer_depth=transformer_blocks,
        dropout=dropout,
    )
