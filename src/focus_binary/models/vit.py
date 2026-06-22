from __future__ import annotations

from typing import Tuple

from focus_binary.utils.logging import get_logger

logger = get_logger(__name__)

try:
    import tensorflow as tf
except Exception:  # pragma: no cover
    tf = None


def _mlp(x, hidden_units, dropout_rate):
    for units in hidden_units:
        x = tf.keras.layers.Dense(units, activation=tf.nn.gelu)(x)
        if dropout_rate and dropout_rate > 0:
            x = tf.keras.layers.Dropout(dropout_rate)(x)
    return x


if tf is not None:
    @tf.keras.utils.register_keras_serializable(package="focus_binary")
    class _CLSToken(tf.keras.layers.Layer):
        def __init__(self, embed_dim: int, **kwargs):
            super().__init__(**kwargs)
            self.embed_dim = embed_dim

        def build(self, input_shape):
            self.cls = self.add_weight(
                name="cls",
                shape=(1, 1, self.embed_dim),
                initializer="zeros",
                trainable=True,
            )

        def call(self, x):
            batch = tf.shape(x)[0]
            cls = tf.tile(self.cls, [batch, 1, 1])
            return tf.concat([cls, x], axis=1)

        def get_config(self):
            config = super().get_config()
            config.update({"embed_dim": self.embed_dim})
            return config


    @tf.keras.utils.register_keras_serializable(package="focus_binary")
    class _PositionalEmbedding(tf.keras.layers.Layer):
        def __init__(self, num_tokens: int, embed_dim: int, **kwargs):
            super().__init__(**kwargs)
            self.num_tokens = num_tokens
            self.embed_dim = embed_dim

        def build(self, input_shape):
            self.positional = self.add_weight(
                name="positional",
                shape=(1, self.num_tokens, self.embed_dim),
                initializer="random_normal",
                trainable=True,
            )

        def call(self, x):
            return x + self.positional

        def get_config(self):
            config = super().get_config()
            config.update({"num_tokens": self.num_tokens, "embed_dim": self.embed_dim})
            return config
else:  # pragma: no cover
    _CLSToken = None
    _PositionalEmbedding = None


def _transformer_encoder(x, embed_dim: int, num_heads: int, mlp_dim: int, dropout: float):
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
    y = _mlp(y, hidden_units=(mlp_dim, embed_dim), dropout_rate=dropout)
    return tf.keras.layers.Add()([x, y])


def build_vit(
    input_shape: Tuple[int, int, int] = (224, 224, 3),
    num_classes: int = 1,
    patch_size: int = 16,
    embed_dim: int = 96,
    num_heads: int = 4,
    depth: int = 4,
    mlp_dim: int = 192,
    dropout: float = 0.1,
    use_cls_token: bool = False,
):
    """Small Vision Transformer.

    Expects inputs normalized to [0,1] from the data pipeline.
    """

    if tf is None:
        logger.warning("TensorFlow not installed; returning placeholder model spec for vit")
        return {
            "model": "vit",
            "input_shape": input_shape,
            "patch_size": patch_size,
            "depth": depth,
        }

    if patch_size not in (8, 16):
        raise ValueError("patch_size must be 8 or 16")
    if embed_dim < 64 or embed_dim > 256:
        raise ValueError("embed_dim must be in [64, 256]")
    if num_heads < 2 or num_heads > 8:
        raise ValueError("num_heads must be in [2, 8]")
    if embed_dim % num_heads != 0:
        raise ValueError("embed_dim must be divisible by num_heads")
    if depth < 2 or depth > 8:
        raise ValueError("depth must be in [2, 8]")
    if mlp_dim < 128 or mlp_dim > 512:
        raise ValueError("mlp_dim must be in [128, 512]")

    height, width, _ = input_shape
    if height % patch_size != 0 or width % patch_size != 0:
        raise ValueError("input_shape must be divisible by patch_size")

    inputs = tf.keras.layers.Input(shape=input_shape)

    x = tf.keras.layers.Conv2D(
        filters=embed_dim,
        kernel_size=patch_size,
        strides=patch_size,
        padding="valid",
        name="patchify",
    )(inputs)
    num_patches = (height // patch_size) * (width // patch_size)
    x = tf.keras.layers.Reshape((num_patches, embed_dim))(x)

    if use_cls_token:
        x = _CLSToken(embed_dim)(x)
        num_tokens = num_patches + 1
    else:
        num_tokens = num_patches

    x = _PositionalEmbedding(num_tokens, embed_dim)(x)
    if dropout and dropout > 0:
        x = tf.keras.layers.Dropout(dropout)(x)

    for _ in range(depth):
        x = _transformer_encoder(x, embed_dim=embed_dim, num_heads=num_heads, mlp_dim=mlp_dim, dropout=dropout)

    x = tf.keras.layers.LayerNormalization(epsilon=1e-6)(x)
    if use_cls_token:
        x = x[:, 0]
    else:
        x = tf.keras.layers.GlobalAveragePooling1D()(x)

    if dropout and dropout > 0:
        x = tf.keras.layers.Dropout(dropout)(x)

    activation = "sigmoid" if num_classes == 1 else "softmax"
    outputs = tf.keras.layers.Dense(num_classes, activation=activation)(x)
    model = tf.keras.Model(inputs=inputs, outputs=outputs, name="vit_small")
    model.compile(
        optimizer=tf.keras.optimizers.Adam(1e-4),
        loss="binary_crossentropy",
        metrics=[tf.keras.metrics.BinaryAccuracy(name="binary_accuracy"), tf.keras.metrics.AUC(name="auc")],
    )
    return model
