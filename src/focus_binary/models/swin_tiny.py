from __future__ import annotations

from typing import Tuple

from focus_binary.utils.logging import get_logger

logger = get_logger(__name__)

try:
    import tensorflow as tf
except Exception:  # pragma: no cover
    tf = None


def _compile_model(model, optimizer: str, learning_rate: float):
    if tf is None:
        return model

    if isinstance(optimizer, str):
        opt_name = optimizer.lower()
        if opt_name == "adam":
            opt = tf.keras.optimizers.Adam(learning_rate=learning_rate)
        elif opt_name == "rmsprop":
            opt = tf.keras.optimizers.RMSprop(learning_rate=learning_rate)
        else:
            raise ValueError("optimizer must be 'adam' or 'rmsprop'")
    else:
        opt = optimizer

    model.compile(
        optimizer=opt,
        loss=tf.keras.losses.BinaryCrossentropy(),
        metrics=[
            tf.keras.metrics.AUC(name="auc"),
            tf.keras.metrics.BinaryAccuracy(name="binary_accuracy"),
            tf.keras.metrics.Precision(name="precision"),
            tf.keras.metrics.Recall(name="recall"),
            tf.keras.metrics.FalsePositives(name="false_positives"),
            tf.keras.metrics.FalseNegatives(name="false_negatives"),
        ],
    )
    return model


if tf is not None:
    @tf.keras.utils.register_keras_serializable(package="focus_binary")
    class WindowPartition(tf.keras.layers.Layer):
        def __init__(self, window_size: int, **kwargs):
            super().__init__(**kwargs)
            self.window_size = int(window_size)

        def call(self, x):
            ws = self.window_size
            shape = tf.shape(x)
            batch, height, width, channels = shape[0], shape[1], shape[2], shape[3]
            x = tf.reshape(x, (batch, height // ws, ws, width // ws, ws, channels))
            x = tf.transpose(x, perm=(0, 1, 3, 2, 4, 5))
            return tf.reshape(x, (-1, ws * ws, channels))

        def get_config(self):
            cfg = super().get_config()
            cfg.update({"window_size": self.window_size})
            return cfg


    @tf.keras.utils.register_keras_serializable(package="focus_binary")
    class WindowReverse(tf.keras.layers.Layer):
        def __init__(self, window_size: int, height: int, width: int, **kwargs):
            super().__init__(**kwargs)
            self.window_size = int(window_size)
            self.height = int(height)
            self.width = int(width)

        def call(self, windows):
            ws = self.window_size
            height = self.height
            width = self.width
            num_windows = (height // ws) * (width // ws)
            batch = tf.shape(windows)[0] // num_windows
            channels = tf.shape(windows)[-1]
            x = tf.reshape(windows, (batch, height // ws, width // ws, ws, ws, channels))
            x = tf.transpose(x, perm=(0, 1, 3, 2, 4, 5))
            return tf.reshape(x, (batch, height, width, channels))

        def get_config(self):
            cfg = super().get_config()
            cfg.update({"window_size": self.window_size, "height": self.height, "width": self.width})
            return cfg
else:  # pragma: no cover
    WindowPartition = None
    WindowReverse = None


def _window_attention_block(
    x: "tf.Tensor",
    *,
    embed_dim: int,
    num_heads: int,
    window_size: int,
    mlp_dim: int,
    dropout: float,
    height: int,
    width: int,
) -> "tf.Tensor":
    shortcut = x
    x = tf.keras.layers.LayerNormalization(epsilon=1e-6)(x)
    x = WindowPartition(window_size)(x)
    x = tf.keras.layers.MultiHeadAttention(
        num_heads=num_heads,
        key_dim=embed_dim // num_heads,
        dropout=dropout,
    )(x, x)
    if dropout and dropout > 0:
        x = tf.keras.layers.Dropout(dropout)(x)
    x = WindowReverse(window_size, height, width)(x)
    x = tf.keras.layers.Add()([shortcut, x])

    y = tf.keras.layers.LayerNormalization(epsilon=1e-6)(x)
    y = tf.keras.layers.Dense(mlp_dim, activation=tf.nn.gelu)(y)
    if dropout and dropout > 0:
        y = tf.keras.layers.Dropout(dropout)(y)
    y = tf.keras.layers.Dense(embed_dim)(y)
    if dropout and dropout > 0:
        y = tf.keras.layers.Dropout(dropout)(y)
    return tf.keras.layers.Add()([x, y])


def build_swin_tiny(
    input_shape: Tuple[int, int, int] = (224, 224, 3),
    num_classes: int = 1,
    patch_size: int = 4,
    window_size: int = 7,
    embed_dim: int = 96,
    depth: int = 2,
    num_heads: int = 3,
    mlp_dim: int = 192,
    dropout: float = 0.1,
    optimizer: str = "adam",
    learning_rate: float = 1e-4,
):
    """Lightweight Swin-style model using windowed attention blocks (no shifted windows)."""

    if tf is None:
        logger.warning("TensorFlow not installed; returning placeholder model spec for swin_tiny")
        return {
            "model": "swin_tiny",
            "input_shape": input_shape,
            "patch_size": patch_size,
            "window_size": window_size,
            "embed_dim": embed_dim,
            "depth": depth,
        }

    if embed_dim % num_heads != 0:
        raise ValueError("embed_dim must be divisible by num_heads")
    if depth < 1:
        raise ValueError("depth must be >= 1")

    height, width, _ = input_shape
    if height % patch_size != 0 or width % patch_size != 0:
        raise ValueError("input_shape must be divisible by patch_size")

    grid_h = height // patch_size
    grid_w = width // patch_size
    if grid_h % window_size != 0 or grid_w % window_size != 0:
        raise ValueError("Window size must divide patch grid dimensions")

    inputs = tf.keras.Input(shape=input_shape)
    x = tf.keras.layers.Conv2D(embed_dim, kernel_size=patch_size, strides=patch_size, padding="valid")(inputs)

    for _ in range(depth):
        x = _window_attention_block(
            x,
            embed_dim=embed_dim,
            num_heads=num_heads,
            window_size=window_size,
            mlp_dim=mlp_dim,
            dropout=dropout,
            height=grid_h,
            width=grid_w,
        )

    x = tf.keras.layers.LayerNormalization(epsilon=1e-6)(x)
    x = tf.keras.layers.GlobalAveragePooling2D()(x)
    if dropout and dropout > 0:
        x = tf.keras.layers.Dropout(dropout)(x)
    outputs = tf.keras.layers.Dense(num_classes, activation="sigmoid")(x)
    model = tf.keras.Model(inputs, outputs, name="swin_tiny")
    return _compile_model(model, optimizer=optimizer, learning_rate=learning_rate)
