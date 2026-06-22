from __future__ import annotations

from typing import Any, Tuple

from focus_binary.utils.logging import get_logger

logger = get_logger(__name__)

try:
    import tensorflow as tf
except Exception:  # pragma: no cover
    tf = None


def _squeeze_excitation(x, ratio: int = 16):
    filters = int(x.shape[-1])
    hidden = max(filters // ratio, 1)
    se = tf.keras.layers.GlobalAveragePooling2D()(x)
    se = tf.keras.layers.Dense(hidden, activation="relu")(se)
    se = tf.keras.layers.Dense(filters, activation="sigmoid")(se)
    se = tf.keras.layers.Reshape((1, 1, filters))(se)
    return tf.keras.layers.Multiply()([x, se])


def _spatial_attention(x, kernel_size: int = 7):
    mean_map = tf.keras.layers.Lambda(lambda t: tf.reduce_mean(t, axis=-1, keepdims=True))(x)
    max_map = tf.keras.layers.Lambda(lambda t: tf.reduce_max(t, axis=-1, keepdims=True))(x)
    stacked = tf.keras.layers.Concatenate(axis=-1)([mean_map, max_map])
    spatial = tf.keras.layers.Conv2D(1, kernel_size, padding="same", activation="sigmoid")(stacked)
    return tf.keras.layers.Multiply()([x, spatial])


def _compile_model(model, optimizer: str, learning_rate: float):
    if tf is None:
        return model

    opt_name = optimizer.lower() if isinstance(optimizer, str) else None
    if opt_name == "adam":
        opt = tf.keras.optimizers.Adam(learning_rate=learning_rate)
    elif opt_name == "rmsprop":
        opt = tf.keras.optimizers.RMSprop(learning_rate=learning_rate)
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


def _hp_get(hp: Any, name: str, default: Any):
    if hp is None:
        return default
    if isinstance(hp, dict):
        return hp.get(name, default)
    if hasattr(hp, "get"):
        try:
            return hp.get(name, default)
        except Exception:
            return default
    return default


def _hp_choice(hp: Any, name: str, values: list, default: Any):
    if hp is None:
        return default
    if hasattr(hp, "Choice"):
        return hp.Choice(name, values=values, default=default)
    return _hp_get(hp, name, default)


def _hp_int(hp: Any, name: str, min_value: int, max_value: int, default: int, step: int = 1):
    if hp is None:
        return default
    if hasattr(hp, "Int"):
        return hp.Int(name, min_value=min_value, max_value=max_value, step=step, default=default)
    return int(_hp_get(hp, name, default))


def _hp_float(hp: Any, name: str, min_value: float, max_value: float, default: float, step: float | None = None):
    if hp is None:
        return default
    if hasattr(hp, "Float"):
        return hp.Float(name, min_value=min_value, max_value=max_value, step=step, default=default)
    return float(_hp_get(hp, name, default))


def _hp_bool(hp: Any, name: str, default: bool):
    if hp is None:
        return default
    if hasattr(hp, "Boolean"):
        return hp.Boolean(name, default=default)
    return bool(_hp_get(hp, name, default))


def build_cnn_attention(
    hp: Any,
    input_shape: Tuple[int, int, int] = (224, 224, 3),
    attention_type: str = "se",
):
    """CNN backbone augmented with optional SE/CBAM attention."""

    if tf is None:
        logger.warning("TensorFlow not installed; returning placeholder model spec for cnn_attention")
        return {
            "model": "cnn_attention",
            "attention_type": attention_type,
            "input_shape": input_shape,
        }

    num_blocks = _hp_int(hp, "num_blocks", 2, 5, default=3)
    filters_base = _hp_int(hp, "filters_base", 16, 64, default=32, step=8)
    kernel_size = _hp_choice(hp, "kernel_size", [3, 5], default=3)
    dropout = _hp_float(hp, "dropout", 0.0, 0.5, default=0.2, step=0.1)
    batchnorm = _hp_bool(hp, "batchnorm", default=True)
    dense_units = _hp_int(hp, "dense_units", 0, 256, default=0, step=32)
    l2_reg = _hp_float(hp, "l2_reg", 0.0, 1e-3, default=0.0)
    optimizer = _hp_choice(hp, "optimizer", ["adam", "rmsprop"], default="adam")
    learning_rate = _hp_float(hp, "learning_rate", 1e-4, 1e-2, default=1e-3)
    se_ratio = _hp_int(hp, "se_ratio", 4, 16, default=8, step=4)
    attention_placement = _hp_choice(hp, "attention_placement", ["all", "last_two"], default="last_two")
    spatial_kernel = _hp_choice(hp, "spatial_kernel", [3, 7], default=7)

    if kernel_size not in (3, 5):
        raise ValueError("kernel_size must be 3 or 5")
    if attention_type not in {"se", "cbam", "none"}:
        raise ValueError("attention_type must be 'se', 'cbam', or 'none'")

    l2 = tf.keras.regularizers.l2(l2_reg) if l2_reg and l2_reg > 0 else None
    inputs = tf.keras.Input(shape=input_shape)
    x = inputs

    for block in range(num_blocks):
        filters = int(filters_base * (2**block))
        x = tf.keras.layers.Conv2D(
            filters,
            kernel_size,
            padding="same",
            use_bias=not batchnorm,
            kernel_regularizer=l2,
        )(x)
        if batchnorm:
            x = tf.keras.layers.BatchNormalization()(x)
        x = tf.keras.layers.ReLU()(x)

        use_attention = attention_type != "none"
        if attention_placement == "last_two":
            use_attention = use_attention and block >= max(num_blocks - 2, 0)
        if use_attention:
            x = _squeeze_excitation(x, ratio=se_ratio)
            if attention_type == "cbam":
                x = _spatial_attention(x, kernel_size=spatial_kernel)

        x = tf.keras.layers.MaxPooling2D(pool_size=2)(x)
        if dropout and dropout > 0:
            x = tf.keras.layers.Dropout(dropout)(x)

    x = tf.keras.layers.GlobalAveragePooling2D()(x)
    if dense_units > 0:
        x = tf.keras.layers.Dense(dense_units, activation="relu", kernel_regularizer=l2)(x)
        if dropout and dropout > 0:
            x = tf.keras.layers.Dropout(dropout)(x)

    outputs = tf.keras.layers.Dense(1, activation="sigmoid", kernel_regularizer=l2)(x)
    model = tf.keras.Model(inputs, outputs, name=f"cnn_attention_{attention_type}")
    return _compile_model(model, optimizer=optimizer, learning_rate=learning_rate)


def build_cnn_with_attention(
    input_shape: Tuple[int, int, int] = (224, 224, 3),
    attention: str = "se",
):
    """Backward-compatible wrapper for the attention CNN."""
    return build_cnn_attention(hp=None, input_shape=input_shape, attention_type=attention)
