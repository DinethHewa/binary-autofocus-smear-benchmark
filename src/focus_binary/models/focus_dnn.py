from __future__ import annotations

from typing import Any

from focus_binary.utils.logging import get_logger

logger = get_logger(__name__)

try:
    import tensorflow as tf
except Exception:  # pragma: no cover
    tf = None


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


def _hp_int(hp: Any, name: str, min_value: int, max_value: int, default: int):
    if hp is None:
        return default
    if hasattr(hp, "Int"):
        return hp.Int(name, min_value=min_value, max_value=max_value, default=default)
    return int(_hp_get(hp, name, default))


def _hp_float(hp: Any, name: str, min_value: float, max_value: float, default: float, step: float | None = None):
    if hp is None:
        return default
    if hasattr(hp, "Float"):
        return hp.Float(name, min_value=min_value, max_value=max_value, step=step, default=default)
    return float(_hp_get(hp, name, default))


def _hp_choice(hp: Any, name: str, values: list, default: Any):
    if hp is None:
        return default
    if hasattr(hp, "Choice"):
        return hp.Choice(name, values=values, default=default)
    return _hp_get(hp, name, default)


def build_focus_dnn(hp: Any, input_dim: int):
    """MLP for focus-measure vectors."""
    if tf is None:
        logger.warning("TensorFlow not installed; returning placeholder model spec for focus_dnn")
        return {"model": "focus_dnn", "input_dim": input_dim}

    num_layers = _hp_int(hp, "num_layers", 1, 5, default=2)
    units = _hp_int(hp, "units", 16, 256, default=64)
    activation = _hp_choice(hp, "activation", ["relu", "gelu"], default="relu")
    dropout = _hp_float(hp, "dropout", 0.0, 0.5, default=0.1, step=0.1)
    l2_reg = _hp_float(hp, "l2", 0.0, 1e-3, default=0.0)
    learning_rate = _hp_float(hp, "learning_rate", 1e-4, 1e-2, default=1e-3)
    optimizer = _hp_choice(hp, "optimizer", ["adam", "rmsprop"], default="adam")

    l2 = tf.keras.regularizers.l2(l2_reg) if l2_reg and l2_reg > 0 else None
    inputs = tf.keras.Input(shape=(input_dim,), name="focus_vector")
    x = inputs
    for _ in range(num_layers):
        x = tf.keras.layers.Dense(units, activation=activation, kernel_regularizer=l2)(x)
        if dropout and dropout > 0:
            x = tf.keras.layers.Dropout(dropout)(x)

    outputs = tf.keras.layers.Dense(1, activation="sigmoid", kernel_regularizer=l2)(x)
    model = tf.keras.Model(inputs, outputs, name="focus_dnn")

    if optimizer == "adam":
        opt = tf.keras.optimizers.Adam(learning_rate=learning_rate)
    else:
        opt = tf.keras.optimizers.RMSprop(learning_rate=learning_rate)

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
