from __future__ import annotations

from typing import Tuple

from focus_binary.utils.logging import get_logger

logger = get_logger(__name__)

try:  # Optional TF import
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


def build_cnn_baseline(
    input_shape: Tuple[int, int, int] = (224, 224, 3),
    num_blocks: int = 3,
    filters_base: int = 32,
    kernel_size: int = 3,
    dropout: float = 0.2,
    batchnorm: bool = True,
    dense_units: int = 0,
    l2_reg: float = 0.0,
    optimizer: str = "adam",
    learning_rate: float = 1e-3,
):
    """Lightweight CNN with tunable hyperparameters for binary classification."""

    if tf is None:
        logger.warning("TensorFlow not installed; returning placeholder model spec for cnn_baseline")
        return {
            "model": "cnn_baseline",
            "input_shape": input_shape,
            "num_blocks": num_blocks,
            "filters_base": filters_base,
            "kernel_size": kernel_size,
            "dropout": dropout,
            "batchnorm": batchnorm,
            "dense_units": dense_units,
            "l2_reg": l2_reg,
            "optimizer": optimizer,
        }

    if num_blocks < 2 or num_blocks > 5:
        raise ValueError("num_blocks must be in [2, 5]")
    if kernel_size not in (3, 5):
        raise ValueError("kernel_size must be 3 or 5")
    if filters_base < 8:
        raise ValueError("filters_base must be >= 8")
    if dense_units < 0:
        raise ValueError("dense_units must be >= 0")

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
        x = tf.keras.layers.MaxPooling2D(pool_size=2)(x)
        if dropout and dropout > 0:
            x = tf.keras.layers.Dropout(dropout)(x)

    x = tf.keras.layers.GlobalAveragePooling2D()(x)
    if dense_units > 0:
        x = tf.keras.layers.Dense(dense_units, activation="relu", kernel_regularizer=l2)(x)
        if dropout and dropout > 0:
            x = tf.keras.layers.Dropout(dropout)(x)

    outputs = tf.keras.layers.Dense(1, activation="sigmoid", kernel_regularizer=l2)(x)
    model = tf.keras.Model(inputs, outputs, name="cnn_baseline")
    return _compile_model(model, optimizer=optimizer, learning_rate=learning_rate)


def build_cnn_minimal(input_shape: Tuple[int, int, int] = (224, 224, 3)):
    """Fixed minimal baseline for sanity checks."""
    return build_cnn_baseline(
        input_shape=input_shape,
        num_blocks=2,
        filters_base=16,
        kernel_size=3,
        dropout=0.1,
        batchnorm=False,
        dense_units=0,
        l2_reg=0.0,
        optimizer="adam",
        learning_rate=1e-3,
    )
