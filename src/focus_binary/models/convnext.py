from __future__ import annotations

from typing import Iterable, Tuple

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


def _has_keras_convnext() -> bool:
    if tf is None:
        return False
    return hasattr(tf.keras.applications, "ConvNeXtTiny")


def _get_keras_preprocess():
    if tf is None:
        return None
    try:
        from tensorflow.keras.applications import convnext
    except Exception:  # pragma: no cover
        return None
    return getattr(convnext, "preprocess_input", None)


if tf is not None:
    @tf.keras.utils.register_keras_serializable(package="focus_binary")
    class ConvNeXtPreprocess(tf.keras.layers.Layer):
        def call(self, inputs):
            try:
                from tensorflow.keras.applications import convnext
            except Exception:  # pragma: no cover
                return inputs
            return convnext.preprocess_input(inputs * 255.0)

        def get_config(self):
            return super().get_config()


def _scaled_depths(depths: Iterable[int], depth_mult: float) -> Tuple[int, ...]:
    scaled = []
    for depth in depths:
        scaled_depth = max(1, int(round(depth * depth_mult)))
        scaled.append(scaled_depth)
    return tuple(scaled)


def _scaled_dims(dims: Iterable[int], width_mult: float) -> Tuple[int, ...]:
    scaled = []
    for dim in dims:
        scaled_dim = max(16, int(round(dim * width_mult)))
        scaled.append(scaled_dim)
    return tuple(scaled)


def _convnext_block(x, dim: int, drop_rate: float) -> "tf.Tensor":
    shortcut = x
    x = tf.keras.layers.DepthwiseConv2D(kernel_size=7, padding="same")(x)
    x = tf.keras.layers.LayerNormalization(epsilon=1e-6)(x)
    x = tf.keras.layers.Dense(4 * dim, activation=tf.nn.gelu)(x)
    x = tf.keras.layers.Dense(dim)(x)
    if drop_rate and drop_rate > 0:
        x = tf.keras.layers.Dropout(drop_rate)(x)
    return tf.keras.layers.Add()([shortcut, x])


def _convnext_downsample(x, dim: int) -> "tf.Tensor":
    x = tf.keras.layers.LayerNormalization(epsilon=1e-6)(x)
    return tf.keras.layers.Conv2D(dim, kernel_size=2, strides=2, padding="same")(x)


def _build_minimal_convnext(
    inputs: "tf.Tensor",
    *,
    depths: Tuple[int, ...],
    dims: Tuple[int, ...],
    drop_rate: float,
) -> "tf.Tensor":
    x = tf.keras.layers.Conv2D(dims[0], kernel_size=4, strides=4, padding="same")(inputs)
    x = tf.keras.layers.LayerNormalization(epsilon=1e-6)(x)

    for stage, depth in enumerate(depths):
        for _ in range(depth):
            x = _convnext_block(x, dims[stage], drop_rate=drop_rate)
        if stage < len(depths) - 1:
            x = _convnext_downsample(x, dims[stage + 1])
    return x


def build_convnext(
    input_shape: Tuple[int, int, int] = (224, 224, 3),
    num_classes: int = 1,
    pooling: str = "avg",
    head_units: int = 0,
    dropout: float = 0.1,
    optimizer: str = "adam",
    learning_rate: float = 1e-4,
    weights: str | None = "imagenet",
    use_keras_app: bool = True,
    depth_mult: float = 1.0,
    width_mult: float = 1.0,
):
    """ConvNeXt family model using ConvNeXtTiny if available, otherwise a minimal fallback."""

    if tf is None:
        logger.warning("TensorFlow not installed; returning placeholder model spec for convnext")
        return {
            "model": "convnext",
            "input_shape": input_shape,
            "pooling": pooling,
            "head_units": head_units,
            "dropout": dropout,
            "weights": weights,
        }

    if pooling not in {"avg", "max"}:
        raise ValueError("pooling must be 'avg' or 'max'")

    inputs = tf.keras.Input(shape=input_shape)
    preprocess_fn = _get_keras_preprocess() if _has_keras_convnext() else None
    use_keras = bool(use_keras_app and _has_keras_convnext())

    if use_keras:
        backbone = tf.keras.applications.ConvNeXtTiny(
            include_top=False,
            weights=weights,
            input_shape=input_shape,
        )
        x = inputs
        if preprocess_fn is not None:
            x = ConvNeXtPreprocess(name="convnext_preprocess")(x)
        x = backbone(x)
    else:
        backbone = None
        base_depths = (2, 2, 6, 2)
        base_dims = (64, 128, 256, 512)
        depths = _scaled_depths(base_depths, depth_mult)
        dims = _scaled_dims(base_dims, width_mult)
        x = _build_minimal_convnext(inputs, depths=depths, dims=dims, drop_rate=dropout)

    if pooling == "avg":
        x = tf.keras.layers.GlobalAveragePooling2D()(x)
    else:
        x = tf.keras.layers.GlobalMaxPooling2D()(x)
    if head_units > 0:
        x = tf.keras.layers.Dense(head_units, activation="relu")(x)
        if dropout and dropout > 0:
            x = tf.keras.layers.Dropout(dropout)(x)
    outputs = tf.keras.layers.Dense(num_classes, activation="sigmoid")(x)
    model = tf.keras.Model(inputs, outputs, name="convnext")
    model.backbone = backbone  # type: ignore[attr-defined]
    return _compile_model(model, optimizer=optimizer, learning_rate=learning_rate)
