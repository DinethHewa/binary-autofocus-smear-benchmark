from __future__ import annotations

from typing import Any, Tuple

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


def _freeze_backbone(base_model):
    for layer in base_model.layers:
        if isinstance(layer, tf.keras.layers.BatchNormalization):
            layer.trainable = False
        else:
            layer.trainable = False


def _unfreeze_top_layers(base_model, n_layers: int):
    if n_layers <= 0:
        return
    for layer in base_model.layers[-n_layers:]:
        if isinstance(layer, tf.keras.layers.BatchNormalization):
            layer.trainable = False
        else:
            layer.trainable = True


def _custom_cnn_encoder(
    hp: Any,
    inputs: tf.Tensor,
    name: str = "custom_cnn",
) -> tf.Tensor:
    num_blocks = _hp_int(hp, "num_blocks", 2, 5, default=3)
    filters_base = _hp_int(hp, "filters_base", 16, 64, default=32)
    kernel_size = _hp_choice(hp, "kernel_size", [3, 5], default=3)
    dropout = _hp_float(hp, "dropout", 0.0, 0.5, default=0.2, step=0.1)
    batchnorm = bool(_hp_get(hp, "batchnorm", True))
    l2_reg = _hp_float(hp, "l2", 0.0, 1e-3, default=0.0)

    l2 = tf.keras.regularizers.l2(l2_reg) if l2_reg and l2_reg > 0 else None
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
    x = tf.keras.layers.GlobalAveragePooling2D(name=f"{name}_gap")(x)
    return x


def build_cnn_focus_hybrid(
    hp: Any,
    input_shape: Tuple[int, int, int],
    focus_dim: int,
):
    """CNN image branch fused with focus-vector branch."""
    if tf is None:
        logger.warning("TensorFlow not installed; returning placeholder model spec for cnn_focus_hybrid")
        return {"model": "cnn_focus_hybrid", "input_shape": input_shape, "focus_dim": focus_dim}

    backbone_choice = _hp_choice(
        hp,
        "backbone_choice",
        ["custom_cnn", "mobilenetv2", "efficientnetb0"],
        default="custom_cnn",
    )
    focus_units = _hp_int(hp, "focus_units", 0, 128, default=64)
    fusion_layers = _hp_int(hp, "fusion_layers", 1, 3, default=1)
    fusion_units = _hp_int(hp, "fusion_units", 32, 512, default=128)
    fusion_dropout = _hp_float(hp, "fusion_dropout", 0.0, 0.5, default=0.2, step=0.1)
    unfreeze_top = _hp_int(hp, "unfreeze_top", 0, 20, default=0)
    optimizer = _hp_choice(hp, "optimizer", ["adam", "rmsprop"], default="adam")
    learning_rate = _hp_float(hp, "learning_rate", 1e-4, 1e-2, default=1e-3)

    image_in = tf.keras.Input(shape=input_shape, name="image")
    focus_in = tf.keras.Input(shape=(focus_dim,), name="focus_vector")

    if backbone_choice == "custom_cnn":
        img_emb = _custom_cnn_encoder(hp, image_in)
    else:
        input_channels = int(input_shape[-1]) if input_shape and len(input_shape) == 3 else 3
        require_rgb = input_channels != 3
        backbone_input_shape = (input_shape[0], input_shape[1], 3) if require_rgb else input_shape
        if backbone_choice == "mobilenetv2":
            try:
                base = tf.keras.applications.MobileNetV2(
                    include_top=False,
                    weights="imagenet",
                    input_shape=backbone_input_shape,
                    pooling="avg",
                )
            except ValueError as exc:
                if "Shape mismatch" in str(exc):
                    logger.warning(
                        "imagenet weights incompatible with input channels; falling back to random init",
                        extra={"backbone": "MobileNetV2", "input_channels": input_channels},
                    )
                    require_rgb = False
                    base = tf.keras.applications.MobileNetV2(
                        include_top=False,
                        weights=None,
                        input_shape=input_shape,
                        pooling="avg",
                    )
                else:
                    raise
        else:
            try:
                base = tf.keras.applications.EfficientNetB0(
                    include_top=False,
                    weights="imagenet",
                    input_shape=backbone_input_shape,
                    pooling="avg",
                )
            except ValueError as exc:
                if "Shape mismatch" in str(exc):
                    logger.warning(
                        "imagenet weights incompatible with input channels; falling back to random init",
                        extra={"backbone": "EfficientNetB0", "input_channels": input_channels},
                    )
                    require_rgb = False
                    base = tf.keras.applications.EfficientNetB0(
                        include_top=False,
                        weights=None,
                        input_shape=input_shape,
                        pooling="avg",
                    )
                else:
                    raise
        _freeze_backbone(base)
        if unfreeze_top > 0:
            _unfreeze_top_layers(base, unfreeze_top)
        x = image_in
        if require_rgb:
            if input_channels == 1:
                x = tf.keras.layers.Lambda(lambda t: tf.image.grayscale_to_rgb(t), name="to_rgb")(x)
            else:
                x = tf.keras.layers.Lambda(lambda t: t[..., :3], name="to_rgb")(x)
        img_emb = base(x, training=False)

    focus_branch = focus_in
    if focus_units > 0:
        focus_branch = tf.keras.layers.Dense(focus_units, activation="relu")(focus_branch)
        if fusion_dropout and fusion_dropout > 0:
            focus_branch = tf.keras.layers.Dropout(fusion_dropout)(focus_branch)

    fused = tf.keras.layers.Concatenate()([img_emb, focus_branch])
    x = fused
    for _ in range(fusion_layers):
        x = tf.keras.layers.Dense(fusion_units, activation="relu")(x)
        if fusion_dropout and fusion_dropout > 0:
            x = tf.keras.layers.Dropout(fusion_dropout)(x)

    outputs = tf.keras.layers.Dense(1, activation="sigmoid")(x)
    model = tf.keras.Model(inputs=[image_in, focus_in], outputs=outputs, name="cnn_focus_hybrid")

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
