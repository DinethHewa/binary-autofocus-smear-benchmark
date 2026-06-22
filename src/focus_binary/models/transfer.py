from __future__ import annotations

import re
from dataclasses import dataclass
from typing import Any, Callable, Dict, List, Optional

from focus_binary.utils.logging import get_logger

logger = get_logger(__name__)

try:  # Optional TF import
    import tensorflow as tf
except Exception:  # pragma: no cover
    tf = None


@dataclass(frozen=True)
class TransferBackboneSpec:
    name: str
    constructor: Callable[..., Any]
    preprocess_input: Callable[..., Any]


TRANSFER_BACKBONES_ALL = [
    "MobileNet",
    "MobileNetV2",
    "MobileNetV3Small",
    "MobileNetV3Large",
    "NASNetMobile",
    "EfficientNetB0",
    "EfficientNetB1",
    "EfficientNetB2",
    "EfficientNetB3",
    "EfficientNetV2B0",
    "EfficientNetV2S",
    "DenseNet121",
    "ResNet50",
    "ResNet50V2",
    "InceptionV3",
]

TRANSFER_BACKBONES_LIGHT = [
    "MobileNetV2",
    "MobileNetV3Small",
    "MobileNetV3Large",
    "EfficientNetB0",
    "EfficientNetB1",
    "EfficientNetV2B0",
    "NASNetMobile",
    "MobileNet",
]
# ResNet/DenseNet/Inception are excluded here to keep the default set lightweight.


def _build_backbones() -> Dict[str, TransferBackboneSpec]:
    if tf is None:
        return {}

    app = tf.keras.applications
    return {
        "MobileNet": TransferBackboneSpec("MobileNet", app.MobileNet, app.mobilenet.preprocess_input),
        "MobileNetV2": TransferBackboneSpec("MobileNetV2", app.MobileNetV2, app.mobilenet_v2.preprocess_input),
        "MobileNetV3Small": TransferBackboneSpec(
            "MobileNetV3Small", app.MobileNetV3Small, app.mobilenet_v3.preprocess_input
        ),
        "MobileNetV3Large": TransferBackboneSpec(
            "MobileNetV3Large", app.MobileNetV3Large, app.mobilenet_v3.preprocess_input
        ),
        "NASNetMobile": TransferBackboneSpec("NASNetMobile", app.NASNetMobile, app.nasnet.preprocess_input),
        "EfficientNetB0": TransferBackboneSpec("EfficientNetB0", app.EfficientNetB0, app.efficientnet.preprocess_input),
        "EfficientNetB1": TransferBackboneSpec("EfficientNetB1", app.EfficientNetB1, app.efficientnet.preprocess_input),
        "EfficientNetB2": TransferBackboneSpec("EfficientNetB2", app.EfficientNetB2, app.efficientnet.preprocess_input),
        "EfficientNetB3": TransferBackboneSpec("EfficientNetB3", app.EfficientNetB3, app.efficientnet.preprocess_input),
        "EfficientNetV2B0": TransferBackboneSpec(
            "EfficientNetV2B0", app.EfficientNetV2B0, app.efficientnet_v2.preprocess_input
        ),
        "EfficientNetV2S": TransferBackboneSpec(
            "EfficientNetV2S", app.EfficientNetV2S, app.efficientnet_v2.preprocess_input
        ),
        "DenseNet121": TransferBackboneSpec("DenseNet121", app.DenseNet121, app.densenet.preprocess_input),
        "ResNet50": TransferBackboneSpec("ResNet50", app.ResNet50, app.resnet.preprocess_input),
        "ResNet50V2": TransferBackboneSpec("ResNet50V2", app.ResNet50V2, app.resnet_v2.preprocess_input),
        "InceptionV3": TransferBackboneSpec("InceptionV3", app.InceptionV3, app.inception_v3.preprocess_input),
    }


BACKBONES: Dict[str, TransferBackboneSpec] = _build_backbones()


def list_transfer_backbones() -> List[str]:
    return list(TRANSFER_BACKBONES_ALL)


def get_preprocess(backbone_name: str) -> Callable[..., Any]:
    if backbone_name not in BACKBONES:
        raise KeyError(f"Unknown backbone '{backbone_name}'. Available: {list(TRANSFER_BACKBONES_ALL)}")
    return BACKBONES[backbone_name].preprocess_input


def _is_batch_norm(layer: Any) -> bool:
    if tf is None:
        return False
    return isinstance(layer, tf.keras.layers.BatchNormalization)


def set_backbone_trainable(backbone_model: Any, trainable: bool) -> None:
    for layer in backbone_model.layers:
        layer.trainable = trainable
        if _is_batch_norm(layer):
            layer.trainable = False


def _first_int(groups: tuple[str, ...]) -> Optional[int]:
    for item in groups:
        if item is not None:
            return int(item)
    return None


def _block_id_for_layer(backbone_name: str, layer_name: str) -> Optional[int]:
    name = backbone_name.lower()
    if "efficientnet" in name:
        match = re.search(r"block(\d+)", layer_name)
        return int(match.group(1)) if match else None
    if name == "mobilenetv2":
        match = re.search(r"block_(\d+)", layer_name)
        return int(match.group(1)) if match else None
    if name.startswith("mobilenetv3"):
        match = re.search(r"expanded_conv(?:_(\d+))?", layer_name)
        if not match:
            return None
        if match.group(1) is None:
            return 0
        return int(match.group(1))
    if name == "mobilenet":
        match = re.search(r"conv_pw_(\d+)|conv_dw_(\d+)", layer_name)
        if not match:
            return None
        return _first_int(match.groups())
    if name.startswith("resnet50"):
        match = re.search(r"conv(\d+)_block", layer_name)
        return int(match.group(1)) if match else None
    if name == "densenet121":
        match = re.search(r"conv(\d+)_block", layer_name)
        if match:
            return int(match.group(1))
        match = re.search(r"pool(\d+)", layer_name)
        return int(match.group(1)) if match else None
    if name == "inceptionv3":
        match = re.search(r"mixed(\d+)", layer_name)
        return int(match.group(1)) if match else None
    if name == "nasnetmobile":
        match = re.search(r"(normal_cell|reduction_cell)_(\d+)", layer_name)
        if not match:
            return None
        return int(match.group(2))
    return None


def _fallback_unfreeze(backbone_model: Any, n_blocks: int) -> int:
    k = max(1, n_blocks * 20)
    non_bn_layers = [layer for layer in backbone_model.layers if not _is_batch_norm(layer)]
    for layer in non_bn_layers[:-k]:
        layer.trainable = False
    for layer in non_bn_layers[-k:]:
        layer.trainable = True
    return min(k, len(non_bn_layers))


def unfreeze_top_n_blocks(backbone_model: Any, backbone_name: str, n_blocks: int) -> None:
    set_backbone_trainable(backbone_model, False)
    if n_blocks <= 0:
        return

    block_ids: List[Optional[int]] = []
    for layer in backbone_model.layers:
        block_ids.append(_block_id_for_layer(backbone_name, layer.name))

    unique_ids = sorted({bid for bid in block_ids if bid is not None})
    if not unique_ids:
        count = _fallback_unfreeze(backbone_model, n_blocks)
        logger.info(
            "fallback unfreeze used",
            extra={"backbone": backbone_name, "blocks": n_blocks, "layers_unfrozen": count},
        )
        return

    selected = set(unique_ids[-n_blocks:])
    for layer, block_id in zip(backbone_model.layers, block_ids):
        if block_id in selected and not _is_batch_norm(layer):
            layer.trainable = True
        else:
            layer.trainable = False

    logger.info(
        "unfreeze blocks",
        extra={"backbone": backbone_name, "blocks": n_blocks, "selected_blocks": sorted(selected)},
    )


def compile_transfer_model(model: Any, lr: float, label_smoothing: float) -> Any:
    if tf is None:
        return model
    model.compile(
        optimizer=tf.keras.optimizers.Adam(learning_rate=lr),
        loss=tf.keras.losses.BinaryCrossentropy(label_smoothing=label_smoothing),
        metrics=[
            tf.keras.metrics.BinaryAccuracy(name="binary_accuracy"),
            tf.keras.metrics.AUC(name="auc"),
        ],
    )
    return model


def build_transfer_model(
    backbone: str,
    input_size: int = 224,
    input_channels: int = 3,
    pooling: str = "avg",
    head_units: int = 0,
    dropout: float = 0.0,
    base_trainable: int = 0,
    lr: float = 1e-3,
    label_smoothing: float = 0.0,
    weights: str = "imagenet",
    num_classes: int = 1,
) -> Dict[str, Any]:
    """Build a transfer learning model and return model + preprocess handle.

    Preprocessing is expected to be applied in the input pipeline using the
    returned preprocess_input callable. Avoid applying it inside the model
    to prevent double-normalization.
    """

    if tf is None:
        logger.warning("TensorFlow not installed; returning placeholder spec for transfer model")
        return {
            "model": {
                "family": "transfer",
                "backbone": backbone,
                "input_size": input_size,
                "pooling": pooling,
            },
            "preprocess_input": None,
            "backbone_model": None,
            "backbone_name": backbone,
            "preprocess_in_model": False,
            "base_trainable": base_trainable,
        }

    if backbone not in BACKBONES:
        raise KeyError(f"Unknown backbone '{backbone}'. Available: {list(TRANSFER_BACKBONES_ALL)}")

    spec = BACKBONES[backbone]
    input_channels = int(input_channels) if input_channels else 3
    model_input_shape = (input_size, input_size, input_channels)
    require_rgb = input_channels != 3 and weights not in {None, "none"}
    backbone_input_shape = (input_size, input_size, 3) if require_rgb else model_input_shape

    try:
        base_model = spec.constructor(
            include_top=False,
            weights=weights,
            input_shape=backbone_input_shape,
            pooling=None,
        )
    except ValueError as exc:
        if "Shape mismatch" in str(exc) and weights not in {None, "none"}:
            logger.warning(
                "imagenet weights incompatible with input channels; falling back to random init",
                extra={"backbone": backbone, "input_channels": input_channels},
            )
            weights = None
            require_rgb = False
            backbone_input_shape = model_input_shape
            base_model = spec.constructor(
                include_top=False,
                weights=None,
                input_shape=backbone_input_shape,
                pooling=None,
            )
        else:
            raise
    set_backbone_trainable(base_model, False)

    inputs = tf.keras.Input(shape=model_input_shape)
    x = inputs
    if require_rgb:
        if input_channels == 1:
            x = tf.keras.layers.Lambda(lambda t: tf.image.grayscale_to_rgb(t), name="to_rgb")(x)
        else:
            x = tf.keras.layers.Lambda(lambda t: t[..., :3], name="to_rgb")(x)
    x = base_model(x, training=False)

    if pooling == "avg":
        x = tf.keras.layers.GlobalAveragePooling2D()(x)
    elif pooling == "max":
        x = tf.keras.layers.GlobalMaxPooling2D()(x)
    else:
        raise ValueError("pooling must be 'avg' or 'max'")

    if dropout and dropout > 0:
        x = tf.keras.layers.Dropout(dropout)(x)
    if head_units and head_units > 0:
        x = tf.keras.layers.Dense(head_units, activation="relu")(x)
        if dropout and dropout > 0:
            x = tf.keras.layers.Dropout(dropout)(x)

    outputs = tf.keras.layers.Dense(num_classes, activation="sigmoid")(x)
    model = tf.keras.Model(inputs, outputs, name=f"transfer_{backbone.lower()}")

    return {
        "model": model,
        "preprocess_input": spec.preprocess_input,
        "backbone_model": base_model,
        "backbone_name": backbone,
        "preprocess_in_model": False,
        "base_trainable": base_trainable,
        "input_size": input_size,
        "input_channels": input_channels,
        "pooling": pooling,
        "label_smoothing": label_smoothing,
        "lr": lr,
    }


def train_with_finetune_schedule(
    model: Any,
    backbone_model: Any,
    train_ds: Any,
    val_ds: Any,
    *,
    phase1_epochs: int,
    phase2_epochs: int,
    phase1_lr: float,
    phase2_lr: float,
    base_trainable_blocks: int,
    backbone_name: str,
    label_smoothing: float,
    callbacks: Optional[List[Any]] = None,
    class_weight: Optional[Dict[int, float]] = None,
) -> Dict[str, Any]:
    """Two-phase fine-tuning schedule with BN frozen throughout."""

    callbacks = callbacks or []
    histories: Dict[str, Any] = {}

    logger.info("phase1: head training", extra={"epochs": phase1_epochs, "lr": phase1_lr})
    set_backbone_trainable(backbone_model, False)
    compile_transfer_model(model, lr=phase1_lr, label_smoothing=label_smoothing)
    histories["phase1"] = model.fit(
        train_ds,
        validation_data=val_ds,
        epochs=phase1_epochs,
        callbacks=callbacks,
        class_weight=class_weight,
    )

    if phase2_epochs <= 0:
        return histories

    logger.info(
        "phase2: fine-tuning",
        extra={"epochs": phase2_epochs, "lr": phase2_lr, "trainable_blocks": base_trainable_blocks},
    )
    if base_trainable_blocks > 0:
        unfreeze_top_n_blocks(backbone_model, backbone_name, base_trainable_blocks)
    else:
        set_backbone_trainable(backbone_model, False)

    compile_transfer_model(model, lr=phase2_lr, label_smoothing=label_smoothing)
    histories["phase2"] = model.fit(
        train_ds,
        validation_data=val_ds,
        epochs=phase2_epochs,
        callbacks=callbacks,
        class_weight=class_weight,
    )
    return histories
