from __future__ import annotations

import argparse
from typing import List

from focus_binary.models.transfer import (
    TRANSFER_BACKBONES_ALL,
    TRANSFER_BACKBONES_LIGHT,
    build_transfer_model,
    unfreeze_top_n_blocks,
)
from focus_binary.utils.logging import get_logger

logger = get_logger(__name__)

try:
    import tensorflow as tf
except Exception:  # pragma: no cover
    tf = None


def parse_args(argv: list[str] | None = None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Smoke test transfer backbones.")
    parser.add_argument("--set", default="light", choices=["light", "all"], help="Backbone set to test")
    parser.add_argument("--input-size", type=int, default=224, help="Input size for model creation")
    parser.add_argument("--batch", type=int, default=2, help="Batch size for forward pass")
    return parser.parse_args(argv)


def _count_trainable_non_bn(model) -> int:
    return sum(
        1
        for layer in model.layers
        if layer.trainable and not isinstance(layer, tf.keras.layers.BatchNormalization)
    )


def _run_backbone(backbone: str, input_size: int, batch: int) -> None:
    spec = build_transfer_model(
        backbone=backbone,
        input_size=input_size,
        weights=None,
        head_units=0,
        dropout=0.0,
    )
    model = spec["model"]
    preprocess = spec.get("preprocess_input")

    if spec.get("preprocess_in_model") is True:
        raise AssertionError("Preprocessing should be applied in the input pipeline only.")

    x = tf.random.uniform((batch, input_size, input_size, 3), dtype=tf.float32)
    if preprocess is not None:
        x = preprocess(x * 255.0)
    y = model(x, training=False)
    if y.shape[-1] != 1:
        raise AssertionError(f"{backbone}: expected output shape (batch, 1), got {y.shape}")

    unfreeze_top_n_blocks(spec["backbone_model"], backbone, n_blocks=2)
    trainable = _count_trainable_non_bn(spec["backbone_model"])
    if trainable == 0:
        raise AssertionError(f"{backbone}: no trainable layers after unfreeze.")

    logger.info("smoke pass", extra={"backbone": backbone, "trainable_layers": trainable})


def main(argv: list[str] | None = None) -> None:
    args = parse_args(argv)
    if tf is None:
        raise ImportError("TensorFlow is required to run transfer smoke tests.")

    backbones: List[str] = (
        TRANSFER_BACKBONES_LIGHT if args.set == "light" else TRANSFER_BACKBONES_ALL
    )
    for backbone in backbones:
        _run_backbone(backbone, input_size=args.input_size, batch=args.batch)

    print(f"Smoke test complete for {len(backbones)} backbones.")


if __name__ == "__main__":
    main()
