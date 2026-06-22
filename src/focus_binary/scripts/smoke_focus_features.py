from __future__ import annotations

import argparse
from pathlib import Path

import pandas as pd

from focus_binary import paths
from focus_binary.data.tfdata_features import build_feature_datasets
from focus_binary.models.focus_dnn import build_focus_dnn
from focus_binary.utils.io import load_yaml
from focus_binary.utils.logging import get_logger

logger = get_logger(__name__)

try:
    import tensorflow as tf
except Exception:  # pragma: no cover
    tf = None


def parse_args(argv: list[str] | None = None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Smoke test for focus feature pipeline.")
    parser.add_argument("--manifest", required=True, help="Manifest CSV with splits")
    parser.add_argument("--split", default="train", help="Split to sample from")
    parser.add_argument("--n-samples", type=int, default=16, help="Number of samples to load")
    parser.add_argument("--batch-size", type=int, default=16, help="Batch size")
    parser.add_argument("--input-size", type=int, default=224, help="Input size for resizing")
    return parser.parse_args(argv)


def main(argv: list[str] | None = None) -> int:
    args = parse_args(argv)
    if tf is None:
        raise ImportError("TensorFlow is required for smoke test.")

    manifest_path = Path(args.manifest)
    if not manifest_path.is_absolute():
        manifest_path = paths.PROJECT_ROOT / manifest_path

    df = pd.read_csv(manifest_path)
    if "split" in df.columns:
        df = df[df["split"] == args.split]
    if df.empty:
        raise ValueError("No rows found for requested split.")

    cfg = {}
    try:
        cfg = load_yaml(paths.CONFIG_DIR / "default.yaml")
    except FileNotFoundError:
        cfg = {}
    enabled_measures = cfg.get("enabled_focus_measures", ["lapvar", "tenengrad", "brenner", "sml"])

    sample_path = manifest_path.parent / "_smoke_focus_manifest.csv"
    df.head(args.n_samples).to_csv(sample_path, index=False)

    ds = build_feature_datasets(
        manifest_csv=sample_path,
        split=args.split,
        batch_size=args.batch_size,
        input_size=args.input_size,
        image_mode="rgb",
        enabled_measures=enabled_measures,
        augment_images=False,
        shuffle=False,
        seed=42,
        compute_from_augmented=False,
    )

    batch = next(iter(ds))
    (images, focus_vecs), labels = batch
    focus_dim = int(focus_vecs.shape[-1])

    model = build_focus_dnn(None, input_dim=focus_dim)
    preds = model(focus_vecs, training=False)

    logger.info(
        "focus feature smoke test",
        extra={
            "images_shape": tuple(images.shape),
            "focus_shape": tuple(focus_vecs.shape),
            "labels_shape": tuple(labels.shape),
            "preds_shape": tuple(preds.shape),
            "metrics": model.metrics_names,
        },
    )
    print(f"images: {tuple(images.shape)}")
    print(f"focus_vectors: {tuple(focus_vecs.shape)}")
    print(f"preds: {tuple(preds.shape)}")
    print(f"metrics: {model.metrics_names}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
