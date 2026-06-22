from __future__ import annotations

import argparse
import os
import sys
from pathlib import Path

import numpy as np
import pandas as pd

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))
os.environ.setdefault("TF_CPP_MIN_LOG_LEVEL", "2")

from revision_utils import (  # noqa: E402
    append_missing,
    best_baseline_model,
    build_metadata,
    config_hash,
    display_model_name,
    ensure_dir,
    fresh_all,
    load_config,
    load_manifest,
    load_predictions,
    repo_path,
    safe_write_csv,
    safe_write_text,
    save_metadata_for_outputs,
    select_prediction_source,
    standardize_model_name,
)


SCRIPT_NAME = Path(__file__).name
SCHEMA = [
    "sample_id",
    "dataset",
    "stack_id",
    "true_label",
    "probability_focused",
    "predicted_label_default_0p5",
    "model_code_name",
    "model_display_name",
    "seed",
    "split",
]


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Recover validation prediction probabilities without retraining.")
    parser.add_argument("--config", default="configs/bspc_revision.yaml")
    parser.add_argument("--force", action="store_true")
    parser.add_argument("--allow-retrain", action="store_true")
    return parser.parse_args()


def _checkpoint_for(config: dict, family: str) -> Path:
    return repo_path((config.get("paths") or {}).get("runs_dir", "runs")) / family / "best_model.keras"


def _load_existing_predictions(config: dict, manifest: pd.DataFrame, family: str, split: str) -> tuple[pd.DataFrame, Path | None, str]:
    source, message = select_prediction_source(config, family, split)
    if source is None:
        return pd.DataFrame(), None, message or f"Missing {family} {split} predictions"
    try:
        df = load_predictions(source, model_code_name=family, split=split, config=config, manifest=manifest)
    except Exception as exc:
        return pd.DataFrame(), source, f"Could not load {source}: {exc}"
    if family == "classical_ml":
        model_name = best_baseline_model((config.get("prediction_sources") or {}).get("classical_metrics"), split="val")
        if model_name and "source_model" in df.columns:
            df = df[df["source_model"] == model_name].copy()
            df["selected_submodel"] = model_name
    if family == "threshold_baselines":
        model_name = best_baseline_model((config.get("prediction_sources") or {}).get("threshold_metrics"), split="val")
        if model_name and "source_model" in df.columns:
            df = df[df["source_model"] == model_name].copy()
            df["selected_submodel"] = model_name
    if df.empty:
        return df, source, f"No usable {split} rows for {family} in {source}"
    return df, source, "reused_existing_predictions"


def _custom_objects() -> dict:
    objects = {}
    try:
        from focus_binary.models.vit import _CLSToken, _PositionalEmbedding

        if _CLSToken is not None:
            objects["_CLSToken"] = _CLSToken
        if _PositionalEmbedding is not None:
            objects["_PositionalEmbedding"] = _PositionalEmbedding
    except Exception:
        pass
    try:
        from focus_binary.models.swin_tiny import WindowPartition, WindowReverse

        if WindowPartition is not None:
            objects["WindowPartition"] = WindowPartition
        if WindowReverse is not None:
            objects["WindowReverse"] = WindowReverse
    except Exception:
        pass
    return objects


def _as_probabilities(preds) -> np.ndarray:
    arr = np.asarray(preds)
    if arr.ndim > 1:
        if arr.shape[-1] == 1:
            arr = arr.reshape(-1)
        elif arr.shape[-1] >= 2:
            arr = arr[..., 1]
    return arr.reshape(-1).astype(float)


def _evaluate_checkpoint(config: dict, manifest: pd.DataFrame, family: str, split: str) -> pd.DataFrame:
    import tensorflow as tf
    from focus_binary.data.tfdata import build_datasets
    from focus_binary.data.tfdata_features import build_feature_datasets
    from focus_binary.models.transfer import get_preprocess

    checkpoint = _checkpoint_for(config, family)
    if not checkpoint.exists():
        raise FileNotFoundError(checkpoint)
    try:
        model = tf.keras.models.load_model(checkpoint, custom_objects=_custom_objects(), compile=False, safe_mode=False)
    except TypeError:
        model = tf.keras.models.load_model(checkpoint, custom_objects=_custom_objects(), compile=False)

    hparams_path = checkpoint.parent / "best_hparams.json"
    hparams = {}
    if hparams_path.exists():
        import json

        hparams = json.loads(hparams_path.read_text(encoding="utf-8"))
    input_size = int(hparams.get("input_size", 224))
    manifest_path = repo_path((config.get("paths") or {}).get("manifest"))
    cfg = {}
    enabled_measures = ["lapvar", "tenengrad", "brenner", "sml"]
    try:
        import yaml

        default_cfg = repo_path("configs/default.yaml")
        if default_cfg and default_cfg.exists():
            cfg = yaml.safe_load(default_cfg.read_text(encoding="utf-8")) or {}
            enabled_measures = cfg.get("enabled_focus_measures", enabled_measures)
    except Exception:
        pass
    split_df = manifest[manifest["split"].astype(str).str.lower() == split.lower()].reset_index(drop=True)
    if split_df.empty:
        raise ValueError(f"No {split} rows in manifest.")

    if family in {"focus_dnn", "cnn_focus_hybrid"}:
        ds = build_feature_datasets(
            manifest_csv=manifest_path,
            split=split,
            batch_size=16,
            input_size=input_size,
            image_mode="rgb",
            enabled_measures=enabled_measures,
            augment_images=False,
            shuffle=False,
            seed=42,
            compute_from_augmented=bool(cfg.get("focus_vector_from_augmented", False)),
        )
        if family == "focus_dnn":
            ds = ds.map(lambda x, y: (x[1], y), num_parallel_calls=tf.data.AUTOTUNE)
    else:
        ds = build_datasets(
            manifest_csv=manifest_path,
            split=split,
            batch_size=16,
            input_size=input_size,
            image_mode="rgb",
            augment=False,
            shuffle=False,
            seed=42,
            force_rgb=True,
        )
    if family == "transfer":
        preprocess = get_preprocess(str(hparams.get("backbone", "MobileNetV2")))

        def _apply_preprocess(img, label):
            return preprocess(img * 255.0), label

        ds = ds.map(_apply_preprocess, num_parallel_calls=tf.data.AUTOTUNE)

    y_prob = _as_probabilities(model.predict(ds, verbose=0))
    if len(y_prob) != len(split_df):
        raise ValueError(f"Predicted {len(y_prob)} rows but manifest split has {len(split_df)} rows")
    out = pd.DataFrame(
        {
            "sample_id": split_df.get("sample_id", split_df["image_path"].map(lambda x: Path(str(x)).stem)),
            "dataset": split_df["dataset"],
            "image_path": split_df["image_path"],
            "true_label": split_df["label"].astype(int),
            "probability_focused": y_prob,
            "split": split,
            "model_code_name": family,
            "model_display_name": display_model_name(family, config),
            "seed": "",
        }
    )
    for col in ["stack_id", "patient_id", "source"]:
        if col in split_df.columns:
            out[col] = split_df[col]
    out["dataset"] = out["dataset"].map(lambda x: config.get("datasets", {}).get("code_to_display", {}).get(x, x))
    out["predicted_label_default_0p5"] = (out["probability_focused"] >= 0.5).astype(int)
    return out


def _standard_schema(df: pd.DataFrame, family: str, split: str, config: dict) -> pd.DataFrame:
    out = df.copy()
    out["model_code_name"] = family
    out["model_display_name"] = display_model_name(family, config)
    out["split"] = split
    if "stack_id" not in out.columns:
        out["stack_id"] = ""
    if "seed" not in out.columns:
        out["seed"] = ""
    out["predicted_label_default_0p5"] = (out["probability_focused"].astype(float) >= 0.5).astype(int)
    for col in SCHEMA:
        if col not in out.columns:
            out[col] = ""
    keep = SCHEMA + [c for c in ["image_path", "dataset_code", "source_model", "selected_submodel"] if c in out.columns]
    return out[keep].reset_index(drop=True)


def main() -> int:
    args = parse_args()
    config = load_config(args.config)
    cfg_hash = config_hash(config)
    out_dir = ensure_dir((config.get("subdirs") or {}).get("predictions", "revision_outputs/predictions"))
    logs_dir = ensure_dir((config.get("subdirs") or {}).get("logs", "revision_outputs/logs"))
    families = [standardize_model_name(m) for m in config.get("model_families", [])]
    outputs = [out_dir / f"validation_predictions_{m}.csv" for m in families]
    outputs.extend([out_dir / f"test_predictions_{m}.csv" for m in families])
    outputs.extend([out_dir / "validation_prediction_inventory.csv", logs_dir / "validation_prediction_recovery.log"])
    input_files = [args.config, repo_path((config.get("paths") or {}).get("manifest"))]
    for family in families:
        for split in ["val", "test"]:
            source, _ = select_prediction_source(config, family, split)
            if source:
                input_files.append(source)
        input_files.append(_checkpoint_for(config, family))

    if not args.force and fresh_all(outputs, input_files, cfg_hash):
        print("validation prediction outputs are fresh; skipping")
        return 0

    manifest = load_manifest(config=config)
    inventory = []
    log_lines = []
    generated_outputs = []

    for family in families:
        for split in ["val", "test"]:
            out_path = out_dir / (f"validation_predictions_{family}.csv" if split == "val" else f"test_predictions_{family}.csv")
            df, source, status = _load_existing_predictions(config, manifest, family, split)
            recovery_method = "existing_prediction_csv"
            message = status
            if df.empty and family not in {"classical_ml", "threshold_baselines"}:
                checkpoint = _checkpoint_for(config, family)
                if checkpoint.exists():
                    try:
                        df = _evaluate_checkpoint(config, manifest, family, split)
                        source = checkpoint
                        recovery_method = "checkpoint_evaluation"
                        message = "evaluated_existing_checkpoint"
                    except Exception as exc:
                        message = f"checkpoint_evaluation_failed: {exc}"
                        append_missing(f"Validation recovery failed for {family} {split}: {exc}", config)
                else:
                    message = f"missing_predictions_and_checkpoint: {family} {split}"
                    append_missing(message, config)
            elif df.empty:
                append_missing(message or f"Missing {family} {split} predictions", config)

            if not df.empty:
                std = _standard_schema(df, family, split, config)
                safe_write_csv(std, out_path)
                generated_outputs.append(out_path)
                row_count = len(std)
                final_status = "recovered"
            else:
                row_count = 0
                final_status = "missing_retrain_required" if args.allow_retrain else "missing_no_retrain"

            inventory.append(
                {
                    "model_code_name": family,
                    "model_display_name": display_model_name(family, config),
                    "split": split,
                    "status": final_status,
                    "recovery_method": recovery_method if row_count else "",
                    "source_path": str(source) if source else "",
                    "output_path": str(out_path) if row_count else "",
                    "rows": row_count,
                    "checkpoint_path": str(_checkpoint_for(config, family)),
                    "checkpoint_exists": _checkpoint_for(config, family).exists(),
                    "message": message,
                }
            )
            log_lines.append(f"{family}\t{split}\t{final_status}\t{message}")

    inventory_path = out_dir / "validation_prediction_inventory.csv"
    log_path = logs_dir / "validation_prediction_recovery.log"
    safe_write_csv(pd.DataFrame(inventory), inventory_path)
    safe_write_text("\n".join(log_lines) + "\n", log_path)
    generated_outputs.extend([inventory_path, log_path])
    metadata = build_metadata(script_name=SCRIPT_NAME, input_files=input_files, cfg_hash=cfg_hash, args=args)
    save_metadata_for_outputs(generated_outputs, metadata)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
