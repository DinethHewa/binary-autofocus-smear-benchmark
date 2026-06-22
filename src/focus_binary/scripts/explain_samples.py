from __future__ import annotations

import argparse
from pathlib import Path
from typing import List

import pandas as pd

from focus_binary import paths
from focus_binary.explain.protocol import run_explainability, _faithfulness_metrics  # type: ignore
from focus_binary.classical_ml.models import build_classical_models, compute_focus_vectors, predict_probabilities
from focus_binary.classical_ml.explain import explain_classical_model
from focus_binary.baselines.threshold import build_composite_scores, select_threshold
from focus_binary.eval.metrics import compute_metrics
from focus_binary.models.transfer import get_preprocess
from focus_binary.utils.io import load_yaml, save_json
from focus_binary.utils.logging import get_logger

logger = get_logger(__name__)

try:
    import tensorflow as tf
except Exception:  # pragma: no cover
    tf = None


def parse_args(argv: List[str] | None = None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Generate explanation heatmaps for sample images.")
    parser.add_argument("--model-path", default=None, help="Path to model (.keras) for deep families")
    parser.add_argument("--manifest", required=True, help="Manifest CSV with splits")
    parser.add_argument("--out-dir", required=True, help="Output directory for explanations")
    parser.add_argument(
        "--family",
        required=True,
        choices=[
            "cnn",
            "cnn_attention",
            "transfer",
            "vit",
            "hybrid_vit",
            "focus_dnn",
            "cnn_focus_hybrid",
            "convnext",
            "swin",
            "classical_ml",
            "threshold_baselines",
        ],
    )
    parser.add_argument("--split", default="test", help="Split to sample from")
    parser.add_argument("--n-samples", type=int, default=32, help="Number of samples to explain")
    parser.add_argument("--seed", type=int, default=0, help="Sampling seed")
    parser.add_argument("--backbone", default=None, help="Transfer backbone name for preprocessing")
    parser.add_argument("--layer", default=None, help="Target conv layer for Grad-CAM")
    parser.add_argument("--faithfulness-samples", type=int, default=50, help="Samples per dataset for faithfulness")
    parser.add_argument("--faithfulness", action=argparse.BooleanOptionalAction, default=True)
    return parser.parse_args(argv)


def _load_model(model_path: Path):
    if tf is None:
        raise ImportError("TensorFlow is required for explanations.")
    from focus_binary.models.vit import _CLSToken, _PositionalEmbedding  # type: ignore
    from focus_binary.models.swin_tiny import WindowPartition, WindowReverse  # type: ignore
    from focus_binary.models.convnext import ConvNeXtPreprocess  # type: ignore

    custom_objects = {}
    if _CLSToken is not None:
        custom_objects["_CLSToken"] = _CLSToken
    if _PositionalEmbedding is not None:
        custom_objects["_PositionalEmbedding"] = _PositionalEmbedding
    if WindowPartition is not None:
        custom_objects["WindowPartition"] = WindowPartition
    if WindowReverse is not None:
        custom_objects["WindowReverse"] = WindowReverse
    if ConvNeXtPreprocess is not None:
        custom_objects["ConvNeXtPreprocess"] = ConvNeXtPreprocess

    if custom_objects:
        return tf.keras.models.load_model(model_path, custom_objects=custom_objects)
    return tf.keras.models.load_model(model_path)


def main(argv: List[str] | None = None) -> Path:
    args = parse_args(argv)
    if tf is None:
        raise ImportError("TensorFlow is required for explanations.")

    out_dir = Path(args.out_dir) / args.family
    out_dir.mkdir(parents=True, exist_ok=True)

    manifest = pd.read_csv(Path(args.manifest))
    subset = manifest[manifest["split"] == args.split] if "split" in manifest.columns else manifest
    if subset.empty:
        raise ValueError("No samples found for the requested split.")

    cfg = {}
    try:
        cfg = load_yaml(paths.CONFIG_DIR / "default.yaml")
    except FileNotFoundError:
        cfg = {}
    enabled_measures = cfg.get("enabled_focus_measures", ["lapvar", "tenengrad", "brenner", "sml"])

    def _resolve_input_size() -> int:
        model_cfg = cfg.get("model") or {}
        if "input_size" in model_cfg:
            return int(model_cfg["input_size"])
        img_size = cfg.get("img_size")
        if isinstance(img_size, (list, tuple)) and len(img_size) >= 2:
            return int(img_size[0])
        return 224

    if args.family in {"classical_ml", "threshold_baselines"}:
        if "split" not in manifest.columns:
            raise ValueError("Manifest must include split column for classical/threshold explainability.")
        input_size = _resolve_input_size()
        train_df = manifest[manifest["split"] == "train"].reset_index(drop=True)
        val_df = manifest[manifest["split"] == "val"].reset_index(drop=True)
        if val_df.empty:
            val_df = train_df.copy()
            val_df["split"] = "val"

        all_vectors = compute_focus_vectors(
            manifest["image_path"].astype(str).tolist(),
            input_size=input_size,
            enabled_measures=enabled_measures,
            batch_size=64,
            manifest_path=Path(args.manifest),
        )
        train_idx = manifest.index[manifest["split"] == "train"].to_numpy()
        val_idx = manifest.index[manifest["split"] == "val"].to_numpy()
        X_train = all_vectors[train_idx]
        X_val = all_vectors[val_idx]
        y_train = manifest.loc[train_idx, "label"].to_numpy().astype(int)
        y_val = manifest.loc[val_idx, "label"].to_numpy().astype(int)

        if args.family == "classical_ml":
            models = build_classical_models(seed=args.seed)
            best_name = None
            best_auc = float("-inf")
            best_model = None
            for name, model in models.items():
                model.fit(X_train, y_train)
                y_prob_val = predict_probabilities(model, X_val)
                val_auc = compute_metrics(y_val, y_prob_val).get("auc", float("nan"))
                if val_auc > best_auc:
                    best_auc = val_auc
                    best_name = name
                    best_model = model
            if best_model is None:
                raise RuntimeError("No classical model trained.")

            explain = explain_classical_model(best_model, X_val, y_val, enabled_measures, seed=args.seed)
            ranked = explain.get("permutation_importance") or explain.get("tree_importances") or explain.get("coefficients") or []
            feature_rank = [enabled_measures.index(row["feature"]) for row in ranked if row["feature"] in enabled_measures]

            faithfulness = _faithfulness_metrics(
                model=best_model,
                source_df=subset,
                model_family="classical_ml",
                input_size=(input_size, input_size),
                preprocess_fn=None,
                enabled_measures=enabled_measures,
                per_dataset_samples=args.faithfulness_samples,
                seed=args.seed,
                layer_name=None,
                feature_rank=feature_rank,
            )
            metrics_rows = faithfulness.get("summary_rows", [])
            samples = faithfulness.get("sample_rows", [])
            metrics_path = out_dir / "explainability_metrics.csv"
            pd.DataFrame(metrics_rows).to_csv(metrics_path, index=False)
            samples_path = out_dir / "faithfulness_samples.csv"
            pd.DataFrame(samples).to_csv(samples_path, index=False)

            del_vals = [row.get("deletion_auc_feature") for row in samples if row.get("deletion_auc_feature") is not None]
            del_vals = [v for v in del_vals if v == v]
            del_mean = float(np.mean(del_vals)) if del_vals else float("nan")
            del_std = float(np.std(del_vals)) if del_vals else float("nan")

            summary_out = {
                "family": args.family,
                "split": args.split,
                "n_samples": len(samples),
                "deletion_auc_feature_mean": del_mean,
                "deletion_auc_feature_std": del_std,
                "top_features": explain.get("top_features", []),
                "explainability_metrics_csv": str(metrics_path),
                "faithfulness_samples_csv": str(samples_path),
                "best_model": best_name,
                "out_dir": str(out_dir),
            }
            summary_path = out_dir / "explainability_summary.json"
            save_json(summary_out, summary_path)
            logger.info("classical explainability complete", extra={"summary": str(summary_path)})
            return summary_path

        if args.family == "threshold_baselines":
            scores = {}
            for idx, name in enumerate(enabled_measures):
                scores[name] = X_val[:, idx]
            composite_val, weights = build_composite_scores(X_train, X_val)
            composite_test, _ = build_composite_scores(X_train, all_vectors[val_idx])
            scores["composite"] = composite_val

            best_name = None
            best_auc = float("-inf")
            best_threshold = 0.5
            for name, vals in scores.items():
                threshold = select_threshold(y_val, vals, metric="f1")
                val_auc = compute_metrics(y_val, vals, threshold=threshold).get("auc", float("nan"))
                if val_auc > best_auc:
                    best_auc = val_auc
                    best_name = name
                    best_threshold = threshold

            if best_name is None:
                raise RuntimeError("No threshold baseline selected.")

            if best_name == "composite":
                mean = X_train.mean(axis=0)
                std = X_train.std(axis=0)
                std = np.where(std == 0, 1.0, std)

                def _predict_fn(x):
                    z = (x - mean) / std
                    return z.mean(axis=1)

                feature_rank = list(np.argsort(-np.abs(weights)))
            else:
                idx = enabled_measures.index(best_name)

                def _predict_fn(x):
                    return x[:, idx]

                feature_rank = [idx] + [i for i in range(len(enabled_measures)) if i != idx]

            faithfulness = _faithfulness_metrics(
                model=None,
                source_df=subset,
                model_family="threshold_baselines",
                input_size=(input_size, input_size),
                preprocess_fn=None,
                enabled_measures=enabled_measures,
                per_dataset_samples=args.faithfulness_samples,
                seed=args.seed,
                layer_name=None,
                feature_rank=feature_rank,
                predict_fn=_predict_fn,
            )
            metrics_rows = faithfulness.get("summary_rows", [])
            samples = faithfulness.get("sample_rows", [])
            metrics_path = out_dir / "explainability_metrics.csv"
            pd.DataFrame(metrics_rows).to_csv(metrics_path, index=False)
            samples_path = out_dir / "faithfulness_samples.csv"
            pd.DataFrame(samples).to_csv(samples_path, index=False)

            del_vals = [row.get("deletion_auc_feature") for row in samples if row.get("deletion_auc_feature") is not None]
            del_vals = [v for v in del_vals if v == v]
            del_mean = float(np.mean(del_vals)) if del_vals else float("nan")
            del_std = float(np.std(del_vals)) if del_vals else float("nan")

            summary_out = {
                "family": args.family,
                "split": args.split,
                "n_samples": len(samples),
                "deletion_auc_feature_mean": del_mean,
                "deletion_auc_feature_std": del_std,
                "best_measure": best_name,
                "threshold": float(best_threshold),
                "explainability_metrics_csv": str(metrics_path),
                "faithfulness_samples_csv": str(samples_path),
                "out_dir": str(out_dir),
            }
            summary_path = out_dir / "explainability_summary.json"
            save_json(summary_out, summary_path)
            logger.info("threshold explainability complete", extra={"summary": str(summary_path)})
            return summary_path

    if args.model_path is None:
        raise ValueError("--model-path is required for deep families")

    model = _load_model(Path(args.model_path))
    preprocess_fn = get_preprocess(args.backbone) if args.family == "transfer" and args.backbone else None

    summary = run_explainability(
        model=model,
        sample_batch=subset,
        model_family=args.family,
        out_dir=out_dir,
        n_samples=args.n_samples,
        seed=args.seed,
        preprocess_fn=preprocess_fn,
        layer_name=args.layer,
        enabled_measures=enabled_measures,
        faithfulness_samples_per_dataset=args.faithfulness_samples,
        run_faithfulness=args.faithfulness,
    )

    threshold = 0.5
    records = summary.get("records", [])
    fp = [r for r in records if r.get("label") == 0 and r.get("pred", 0.0) >= threshold]
    fn = [r for r in records if r.get("label") == 1 and r.get("pred", 0.0) < threshold]
    fp = sorted(fp, key=lambda r: r.get("pred", 0.0), reverse=True)[:3]
    fn = sorted(fn, key=lambda r: r.get("pred", 0.0))[:3]

    metrics_rows = [
        {"metric": "stability_mean", "value": summary.get("stability_mean")},
        {"metric": "stability_std", "value": summary.get("stability_std")},
        {"metric": "deletion_auc_mean", "value": summary.get("deletion_auc_mean")},
        {"metric": "deletion_auc_std", "value": summary.get("deletion_auc_std")},
        {"metric": "insertion_auc_mean", "value": summary.get("insertion_auc_mean")},
        {"metric": "insertion_auc_std", "value": summary.get("insertion_auc_std")},
        {"metric": "deletion_auc_feature_mean", "value": summary.get("deletion_auc_feature_mean")},
        {"metric": "deletion_auc_feature_std", "value": summary.get("deletion_auc_feature_std")},
    ]
    if summary.get("top_features"):
        metrics_rows.append({"metric": "top_features", "value": ",".join(summary["top_features"])})
    if summary.get("ablation"):
        for key, value in summary["ablation"].items():
            metrics_rows.append({"metric": key, "value": value})

    metrics_path = out_dir / "explainability_metrics.csv"
    pd.DataFrame(metrics_rows).to_csv(metrics_path, index=False)

    summary_out = {
        "family": args.family,
        "split": args.split,
        "n_samples": summary.get("n_samples", 0),
        "stability_mean": summary.get("stability_mean"),
        "stability_std": summary.get("stability_std"),
        "deletion_auc_mean": summary.get("deletion_auc_mean"),
        "deletion_auc_std": summary.get("deletion_auc_std"),
        "insertion_auc_mean": summary.get("insertion_auc_mean"),
        "insertion_auc_std": summary.get("insertion_auc_std"),
        "deletion_auc_feature_mean": summary.get("deletion_auc_feature_mean"),
        "deletion_auc_feature_std": summary.get("deletion_auc_feature_std"),
        "records_csv": summary.get("records_csv"),
        "faithfulness_samples_csv": summary.get("faithfulness_samples_csv"),
        "explainability_metrics_csv": summary.get("explainability_metrics_csv"),
        "feature_importance_csv": summary.get("feature_importance_csv"),
        "top_features": summary.get("top_features", []),
        "false_positives": fp,
        "false_negatives": fn,
        "metrics_csv": str(metrics_path),
        "out_dir": str(out_dir),
        "hybrid_branch_ablation_csv": summary.get("hybrid_ablation_csv"),
    }
    summary_path = out_dir / "explainability_summary.json"
    save_json(summary_out, summary_path)
    logger.info("explanations written", extra={"summary": str(summary_path)})
    return summary_path


if __name__ == "__main__":
    main()
