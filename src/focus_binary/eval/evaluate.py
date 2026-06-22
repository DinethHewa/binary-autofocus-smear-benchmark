from __future__ import annotations

from collections.abc import Mapping
from pathlib import Path
from typing import Any, Dict, Iterable, List, Optional, Tuple

import numpy as np
import pandas as pd

from focus_binary import paths
from focus_binary.calib.calibration import (
    choose_threshold,
    compute_brier,
    compute_ece,
    reliability_bins,
)
from focus_binary.eval.metrics import compute_metrics
from focus_binary.utils.io import save_json
from focus_binary.utils.logging import get_logger

logger = get_logger(__name__)


def _to_numpy(values) -> np.ndarray:
    if hasattr(values, "numpy"):
        return values.numpy()
    return np.asarray(values)


def _resolve_dataset_input(ds_input: Any) -> Tuple[Any, Optional[pd.DataFrame]]:
    if isinstance(ds_input, tuple) and len(ds_input) == 2 and isinstance(ds_input[1], pd.DataFrame):
        return ds_input[0], ds_input[1]
    return ds_input, None


def _predict_probs(model: Any, dataset: Any) -> np.ndarray:
    preds = model.predict(dataset)
    preds = np.asarray(preds)
    if preds.ndim > 1:
        if preds.shape[-1] == 1:
            preds = preds.reshape(-1)
        elif preds.shape[-1] >= 2:
            preds = preds[..., 1]
    return preds.reshape(-1)


def _extract_labels_and_meta(
    dataset: Iterable,
    dataset_name: str,
) -> Tuple[np.ndarray, List[str], List[str], List[str]]:
    y_true: List[int] = []
    image_paths: List[str] = []
    datasets: List[str] = []
    splits: List[str] = []

    for batch in dataset:
        x = batch
        meta = None
        labels = None

        if isinstance(batch, (tuple, list)):
            if len(batch) >= 2:
                labels = batch[1]
            if len(batch) >= 3:
                meta = batch[2]
        elif isinstance(batch, Mapping):
            labels = batch.get("label") or batch.get("labels") or batch.get("y")
            meta = batch

        if labels is None:
            raise ValueError("Dataset batches must include labels for evaluation.")

        labels_np = _to_numpy(labels).reshape(-1).astype(int).tolist()
        y_true.extend(labels_np)

        if isinstance(meta, Mapping):
            meta_paths = meta.get("image_path")
            meta_datasets = meta.get("dataset")
            meta_splits = meta.get("split")

            if meta_paths is not None:
                image_paths.extend([str(p) for p in _to_numpy(meta_paths).reshape(-1)])
            if meta_datasets is not None:
                datasets.extend([str(d) for d in _to_numpy(meta_datasets).reshape(-1)])
            if meta_splits is not None:
                splits.extend([str(s) for s in _to_numpy(meta_splits).reshape(-1)])

    if not image_paths:
        image_paths = [""] * len(y_true)
    if not datasets:
        datasets = [dataset_name] * len(y_true)
    if not splits:
        splits = [""] * len(y_true)

    return np.asarray(y_true), image_paths, datasets, splits


def _rows_from_meta(
    y_true: np.ndarray,
    y_prob: np.ndarray,
    image_paths: List[str],
    datasets: List[str],
    splits: List[str],
) -> List[Dict[str, Any]]:
    n = min(len(y_true), len(y_prob), len(image_paths), len(datasets), len(splits))
    rows = []
    for i in range(n):
        rows.append(
            {
                "image_path": image_paths[i],
                "y_true": int(y_true[i]),
                "y_prob": float(y_prob[i]),
                "split": splits[i],
                "dataset": datasets[i],
            }
        )
    return rows


def _calibration_payload(y_true: np.ndarray, y_prob: np.ndarray, n_bins: int = 15) -> Dict[str, Any]:
    return {
        "ece": compute_ece(y_true, y_prob, n_bins=n_bins),
        "brier": compute_brier(y_true, y_prob),
        "reliability": reliability_bins(y_true, y_prob, n_bins=n_bins),
    }


def _evaluate_dataset(
    model: Any,
    dataset: Any,
    threshold: float,
    dataset_name: str,
    meta_df: Optional[pd.DataFrame],
) -> Tuple[np.ndarray, np.ndarray, List[Dict[str, Any]]]:
    y_prob = _predict_probs(model, dataset)

    if meta_df is not None:
        meta_df = meta_df.reset_index(drop=True)
        y_true = meta_df["label"].to_numpy().astype(int) if "label" in meta_df.columns else None
        image_paths = meta_df.get("image_path", pd.Series([""] * len(meta_df))).astype(str).tolist()
        datasets = meta_df.get("dataset", pd.Series([dataset_name] * len(meta_df))).astype(str).tolist()
        splits = meta_df.get("split", pd.Series([""] * len(meta_df))).astype(str).tolist()
        if y_true is None:
            y_true, image_paths, datasets, splits = _extract_labels_and_meta(dataset, dataset_name)
    else:
        y_true, image_paths, datasets, splits = _extract_labels_and_meta(dataset, dataset_name)

    rows = _rows_from_meta(y_true, y_prob, image_paths, datasets, splits)
    return y_true, y_prob, rows


def evaluate_model(
    model: Any,
    ds_test: Any,
    threshold: float = 0.5,
    preds_path: str | Path | None = None,
    y_true_val: np.ndarray | None = None,
    y_prob_val: np.ndarray | None = None,
    threshold_metric: str = "f1",
    reliability_path: str | Path | None = None,
) -> Dict[str, Any]:
    """Evaluate a model on a dataset or mapping of datasets.

    If validation labels/probabilities are provided, a threshold is selected on
    validation data and applied to test metrics.
    """

    per_dataset: Dict[str, Dict[str, Any]] = {}
    all_true: List[np.ndarray] = []
    all_prob: List[np.ndarray] = []
    all_rows: List[Dict[str, Any]] = []
    threshold_used = float(threshold)
    reliability_payload: Dict[str, Any] = {}
    val_metrics: Dict[str, Any] | None = None

    if y_true_val is not None and y_prob_val is not None:
        threshold_used = choose_threshold(y_true_val, y_prob_val, metric=threshold_metric)
        val_metrics = compute_metrics(y_true_val, y_prob_val, threshold=threshold_used)
        val_calib = _calibration_payload(np.asarray(y_true_val), np.asarray(y_prob_val))
        val_metrics.update({k: val_calib[k] for k in ("ece", "brier")})
        reliability_payload["val"] = val_calib["reliability"]

    if isinstance(ds_test, Mapping):
        for dataset_name, ds_input in ds_test.items():
            dataset, meta_df = _resolve_dataset_input(ds_input)
            y_true, y_prob, rows = _evaluate_dataset(
                model,
                dataset,
                threshold,
                dataset_name=dataset_name,
                meta_df=meta_df,
            )
            metrics = compute_metrics(y_true, y_prob, threshold=threshold_used)
            metrics.update(
                {
                    "ece": compute_ece(y_true, y_prob),
                    "brier": compute_brier(y_true, y_prob),
                }
            )
            metrics["dataset"] = dataset_name
            per_dataset[dataset_name] = metrics
            all_true.append(y_true)
            all_prob.append(y_prob)
            all_rows.extend(rows)
    else:
        dataset, meta_df = _resolve_dataset_input(ds_test)
        y_true, y_prob, rows = _evaluate_dataset(
            model,
            dataset,
            threshold,
            dataset_name="",
            meta_df=meta_df,
        )
        all_true.append(y_true)
        all_prob.append(y_prob)
        all_rows.extend(rows)

    y_true_all = np.concatenate(all_true) if all_true else np.asarray([])
    y_prob_all = np.concatenate(all_prob) if all_prob else np.asarray([])
    overall = compute_metrics(y_true_all, y_prob_all, threshold=threshold_used) if len(y_true_all) else {}
    if len(y_true_all):
        calib = _calibration_payload(y_true_all, y_prob_all)
        overall.update({k: calib[k] for k in ("ece", "brier")})
        reliability_payload["test"] = calib["reliability"]

    preds_path = Path(preds_path) if preds_path else (paths.ARTIFACT_DIR / "predictions.csv")
    if not preds_path.is_absolute():
        preds_path = paths.PROJECT_ROOT / preds_path
    preds_path.parent.mkdir(parents=True, exist_ok=True)
    pd.DataFrame(all_rows).to_csv(preds_path, index=False)

    if reliability_payload:
        reliability_path = Path(reliability_path) if reliability_path else preds_path.with_name("reliability.json")
        if not reliability_path.is_absolute():
            reliability_path = paths.PROJECT_ROOT / reliability_path
        save_json(
            {
                "threshold": threshold_used,
                "threshold_metric": threshold_metric,
                "reliability": reliability_payload,
            },
            reliability_path,
        )

    return {
        "overall": overall,
        "per_dataset": per_dataset,
        "predictions_csv": str(preds_path),
        "threshold": threshold_used,
        "threshold_metric": threshold_metric,
        "val_metrics": val_metrics,
    }
