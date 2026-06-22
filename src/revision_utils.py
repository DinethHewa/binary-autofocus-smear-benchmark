from __future__ import annotations

import argparse
import hashlib
import json
import math
import os
import subprocess
import sys
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Callable, Iterable

import numpy as np
import pandas as pd

try:
    import yaml
except Exception:  # pragma: no cover
    yaml = None

try:
    from sklearn.metrics import matthews_corrcoef, roc_auc_score
except Exception:  # pragma: no cover
    matthews_corrcoef = None
    roc_auc_score = None

try:
    from scipy.stats import binomtest, chi2
except Exception:  # pragma: no cover
    binomtest = None
    chi2 = None


PROJECT_ROOT = Path(__file__).resolve().parents[1]

MODEL_DISPLAY_NAMES = {
    "cnn": "CNN",
    "cnn_attention": "Attention-CNN",
    "cnn_focus_hybrid": "CNN--focus hybrid",
    "transfer": "Transfer learning",
    "hybrid_vit": "Hybrid ViT",
    "focus_dnn": "Focus-DNN",
    "classical_ml": "Classical ML",
    "threshold_baselines": "Threshold baselines",
    "vit": "ViT",
}

MODEL_ALIASES = {
    "attention-cnn": "cnn_attention",
    "attention_cnn": "cnn_attention",
    "cnn-focus-hybrid": "cnn_focus_hybrid",
    "cnn_focus": "cnn_focus_hybrid",
    "cnn--focus hybrid": "cnn_focus_hybrid",
    "transfer learning": "transfer",
    "hybrid vit": "hybrid_vit",
    "focus-dnn": "focus_dnn",
    "classical": "classical_ml",
    "classical_ml:gradient_boosting": "classical_ml",
    "threshold": "threshold_baselines",
    "threshold baseline": "threshold_baselines",
    "threshold_baseline": "threshold_baselines",
}

DATASET_DISPLAY = {
    "wbc": "WBC",
    "WBC": "WBC",
    "WBC-MF": "WBC",
    "tbf_imgs": "TBF",
    "TBF": "TBF",
    "pbs_imgs": "PBS",
    "PBS": "PBS",
    "bma": "BMA",
    "BMA": "BMA",
    "TBSI": "TBSI",
    "TBI": "TBSI",
}


def repo_path(path: str | Path | None) -> Path | None:
    if path is None:
        return None
    p = Path(path)
    if p.is_absolute():
        return p
    return PROJECT_ROOT / p


def load_config(path: str | Path) -> dict[str, Any]:
    if yaml is None:
        raise ImportError("PyYAML is required to load revision configs.")
    p = repo_path(path)
    if p is None or not p.exists():
        raise FileNotFoundError(path)
    with p.open("r", encoding="utf-8") as f:
        cfg = yaml.safe_load(f) or {}
    cfg["_config_path"] = str(p)
    return cfg


def config_hash(config: dict[str, Any]) -> str:
    clean = {k: v for k, v in dict(config).items() if not str(k).startswith("_")}
    payload = json.dumps(clean, sort_keys=True, default=str, separators=(",", ":"))
    return hashlib.sha256(payload.encode("utf-8")).hexdigest()


def ensure_dir(path: str | Path) -> Path:
    p = repo_path(path) or Path(path)
    p.mkdir(parents=True, exist_ok=True)
    return p


def metadata_path(output_path: str | Path) -> Path:
    p = repo_path(output_path) or Path(output_path)
    return p.with_name(p.name + ".metadata.json")


def _mtime(path: str | Path) -> float:
    p = repo_path(path) or Path(path)
    return p.stat().st_mtime


def output_is_fresh(output_path: str | Path, input_paths: Iterable[str | Path], config_hash_value: str | None) -> bool:
    out = repo_path(output_path) or Path(output_path)
    if not out.exists():
        return False
    meta = metadata_path(out)
    if not meta.exists():
        return False
    try:
        with meta.open("r", encoding="utf-8") as f:
            metadata = json.load(f)
    except Exception:
        return False
    if config_hash_value and metadata.get("config_hash") != config_hash_value:
        return False
    out_mtime = out.stat().st_mtime
    for item in input_paths:
        if item is None:
            continue
        p = repo_path(item) or Path(item)
        if p.exists() and p.stat().st_mtime > out_mtime:
            return False
    return True


def git_commit_hash() -> str | None:
    try:
        result = subprocess.run(
            ["git", "rev-parse", "HEAD"],
            cwd=PROJECT_ROOT,
            capture_output=True,
            text=True,
            check=False,
        )
    except Exception:
        return None
    if result.returncode != 0:
        return None
    value = result.stdout.strip()
    return value or None


def save_run_metadata(output_path: str | Path, metadata: dict[str, Any]) -> Path:
    out = repo_path(output_path) or Path(output_path)
    meta = dict(metadata)
    meta.setdefault("timestamp", datetime.now(timezone.utc).isoformat())
    meta.setdefault("git_commit_hash", git_commit_hash())
    safe_write_json(meta, metadata_path(out))
    return metadata_path(out)


def build_metadata(
    *,
    script_name: str,
    input_files: Iterable[str | Path],
    cfg_hash: str,
    args: argparse.Namespace | dict[str, Any] | None = None,
) -> dict[str, Any]:
    inputs = []
    mtimes = {}
    for item in input_files:
        if item is None:
            continue
        p = repo_path(item) or Path(item)
        inputs.append(str(p))
        if p.exists():
            mtimes[str(p)] = p.stat().st_mtime
    if isinstance(args, argparse.Namespace):
        arg_payload = vars(args)
    else:
        arg_payload = args or {}
    return {
        "script_name": script_name,
        "input_files": inputs,
        "input_file_modification_times": mtimes,
        "config_hash": cfg_hash,
        "command_line_arguments": arg_payload,
    }


def standardize_model_name(code_name: str | None) -> str:
    if code_name is None or (isinstance(code_name, float) and pd.isna(code_name)):
        return ""
    name = str(code_name).strip()
    if not name:
        return ""
    key = name.lower().replace("-", "_")
    if key in MODEL_DISPLAY_NAMES:
        return key
    if key in MODEL_ALIASES:
        return MODEL_ALIASES[key]
    if ":" in key:
        family = key.split(":", 1)[0]
        if family in MODEL_DISPLAY_NAMES:
            return family
    return key


def display_model_name(code_name: str | None, config: dict[str, Any] | None = None) -> str:
    family = standardize_model_name(code_name)
    if config:
        names = config.get("publication_display_names") or {}
        if family in names:
            return str(names[family])
    return MODEL_DISPLAY_NAMES.get(family, family)


def standardize_dataset_name(name: str | None, config: dict[str, Any] | None = None) -> str:
    if name is None or (isinstance(name, float) and pd.isna(name)):
        return ""
    raw = str(name).strip()
    if config:
        mapping = (config.get("datasets") or {}).get("code_to_display") or {}
        if raw in mapping:
            return str(mapping[raw])
    return DATASET_DISPLAY.get(raw, DATASET_DISPLAY.get(raw.lower(), raw))


def load_manifest(path: str | Path | None = None, config: dict[str, Any] | None = None) -> pd.DataFrame:
    if path is None and config is not None:
        path = (config.get("paths") or {}).get("manifest")
    if path is None:
        path = PROJECT_ROOT / "data" / "manifest_with_splits.csv"
    p = repo_path(path) or Path(path)
    if not p.exists():
        raise FileNotFoundError(p)
    df = pd.read_csv(p)
    if "sample_id" not in df.columns:
        if "image_path" in df.columns:
            df["sample_id"] = df["image_path"].map(lambda x: Path(str(x)).stem)
        else:
            df["sample_id"] = np.arange(len(df)).astype(str)
    if "dataset_display" not in df.columns and "dataset" in df.columns:
        df["dataset_display"] = df["dataset"].map(lambda x: standardize_dataset_name(x, config))
    return df


def _first_present(df: pd.DataFrame, candidates: list[str]) -> str | None:
    for col in candidates:
        if col in df.columns:
            return col
    return None


def load_predictions(
    path: str | Path,
    *,
    model_code_name: str | None = None,
    model_display_name: str | None = None,
    split: str | None = None,
    config: dict[str, Any] | None = None,
    manifest: pd.DataFrame | None = None,
) -> pd.DataFrame:
    p = repo_path(path) or Path(path)
    if not p.exists():
        raise FileNotFoundError(p)
    raw = pd.read_csv(p)
    df = raw.copy()

    y_col = _first_present(df, ["true_label", "y_true", "label", "target"])
    p_col = _first_present(df, ["probability_focused", "y_prob", "prob", "probability", "score"])
    if y_col is None or p_col is None:
        raise ValueError(f"Prediction file lacks label/probability columns: {p}")

    out = pd.DataFrame()
    if "sample_id" in df.columns:
        out["sample_id"] = df["sample_id"].astype(str)
    elif "image_path" in df.columns:
        out["sample_id"] = df["image_path"].map(lambda x: Path(str(x)).stem)
    else:
        out["sample_id"] = np.arange(len(df)).astype(str)

    if "dataset" in df.columns:
        out["dataset"] = df["dataset"].map(lambda x: standardize_dataset_name(x, config))
        out["dataset_code"] = df["dataset"].astype(str)
    else:
        out["dataset"] = ""
        out["dataset_code"] = ""

    if "image_path" in df.columns:
        out["image_path"] = df["image_path"].astype(str)
    out["true_label"] = pd.to_numeric(df[y_col], errors="coerce").fillna(0).astype(int)
    out["probability_focused"] = pd.to_numeric(df[p_col], errors="coerce").astype(float)
    out["probability_focused"] = out["probability_focused"].clip(0.0, 1.0)

    if "split" in df.columns:
        out["split"] = df["split"].astype(str)
    else:
        out["split"] = split or ""
    if split is not None:
        out = out[out["split"].astype(str).str.lower() == split.lower()].copy()

    source_family = model_code_name
    if source_family is None:
        source_family = _first_present(df, ["model_code_name", "family"])
        if source_family is not None:
            source_family = str(df[source_family].dropna().iloc[0]) if len(df[source_family].dropna()) else None
    source_family = standardize_model_name(source_family or "")
    out["model_code_name"] = source_family
    out["model_display_name"] = model_display_name or display_model_name(source_family, config)

    if "model" in df.columns:
        out["source_model"] = df["model"].astype(str)
    elif "model_name" in df.columns:
        out["source_model"] = df["model_name"].astype(str)

    if "seed" in df.columns:
        out["seed"] = df["seed"]
    else:
        out["seed"] = ""
    if "stack_id" in df.columns:
        out["stack_id"] = df["stack_id"].astype(str)
    if "patient_id" in df.columns:
        out["patient_id"] = df["patient_id"].astype(str)
    if "source" in df.columns:
        out["source"] = df["source"].astype(str)

    if manifest is not None and "image_path" in out.columns and "image_path" in manifest.columns:
        wanted = ["image_path"]
        for col in ["stack_id", "patient_id", "source"]:
            if col in manifest.columns and col not in out.columns:
                wanted.append(col)
        if len(wanted) > 1:
            merge_df = manifest[wanted].drop_duplicates("image_path")
            out = out.merge(merge_df, on="image_path", how="left")

    out["predicted_label_default_0p5"] = (out["probability_focused"] >= 0.5).astype(int)
    return out.reset_index(drop=True)


def _manual_auc(y_true: np.ndarray, y_prob: np.ndarray) -> float:
    y_true = np.asarray(y_true).astype(int)
    y_prob = np.asarray(y_prob).astype(float)
    pos = y_true == 1
    neg = y_true == 0
    n_pos = int(pos.sum())
    n_neg = int(neg.sum())
    if n_pos == 0 or n_neg == 0:
        return float("nan")
    order = np.argsort(y_prob)
    ranks = np.empty_like(order, dtype=float)
    ranks[order] = np.arange(1, len(y_prob) + 1)
    unique_vals, inverse, counts = np.unique(y_prob, return_inverse=True, return_counts=True)
    if len(unique_vals) < len(y_prob):
        for idx, count in enumerate(counts):
            if count > 1:
                tie_mask = inverse == idx
                ranks[tie_mask] = ranks[tie_mask].mean()
    rank_sum_pos = ranks[pos].sum()
    return float((rank_sum_pos - n_pos * (n_pos + 1) / 2) / (n_pos * n_neg))


def _auc(y_true: np.ndarray, y_prob: np.ndarray) -> float:
    y_true = np.asarray(y_true).astype(int)
    y_prob = np.asarray(y_prob).astype(float)
    if len(np.unique(y_true)) < 2:
        return float("nan")
    if roc_auc_score is not None:
        try:
            return float(roc_auc_score(y_true, y_prob))
        except Exception:
            pass
    return _manual_auc(y_true, y_prob)


def compute_ece(y_true: Iterable[int], y_prob: Iterable[float], n_bins: int = 10) -> float:
    y_true_arr = np.asarray(list(y_true), dtype=float).reshape(-1)
    y_prob_arr = np.asarray(list(y_prob), dtype=float).reshape(-1)
    if len(y_true_arr) == 0:
        return float("nan")
    y_prob_arr = np.clip(y_prob_arr, 0.0, 1.0)
    bins = np.linspace(0.0, 1.0, n_bins + 1)
    ece = 0.0
    for idx in range(n_bins):
        lo, hi = bins[idx], bins[idx + 1]
        if idx == n_bins - 1:
            mask = (y_prob_arr >= lo) & (y_prob_arr <= hi)
        else:
            mask = (y_prob_arr >= lo) & (y_prob_arr < hi)
        if not np.any(mask):
            continue
        conf = float(y_prob_arr[mask].mean())
        acc = float(y_true_arr[mask].mean())
        ece += float(mask.mean()) * abs(acc - conf)
    return float(ece)


def compute_brier(y_true: Iterable[int], y_prob: Iterable[float]) -> float:
    y_true_arr = np.asarray(list(y_true), dtype=float).reshape(-1)
    y_prob_arr = np.asarray(list(y_prob), dtype=float).reshape(-1)
    if len(y_true_arr) == 0:
        return float("nan")
    return float(np.mean((np.clip(y_prob_arr, 0.0, 1.0) - y_true_arr) ** 2))


def compute_binary_metrics(y_true: Iterable[int], y_prob: Iterable[float], threshold: float = 0.5) -> dict[str, Any]:
    y_true_arr = np.asarray(list(y_true), dtype=int).reshape(-1)
    y_prob_arr = np.asarray(list(y_prob), dtype=float).reshape(-1)
    if len(y_true_arr) != len(y_prob_arr):
        raise ValueError("y_true and y_prob must have the same length")
    if len(y_true_arr) == 0:
        return {
            "n": 0,
            "TP": 0,
            "TN": 0,
            "FP": 0,
            "FN": 0,
            "accuracy": float("nan"),
            "precision": float("nan"),
            "recall": float("nan"),
            "specificity": float("nan"),
            "balanced_accuracy": float("nan"),
            "MCC": float("nan"),
            "AUC": float("nan"),
            "F1": float("nan"),
            "ECE": float("nan"),
            "Brier score": float("nan"),
        }
    y_hat = (y_prob_arr >= threshold).astype(int)
    tp = int(((y_hat == 1) & (y_true_arr == 1)).sum())
    tn = int(((y_hat == 0) & (y_true_arr == 0)).sum())
    fp = int(((y_hat == 1) & (y_true_arr == 0)).sum())
    fn = int(((y_hat == 0) & (y_true_arr == 1)).sum())
    n = int(len(y_true_arr))
    accuracy = (tp + tn) / n
    precision = tp / (tp + fp) if (tp + fp) else 0.0
    recall = tp / (tp + fn) if (tp + fn) else 0.0
    specificity = tn / (tn + fp) if (tn + fp) else 0.0
    balanced_accuracy = (recall + specificity) / 2.0
    f1 = (2 * precision * recall / (precision + recall)) if (precision + recall) else 0.0
    if matthews_corrcoef is not None and len(np.unique(y_hat)) > 1 and len(np.unique(y_true_arr)) > 1:
        mcc = float(matthews_corrcoef(y_true_arr, y_hat))
    else:
        denom = math.sqrt((tp + fp) * (tp + fn) * (tn + fp) * (tn + fn))
        mcc = ((tp * tn - fp * fn) / denom) if denom else 0.0
    return {
        "n": n,
        "TP": tp,
        "TN": tn,
        "FP": fp,
        "FN": fn,
        "predicted_positive_count": int(y_hat.sum()),
        "predicted_negative_count": int((1 - y_hat).sum()),
        "accuracy": float(accuracy),
        "precision": float(precision),
        "recall": float(recall),
        "specificity": float(specificity),
        "balanced_accuracy": float(balanced_accuracy),
        "MCC": float(mcc),
        "AUC": _auc(y_true_arr, y_prob_arr),
        "F1": float(f1),
        "ECE": compute_ece(y_true_arr, y_prob_arr),
        "Brier score": compute_brier(y_true_arr, y_prob_arr),
        "false_acceptance_rate": fp / (fp + tn) if (fp + tn) else float("nan"),
        "false_rejection_rate": fn / (fn + tp) if (fn + tp) else float("nan"),
    }


def _metric_value(metric: str | Callable[[np.ndarray, np.ndarray], float], y_true: np.ndarray, y_prob: np.ndarray, threshold: float) -> float:
    if callable(metric):
        return float(metric(y_true, y_prob))
    metrics = compute_binary_metrics(y_true, y_prob, threshold=threshold)
    aliases = {
        "auc": "AUC",
        "f1": "F1",
        "mcc": "MCC",
        "ece": "ECE",
        "brier": "Brier score",
        "brier_score": "Brier score",
    }
    key = aliases.get(metric.lower(), metric)
    return float(metrics.get(key, float("nan")))


def bootstrap_ci(
    y_true: Iterable[int],
    y_prob: Iterable[float],
    metric: str | Callable[[np.ndarray, np.ndarray], float],
    *,
    n_iterations: int = 2000,
    confidence_level: float = 0.95,
    seed: int = 42,
    threshold: float = 0.5,
) -> dict[str, float]:
    y_true_arr = np.asarray(list(y_true), dtype=int).reshape(-1)
    y_prob_arr = np.asarray(list(y_prob), dtype=float).reshape(-1)
    if len(y_true_arr) == 0:
        return {"estimate": float("nan"), "ci_low": float("nan"), "ci_high": float("nan")}
    estimate = _metric_value(metric, y_true_arr, y_prob_arr, threshold)
    rng = np.random.default_rng(seed)
    values = []
    for _ in range(int(n_iterations)):
        idx = rng.integers(0, len(y_true_arr), len(y_true_arr))
        yt = y_true_arr[idx]
        yp = y_prob_arr[idx]
        try:
            value = _metric_value(metric, yt, yp, threshold)
        except Exception:
            value = float("nan")
        if not np.isnan(value):
            values.append(value)
    if not values:
        return {"estimate": float(estimate), "ci_low": float("nan"), "ci_high": float("nan")}
    alpha = 1.0 - confidence_level
    low, high = np.quantile(values, [alpha / 2.0, 1.0 - alpha / 2.0])
    return {"estimate": float(estimate), "ci_low": float(low), "ci_high": float(high)}


def paired_mcnemar(y_true: Iterable[int], pred_a: Iterable[int], pred_b: Iterable[int]) -> dict[str, float]:
    y = np.asarray(list(y_true), dtype=int)
    a = np.asarray(list(pred_a), dtype=int)
    b = np.asarray(list(pred_b), dtype=int)
    correct_a = a == y
    correct_b = b == y
    b_count = int((correct_a & ~correct_b).sum())
    c_count = int((~correct_a & correct_b).sum())
    n = b_count + c_count
    if n == 0:
        return {"b": b_count, "c": c_count, "statistic": 0.0, "p_value": 1.0, "method": "exact_binomial"}
    if binomtest is not None:
        p_value = float(binomtest(min(b_count, c_count), n=n, p=0.5, alternative="two-sided").pvalue)
        stat = float((abs(b_count - c_count) - 1) ** 2 / n) if n else 0.0
        return {"b": b_count, "c": c_count, "statistic": stat, "p_value": p_value, "method": "exact_binomial"}
    stat = float((abs(b_count - c_count) - 1) ** 2 / n)
    if chi2 is not None:
        p_value = float(chi2.sf(stat, 1))
    else:
        p_value = float(math.erfc(math.sqrt(stat / 2.0)))
    return {"b": b_count, "c": c_count, "statistic": stat, "p_value": p_value, "method": "chi_square_continuity"}


def auc_difference_bootstrap(
    y_true: Iterable[int],
    prob_a: Iterable[float],
    prob_b: Iterable[float],
    *,
    n_iterations: int = 2000,
    confidence_level: float = 0.95,
    seed: int = 42,
) -> dict[str, float]:
    y = np.asarray(list(y_true), dtype=int)
    a = np.asarray(list(prob_a), dtype=float)
    b = np.asarray(list(prob_b), dtype=float)
    if len(y) == 0:
        return {"auc_a": float("nan"), "auc_b": float("nan"), "auc_difference": float("nan"), "ci_low": float("nan"), "ci_high": float("nan"), "p_value": float("nan")}
    auc_a = _auc(y, a)
    auc_b = _auc(y, b)
    observed = auc_a - auc_b
    rng = np.random.default_rng(seed)
    diffs = []
    for _ in range(int(n_iterations)):
        idx = rng.integers(0, len(y), len(y))
        if len(np.unique(y[idx])) < 2:
            continue
        diffs.append(_auc(y[idx], a[idx]) - _auc(y[idx], b[idx]))
    if not diffs:
        return {"auc_a": float(auc_a), "auc_b": float(auc_b), "auc_difference": float(observed), "ci_low": float("nan"), "ci_high": float("nan"), "p_value": float("nan")}
    alpha = 1.0 - confidence_level
    low, high = np.quantile(diffs, [alpha / 2.0, 1.0 - alpha / 2.0])
    diffs_arr = np.asarray(diffs)
    p_value = 2.0 * min(float(np.mean(diffs_arr <= 0.0)), float(np.mean(diffs_arr >= 0.0)))
    return {
        "auc_a": float(auc_a),
        "auc_b": float(auc_b),
        "auc_difference": float(observed),
        "ci_low": float(low),
        "ci_high": float(high),
        "p_value": float(min(max(p_value, 0.0), 1.0)),
        "method": "paired_bootstrap",
    }


def safe_write_csv(df: pd.DataFrame, path: str | Path, **kwargs: Any) -> Path:
    p = repo_path(path) or Path(path)
    p.parent.mkdir(parents=True, exist_ok=True)
    tmp = p.with_name(p.name + ".tmp")
    df.to_csv(tmp, index=False, **kwargs)
    os.replace(tmp, p)
    return p


def safe_write_json(data: Any, path: str | Path, **kwargs: Any) -> Path:
    p = repo_path(path) or Path(path)
    p.parent.mkdir(parents=True, exist_ok=True)
    tmp = p.with_name(p.name + ".tmp")
    with tmp.open("w", encoding="utf-8") as f:
        json.dump(data, f, indent=2, sort_keys=True, default=str, **kwargs)
        f.write("\n")
    os.replace(tmp, p)
    return p


def safe_write_text(text: str, path: str | Path) -> Path:
    p = repo_path(path) or Path(path)
    p.parent.mkdir(parents=True, exist_ok=True)
    tmp = p.with_name(p.name + ".tmp")
    tmp.write_text(text, encoding="utf-8")
    os.replace(tmp, p)
    return p


def append_missing(message: str, config: dict[str, Any] | None = None) -> None:
    out_dir = "revision_outputs/logs"
    if config is not None:
        out_dir = (config.get("subdirs") or {}).get("logs", out_dir)
    ensure_dir(out_dir)
    path = repo_path(out_dir) / "missing_artifacts.log"
    timestamp = datetime.now(timezone.utc).isoformat()
    with path.open("a", encoding="utf-8") as f:
        f.write(f"{timestamp}\t{message}\n")


def threshold_grid(config: dict[str, Any]) -> np.ndarray:
    grid = config.get("threshold_grid") or {}
    start = float(grid.get("start", 0.01))
    stop = float(grid.get("stop", 0.99))
    step = float(grid.get("step", 0.01))
    n = int(round((stop - start) / step)) + 1
    return np.round(start + np.arange(n) * step, 10)


def best_baseline_model(metrics_path: str | Path, *, split: str = "val") -> str | None:
    p = repo_path(metrics_path) or Path(metrics_path)
    if not p.exists():
        return None
    df = pd.read_csv(p)
    if "model" not in df.columns:
        return None
    work = df.copy()
    if "split" in work.columns:
        split_rows = work[work["split"].astype(str).str.lower() == split.lower()]
        if split_rows.empty and split != "test":
            split_rows = work[work["split"].astype(str).str.lower() == "test"]
        work = split_rows if not split_rows.empty else work
    if "dataset" in work.columns:
        all_rows = work[work["dataset"].astype(str).str.lower().isin(["all", "pooled"])]
        work = all_rows if not all_rows.empty else work
    sort_cols = [c for c in ["AUC", "auc", "F1", "f1", "balanced_accuracy", "accuracy"] if c in work.columns]
    if sort_cols:
        work = work.sort_values(sort_cols, ascending=[False] * len(sort_cols))
    return str(work.iloc[0]["model"]) if not work.empty else None


def select_prediction_source(config: dict[str, Any], model: str, split: str) -> tuple[Path | None, str | None]:
    paths = config.get("paths") or {}
    sources = config.get("prediction_sources") or {}
    model = standardize_model_name(model)
    if model in {"classical_ml", "threshold_baselines"}:
        key = "classical_predictions" if model == "classical_ml" else "threshold_predictions"
        p = repo_path(sources.get(key) or paths.get(f"{model}_dir"))
        if p and p.exists():
            return p, None
        return None, f"Missing {model} predictions: {p}"
    pattern_key = "deep_validation_pattern" if split == "val" else "deep_test_pattern"
    p = repo_path(str(sources.get(pattern_key, "")).format(model=model))
    if p and p.exists():
        return p, None
    if split == "test":
        fallback = repo_path(str(sources.get("final_test_pattern", "")).format(model=model))
        if fallback and fallback.exists():
            return fallback, None
    return None, f"Missing {model} {split} predictions"


def parse_args_config(description: str) -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=description)
    parser.add_argument("--config", default="configs/bspc_revision.yaml")
    parser.add_argument("--force", action="store_true")
    return parser.parse_args()


def save_metadata_for_outputs(outputs: Iterable[str | Path], metadata: dict[str, Any]) -> None:
    for out in outputs:
        p = repo_path(out) or Path(out)
        if p.exists():
            save_run_metadata(p, metadata)


def fresh_all(outputs: Iterable[str | Path], inputs: Iterable[str | Path], cfg_hash: str) -> bool:
    return all(output_is_fresh(out, inputs, cfg_hash) for out in outputs)


def current_command(args: argparse.Namespace | None = None) -> str:
    return " ".join([Path(sys.executable).name, *sys.argv])
