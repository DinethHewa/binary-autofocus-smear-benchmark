from __future__ import annotations

import argparse
from pathlib import Path
from typing import Dict, List, Tuple

import numpy as np
import pandas as pd

from focus_binary.eval.metrics import compute_metrics
from focus_binary.stats.bootstrap import bootstrap_difference
from focus_binary.stats.tests import friedman_test, mcnemar_test, nemenyi_posthoc
from focus_binary.utils.io import save_json
from focus_binary.utils.logging import get_logger

logger = get_logger(__name__)


def parse_args(argv: List[str] | None = None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Statistical comparison of model results.")
    parser.add_argument("--results-dir", default=None, help="Directory with per-dataset metrics or multiseed outputs")
    parser.add_argument("--metric", default="auc", help="Metric to compare (default: auc)")
    parser.add_argument("--out-dir", default=None, help="Output directory for stats files")

    parser.add_argument("--preds-a", default=None, help="Predictions CSV for model A (pairwise mode)")
    parser.add_argument("--preds-b", default=None, help="Predictions CSV for model B (pairwise mode)")
    parser.add_argument("--out", default=None, help="Output JSON summary (pairwise mode)")
    parser.add_argument("--threshold", type=float, default=0.5, help="Decision threshold (pairwise mode)")
    parser.add_argument("--bootstrap", type=int, default=1000, help="Bootstrap iterations (pairwise mode)")
    return parser.parse_args(argv)


def _pairwise_mode(args: argparse.Namespace) -> Path:
    if args.preds_a is None or args.preds_b is None or args.out is None:
        raise ValueError("Pairwise mode requires --preds-a, --preds-b, and --out")

    df_a = pd.read_csv(Path(args.preds_a))
    df_b = pd.read_csv(Path(args.preds_b))

    if len(df_a) != len(df_b):
        raise ValueError("Prediction files must have the same number of rows.")

    y_true = df_a["y_true"].to_numpy().astype(int)
    y_prob_a = df_a["y_prob"].to_numpy().astype(float)
    y_prob_b = df_b["y_prob"].to_numpy().astype(float)

    metrics_a = compute_metrics(y_true, y_prob_a, threshold=args.threshold)
    metrics_b = compute_metrics(y_true, y_prob_b, threshold=args.threshold)

    y_pred_a = (y_prob_a >= args.threshold).astype(int)
    y_pred_b = (y_prob_b >= args.threshold).astype(int)
    mcnemar = mcnemar_test(y_true, y_pred_a, y_pred_b)

    auc_diff = bootstrap_difference(
        y_true,
        y_prob_a,
        y_prob_b,
        metric_fn=lambda yt, yp: compute_metrics(yt, yp)["auc"],
        n_boot=args.bootstrap,
    )
    f1_diff = bootstrap_difference(
        y_true,
        y_prob_a,
        y_prob_b,
        metric_fn=lambda yt, yp: compute_metrics(yt, yp, threshold=args.threshold)["f1"],
        n_boot=args.bootstrap,
    )

    summary = {
        "metrics_a": metrics_a,
        "metrics_b": metrics_b,
        "mcnemar": mcnemar,
        "auc_diff": auc_diff,
        "f1_diff": f1_diff,
    }

    out_path = Path(args.out)
    save_json(summary, out_path)
    logger.info("stats comparison complete", extra={"out": str(out_path)})
    return out_path


def _load_per_dataset_from_compare_best(results_dir: Path, metric: str) -> pd.DataFrame:
    path = results_dir / "per_dataset_metrics.csv"
    if not path.exists():
        return pd.DataFrame()
    df = pd.read_csv(path)
    if metric not in df.columns:
        raise KeyError(f"Metric '{metric}' not found in {path}")
    return df[["family", "dataset", metric]].copy()


def _load_per_dataset_from_multiseed(results_dir: Path, metric: str) -> pd.DataFrame:
    rows = []
    for family_dir in sorted(results_dir.iterdir()):
        metrics_path = family_dir / "multiseed_metrics.csv"
        if not metrics_path.exists():
            continue
        df = pd.read_csv(metrics_path)
        if metric not in df.columns:
            raise KeyError(f"Metric '{metric}' not found in {metrics_path}")
        df = df[df["split"] == "test"]
        df = df[df["dataset"] != "all"]
        grouped = df.groupby("dataset")[metric].mean().reset_index()
        grouped["family"] = family_dir.name
        rows.append(grouped)
    if not rows:
        return pd.DataFrame()
    return pd.concat(rows, ignore_index=True)


def _prepare_scores_matrix(df: pd.DataFrame, metric: str) -> Tuple[np.ndarray, List[str], List[str]]:
    pivot = df.pivot_table(index="dataset", columns="family", values=metric, aggfunc="mean")
    if pivot.isna().any(axis=1).any():
        missing = pivot[pivot.isna().any(axis=1)]
        logger.warning("Dropping datasets with missing families", extra={"datasets": list(missing.index)})
    pivot = pivot.dropna(axis=0, how="any")
    if pivot.empty:
        raise ValueError("No complete dataset rows found for Friedman test.")
    datasets = list(pivot.index)
    families = list(pivot.columns)
    scores = pivot.to_numpy()
    return scores, datasets, families


def _write_summary_md(
    out_dir: Path,
    metric: str,
    families: List[str],
    n_datasets: int,
    friedman: Dict[str, float | np.ndarray],
    nemenyi: Dict[str, object],
    sig_matrix: pd.DataFrame,
) -> Path:
    lines = [
        "# Statistical Comparison",
        "",
        f"Metric: {metric}",
        f"Models compared: {len(families)}",
        f"Datasets: {n_datasets}",
        "",
        f"Friedman statistic: {friedman['statistic']:.4f}",
        f"Friedman p-value: {friedman['p_value']:.4f}",
        f"Nemenyi critical difference (alpha=0.05): {nemenyi['critical_difference']:.4f}",
        "",
        "Significant differences (Nemenyi):",
    ]

    pairs = []
    for i, fam_i in enumerate(families):
        for j in range(i + 1, len(families)):
            if bool(sig_matrix.iloc[i, j]):
                pairs.append(f"- {fam_i} vs {families[j]}")

    if pairs:
        lines.extend(pairs)
    else:
        lines.append("- None above critical difference")

    out_path = out_dir / "stats_summary.md"
    out_path.write_text("\n".join(lines))
    return out_path


def _results_mode(args: argparse.Namespace) -> Path:
    if args.results_dir is None:
        raise ValueError("--results-dir is required for results mode.")

    results_dir = Path(args.results_dir)
    if not results_dir.exists():
        raise FileNotFoundError(f"results-dir not found: {results_dir}")

    metric = args.metric
    df = _load_per_dataset_from_compare_best(results_dir, metric)
    if df.empty:
        df = _load_per_dataset_from_multiseed(results_dir, metric)

    if df.empty:
        raise RuntimeError("No per-dataset metrics found in results-dir.")

    scores, datasets, families = _prepare_scores_matrix(df, metric)
    friedman = friedman_test(scores)
    nemenyi = nemenyi_posthoc(scores)

    ranks = nemenyi["avg_ranks"]
    ranks_df = pd.DataFrame({"family": families, "avg_rank": ranks})
    ranks_df = ranks_df.sort_values("avg_rank", ascending=True).reset_index(drop=True)

    sig = pd.DataFrame(nemenyi["significance"], index=families, columns=families)

    out_dir = Path(args.out_dir) if args.out_dir else results_dir
    out_dir.mkdir(parents=True, exist_ok=True)

    ranks_path = out_dir / "stats_ranks.csv"
    ranks_df.to_csv(ranks_path, index=False)

    sig_path = out_dir / "stats_significance.csv"
    sig.to_csv(sig_path)

    summary_path = _write_summary_md(out_dir, metric, families, len(datasets), friedman, nemenyi, sig)

    logger.info(
        "stats comparison complete",
        extra={"ranks": str(ranks_path), "significance": str(sig_path), "summary": str(summary_path)},
    )
    return summary_path


def main(argv: List[str] | None = None) -> Path:
    args = parse_args(argv)
    if args.preds_a or args.preds_b or args.out:
        return _pairwise_mode(args)
    return _results_mode(args)


if __name__ == "__main__":
    main()
