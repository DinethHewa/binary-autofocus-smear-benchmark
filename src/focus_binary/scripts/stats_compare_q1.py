from __future__ import annotations

import argparse
from pathlib import Path
from typing import Dict, List, Tuple

import numpy as np
import pandas as pd

from focus_binary.eval.metrics import compute_metrics
from focus_binary.stats.bootstrap import paired_bootstrap_test
from focus_binary.stats.tests import friedman_test, nemenyi_posthoc
from focus_binary.utils.logging import get_logger
from focus_binary.utils.io import save_json

logger = get_logger(__name__)


def parse_args(argv: List[str] | None = None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Q1 statistical comparison using LODO summaries.")
    parser.add_argument(
        "--lodo-summary",
        default="./reports/lodo/lodo_summary_all.csv",
        help="Path to lodo_summary_all.csv",
    )
    parser.add_argument("--out-dir", default="./reports/final_q1", help="Output directory")
    return parser.parse_args(argv)


def _prepare_matrix(df: pd.DataFrame, metric: str) -> Tuple[np.ndarray, List[str], List[str]]:
    pivot = df.pivot_table(index="heldout_dataset", columns="family", values=metric, aggfunc="mean")
    if pivot.isna().any(axis=1).any():
        missing = pivot[pivot.isna().any(axis=1)]
        logger.warning("Dropping heldouts with missing families", extra={"heldouts": list(missing.index)})
    pivot = pivot.dropna(axis=0, how="any")
    if pivot.empty:
        raise ValueError(f"No complete heldout rows for {metric}.")
    return pivot.to_numpy(), list(pivot.index), list(pivot.columns)


def _write_ranks(out_dir: Path, metric: str, families: List[str], nemenyi: Dict[str, object]) -> None:
    ranks = np.asarray(nemenyi["avg_ranks"], dtype=float)
    ranks_df = pd.DataFrame({"family": families, "avg_rank": ranks})
    ranks_df = ranks_df.sort_values("avg_rank", ascending=True)
    ranks_df.to_csv(out_dir / f"stats_{metric}_ranks.csv", index=False)

    sig = pd.DataFrame(nemenyi["significance"], index=families, columns=families)
    sig.to_csv(out_dir / f"stats_{metric}_significance.csv")


def _load_predictions(base_dir: Path, family: str) -> pd.DataFrame:
    rows = []
    family_dir = base_dir / family
    if not family_dir.exists():
        return pd.DataFrame()
    for heldout_dir in sorted(family_dir.glob("heldout_*")):
        for seed_dir in sorted(heldout_dir.glob("seed_*")):
            preds_path = seed_dir / "predictions.csv"
            if preds_path.exists():
                df = pd.read_csv(preds_path)
                rows.append(df)
    if not rows:
        return pd.DataFrame()
    df_all = pd.concat(rows, ignore_index=True)
    grouped = (
        df_all.groupby(["image_path", "heldout_dataset", "dataset"])[["y_true", "y_prob"]]
        .mean()
        .reset_index()
    )
    return grouped


def _bootstrap_top2(base_dir: Path, families: List[str]) -> Dict[str, object]:
    if len(families) < 2:
        return {"error": "Need at least 2 families for bootstrap."}

    fam_a, fam_b = families[0], families[1]
    preds_a = _load_predictions(base_dir, fam_a)
    preds_b = _load_predictions(base_dir, fam_b)
    if preds_a.empty or preds_b.empty:
        return {"error": "Missing predictions for bootstrap."}

    merged = preds_a.merge(
        preds_b,
        on=["image_path", "heldout_dataset", "dataset"],
        suffixes=("_a", "_b"),
    )
    if merged.empty:
        return {"error": "No overlapping predictions for bootstrap."}

    y_true = merged["y_true_a"].to_numpy().astype(int)
    y_prob_a = merged["y_prob_a"].to_numpy().astype(float)
    y_prob_b = merged["y_prob_b"].to_numpy().astype(float)

    result = paired_bootstrap_test(
        y_true,
        y_prob_a,
        y_prob_b,
        metric_fn=lambda yt, yp: compute_metrics(yt, yp)["auc"],
        n=2000,
        seed=42,
    )
    result.update({"family_a": fam_a, "family_b": fam_b})
    return result


def _write_summary_md(
    out_dir: Path,
    metric: str,
    families: List[str],
    friedman: Dict[str, object],
    nemenyi: Dict[str, object],
) -> None:
    lines = [
        f"## {metric.upper()} Statistics",
        "",
        f"Families compared: {len(families)}",
        f"Friedman statistic: {friedman['statistic']:.4f}",
        f"Friedman p-value: {friedman['p_value']:.4f}",
        "Nemenyi posthoc uses alpha=0.05.",
        f"Critical difference: {nemenyi['critical_difference']:.4f}",
        "",
    ]
    out_path = out_dir / f"stats_{metric}_summary.md"
    out_path.write_text("\n".join(lines))


def main(argv: List[str] | None = None) -> Path:
    args = parse_args(argv)
    lodo_path = Path(args.lodo_summary)
    out_dir = Path(args.out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)

    if not lodo_path.exists():
        raise FileNotFoundError(f"LODO summary not found: {lodo_path}")

    lodo_df = pd.read_csv(lodo_path)
    if lodo_df.empty:
        raise ValueError("LODO summary is empty.")

    results = {}
    for metric in ["auc", "f1"]:
        scores, heldouts, families = _prepare_matrix(lodo_df, metric)
        friedman = friedman_test(scores)
        nemenyi = nemenyi_posthoc(scores)
        _write_ranks(out_dir, metric, families, nemenyi)
        _write_summary_md(out_dir, metric, families, friedman, nemenyi)
        results[metric] = {"friedman": friedman, "nemenyi": nemenyi, "families": families}

    # Bootstrap fallback on top-2 AUC families
    auc_means = (
        lodo_df.groupby("family")["auc"].mean().sort_values(ascending=False).reset_index()
    )
    top2 = auc_means["family"].tolist()[:2]
    bootstrap = _bootstrap_top2(lodo_path.parent, top2)
    save_json(bootstrap, out_dir / "stats_bootstrap_top2.json")

    summary_lines = [
        "# Statistical Summary",
        "",
        "Tests: Friedman + Nemenyi (alpha=0.05), bootstrap for top-2 AUC.",
    ]
    for metric, entry in results.items():
        friedman = entry["friedman"]
        summary_lines.append(
            f"- {metric.upper()}: Friedman p={friedman['p_value']:.4f} (stat={friedman['statistic']:.4f})"
        )
    summary_lines.append("")
    summary_lines.append("Bootstrap top-2 AUC delta: see stats_bootstrap_top2.json")
    (out_dir / "stats_summary.md").write_text("\n".join(summary_lines))

    logger.info("Q1 stats complete", extra={"out_dir": str(out_dir)})
    return out_dir


if __name__ == "__main__":
    main()
