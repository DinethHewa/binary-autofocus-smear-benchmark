from __future__ import annotations

from typing import Dict, Tuple

import pandas as pd

from focus_binary.utils.logging import get_logger

logger = get_logger(__name__)


def dataset_balance_report(
    df: pd.DataFrame,
    split_col: str = "split",
    label_col: str = "label",
    dataset_col: str = "dataset",
) -> pd.DataFrame:
    counts = (
        df.groupby([dataset_col, split_col, label_col])
        .size()
        .rename("count")
        .reset_index()
    )
    totals = counts.groupby([dataset_col, split_col])["count"].sum().rename("total")
    counts = counts.merge(totals.reset_index(), on=[dataset_col, split_col])
    counts["ratio"] = counts["count"] / counts["total"].clip(lower=1)
    return counts


def report_and_check_imbalance(
    df: pd.DataFrame,
    split_col: str = "split",
    label_col: str = "label",
    dataset_col: str = "dataset",
    extreme_threshold: float = 0.2,
) -> Tuple[pd.DataFrame, bool]:
    report = dataset_balance_report(df, split_col=split_col, label_col=label_col, dataset_col=dataset_col)
    print("Class balance per dataset/split:")
    for _, row in report.iterrows():
        print(
            f"  {row[dataset_col]} split={row[split_col]} label={row[label_col]} "
            f"count={row['count']} ratio={row['ratio']:.3f}"
        )

    extreme = False
    for _, row in report.iterrows():
        if row["ratio"] < extreme_threshold or row["ratio"] > 1 - extreme_threshold:
            logger.warning(
                "Extreme class imbalance detected",
                extra={
                    "dataset": row[dataset_col],
                    "split": row[split_col],
                    "label": int(row[label_col]),
                    "ratio": float(row["ratio"]),
                },
            )
            extreme = True
    return report, extreme


def compute_class_weights(
    df: pd.DataFrame,
    split: str = "train",
    split_col: str = "split",
    label_col: str = "label",
) -> Dict[int, float]:
    subset = df[df[split_col] == split]
    counts = subset[label_col].value_counts().to_dict()
    total = sum(counts.values())
    weights: Dict[int, float] = {}
    for label in (0, 1):
        count = counts.get(label, 0)
        if count == 0:
            weights[label] = 0.0
        else:
            weights[label] = total / (2.0 * count)
    return weights
