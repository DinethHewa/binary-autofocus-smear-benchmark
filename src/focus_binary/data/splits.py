from __future__ import annotations

import json
from dataclasses import dataclass
from datetime import datetime
from pathlib import Path
from typing import Dict, Iterable, Tuple

import numpy as np
import pandas as pd

from focus_binary.utils.logging import get_logger

logger = get_logger(__name__)


@dataclass
class SplitConfig:
    seed: int = 42
    train: float = 0.6
    val: float = 0.2
    test: float = 0.2
    group_col: str = "stack_id"
    stratify_col: str = "label"
    by_dataset: bool = True
    strategy: str = "group_stratified"

    def validate(self) -> "SplitConfig":
        if self.train <= 0 or self.val <= 0 or self.test <= 0:
            raise ValueError("train/val/test must be > 0")
        total = self.train + self.val + self.test
        if abs(total - 1.0) > 1e-3:
            if abs(total - 1.0) < 0.1:
                self.train /= total
                self.val /= total
                self.test /= total
            else:
                raise ValueError("train+val+test must sum to 1 (or close enough to normalize)")
        return self


def _normalize_split(train: float, val: float, test: float) -> Tuple[float, float, float]:
    total = train + val + test
    if total <= 0:
        raise ValueError("train+val+test must be > 0")
    return train / total, val / total, test / total


def _group_labels(df: pd.DataFrame, group_col: str, stratify_col: str) -> Dict[str, int]:
    means = df.groupby(group_col)[stratify_col].mean()
    return {group: int(val >= 0.5) for group, val in means.items()}


def _allocate(groups: Iterable[str], labels: Dict[str, int], rng: np.random.Generator, train_p: float, val_p: float):
    train_groups, val_groups, test_groups = [], [], []
    for label in sorted(set(labels.values())):
        label_groups = [g for g in groups if labels[g] == label]
        rng.shuffle(label_groups)
        n = len(label_groups)
        if n == 0:
            continue
        n_train = int(round(train_p * n))
        n_val = int(round(val_p * n))
        if n_train + n_val > n:
            n_val = max(0, n - n_train)
        n_test = n - n_train - n_val
        # ensure at least one group if possible
        if n_train == 0 and n > 0:
            n_train = min(1, n)
            if n_test > 0:
                n_test -= 1
        train_groups.extend(label_groups[:n_train])
        val_groups.extend(label_groups[n_train : n_train + n_val])
        test_groups.extend(label_groups[n_train + n_val :])
    return train_groups, val_groups, test_groups


def split_manifest(
    manifest_df: pd.DataFrame,
    seed: int | SplitConfig | None = None,
    train: float = 0.7,
    val: float = 0.15,
    test: float = 0.15,
    group_col: str = "stack_id",
    stratify_col: str = "label",
    by_dataset: bool = True,
) -> pd.DataFrame:
    """Deterministic, group-respecting split with approximate stratification."""

    if isinstance(seed, SplitConfig):
        cfg = seed.validate()
        seed = cfg.seed
        train = cfg.train
        val = cfg.val
        test = cfg.test
        group_col = cfg.group_col
        stratify_col = cfg.stratify_col
        by_dataset = cfg.by_dataset
    elif seed is None:
        seed = 42

    train_p, val_p, test_p = _normalize_split(train, val, test)
    rng = np.random.default_rng(seed)
    df = manifest_df.copy()
    df["split"] = "unassigned"

    if by_dataset and "dataset" not in df.columns:
        raise KeyError("dataset column required when by_dataset=True")
    if group_col not in df.columns:
        raise KeyError(f"missing group column: {group_col}")
    if stratify_col not in df.columns:
        raise KeyError(f"missing stratify column: {stratify_col}")

    datasets = sorted(df["dataset"].unique()) if by_dataset else ["__all__"]

    for dataset in datasets:
        if dataset == "__all__":
            subset_mask = pd.Series(True, index=df.index)
            subset = df
        else:
            subset_mask = df["dataset"] == dataset
            subset = df[subset_mask]

        group_labels = _group_labels(subset, group_col, stratify_col)
        groups = list(group_labels.keys())
        rng.shuffle(groups)
        train_groups, val_groups, test_groups = _allocate(groups, group_labels, rng, train_p, val_p)

        df.loc[subset_mask & df[group_col].isin(train_groups), "split"] = "train"
        df.loc[subset_mask & df[group_col].isin(val_groups), "split"] = "val"
        df.loc[subset_mask & df[group_col].isin(test_groups), "split"] = "test"

        logger.info(
            "split dataset",
            extra={
                "dataset": dataset,
                "groups": len(groups),
                "train_groups": len(train_groups),
                "val_groups": len(val_groups),
                "test_groups": len(test_groups),
            },
        )

    unassigned = df[df["split"] == "unassigned"]
    if not unassigned.empty:
        logger.warning("unassigned rows remain after split", extra={"count": len(unassigned)})

    return df


def check_no_leak(
    df: pd.DataFrame,
    group_col: str = "stack_id",
    split_col: str = "split",
    dataset_col: str = "dataset",
) -> bool:
    def _check(subset: pd.DataFrame) -> bool:
        groups_by_split = {
            split: set(subset[subset[split_col] == split][group_col].unique())
            for split in subset[split_col].unique()
        }
        splits = list(groups_by_split.keys())
        for i, s1 in enumerate(splits):
            for s2 in splits[i + 1 :]:
                if groups_by_split[s1] & groups_by_split[s2]:
                    return False
        return True

    if dataset_col in df.columns:
        for dataset in df[dataset_col].dropna().unique():
            subset = df[df[dataset_col] == dataset]
            if not _check(subset):
                return False
        return True

    return _check(df)


def leakage_check(
    df: pd.DataFrame,
    group_col: str = "stack_id",
    split_col: str = "split",
    dataset_col: str = "dataset",
) -> bool:
    passed = check_no_leak(df, group_col=group_col, split_col=split_col, dataset_col=dataset_col)
    print("Leakage check: PASS" if passed else "Leakage check: FAIL")
    return passed


def assert_no_leak(
    df: pd.DataFrame,
    group_col: str = "stack_id",
    split_col: str = "split",
    dataset_col: str = "dataset",
) -> None:
    passed = leakage_check(df, group_col=group_col, split_col=split_col, dataset_col=dataset_col)
    if not passed:
        raise AssertionError("Leakage detected across splits")


def write_split_meta(
    path: Path,
    seed: int,
    train: float,
    val: float,
    test: float,
    group_col: str,
    stratify_col: str,
    by_dataset: bool,
    strategy: str = "group_majority_label",
) -> Path:
    meta = {
        "seed": seed,
        "train": train,
        "val": val,
        "test": test,
        "group_col": group_col,
        "stratify_col": stratify_col,
        "by_dataset": by_dataset,
        "strategy": strategy,
        "created_at": datetime.utcnow().isoformat() + "Z",
    }
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(meta, indent=2))
    return path
