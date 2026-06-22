from __future__ import annotations

from pathlib import Path
from typing import Iterable, Mapping, Optional, Sequence, Tuple

import pandas as pd

from focus_binary.utils.logging import get_logger

logger = get_logger(__name__)


LEADERBOARD_COLUMNS = [
    "family",
    "model_name",
    "params_count",
    "latency_ms_mean",
    "latency_ms_p95",
    "tuning_walltime_s",
    "training_walltime_s",
    "hardware",
    "input_size",
    "auc",
    "f1",
    "acc",
    "precision",
    "recall",
    "fp",
    "fn",
]


def _normalize_results(results: Iterable[Mapping]) -> pd.DataFrame:
    return pd.DataFrame(list(results))


def _coalesce(row: pd.Series, keys: Sequence[str]) -> Optional[float]:
    for key in keys:
        if key in row and pd.notna(row[key]):
            return row[key]
    return None


def build_leaderboard(results: Iterable[Mapping], include_dataset_rows: bool = True) -> pd.DataFrame:
    df = _normalize_results(results)
    if df.empty:
        return pd.DataFrame(columns=LEADERBOARD_COLUMNS)

    rows = []
    for _, row in df.iterrows():
        acc = _coalesce(row, ["acc", "accuracy", "binary_accuracy"])
        auc = _coalesce(row, ["auc", "test_auc"])
        f1 = _coalesce(row, ["f1", "test_f1"])
        precision = _coalesce(row, ["precision"])
        recall = _coalesce(row, ["recall"])
        fp = _coalesce(row, ["fp", "false_positives"])
        fn = _coalesce(row, ["fn", "false_negatives"])
        model_name = row.get("model_name", row.get("model", ""))
        dataset = row.get("dataset", "")
        if include_dataset_rows and dataset:
            model_name = f"{model_name} [{dataset}]"

        rows.append(
            {
                "family": row.get("family", ""),
                "model_name": model_name,
                "params_count": row.get("params_count", row.get("params", None)),
                "latency_ms_mean": row.get("latency_ms_mean", None),
                "latency_ms_p95": row.get("latency_ms_p95", None),
                "tuning_walltime_s": row.get("tuning_walltime_s", None),
                "training_walltime_s": row.get("training_walltime_s", None),
                "hardware": row.get("hardware", None),
                "input_size": row.get("input_size", ""),
                "auc": auc,
                "f1": f1,
                "acc": acc,
                "precision": precision,
                "recall": recall,
                "fp": fp,
                "fn": fn,
            }
        )

    return pd.DataFrame(rows, columns=LEADERBOARD_COLUMNS)


def rank_leaderboard(df: pd.DataFrame) -> pd.DataFrame:
    if df.empty:
        return df
    ranked = df.copy()
    ranked["auc"] = pd.to_numeric(ranked["auc"], errors="coerce")
    ranked["f1"] = pd.to_numeric(ranked["f1"], errors="coerce")
    ranked["params_count"] = pd.to_numeric(ranked["params_count"], errors="coerce")
    ranked["latency_ms_mean"] = pd.to_numeric(ranked["latency_ms_mean"], errors="coerce")
    ranked = ranked.sort_values(
        by=["auc", "f1", "latency_ms_mean", "params_count"],
        ascending=[False, False, True, True],
        na_position="last",
    )
    ranked.insert(0, "rank", range(1, len(ranked) + 1))
    return ranked


def write_leaderboard(results: Iterable[Mapping], path: Path) -> Path:
    leaderboard = build_leaderboard(results, include_dataset_rows=True)
    path.parent.mkdir(parents=True, exist_ok=True)
    leaderboard.to_csv(path, index=False)
    logger.info("wrote leaderboard", extra={"path": str(path), "rows": len(leaderboard)})
    return path


def write_markdown_summary(leaderboard: pd.DataFrame, path: Path) -> Path:
    ranked = rank_leaderboard(leaderboard)
    top = ranked.head(10)
    lines = [
        "# Evaluation Summary",
        "",
        "Ranking: pooled test AUC desc, F1 desc, latency_ms_mean asc, params_count asc.",
        "Tuning objective: val_auc primary, val_f1 secondary.",
        "",
        "## Leaderboard (Top 10)",
        "",
    ]
    if top.empty:
        lines.append("No results available.")
    else:
        header = "| rank | family | model_name | auc | f1 | acc | params_count | latency_ms_mean | latency_ms_p95 |"
        sep = "|---|---|---|---|---|---|---|---|---|"
        lines.extend([header, sep])
        for _, row in top.iterrows():
            lines.append(
                f"| {row.get('rank','')} | {row.get('family','')} | {row.get('model_name','')} | "
                f"{row.get('auc','')} | {row.get('f1','')} | {row.get('acc','')} | {row.get('params_count','')} | "
                f"{row.get('latency_ms_mean','')} | {row.get('latency_ms_p95','')} |"
            )
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text("\n".join(lines))
    logger.info("wrote markdown summary", extra={"path": str(path)})
    return path


def write_report(results: Iterable[Mapping], path: Path, markdown_path: Optional[Path] = None) -> Tuple[Path, Optional[Path]]:
    """Write CSV/JSON summaries; if metrics are present, also create a markdown summary."""

    df = _normalize_results(results)
    path.parent.mkdir(parents=True, exist_ok=True)

    if path.suffix.lower() == ".json":
        path.write_text(df.to_json(orient="records", indent=2))
        logger.info("wrote evaluation report", extra={"path": str(path), "rows": len(df)})
        return path, None

    if any(col in df.columns for col in ("auc", "f1", "accuracy", "precision", "recall")):
        leaderboard = build_leaderboard(results, include_dataset_rows=True)
        leaderboard.to_csv(path, index=False)
        md_path = markdown_path or path.with_suffix(".md")
        write_markdown_summary(leaderboard, md_path)
        return path, md_path

    df.to_csv(path, index=False)
    logger.info("wrote evaluation report", extra={"path": str(path), "rows": len(df)})
    return path, None
