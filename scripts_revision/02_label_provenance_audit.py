from __future__ import annotations

import argparse
import re
import sys
from pathlib import Path

import numpy as np
import pandas as pd

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))

from revision_utils import (  # noqa: E402
    build_metadata,
    config_hash,
    ensure_dir,
    fresh_all,
    load_config,
    load_manifest,
    repo_path,
    safe_write_csv,
    safe_write_text,
    save_metadata_for_outputs,
    standardize_dataset_name,
)


SCRIPT_NAME = Path(__file__).name


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Audit focused/unfocused label provenance from manifest metadata.")
    parser.add_argument("--config", default="configs/bspc_revision.yaml")
    parser.add_argument("--force", action="store_true")
    return parser.parse_args()


def _first_col(df: pd.DataFrame, names: list[str]) -> str | None:
    lower = {c.lower(): c for c in df.columns}
    for name in names:
        if name.lower() in lower:
            return lower[name.lower()]
    return None


def _parse_focal_plane(path: str) -> int | None:
    text = Path(str(path)).stem
    patterns = [
        r"(?:page|plane|z|slice|focus)[_\- ]?(\d+)",
        r"[_\-](\d+)$",
    ]
    for pattern in patterns:
        match = re.search(pattern, text, flags=re.IGNORECASE)
        if match:
            try:
                return int(match.group(1))
            except Exception:
                return None
    return None


def _provenance_category(row: pd.Series, path: str, columns: set[str]) -> str:
    lower_path = str(path).lower().replace("\\", "/")
    if "label_source" in columns or "label_provenance" in columns:
        return "source-provided focused/unfocused category"
    if "/focused/" in lower_path or "/unfocused/" in lower_path or "\\focused\\" in lower_path:
        return "derived from source directory"
    stack_cols = {"stack_id", "focal_plane_index", "true_focus_index", "distance_from_focus"}
    if stack_cols.intersection(columns) and {"true_focus_index", "focal_plane_index"}.issubset(columns):
        return "derived from z-stack best-focus plane"
    return "unknown / needs manual verification"


def main() -> int:
    args = parse_args()
    config = load_config(args.config)
    cfg_hash = config_hash(config)
    out_dir = ensure_dir((config.get("subdirs") or {}).get("audit", "revision_outputs/audit"))
    outputs = [
        out_dir / "label_provenance_table.csv",
        out_dir / "label_provenance_report.md",
    ]
    manifest_path = repo_path((config.get("paths") or {}).get("manifest"))
    split_path = repo_path((config.get("paths") or {}).get("split_metadata"))
    input_files = [args.config, manifest_path, split_path]
    if not args.force and fresh_all(outputs, input_files, cfg_hash):
        print("label provenance outputs are fresh; skipping")
        return 0

    manifest = load_manifest(config=config)
    columns = {c.lower() for c in manifest.columns}
    url_col = _first_col(manifest, ["source_url", "url", "doi", "source_doi"])
    filename_col = _first_col(manifest, ["original_filename", "filename", "file_name"])
    harmonized_col = _first_col(manifest, ["harmonized_id", "sample_id", "image_id"])
    stack_col = _first_col(manifest, ["stack_id", "focal_stack_id", "stack"])
    focal_col = _first_col(manifest, ["focal_plane_index", "plane_index", "z_index", "focus_index"])
    true_focus_col = _first_col(manifest, ["true_focus_index", "best_focus_index", "in_focus_index"])
    distance_col = _first_col(manifest, ["distance_from_focus", "focus_distance", "z_distance"])
    source_col = _first_col(manifest, ["source", "original_source", "dataset_source"])

    rows = []
    for idx, row in manifest.iterrows():
        path = str(row.get("image_path", ""))
        focal_plane = row.get(focal_col) if focal_col else None
        if pd.isna(focal_plane) if focal_plane is not None else True:
            focal_plane = _parse_focal_plane(path)
        true_focus = row.get(true_focus_col) if true_focus_col else None
        distance = row.get(distance_col) if distance_col else None
        if distance_col is None and focal_plane is not None and true_focus is not None and not pd.isna(true_focus):
            try:
                distance = int(focal_plane) - int(true_focus)
            except Exception:
                distance = None
        rows.append(
            {
                "dataset": standardize_dataset_name(row.get("dataset"), config),
                "original_source": row.get(source_col, "") if source_col else "",
                "source_url_or_doi": row.get(url_col, "") if url_col else "",
                "original_filename": row.get(filename_col, Path(path).name) if filename_col else Path(path).name,
                "harmonized_id": row.get(harmonized_col, row.get("sample_id", idx)) if harmonized_col else row.get("sample_id", idx),
                "label": row.get("label", row.get("true_label", "")),
                "stack_identifier": row.get(stack_col, "") if stack_col else "",
                "focal_plane_index": focal_plane if focal_plane is not None and not pd.isna(focal_plane) else "",
                "true_focus_index": true_focus if true_focus is not None and not pd.isna(true_focus) else "",
                "distance_from_focus": distance if distance is not None and not pd.isna(distance) else "",
                "split_assignment": row.get("split", ""),
                "label_provenance_category": _provenance_category(row, path, columns),
                "image_path": path,
            }
        )

    table = pd.DataFrame(rows)
    safe_write_csv(table, outputs[0])

    has_stack = bool(stack_col)
    has_focal = bool(focal_col) or table["focal_plane_index"].astype(str).ne("").any()
    has_true_focus = bool(true_focus_col)
    has_distance = bool(distance_col) or table["distance_from_focus"].astype(str).ne("").any()
    soft_label_possible = has_stack and has_focal and (has_true_focus or has_distance)
    category_counts = table["label_provenance_category"].value_counts(dropna=False).to_dict()
    split_counts = table["split_assignment"].value_counts(dropna=False).to_dict()

    lines = [
        "# Label Provenance Audit",
        "",
        f"Rows inspected: {len(table)}",
        f"Datasets: {', '.join(sorted(table['dataset'].dropna().astype(str).unique()))}",
        "",
        "## Provenance Categories",
        "",
    ]
    lines.extend([f"- {key}: {value}" for key, value in category_counts.items()])
    lines.extend(["", "## Split Assignments", ""])
    lines.extend([f"- {key}: {value}" for key, value in split_counts.items()])
    lines.extend(["", "## Stack-Distance-Aware Soft Labeling", ""])
    if soft_label_possible:
        lines.append(
            "Stack-distance-aware soft labeling appears possible because the manifest contains stack grouping plus focal-plane metadata and either true-focus indices or distance-from-focus values."
        )
    else:
        lines.append(
            "Stack-distance-aware soft labeling is not fully supported by the current manifest metadata. "
            f"Detected stack grouping: {has_stack}; focal-plane metadata: {has_focal}; "
            f"true-focus index: {has_true_focus}; distance-from-focus: {has_distance}."
        )
    lines.extend(["", "## Notes", ""])
    if table["label_provenance_category"].eq("unknown / needs manual verification").any():
        lines.append("- Some rows need manual verification because no directory- or stack-derived label evidence was detected.")
    else:
        lines.append("- All rows had an inferred provenance category from the available metadata/path structure.")
    safe_write_text("\n".join(lines) + "\n", outputs[1])

    metadata = build_metadata(script_name=SCRIPT_NAME, input_files=input_files, cfg_hash=cfg_hash, args=args)
    metadata["soft_label_possible"] = soft_label_possible
    save_metadata_for_outputs(outputs, metadata)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
