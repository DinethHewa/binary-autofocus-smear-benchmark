"""Shared paths and dataset conventions.

Defaults match the original notebooks inside "Dense binary" and "binary/notebooks".
"""
from __future__ import annotations

import os
from pathlib import Path
from typing import Dict

PROJECT_ROOT = Path(__file__).resolve().parents[2]
CONFIG_DIR = PROJECT_ROOT / "configs"
ARTIFACT_DIR = PROJECT_ROOT / "artifacts"

# Default root for focused/unfocused exports from the notebooks.
_env_output_root = os.environ.get("FOCUS_OUTPUT_ROOT")
DEFAULT_OUTPUT_ROOT = (
    Path(_env_output_root).expanduser()
    if _env_output_root
    else (PROJECT_ROOT.parent / "dataset")
)

# Pretty dataset name -> folder name on disk
DEFAULT_DATASET_MAP: Dict[str, str] = {
    "WBC-MF": "wbc",
    "TBF": "tbf_imgs",
    "PBS": "pbs_imgs",
    "BMA": "bma",
    "TBI": "TBSI",  # folder uses TBSI, keep pretty label TBI
}


def project_root() -> Path:
    return PROJECT_ROOT


def config_dir() -> Path:
    return CONFIG_DIR


def artifact_dir() -> Path:
    return ARTIFACT_DIR


def output_root() -> Path:
    """Location of focused/unfocused image folders on disk."""
    return DEFAULT_OUTPUT_ROOT


def dataset_dir(pretty_name: str, dataset_map: Dict[str, str] | None = None) -> Path:
    dataset_map = dataset_map or DEFAULT_DATASET_MAP
    folder = dataset_map.get(pretty_name, pretty_name)
    return output_root() / folder


def ensure_dir(path: Path) -> Path:
    path.mkdir(parents=True, exist_ok=True)
    return path
