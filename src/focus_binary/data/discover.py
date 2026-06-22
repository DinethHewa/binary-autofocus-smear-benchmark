from __future__ import annotations

import itertools
import re
from dataclasses import dataclass
from pathlib import Path
from typing import Dict, Iterable, List, Sequence, Tuple

from focus_binary.utils.logging import get_logger

logger = get_logger(__name__)


FOCUSED_NAMES = {"focused", "infocus", "in_focus", "in"}
UNFOCUSED_NAMES = {"unfocused", "outfocus", "out_of_focus", "out"}
DEFAULT_IMAGE_EXTS = {".png", ".jpg", ".jpeg", ".tif", ".tiff"}


@dataclass(frozen=True)
class Sample:
    dataset: str
    path: Path
    label: int  # 1 focused, 0 unfocused
    stack_id: str
    patient_id: str = ""
    source: str = "focused_unfocused_output"


@dataclass(frozen=True)
class DatasetScan:
    dataset: str
    samples: Sequence[Sample]
    focused_dirs: Tuple[str, ...]
    unfocused_dirs: Tuple[str, ...]


def discover_datasets(output_root: Path, dataset_names: Iterable[str] | None = None) -> Dict[str, Path]:
    """Return dataset name -> directory path under the output root."""
    root = Path(output_root)
    if dataset_names:
        mapping = {}
        for name in dataset_names:
            candidate = root / name
            if candidate.is_dir():
                mapping[name] = candidate
            else:
                logger.warning("Dataset folder missing", extra={"dataset": name, "path": str(candidate)})
        return mapping

    return {p.name: p for p in sorted(root.iterdir()) if p.is_dir()}


def _class_label_from_name(name: str) -> int | None:
    lowered = name.lower()
    if lowered in FOCUSED_NAMES:
        return 1
    if lowered in UNFOCUSED_NAMES:
        return 0
    return None


def _iter_images(root: Path, exts: Iterable[str]) -> Iterable[Path]:
    for path in root.rglob("*"):
        if path.is_file() and path.suffix.lower() in exts:
            yield path


def infer_stack_id(image_path: Path, class_root: Path, stack_regex: re.Pattern[str] | None = None) -> str:
    """Prefer the first directory under the class folder; fallback to regex or filename prefix."""
    try:
        relative = image_path.relative_to(class_root)
        if len(relative.parts) > 1:
            return relative.parts[0]
    except ValueError:
        pass

    if stack_regex:
        match = stack_regex.search(image_path.name)
        if match:
            if match.lastindex:
                return match.group(1)
            return match.group(0)

    stem = image_path.stem
    for sep in ("_", "-"):
        if sep in stem:
            return stem.split(sep)[0]
    return stem


def scan_datasets(
    output_root: Path,
    dataset_names: Iterable[str] | None = None,
    image_exts: Iterable[str] | None = None,
    stack_regex: re.Pattern[str] | None = None,
    source: str | None = None,
    limit_per_dataset: int | None = None,
) -> List[DatasetScan]:
    """Scan dataset folders for focused/unfocused images.

    Expected layout:
        output_root/<dataset>/<focused|unfocused>/<stack_id>/*.png
    Class folder names are matched case-insensitively against FOCUSED_NAMES/UNFOCUSED_NAMES.
    """

    exts = {ext.lower() if ext.startswith(".") else f".{ext.lower()}" for ext in (image_exts or DEFAULT_IMAGE_EXTS)}
    dataset_roots = discover_datasets(output_root, dataset_names=dataset_names)

    scans: List[DatasetScan] = []
    for dataset, dataset_root in dataset_roots.items():
        samples: List[Sample] = []
        focused_dirs: List[str] = []
        unfocused_dirs: List[str] = []

        class_dirs = [p for p in dataset_root.iterdir() if p.is_dir()]
        for class_dir in class_dirs:
            label = _class_label_from_name(class_dir.name)
            if label is None:
                continue

            if label == 1:
                focused_dirs.append(class_dir.name)
            else:
                unfocused_dirs.append(class_dir.name)

            for img_path in _iter_images(class_dir, exts):
                stack_id = infer_stack_id(img_path, class_dir, stack_regex)
                samples.append(
                    Sample(
                        dataset=dataset,
                        path=img_path.resolve(),
                        label=label,
                        stack_id=stack_id,
                        patient_id="",
                        source=source or output_root.name,
                    )
                )
                if limit_per_dataset and len(samples) >= limit_per_dataset:
                    break
            if limit_per_dataset and len(samples) >= limit_per_dataset:
                break

        if not focused_dirs or not unfocused_dirs:
            logger.warning(
                "Dataset missing expected class folders",
                extra={"dataset": dataset, "focused_dirs": focused_dirs, "unfocused_dirs": unfocused_dirs},
            )

        scans.append(DatasetScan(dataset=dataset, samples=samples, focused_dirs=tuple(focused_dirs), unfocused_dirs=tuple(unfocused_dirs)))
        logger.info(
            "scanned dataset",
            extra={
                "dataset": dataset,
                "folder": str(dataset_root),
                "focused": len([s for s in samples if s.label == 1]),
                "unfocused": len([s for s in samples if s.label == 0]),
            },
        )

    return scans


def flatten_scans(scans: Sequence[DatasetScan]) -> List[Sample]:
    return list(itertools.chain.from_iterable(ds.samples for ds in scans))
