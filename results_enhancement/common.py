from __future__ import annotations

import json
import math
import shutil
import sys
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any, Iterable

import numpy as np
import pandas as pd

try:
    from PIL import Image, ImageDraw
except Exception:  # pragma: no cover
    Image = None
    ImageDraw = None

PROJECT_ROOT = Path(__file__).resolve().parents[1]
SRC_ROOT = PROJECT_ROOT / "src"
if SRC_ROOT.exists() and str(SRC_ROOT) not in sys.path:
    sys.path.insert(0, str(SRC_ROOT))

IMAGE_EXTENSIONS = {".png", ".jpg", ".jpeg", ".bmp", ".tif", ".tiff"}


@dataclass
class PipelineContext:
    project_root: Path
    output_dir: Path
    processing_log: list[str] = field(default_factory=list)
    warnings: list[str] = field(default_factory=list)

    def log(self, message: str) -> None:
        self.processing_log.append(message)

    def warn(self, message: str) -> None:
        if message not in self.warnings:
            self.warnings.append(message)
        self.processing_log.append(f"WARNING: {message}")

    def finalize_logs(self) -> None:
        write_text(
            self.output_dir / "processing_log.md",
            "# Processing Log\n\n" + "\n".join(f"- {line}" for line in self.processing_log),
        )
        warning_lines = self.warnings or ["No additional limitations were recorded beyond those embedded in the output files."]
        write_text(
            self.output_dir / "warnings_and_limitations.md",
            "# Warnings And Limitations\n\n" + "\n".join(f"- {line}" for line in warning_lines),
        )


def ensure_dir(path: str | Path) -> Path:
    target = Path(path)
    target.mkdir(parents=True, exist_ok=True)
    return target


def write_text(path: str | Path, text: str) -> Path:
    target = Path(path)
    target.parent.mkdir(parents=True, exist_ok=True)
    target.write_text(text, encoding="utf-8")
    return target


def write_dataframe(df: pd.DataFrame, path: str | Path) -> Path:
    target = Path(path)
    target.parent.mkdir(parents=True, exist_ok=True)
    df.to_csv(target, index=False)
    return target


def safe_read_csv(path: str | Path, ctx: PipelineContext | None = None, **kwargs: Any) -> pd.DataFrame:
    target = Path(path)
    try:
        return pd.read_csv(target, **kwargs)
    except Exception as exc:  # pragma: no cover - defensive
        if ctx is not None:
            ctx.warn(f"Failed to read CSV '{target}': {exc}")
        return pd.DataFrame()


def read_json(path: str | Path, ctx: PipelineContext | None = None) -> Any:
    target = Path(path)
    try:
        return json.loads(target.read_text(encoding="utf-8"))
    except Exception as exc:  # pragma: no cover - defensive
        if ctx is not None:
            ctx.warn(f"Failed to read JSON '{target}': {exc}")
        return {}


def format_float(value: Any, digits: int = 4) -> str:
    if value is None:
        return "NA"
    try:
        value = float(value)
    except Exception:
        return str(value)
    if np.isnan(value):
        return "NA"
    return f"{value:.{digits}f}"


def safe_div(numerator: float, denominator: float) -> float:
    return float(numerator) / float(denominator) if denominator else float("nan")


def compute_confusion_metrics(tn: float, fp: float, fn: float, tp: float) -> dict[str, float | bool]:
    total = tn + fp + fn + tp
    accuracy = safe_div(tp + tn, total)
    precision = safe_div(tp, tp + fp)
    recall = safe_div(tp, tp + fn)
    specificity = safe_div(tn, tn + fp)
    if math.isnan(precision) or math.isnan(recall) or (precision + recall) == 0:
        f1 = float("nan")
    else:
        f1 = 2.0 * precision * recall / (precision + recall)
    balanced_accuracy = float(np.nanmean([recall, specificity]))
    mcc_den = math.sqrt((tp + fp) * (tp + fn) * (tn + fp) * (tn + fn))
    mcc = safe_div((tp * tn) - (fp * fn), mcc_den)
    fpr = safe_div(fp, fp + tn)
    fnr = safe_div(fn, fn + tp)
    npv = safe_div(tn, tn + fn)
    geometric_mean = math.sqrt(max(recall, 0.0) * max(specificity, 0.0)) if not math.isnan(recall) and not math.isnan(specificity) else float("nan")
    collapsed_positive_only = (tn + fn) == 0 and total > 0
    collapsed_negative_only = (tp + fp) == 0 and total > 0
    specificity_zero_flag = (not math.isnan(specificity)) and specificity == 0.0
    balanced_metric_warning = (
        (not math.isnan(accuracy) and not math.isnan(balanced_accuracy) and abs(accuracy - balanced_accuracy) >= 0.05)
        or collapsed_positive_only
        or collapsed_negative_only
        or specificity_zero_flag
    )
    return {
        "tn": float(tn),
        "fp": float(fp),
        "fn": float(fn),
        "tp": float(tp),
        "n_total": float(total),
        "accuracy": accuracy,
        "precision": precision,
        "recall": recall,
        "sensitivity": recall,
        "specificity": specificity,
        "f1": f1,
        "balanced_accuracy": balanced_accuracy,
        "mcc": mcc,
        "false_positive_rate": fpr,
        "false_negative_rate": fnr,
        "negative_predictive_value": npv,
        "geometric_mean": geometric_mean,
        "positive_prevalence": safe_div(tp + fn, total),
        "negative_prevalence": safe_div(tn + fp, total),
        "collapsed_positive_only": bool(collapsed_positive_only),
        "collapsed_negative_only": bool(collapsed_negative_only),
        "specificity_zero_flag": bool(specificity_zero_flag),
        "balanced_metric_warning": bool(balanced_metric_warning),
    }


def parse_confusion_matrix(value: Any) -> tuple[float, float, float, float] | None:
    if value is None or (isinstance(value, float) and np.isnan(value)):
        return None
    if isinstance(value, (list, tuple)) and len(value) == 2:
        try:
            tn, fp = value[0]
            fn, tp = value[1]
            return float(tn), float(fp), float(fn), float(tp)
        except Exception:
            return None
    if isinstance(value, str):
        cleaned = value.replace("[", " ").replace("]", " ").replace(",", " ")
        parts = [part for part in cleaned.split() if part]
        if len(parts) >= 4:
            try:
                tn, fp, fn, tp = [float(part) for part in parts[:4]]
                return tn, fp, fn, tp
            except Exception:
                return None
    return None


def first_existing(paths: Iterable[Path]) -> Path | None:
    for path in paths:
        if path.exists():
            return path
    return None


def family_display_name(family: str, model_name: str) -> str:
    if model_name.startswith(f"{family}:"):
        return model_name
    if model_name == family:
        return family
    return f"{family}:{model_name}"


def image_file_count(path: Path) -> int:
    if not path.exists() or not path.is_dir():
        return 0
    return sum(1 for item in path.rglob("*") if item.is_file() and item.suffix.lower() in IMAGE_EXTENSIONS)


def copy_text(src: Path, dst: Path) -> None:
    dst.parent.mkdir(parents=True, exist_ok=True)
    shutil.copy2(src, dst)


def save_placeholder_figure(path: str | Path, title: str, message: str, size: tuple[int, int] = (1400, 800)) -> Path:
    target = Path(path)
    target.parent.mkdir(parents=True, exist_ok=True)
    if Image is None or ImageDraw is None:
        target.with_suffix(".txt").write_text(f"{title}\n\n{message}", encoding="utf-8")
        return target

    image = Image.new("RGB", size, color="white")
    draw = ImageDraw.Draw(image)
    text = f"{title}\n\n{message}"
    draw.multiline_text((40, 40), text, fill="black", spacing=8)
    image.save(target)
    return target


def save_image_grid(
    image_paths: list[Path],
    captions: list[str],
    out_path: str | Path,
    title: str,
    cols: int = 3,
    cell_size: tuple[int, int] = (320, 320),
) -> Path:
    target = Path(out_path)
    target.parent.mkdir(parents=True, exist_ok=True)
    if Image is None or ImageDraw is None:
        return save_placeholder_figure(target, title, "Pillow is unavailable, so the image grid could not be rendered.")

    rows = max(1, math.ceil(len(image_paths) / cols))
    width = cols * cell_size[0]
    height = 80 + rows * (cell_size[1] + 40)
    canvas = Image.new("RGB", (width, height), color="white")
    draw = ImageDraw.Draw(canvas)
    draw.text((20, 20), title, fill="black")

    for idx, image_path in enumerate(image_paths):
        image = Image.open(image_path).convert("RGB").resize(cell_size)
        col = idx % cols
        row = idx // cols
        x = col * cell_size[0]
        y = 60 + row * (cell_size[1] + 40)
        canvas.paste(image, (x, y))
        caption = captions[idx] if idx < len(captions) else ""
        draw.text((x + 10, y + cell_size[1] + 8), caption, fill="black")
    canvas.save(target)
    return target
