from __future__ import annotations

from pathlib import Path
from typing import Any, Dict

import json
import yaml

try:  # Optional dependency for model IO.
    import tensorflow as tf
except Exception:  # pragma: no cover
    tf = None


def load_yaml(path: str | Path) -> Dict[str, Any]:
    with open(path, "r") as f:
        return yaml.safe_load(f)


def save_json(data: Dict[str, Any], path: str | Path) -> Path:
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(data, indent=2, default=str))
    return path


def load_json(path: str | Path) -> Dict[str, Any]:
    path = Path(path)
    return json.loads(path.read_text())


def save_model(model: Any, path: str | Path) -> Path:
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    if tf is not None and hasattr(model, "save"):
        model.save(path)
    else:
        path.write_text("model placeholder; TensorFlow not installed")
    return path


def load_model(path: str | Path):
    path = Path(path)
    if tf is None:
        raise ImportError("TensorFlow is not installed; cannot load model.")
    return tf.keras.models.load_model(path)
