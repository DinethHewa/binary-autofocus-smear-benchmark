from __future__ import annotations

import argparse
import io
import json
import os
import sys
from pathlib import Path

import pandas as pd

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))
os.environ.setdefault("TF_CPP_MIN_LOG_LEVEL", "2")

from revision_utils import (  # noqa: E402
    append_missing,
    build_metadata,
    config_hash,
    display_model_name,
    ensure_dir,
    fresh_all,
    load_config,
    repo_path,
    safe_write_csv,
    safe_write_text,
    save_metadata_for_outputs,
    standardize_model_name,
)


SCRIPT_NAME = Path(__file__).name
DEEP_MODELS = ["cnn", "cnn_attention", "cnn_focus_hybrid", "transfer", "hybrid_vit", "focus_dnn", "vit"]


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Export reproducible architecture specifications for model families.")
    parser.add_argument("--config", default="configs/bspc_revision.yaml")
    parser.add_argument("--force", action="store_true")
    return parser.parse_args()


def _read_json(path: Path) -> dict:
    if path.exists():
        try:
            return json.loads(path.read_text(encoding="utf-8"))
        except Exception:
            return {}
    return {}


def _custom_objects() -> dict:
    objects = {}
    try:
        from focus_binary.models.vit import _CLSToken, _PositionalEmbedding

        if _CLSToken is not None:
            objects["_CLSToken"] = _CLSToken
        if _PositionalEmbedding is not None:
            objects["_PositionalEmbedding"] = _PositionalEmbedding
    except Exception:
        pass
    try:
        from focus_binary.models.swin_tiny import WindowPartition, WindowReverse

        if WindowPartition is not None:
            objects["WindowPartition"] = WindowPartition
        if WindowReverse is not None:
            objects["WindowReverse"] = WindowReverse
    except Exception:
        pass
    try:
        from focus_binary.models.convnext import ConvNeXtPreprocess

        if ConvNeXtPreprocess is not None:
            objects["ConvNeXtPreprocess"] = ConvNeXtPreprocess
    except Exception:
        pass
    return objects


def _load_keras_model(model_path: Path):
    try:
        import tensorflow as tf
    except Exception as exc:
        raise ImportError(f"TensorFlow unavailable: {exc}") from exc
    objects = _custom_objects()
    try:
        return tf.keras.models.load_model(model_path, custom_objects=objects, compile=False, safe_mode=False)
    except TypeError:
        return tf.keras.models.load_model(model_path, custom_objects=objects, compile=False)


def _shape_text(layer) -> str:
    shape = getattr(layer, "output_shape", None)
    if shape is None:
        try:
            shape = layer.output.shape
        except Exception:
            shape = ""
    return str(shape)


def _layer_rows(model) -> list[dict]:
    rows = []
    for idx, layer in enumerate(getattr(model, "layers", [])):
        try:
            params = int(layer.count_params())
        except Exception:
            params = 0
        rows.append(
            {
                "layer_index": idx,
                "layer_name": getattr(layer, "name", ""),
                "layer_type": layer.__class__.__name__,
                "output_shape": _shape_text(layer),
                "parameter_count": params,
                "trainable": bool(getattr(layer, "trainable", False)),
            }
        )
    return rows


def _summary_text(model) -> str:
    buffer = io.StringIO()
    try:
        model.summary(print_fn=lambda line: buffer.write(line + "\n"), expand_nested=True)
    except TypeError:
        model.summary(print_fn=lambda line: buffer.write(line + "\n"))
    except Exception as exc:
        buffer.write(f"Unable to render model.summary(): {exc}\n")
    return buffer.getvalue()


def _source_file_for_model(family: str) -> Path | None:
    mapping = {
        "cnn": "cnn_baseline.py",
        "cnn_attention": "cnn_attention.py",
        "cnn_focus_hybrid": "cnn_focus_hybrid.py",
        "transfer": "transfer.py",
        "hybrid_vit": "hybrid.py",
        "focus_dnn": "focus_dnn.py",
        "vit": "vit.py",
    }
    name = mapping.get(family)
    if not name:
        return None
    p = ROOT / "src" / "focus_binary" / "models" / name
    return p if p.exists() else None


def _identify_attention(family: str, hparams: dict, layer_rows: list[dict], source_text: str) -> tuple[str, str, str]:
    text = " ".join([str(r.get("layer_type", "")) + " " + str(r.get("layer_name", "")) for r in layer_rows]).lower()
    source_lower = source_text.lower()
    if family == "cnn_attention":
        attention_type = str(hparams.get("attention_type", "se")).lower()
        placement = str(hparams.get("attention_placement", "last_two"))
        if attention_type == "cbam" or "_spatial_attention" in source_lower:
            return "SE + spatial attention (CBAM-style)", placement, "cnn_attention source/hparams"
        return "SE/channel attention", placement, "cnn_attention source/hparams"
    if "multiheadattention" in text or "multiheadattention" in source_lower:
        return "self-attention", "transformer encoder blocks", "MultiHeadAttention layers/source"
    return "none detected", "", "architecture inspection"


def main() -> int:
    args = parse_args()
    config = load_config(args.config)
    cfg_hash = config_hash(config)
    arch_dir = ensure_dir((config.get("subdirs") or {}).get("architecture", "revision_outputs/architecture"))
    table_dir = ensure_dir((config.get("subdirs") or {}).get("tables", "revision_outputs/tables"))
    summary_path = arch_dir / "model_architecture_summary.md"
    table_path = table_dir / "table_architecture_summary.csv"
    layer_outputs = [arch_dir / f"model_layers_{m}.csv" for m in DEEP_MODELS]
    outputs = [summary_path, table_path, *layer_outputs]
    input_files = [args.config]
    for family in DEEP_MODELS:
        run_dir = repo_path((config.get("paths") or {}).get("runs_dir", "runs")) / family
        input_files.extend([run_dir / "best_model.keras", run_dir / "summary.json", run_dir / "best_hparams.json"])
        src = _source_file_for_model(family)
        if src:
            input_files.append(src)

    if not args.force and fresh_all(outputs, input_files, cfg_hash):
        print("architecture outputs are fresh; skipping")
        return 0

    summary_lines = ["# Model Architecture Specifications", ""]
    table_rows = []
    missing = []

    for family in DEEP_MODELS:
        run_dir = repo_path((config.get("paths") or {}).get("runs_dir", "runs")) / family
        checkpoint = run_dir / "best_model.keras"
        summary_json = _read_json(run_dir / "summary.json")
        hparams = _read_json(run_dir / "best_hparams.json")
        source_file = _source_file_for_model(family)
        source_text = source_file.read_text(encoding="utf-8") if source_file else ""
        loaded = False
        load_error = ""
        layer_rows = []
        model_summary = ""
        params_count = summary_json.get("params_count")

        if checkpoint.exists():
            try:
                model = _load_keras_model(checkpoint)
                loaded = True
                layer_rows = _layer_rows(model)
                model_summary = _summary_text(model)
                try:
                    params_count = int(model.count_params())
                except Exception:
                    pass
            except Exception as exc:
                load_error = str(exc)
                append_missing(f"Could not load {family} checkpoint {checkpoint}: {exc}", config)
        else:
            load_error = "checkpoint missing"
            missing.append(str(checkpoint))
            append_missing(f"Missing checkpoint for architecture export: {checkpoint}", config)

        if not layer_rows:
            layer_rows = [
                {
                    "layer_index": 0,
                    "layer_name": "source_or_summary_only",
                    "layer_type": "unloaded",
                    "output_shape": "",
                    "parameter_count": params_count if params_count is not None else "",
                    "trainable": "",
                }
            ]
            model_summary = "Checkpoint could not be loaded. Architecture information was inferred from source, summary.json, and best_hparams.json.\n"
            if load_error:
                model_summary += f"Load status: {load_error}\n"

        attention, placement, evidence = _identify_attention(family, hparams, layer_rows, source_text)
        safe_write_csv(pd.DataFrame(layer_rows), arch_dir / f"model_layers_{family}.csv")
        summary_lines.extend(
            [
                f"## {display_model_name(family, config)}",
                "",
                f"- Code name: `{family}`",
                f"- Checkpoint: `{checkpoint}`",
                f"- Loaded from checkpoint: {loaded}",
                f"- Attention mechanism: {attention}",
                f"- Attention placement: {placement or 'not applicable/unknown'}",
                f"- Evidence: {evidence}",
                "",
                "```text",
                model_summary.strip(),
                "```",
                "",
            ]
        )
        table_rows.append(
            {
                "model_code_name": family,
                "model_display_name": display_model_name(family, config),
                "checkpoint_path": str(checkpoint),
                "checkpoint_exists": checkpoint.exists(),
                "loaded_from_checkpoint": loaded,
                "load_error": load_error,
                "parameter_count": params_count,
                "input_size": summary_json.get("input_size", hparams.get("input_size", 224)),
                "n_layers": len(layer_rows),
                "attention_mechanism": attention,
                "attention_placement": placement,
                "attention_evidence": evidence,
                "source_file": str(source_file) if source_file else "",
            }
        )

    safe_write_text("\n".join(summary_lines), summary_path)
    safe_write_csv(pd.DataFrame(table_rows), table_path)
    metadata = build_metadata(script_name=SCRIPT_NAME, input_files=input_files, cfg_hash=cfg_hash, args=args)
    metadata["missing_checkpoints"] = missing
    save_metadata_for_outputs(outputs, metadata)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
