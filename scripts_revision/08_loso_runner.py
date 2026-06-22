from __future__ import annotations

import argparse
import json
import os
import subprocess
import sys
from datetime import datetime, timezone
from pathlib import Path

import pandas as pd

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))

from revision_utils import (  # noqa: E402
    build_metadata,
    config_hash,
    display_model_name,
    ensure_dir,
    fresh_all,
    load_config,
    safe_write_csv,
    safe_write_json,
    save_metadata_for_outputs,
    standardize_model_name,
)


SCRIPT_NAME = Path(__file__).name


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Plan or run leave-one-source-out generalization jobs.")
    parser.add_argument("--config", default="configs/bspc_revision.yaml")
    parser.add_argument("--force", action="store_true")
    parser.add_argument("--plan-only", action="store_true", default=True)
    parser.add_argument("--run", action="store_true")
    parser.add_argument("--allow-train", action="store_true")
    return parser.parse_args()


def _command_text(command: list[str]) -> str:
    return subprocess.list2cmdline([str(item) for item in command])


def _folds(config: dict) -> list[tuple[str, str]]:
    data_cfg = config.get("datasets") or {}
    names = data_cfg.get("names") or ["WBC", "TBF", "PBS", "BMA", "TBSI"]
    display_to_code = data_cfg.get("display_to_code") or {}
    return [(str(name), str(display_to_code.get(name, name))) for name in names]


def _checkpoint_candidates(out_dir: Path, model: str, holdout_display: str, holdout_code: str) -> list[Path]:
    return [
        out_dir / "checkpoints" / f"{model}_holdout_{holdout_display}" / "best_model.keras",
        out_dir / model / f"holdout_{holdout_display}" / "best_model.keras",
        out_dir / "training_outputs" / model / f"holdout_{holdout_display}" / "best_model.keras",
        out_dir / "training_outputs" / model / f"holdout_{holdout_code}" / "best_model.keras",
    ]


def _first_existing(paths: list[Path]) -> Path | None:
    for p in paths:
        if p.exists():
            return p
    return None


def _seed_dir_candidates(out_dir: Path, model: str, holdout_display: str, holdout_code: str) -> list[Path]:
    return [
        out_dir / "training_outputs" / model / f"heldout_{holdout_code}" / "seed_42",
        out_dir / "training_outputs" / model / f"heldout_{holdout_display}" / "seed_42",
    ]


def _seed_output_dir(out_dir: Path, model: str, holdout_display: str, holdout_code: str) -> Path | None:
    for seed_dir in _seed_dir_candidates(out_dir, model, holdout_display, holdout_code):
        if (seed_dir / "metrics.json").exists() and (seed_dir / "predictions.csv").exists():
            return seed_dir
    return None


def _job_marker(out_dir: Path, model: str, holdout_display: str) -> Path:
    return out_dir / "job_status" / f"loso_{model}_holdout_{holdout_display}.json"


def _marker_complete(path: Path, force: bool) -> bool:
    if force or not path.exists():
        return False
    try:
        data = json.loads(path.read_text(encoding="utf-8"))
    except Exception:
        return False
    return data.get("status") == "complete" and int(data.get("return_code", 1)) == 0


def _job_command(config: dict, model: str, holdout_code: str, out_dir: Path) -> list[str]:
    manifest = (config.get("paths") or {}).get("manifest", "data/manifest_with_splits.csv")
    runs_dir = (config.get("paths") or {}).get("runs_dir", "runs")
    return [
        sys.executable,
        "-u",
        str(ROOT / "scripts" / "lodo_eval.py"),
        "--families",
        model,
        "--manifest",
        str(manifest),
        "--runs-dir",
        str(runs_dir),
        "--out-dir",
        str(out_dir / "training_outputs"),
        "--heldout",
        holdout_code,
        "--seeds",
        "42",
    ]


def _placeholder(path: Path, message: str) -> None:
    try:
        import matplotlib.pyplot as plt
    except Exception:
        return
    path.parent.mkdir(parents=True, exist_ok=True)
    plt.figure(figsize=(7, 4))
    plt.axis("off")
    plt.text(0.5, 0.5, message, ha="center", va="center", fontsize=12)
    plt.tight_layout()
    plt.savefig(path, dpi=300)
    plt.close()


def _heatmap(df: pd.DataFrame, metric_col: str, path: Path) -> None:
    try:
        import matplotlib.pyplot as plt
        import numpy as np
    except Exception:
        return
    model_col = "model_code_name" if "model_code_name" in df.columns else "family"
    holdout_col = "holdout_dataset" if "holdout_dataset" in df.columns else "heldout_dataset"
    if model_col not in df.columns or holdout_col not in df.columns:
        _placeholder(path, f"LOSO {metric_col} table lacks model/holdout columns")
        return
    pivot = df.pivot_table(index=model_col, columns=holdout_col, values=metric_col, aggfunc="mean")
    fig, ax = plt.subplots(figsize=(8, 4.5))
    im = ax.imshow(pivot.values, aspect="auto", vmin=0, vmax=1, cmap="viridis")
    ax.set_xticks(range(len(pivot.columns)))
    ax.set_xticklabels(pivot.columns, rotation=30, ha="right")
    ax.set_yticks(range(len(pivot.index)))
    ax.set_yticklabels([display_model_name(str(x)) for x in pivot.index])
    ax.set_title(f"LOSO {metric_col}")
    fig.colorbar(im, ax=ax, fraction=0.046, pad=0.04)
    for i in range(pivot.shape[0]):
        for j in range(pivot.shape[1]):
            val = pivot.values[i, j]
            if not np.isnan(val):
                ax.text(j, i, f"{val:.2f}", ha="center", va="center", color="white" if val < 0.55 else "black", fontsize=8)
    plt.tight_layout()
    path.parent.mkdir(parents=True, exist_ok=True)
    plt.savefig(path, dpi=300)
    plt.close()


def _make_heatmaps(metrics_path: Path, fig_dir: Path, table_dir: Path) -> None:
    table_path = table_dir / "table_loso_metrics.csv"
    if not metrics_path.exists():
        safe_write_csv(pd.DataFrame(columns=["model_code_name", "holdout_dataset", "status"]), table_path)
        _placeholder(fig_dir / "fig_loso_heatmap_auc.png", "LOSO AUC metrics not available")
        _placeholder(fig_dir / "fig_loso_heatmap_balanced_accuracy.png", "LOSO balanced accuracy metrics not available")
        return
    df = pd.read_csv(metrics_path)
    safe_write_csv(df, table_path)
    for metric, filename in [("AUC", "fig_loso_heatmap_auc.png"), ("balanced_accuracy", "fig_loso_heatmap_balanced_accuracy.png")]:
        col = metric if metric in df.columns else metric.lower()
        if col not in df.columns:
            _placeholder(fig_dir / filename, f"LOSO {metric} metrics not available")
            continue
        _heatmap(df, col, fig_dir / filename)


def _run_streaming_job(
    *,
    command: list[str],
    marker_path: Path,
    log_path: Path,
    label: str,
    index: int,
    total: int,
) -> int:
    command_text = _command_text(command)
    marker = {
        "status": "running",
        "label": label,
        "command": command_text,
        "started_at": datetime.now(timezone.utc).isoformat(),
        "log_path": str(log_path),
    }
    safe_write_json(marker, marker_path)
    print(f"[LOSO {index}/{total}] START {label}", flush=True)
    print(f"[LOSO {index}/{total}] {command_text}", flush=True)
    log_path.parent.mkdir(parents=True, exist_ok=True)
    env = os.environ.copy()
    env["PYTHONUNBUFFERED"] = "1"
    env.setdefault("PYTHONIOENCODING", "utf-8")
    with log_path.open("w", encoding="utf-8", errors="replace") as log:
        log.write("$ " + command_text + "\n\n")
        log.flush()
        process = subprocess.Popen(
            command,
            cwd=ROOT,
            stdout=subprocess.PIPE,
            stderr=subprocess.STDOUT,
            text=True,
            bufsize=1,
            encoding="utf-8",
            errors="replace",
            env=env,
        )
        try:
            assert process.stdout is not None
            for line in process.stdout:
                print(line, end="", flush=True)
                log.write(line)
                log.flush()
            return_code = process.wait()
        except KeyboardInterrupt:
            process.terminate()
            marker.update({"status": "interrupted", "return_code": None, "ended_at": datetime.now(timezone.utc).isoformat()})
            safe_write_json(marker, marker_path)
            raise
    marker.update(
        {
            "status": "complete" if return_code == 0 else "failed",
            "return_code": return_code,
            "ended_at": datetime.now(timezone.utc).isoformat(),
        }
    )
    safe_write_json(marker, marker_path)
    state = "OK" if return_code == 0 else f"FAILED ({return_code})"
    print(f"[LOSO {index}/{total}] {state} {label}", flush=True)
    return return_code


def _collect_loso_outputs(out_dir: Path, pred_dir: Path, metrics_dir: Path, table_dir: Path, fig_dir: Path, folds: list[tuple[str, str]], models: list[str]) -> None:
    source_summary = out_dir / "training_outputs" / "lodo_summary_all.csv"
    dest_summary = metrics_dir / "loso_metrics.csv"
    if source_summary.exists():
        df = pd.read_csv(source_summary)
        if "model_code_name" not in df.columns and "family" in df.columns:
            df.insert(0, "model_code_name", df["family"])
        if "model_display_name" not in df.columns and "model_code_name" in df.columns:
            df.insert(1, "model_display_name", df["model_code_name"].map(lambda x: display_model_name(str(x))))
        safe_write_csv(df, dest_summary)
    for holdout_display, holdout_code in folds:
        for model in models:
            seed_dir = _seed_output_dir(out_dir, model, holdout_display, holdout_code)
            if seed_dir is None:
                continue
            src_pred = seed_dir / "predictions.csv"
            dest_pred = pred_dir / f"{model}_holdout_{holdout_display}_test.csv"
            pred_df = pd.read_csv(src_pred)
            pred_df["model_code_name"] = model
            pred_df["model_display_name"] = display_model_name(model)
            pred_df["split"] = "test"
            safe_write_csv(pred_df, dest_pred)
    _make_heatmaps(dest_summary, fig_dir, table_dir)


def _build_plan(config: dict, out_dir: Path, pred_dir: Path, metrics_dir: Path, models: list[str], folds: list[tuple[str, str]], args: argparse.Namespace) -> tuple[list[dict], list[dict], list[dict]]:
    plan_rows = []
    status_rows = []
    runnable_jobs = []
    for holdout_display, holdout_code in folds:
        for model in models:
            marker = _job_marker(out_dir, model, holdout_display)
            completed_artifact_dir = _seed_output_dir(out_dir, model, holdout_display, holdout_code)
            completed_by_marker = _marker_complete(marker, args.force)
            checkpoint = _first_existing(_checkpoint_candidates(out_dir, model, holdout_display, holdout_code))
            val_pred = pred_dir / f"{model}_holdout_{holdout_display}_val.csv"
            test_pred = pred_dir / f"{model}_holdout_{holdout_display}_test.csv"
            metrics_file = metrics_dir / "loso_metrics.csv"
            if completed_artifact_dir or completed_by_marker:
                action = "skip_complete"
                status = "complete"
            elif checkpoint and not test_pred.exists():
                action = "evaluate_only"
                status = "checkpoint_exists_predictions_missing"
            elif args.allow_train:
                action = "train_missing"
                status = "training_required_allowed"
            else:
                action = "missing_training_required"
                status = "missing_checkpoint_no_training"
            command = _job_command(config, model, holdout_code, out_dir)
            plan_rows.append(
                {
                    "model_code_name": model,
                    "model_display_name": display_model_name(model, config),
                    "holdout_dataset": holdout_display,
                    "holdout_dataset_code": holdout_code,
                    "checkpoint_path": str(checkpoint or ""),
                    "checkpoint_exists": bool(checkpoint),
                    "validation_predictions": str(val_pred),
                    "validation_predictions_exist": val_pred.exists(),
                    "test_predictions": str(test_pred),
                    "test_predictions_exist": test_pred.exists(),
                    "seed_artifact_dir": str(completed_artifact_dir or ""),
                    "job_marker": str(marker),
                    "job_marker_complete": completed_by_marker,
                    "metrics_file": str(metrics_file),
                    "metrics_exist": metrics_file.exists(),
                    "action": action,
                    "command": _command_text(command),
                }
            )
            status_rows.append(
                {
                    "model_code_name": model,
                    "holdout_dataset": holdout_display,
                    "status": status,
                    "action": action,
                    "train_allowed": bool(args.allow_train),
                    "job_marker": str(marker),
                }
            )
            if args.run and action == "train_missing" and args.allow_train:
                runnable_jobs.append({"model": model, "holdout_display": holdout_display, "holdout_code": holdout_code, "command": command, "marker": marker, "action": action})
    return plan_rows, status_rows, runnable_jobs


def main() -> int:
    args = parse_args()
    if args.run:
        args.plan_only = False
    config = load_config(args.config)
    cfg_hash = config_hash(config)
    out_dir = ensure_dir((config.get("subdirs") or {}).get("loso", "revision_outputs/loso"))
    pred_dir = ensure_dir(out_dir / "predictions")
    metrics_dir = ensure_dir(out_dir / "metrics")
    ensure_dir(out_dir / "job_status")
    table_dir = ensure_dir((config.get("subdirs") or {}).get("tables", "revision_outputs/tables"))
    fig_dir = ensure_dir((config.get("subdirs") or {}).get("figures", "revision_outputs/figures"))
    job_log_dir = ensure_dir((config.get("subdirs") or {}).get("logs", "revision_outputs/logs")) / "jobs"
    ensure_dir(job_log_dir)
    outputs = [
        out_dir / "loso_run_plan.csv",
        out_dir / "loso_status.csv",
        table_dir / "table_loso_metrics.csv",
        fig_dir / "fig_loso_heatmap_auc.png",
        fig_dir / "fig_loso_heatmap_balanced_accuracy.png",
    ]
    input_files = [args.config]
    if not args.force and args.plan_only and fresh_all(outputs, input_files, cfg_hash):
        print("LOSO plan outputs are fresh; skipping", flush=True)
        return 0

    folds = _folds(config)
    models = [standardize_model_name(m) for m in (config.get("priority_loso_models") or [])]
    _collect_loso_outputs(out_dir, pred_dir, metrics_dir, table_dir, fig_dir, folds, models)
    plan_rows, status_rows, runnable_jobs = _build_plan(config, out_dir, pred_dir, metrics_dir, models, folds, args)
    safe_write_csv(pd.DataFrame(plan_rows), outputs[0])
    safe_write_csv(pd.DataFrame(status_rows), outputs[1])

    if args.run:
        print(f"LOSO runner: {len(runnable_jobs)} runnable job(s), {len(plan_rows) - len(runnable_jobs)} skipped/planned.", flush=True)
        failed = 0
        for idx, job in enumerate(runnable_jobs, start=1):
            label = f"{job['model']} holdout {job['holdout_display']} ({job['action']})"
            log_path = job_log_dir / f"loso_{job['model']}_holdout_{job['holdout_display']}.log"
            rc = _run_streaming_job(
                command=job["command"],
                marker_path=job["marker"],
                log_path=log_path,
                label=label,
                index=idx,
                total=len(runnable_jobs),
            )
            _collect_loso_outputs(out_dir, pred_dir, metrics_dir, table_dir, fig_dir, folds, models)
            plan_rows, status_rows, _ = _build_plan(config, out_dir, pred_dir, metrics_dir, models, folds, args)
            safe_write_csv(pd.DataFrame(plan_rows), outputs[0])
            safe_write_csv(pd.DataFrame(status_rows), outputs[1])
            if rc != 0:
                failed += 1
                print(f"Stopping LOSO runner after failed job: {label}", flush=True)
                break
        if failed:
            return 1

    metadata = build_metadata(script_name=SCRIPT_NAME, input_files=input_files, cfg_hash=cfg_hash, args=args)
    save_metadata_for_outputs(outputs, metadata)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
