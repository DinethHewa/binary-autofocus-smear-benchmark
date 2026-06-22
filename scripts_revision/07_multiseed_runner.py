from __future__ import annotations

import argparse
import json
import os
import random
import re
import shutil
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
    repo_path,
    safe_write_csv,
    safe_write_json,
    safe_write_text,
    save_metadata_for_outputs,
    standardize_model_name,
)


SCRIPT_NAME = Path(__file__).name
STACK_COPY_SUFFIX_RE = re.compile(r"\s+\(\d+\)$")
MULTISEED_EVAL_FAMILIES = {
    "cnn",
    "cnn_attention",
    "transfer",
    "vit",
    "hybrid_vit",
    "focus_dnn",
    "cnn_focus_hybrid",
    "convnext",
    "swin",
}


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Plan or run multi-seed evaluation jobs.")
    parser.add_argument("--config", default="configs/bspc_revision.yaml")
    parser.add_argument("--force", action="store_true")
    parser.add_argument("--plan-only", action="store_true", default=True)
    parser.add_argument("--run", action="store_true")
    parser.add_argument("--allow-train", action="store_true")
    return parser.parse_args()


def _command_text(command: list[str]) -> str:
    return subprocess.list2cmdline([str(item) for item in command])


def _checkpoint_candidates(config: dict, model: str, seed: int, out_dir: Path) -> list[Path]:
    runs_dir = repo_path((config.get("paths") or {}).get("runs_dir", "runs"))
    return [
        out_dir / "checkpoints" / f"{model}_seed_{seed}" / "best_model.keras",
        out_dir / model / f"seed_{seed}" / "best_model.keras",
        out_dir / model / f"seed_{seed}" / "final_model.keras",
        runs_dir / f"{model}_seed_{seed}" / "best_model.keras",
        runs_dir / model / f"seed_{seed}" / "best_model.keras",
        runs_dir / model / f"seed_{seed}" / "final_model.keras",
    ]


def _first_existing(paths: list[Path]) -> Path | None:
    for p in paths:
        if p.exists():
            return p
    return None


def _family_out_candidates(out_dir: Path, model: str) -> list[Path]:
    return [
        out_dir / "training_outputs" / model,
        out_dir / "training_outputs" / model / model,  # compatibility with an older wrapper path
    ]


def _seed_dir_candidates(out_dir: Path, model: str, seed: int) -> list[Path]:
    return [family_out / f"seed_{seed}" for family_out in _family_out_candidates(out_dir, model)]


def _seed_output_dir(out_dir: Path, model: str, seed: int) -> Path | None:
    for seed_dir in _seed_dir_candidates(out_dir, model, seed):
        if (seed_dir / "metrics.json").exists() and (seed_dir / "predictions.csv").exists():
            return seed_dir
    return None


def _job_marker(out_dir: Path, model: str, seed: int) -> Path:
    return out_dir / "job_status" / f"multiseed_{model}_seed_{seed}.json"


def _marker_complete(path: Path, force: bool) -> bool:
    if force or not path.exists():
        return False
    try:
        data = json.loads(path.read_text(encoding="utf-8"))
    except Exception:
        return False
    return data.get("status") == "complete" and int(data.get("return_code", 1)) == 0


def _canonical_stack_id(dataset: object, stack_id: object) -> str:
    base = STACK_COPY_SUFFIX_RE.sub("", str(stack_id))
    return f"{dataset}::{base}"


def _allocate_group_splits(groups: list[str], labels: dict[str, int], rng: random.Random) -> tuple[list[str], list[str], list[str]]:
    train_groups: list[str] = []
    val_groups: list[str] = []
    test_groups: list[str] = []
    for label in sorted(set(labels.values())):
        label_groups = [g for g in groups if labels[g] == label]
        rng.shuffle(label_groups)
        n = len(label_groups)
        if n == 0:
            continue
        n_train = int(round(0.70 * n))
        n_val = int(round(0.15 * n))
        if n_train + n_val > n:
            n_val = max(0, n - n_train)
        if n_train == 0:
            n_train = min(1, n)
        train_groups.extend(label_groups[:n_train])
        val_groups.extend(label_groups[n_train : n_train + n_val])
        test_groups.extend(label_groups[n_train + n_val :])
    return train_groups, val_groups, test_groups


def _build_leakage_safe_manifest(source_manifest: Path, dest_manifest: Path, report_path: Path, force: bool, split_seed: int) -> Path:
    if (
        not force
        and dest_manifest.exists()
        and report_path.exists()
        and dest_manifest.stat().st_mtime >= source_manifest.stat().st_mtime
        and report_path.stat().st_mtime >= source_manifest.stat().st_mtime
    ):
        print(f"Multiseed manifest: reusing {dest_manifest}", flush=True)
        return dest_manifest

    df = pd.read_csv(source_manifest)
    required = {"dataset", "image_path", "label", "stack_id"}
    missing = sorted(required - set(df.columns))
    if missing:
        raise KeyError(f"Manifest missing required columns for multiseed split repair: {missing}")

    repaired = df.copy()
    repaired["source_stack_id"] = repaired["stack_id"].astype(str)
    repaired["stack_id"] = [
        _canonical_stack_id(dataset, stack_id)
        for dataset, stack_id in zip(repaired["dataset"], repaired["source_stack_id"])
    ]
    repaired["leakage_group_id"] = repaired["stack_id"]
    repaired["split"] = "unassigned"

    rng = random.Random(split_seed)
    for dataset in sorted(repaired["dataset"].dropna().astype(str).unique()):
        mask = repaired["dataset"].astype(str) == dataset
        subset = repaired.loc[mask]
        group_labels = subset.groupby("stack_id")["label"].mean().map(lambda value: int(value >= 0.5)).to_dict()
        groups = list(group_labels)
        rng.shuffle(groups)
        train_groups, val_groups, test_groups = _allocate_group_splits(groups, group_labels, rng)
        repaired.loc[mask & repaired["stack_id"].isin(train_groups), "split"] = "train"
        repaired.loc[mask & repaired["stack_id"].isin(val_groups), "split"] = "val"
        repaired.loc[mask & repaired["stack_id"].isin(test_groups), "split"] = "test"

    unassigned = int((repaired["split"] == "unassigned").sum())
    if unassigned:
        raise RuntimeError(f"Could not assign split for {unassigned} manifest rows")

    split_counts = (
        repaired.groupby(["dataset", "split", "label"])
        .size()
        .rename("count")
        .reset_index()
        .sort_values(["dataset", "split", "label"])
    )
    safe_write_csv(repaired, dest_manifest)
    safe_write_csv(split_counts, dest_manifest.with_name("leakage_safe_manifest_split_counts.csv"))

    changed_stack = int(
        repaired["source_stack_id"].map(lambda value: bool(STACK_COPY_SUFFIX_RE.search(str(value)))).sum()
    )
    split_counts_text = split_counts.to_string(index=False)
    lines = [
        "# Multiseed Leakage-Safe Manifest",
        "",
        f"Source manifest: `{source_manifest}`",
        f"Output manifest: `{dest_manifest}`",
        "",
        "The multiseed runner uses this derived manifest because the original split can place copied TBF stacks "
        "such as `FieldPos011` and `FieldPos011 (2)` in different splits. The derived manifest canonicalizes "
        "copy suffixes in `stack_id`, then creates a deterministic group split.",
        "",
        f"Rows: {len(repaired)}",
        f"Rows with canonicalized stack IDs: {changed_stack}",
        f"Split seed: {split_seed}",
        "",
        "The downstream multiseed training command runs with `--leakage-check stack_sha1`, so stack overlap and "
        "exact duplicate image leakage still block training. Perceptual-hash-only similarity is not used as a "
        "hard blocker for this training path because it over-flags visually similar microscopy fields.",
        "",
        "## Split Counts",
        "",
        "```",
        split_counts_text,
        "```",
        "",
    ]
    safe_write_text("\n".join(lines), report_path)
    print(f"Multiseed manifest: wrote leakage-safe split to {dest_manifest}", flush=True)
    return dest_manifest


def _multiseed_manifest(config: dict, out_dir: Path, force: bool) -> Path:
    multiseed_cfg = config.get("multiseed") or {}
    source_manifest = repo_path((config.get("paths") or {}).get("manifest", "data/manifest_with_splits.csv"))
    configured_dest = multiseed_cfg.get("leakage_safe_manifest")
    dest_manifest = repo_path(configured_dest) if configured_dest else out_dir / "leakage_safe_manifest.csv"
    if dest_manifest is None:
        dest_manifest = out_dir / "leakage_safe_manifest.csv"
    return _build_leakage_safe_manifest(
        source_manifest=source_manifest,
        dest_manifest=dest_manifest,
        report_path=dest_manifest.with_name("leakage_safe_manifest_report.md"),
        force=force,
        split_seed=int(multiseed_cfg.get("split_seed", 42)),
    )


def _job_command(config: dict, model: str, seed: int, out_dir: Path, action: str, manifest_path: Path, leakage_check: str) -> list[str]:
    runs_dir = (config.get("paths") or {}).get("runs_dir", "runs")
    command = [
        sys.executable,
        "-u",
        str(ROOT / "scripts" / "multiseed_eval.py"),
        "--family",
        model,
        "--manifest",
        str(manifest_path),
        "--runs-dir",
        str(runs_dir),
        "--out-dir",
        str(out_dir / "training_outputs"),
        "--seeds",
        str(seed),
        "--leakage-check",
        leakage_check,
    ]
    if action == "evaluate_only":
        command.append("--no-retrain")
    return command


def _write_status_csv(rows: list[dict], path: Path) -> None:
    safe_write_csv(pd.DataFrame(rows), path)


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
    started = datetime.now(timezone.utc).isoformat()
    marker = {
        "status": "running",
        "label": label,
        "command": command_text,
        "started_at": started,
        "log_path": str(log_path),
    }
    safe_write_json(marker, marker_path)
    print(f"[multiseed {index}/{total}] START {label}", flush=True)
    print(f"[multiseed {index}/{total}] {command_text}", flush=True)
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
            marker.update(
                {
                    "status": "interrupted",
                    "return_code": None,
                    "ended_at": datetime.now(timezone.utc).isoformat(),
                }
            )
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
    print(f"[multiseed {index}/{total}] {state} {label}", flush=True)
    return return_code


def _collect_multiseed_outputs(out_dir: Path, pred_dir: Path, metrics_dir: Path, table_dir: Path, models: list[str], seeds: list[int]) -> None:
    metric_frames = []
    for model in models:
        for family_out in _family_out_candidates(out_dir, model):
            metrics_path = family_out / "multiseed_metrics.csv"
            if metrics_path.exists():
                df = pd.read_csv(metrics_path)
                if "family" not in df.columns and "model_code_name" not in df.columns:
                    df.insert(0, "family", model)
                metric_frames.append(df)
        for seed in seeds:
            seed_dir = _seed_output_dir(out_dir, model, seed)
            if seed_dir is None:
                continue
            src_pred = seed_dir / "predictions.csv"
            dest_pred = pred_dir / f"{model}_seed_{seed}_test.csv"
            try:
                pred_df = pd.read_csv(src_pred)
                pred_df["model_code_name"] = model
                pred_df["model_display_name"] = display_model_name(model)
                pred_df["split"] = "test"
                safe_write_csv(pred_df, dest_pred)
            except Exception:
                shutil.copy2(src_pred, dest_pred)
    if metric_frames:
        metrics_df = pd.concat(metric_frames, ignore_index=True).drop_duplicates()
        safe_write_csv(metrics_df, metrics_dir / "multiseed_metrics.csv")
        group_col = "model_code_name" if "model_code_name" in metrics_df.columns else "family" if "family" in metrics_df.columns else None
        metric_cols = [c for c in ["auc", "AUC", "balanced_accuracy", "MCC", "f1", "F1", "accuracy", "acc"] if c in metrics_df.columns]
        if group_col and metric_cols:
            rows = []
            pooled = metrics_df[metrics_df.get("dataset", "all").astype(str).isin(["all", "pooled"])] if "dataset" in metrics_df.columns else metrics_df
            for model_name, group in pooled.groupby(group_col):
                row = {"model_code_name": model_name, "model_display_name": display_model_name(str(model_name))}
                for col in metric_cols:
                    row[f"{col}_mean"] = group[col].mean()
                    row[f"{col}_std"] = group[col].std()
                row["n_seeds"] = group["seed"].nunique() if "seed" in group.columns else len(group)
                rows.append(row)
            safe_write_csv(pd.DataFrame(rows), table_dir / "table_multiseed_pooled_metrics.csv")
            return
    safe_write_csv(pd.DataFrame(columns=["model_code_name", "model_display_name", "n_seeds", "status"]), table_dir / "table_multiseed_pooled_metrics.csv")


def _build_plan(config: dict, out_dir: Path, pred_dir: Path, metrics_dir: Path, models: list[str], seeds: list[int], args: argparse.Namespace, manifest_path: Path, leakage_check: str) -> tuple[list[dict], list[dict], list[dict]]:
    plan_rows = []
    status_rows = []
    runnable_jobs = []
    for model in models:
        for seed in seeds:
            marker = _job_marker(out_dir, model, seed)
            completed_artifact_dir = _seed_output_dir(out_dir, model, seed)
            completed_by_marker = _marker_complete(marker, args.force)
            checkpoint = _first_existing(_checkpoint_candidates(config, model, seed, out_dir))
            val_pred = pred_dir / f"{model}_seed_{seed}_val.csv"
            test_pred = pred_dir / f"{model}_seed_{seed}_test.csv"
            metrics_file = metrics_dir / "multiseed_metrics.csv"
            unsupported = model not in MULTISEED_EVAL_FAMILIES
            if completed_artifact_dir or completed_by_marker:
                action = "skip_complete"
                status = "complete"
            elif unsupported:
                action = "unsupported_by_multiseed_eval"
                status = "unsupported_optional_model"
            elif checkpoint and not test_pred.exists():
                action = "evaluate_only"
                status = "checkpoint_exists_predictions_missing"
            elif args.allow_train:
                action = "train_missing"
                status = "training_required_allowed"
            else:
                action = "missing_training_required"
                status = "missing_checkpoint_no_training"
            command = (
                _job_command(
                    config,
                    model,
                    seed,
                    out_dir,
                    "evaluate_only" if action == "evaluate_only" else "train_missing",
                    manifest_path,
                    leakage_check,
                )
                if not unsupported
                else []
            )
            row = {
                "model_code_name": model,
                "model_display_name": display_model_name(model, config),
                "seed": seed,
                "manifest_path": str(manifest_path),
                "leakage_check": leakage_check,
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
                "command": _command_text(command) if command else "",
            }
            plan_rows.append(row)
            status_rows.append(
                {
                    "model_code_name": model,
                    "seed": seed,
                    "status": status,
                    "action": action,
                    "train_allowed": bool(args.allow_train),
                    "job_marker": str(marker),
                }
            )
            if command and args.run and action in {"evaluate_only", "train_missing"} and (action != "train_missing" or args.allow_train):
                runnable_jobs.append({"model": model, "seed": seed, "command": command, "marker": marker, "action": action})
    return plan_rows, status_rows, runnable_jobs


def main() -> int:
    args = parse_args()
    if args.run:
        args.plan_only = False
    config = load_config(args.config)
    cfg_hash = config_hash(config)
    out_dir = ensure_dir((config.get("subdirs") or {}).get("multiseed", "revision_outputs/multiseed"))
    pred_dir = ensure_dir(out_dir / "predictions")
    metrics_dir = ensure_dir(out_dir / "metrics")
    ensure_dir(out_dir / "job_status")
    job_log_dir = ensure_dir((config.get("subdirs") or {}).get("logs", "revision_outputs/logs")) / "jobs"
    ensure_dir(job_log_dir)
    table_dir = ensure_dir((config.get("subdirs") or {}).get("tables", "revision_outputs/tables"))
    multiseed_cfg = config.get("multiseed") or {}
    manifest_path = _multiseed_manifest(config, out_dir, args.force)
    leakage_check = str(multiseed_cfg.get("leakage_check", "stack_sha1"))
    outputs = [
        out_dir / "multiseed_run_plan.csv",
        out_dir / "multiseed_status.csv",
        out_dir / "leakage_safe_manifest.csv",
        out_dir / "leakage_safe_manifest_report.md",
        table_dir / "table_multiseed_pooled_metrics.csv",
    ]
    input_files = [args.config, (config.get("paths") or {}).get("manifest", "data/manifest_with_splits.csv")]
    if not args.force and args.plan_only and fresh_all(outputs, input_files, cfg_hash):
        print("multiseed plan outputs are fresh; skipping", flush=True)
        return 0

    models = [standardize_model_name(m) for m in (config.get("priority_multiseed_models") or [])]
    optional = [standardize_model_name(m) for m in (config.get("optional_multiseed_models") or [])]
    models.extend([m for m in optional if m not in models])
    seeds = [int(s) for s in config.get("random_seeds", [11, 22, 33, 44, 55])]

    _collect_multiseed_outputs(out_dir, pred_dir, metrics_dir, table_dir, models, seeds)
    plan_rows, status_rows, runnable_jobs = _build_plan(config, out_dir, pred_dir, metrics_dir, models, seeds, args, manifest_path, leakage_check)
    safe_write_csv(pd.DataFrame(plan_rows), outputs[0])
    _write_status_csv(status_rows, outputs[1])

    if args.run:
        print(f"Multiseed runner: {len(runnable_jobs)} runnable job(s), {len(plan_rows) - len(runnable_jobs)} skipped/planned.", flush=True)
        failed = 0
        for idx, job in enumerate(runnable_jobs, start=1):
            label = f"{job['model']} seed {job['seed']} ({job['action']})"
            log_path = job_log_dir / f"multiseed_{job['model']}_seed_{job['seed']}.log"
            rc = _run_streaming_job(
                command=job["command"],
                marker_path=job["marker"],
                log_path=log_path,
                label=label,
                index=idx,
                total=len(runnable_jobs),
            )
            _collect_multiseed_outputs(out_dir, pred_dir, metrics_dir, table_dir, models, seeds)
            plan_rows, status_rows, _ = _build_plan(config, out_dir, pred_dir, metrics_dir, models, seeds, args, manifest_path, leakage_check)
            safe_write_csv(pd.DataFrame(plan_rows), outputs[0])
            _write_status_csv(status_rows, outputs[1])
            if rc != 0:
                failed += 1
                print(f"Stopping multiseed runner after failed job: {label}", flush=True)
                break
        if failed:
            return 1

    metadata = build_metadata(script_name=SCRIPT_NAME, input_files=input_files, cfg_hash=cfg_hash, args=args)
    save_metadata_for_outputs(outputs, metadata)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
