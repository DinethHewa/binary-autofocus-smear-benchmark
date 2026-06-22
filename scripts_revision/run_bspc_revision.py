from __future__ import annotations

import argparse
import os
import subprocess
import sys
from datetime import datetime, timezone
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))

from revision_utils import ensure_dir, repo_path, safe_write_json, safe_write_text  # noqa: E402


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Master runner for BSPC revision analysis stages.")
    parser.add_argument("--config", default="configs/bspc_revision.yaml")
    parser.add_argument(
        "--stage",
        required=True,
        choices=["audit", "no_retrain", "plan_multiseed", "plan_loso", "train_missing", "all_changed"],
    )
    parser.add_argument("--confirm", action="store_true")
    parser.add_argument("--allow-train", action="store_true")
    parser.add_argument("--force", action="store_true")
    return parser.parse_args()


def _script(name: str) -> Path:
    return ROOT / "scripts_revision" / name


def _commands(args: argparse.Namespace) -> list[list[str]]:
    base = [sys.executable, "-u"]
    cfg = ["--config", args.config]
    force = ["--force"] if args.force else []
    audit = [
        [*base, str(_script("01_anomaly_audit.py")), *cfg, *force],
        [*base, str(_script("02_label_provenance_audit.py")), *cfg, *force],
        [*base, str(_script("03_export_architecture_specs.py")), *cfg, *force],
    ]
    no_retrain = [
        [*base, str(_script("04_recover_validation_predictions.py")), *cfg, *force],
        [*base, str(_script("05_threshold_selection.py")), *cfg, *force],
        [*base, str(_script("06_caops.py")), *cfg, *force],
        [*base, str(_script("09_statistical_analysis.py")), *cfg, *force],
        [*base, str(_script("10_export_metric_equations.py")), *cfg, *force],
        [*base, str(_script("11_make_paper_assets.py")), *cfg, *force],
    ]
    plan_multiseed = [[*base, str(_script("07_multiseed_runner.py")), *cfg, "--plan-only", *force]]
    plan_loso = [[*base, str(_script("08_loso_runner.py")), *cfg, "--plan-only", *force]]
    if args.stage == "audit":
        return audit
    if args.stage == "no_retrain":
        return no_retrain
    if args.stage == "plan_multiseed":
        return plan_multiseed
    if args.stage == "plan_loso":
        return plan_loso
    if args.stage == "train_missing":
        if not args.confirm:
            raise SystemExit("Refusing train_missing without --confirm.")
        return [
            [*base, str(_script("07_multiseed_runner.py")), *cfg, "--run", "--allow-train", *force],
            [*base, str(_script("08_loso_runner.py")), *cfg, "--run", "--allow-train", *force],
        ]
    if args.stage == "all_changed":
        commands = [*audit, *no_retrain, *plan_multiseed, *plan_loso]
        if args.allow_train:
            if not args.confirm:
                raise SystemExit("Refusing all_changed training without --confirm.")
            commands.extend(
                [
                    [*base, str(_script("07_multiseed_runner.py")), *cfg, "--run", "--allow-train", *force],
                    [*base, str(_script("08_loso_runner.py")), *cfg, "--run", "--allow-train", *force],
                ]
            )
        return commands
    raise ValueError(args.stage)


def _log_name(command: list[str], index: int) -> str:
    script_arg = next((item for item in command if item.endswith(".py")), f"command_{index}")
    script = Path(script_arg).stem
    return f"{index:02d}_{script}.log"


def _command_text(command: list[str]) -> str:
    return subprocess.list2cmdline([str(item) for item in command])


def _write_live_status(out_dir: Path, stage: str, config: str, statuses: list[dict], current: dict | None) -> None:
    safe_write_json(
        {
            "stage": stage,
            "config": config,
            "timestamp": datetime.now(timezone.utc).isoformat(),
            "commands": statuses,
            "current_command": current,
            "success": all(item.get("return_code") == 0 for item in statuses) if current is None else False,
        },
        out_dir / "run_status.json",
    )


def _run_streaming(command: list[str], log_path: Path, index: int, total: int) -> dict:
    started = datetime.now(timezone.utc).isoformat()
    command_text = _command_text(command)
    print(f"[{index}/{total}] START {command_text}", flush=True)
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
            log.write("\nInterrupted by user; terminating child process.\n")
            log.flush()
            raise
    ended = datetime.now(timezone.utc).isoformat()
    state = "OK" if return_code == 0 else f"FAILED ({return_code})"
    print(f"[{index}/{total}] {state} {command_text}", flush=True)
    return {
        "command": command_text,
        "return_code": return_code,
        "started_at": started,
        "ended_at": ended,
        "log_path": str(log_path),
    }


def main() -> int:
    args = parse_args()
    out_dir = ensure_dir("revision_outputs")
    logs_dir = ensure_dir("revision_outputs/logs")
    commands = _commands(args)
    statuses = []
    total = len(commands)
    print(f"Running stage '{args.stage}' with {total} command(s). Logs: {logs_dir}", flush=True)
    for idx, command in enumerate(commands, start=1):
        log_path = logs_dir / _log_name(command, idx)
        current = {"index": idx, "total": total, "command": _command_text(command), "log_path": str(log_path)}
        _write_live_status(out_dir, args.stage, args.config, statuses, current)
        status_item = _run_streaming(command, log_path, idx, total)
        statuses.append(status_item)
        _write_live_status(out_dir, args.stage, args.config, statuses, None if idx == total else current)
        if status_item["return_code"] != 0:
            print(
                f"Stopping stage '{args.stage}' after failed command {idx}/{total}. "
                f"Fix the error, then rerun the same command to resume.",
                flush=True,
            )
            break

    status = {
        "stage": args.stage,
        "config": args.config,
        "timestamp": datetime.now(timezone.utc).isoformat(),
        "commands": statuses,
        "success": all(item["return_code"] == 0 for item in statuses),
    }
    safe_write_json(status, out_dir / "run_status.json")

    lines = [
        "# BSPC Revision Run Summary",
        "",
        f"Stage: `{args.stage}`",
        f"Config: `{args.config}`",
        f"Success: {status['success']}",
        "",
        "## Commands",
        "",
    ]
    for item in statuses:
        state = "OK" if item["return_code"] == 0 else f"FAILED ({item['return_code']})"
        lines.append(f"- {state}: `{item['command']}`")
        lines.append(f"  - Log: `{item['log_path']}`")
    safe_write_text("\n".join(lines) + "\n", out_dir / "run_summary.md")
    return 0 if status["success"] else 1


if __name__ == "__main__":
    raise SystemExit(main())
