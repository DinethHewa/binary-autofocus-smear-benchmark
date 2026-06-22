#!/usr/bin/env python
from __future__ import annotations

import json
import sys
from dataclasses import dataclass
from pathlib import Path
from typing import Dict, List, Optional

import yaml


PROJECT_ROOT = Path(__file__).resolve().parents[1]
SRC_ROOT = PROJECT_ROOT / "src"
if str(SRC_ROOT) not in sys.path:
    sys.path.insert(0, str(SRC_ROOT))

from focus_binary.models.registry import available_models
from focus_binary.tuning.spaces import DEFAULT_SPACES

REQUIRED_FILES = [
    "configs/tuning.yaml",
    "configs/default.yaml",
    "scripts/compare_best.py",
    "scripts/run_classical_ml.py",
    "src/focus_binary/models/registry.py",
    "src/focus_binary/tuning/spaces.py",
    "src/focus_binary/classical_ml/models.py",
    "src/focus_binary/classical_ml/explain.py",
    "src/focus_binary/scripts/compare_best.py",
    "src/focus_binary/scripts/run_classical_ml.py",
]


@dataclass
class CheckResult:
    name: str
    status: str
    rationale: str
    files: List[str]


def _add_result(results: List[CheckResult], status: str, name: str, rationale: str, files: Optional[List[str]] = None) -> None:
    results.append(CheckResult(name=name, status=status, rationale=rationale, files=files or []))


def _load_text(path: Path) -> str:
    try:
        return path.read_text()
    except Exception:
        return ""


def _write_reports(payload: Dict[str, object], out_dir: Path) -> None:
    out_dir.mkdir(parents=True, exist_ok=True)
    json_path = out_dir / "repo_status.json"
    md_path = out_dir / "repo_status.md"

    json_path.write_text(json.dumps(payload, indent=2))

    lines = ["# Repo Status", ""]
    registry = payload.get("registry", {})
    if isinstance(registry, dict) and registry:
        lines.append("## Model Registry")
        for family, names in registry.items():
            names_str = ", ".join(names) if isinstance(names, list) else str(names)
            lines.append(f"- {family}: {names_str}")
        lines.append("")

    checks = payload.get("checks", [])
    if checks:
        lines.append("## Checks")
        for idx, check in enumerate(checks, start=1):
            files = ", ".join(check.get("files", [])) if isinstance(check, dict) else ""
            files_str = f" | files: {files}" if files else ""
            lines.append(f"{idx:02d}. **[{check['status']}]** {check['name']} - {check['rationale']}{files_str}")

    md_path.write_text("\n".join(lines))


def main() -> int:
    results: List[CheckResult] = []

    registry = available_models()
    registry_summary = {family: sorted(list(names.keys())) for family, names in registry.items()}
    if registry_summary:
        _add_result(results, "PASS", "Model registry", "Loaded registry families")
    else:
        _add_result(results, "FAIL", "Model registry", "Registry is empty")

    missing_files = [path for path in REQUIRED_FILES if not (PROJECT_ROOT / path).exists()]
    if missing_files:
        _add_result(results, "FAIL", "Required files", "Missing required files", missing_files)
    else:
        _add_result(results, "PASS", "Required files", "All required files present")

    tuning_path = PROJECT_ROOT / "configs/tuning.yaml"
    tunable_families = sorted(DEFAULT_SPACES.keys())
    config_families: List[str] = []
    missing_tuning: List[str] = []
    if tuning_path.exists():
        cfg = yaml.safe_load(tuning_path.read_text()) or {}
        config_families = sorted((cfg.get("families") or {}).keys())
        missing_tuning = [family for family in tunable_families if family not in config_families]
        if missing_tuning:
            _add_result(
                results,
                "FAIL",
                "Tuning config coverage",
                "Missing tunable families in configs/tuning.yaml",
                missing_tuning,
            )
        else:
            _add_result(results, "PASS", "Tuning config coverage", "All tunable families present in configs/tuning.yaml")
    else:
        _add_result(results, "FAIL", "Tuning config coverage", "configs/tuning.yaml missing", ["configs/tuning.yaml"])

    compare_path = PROJECT_ROOT / "src/focus_binary/scripts/compare_best.py"
    compare_text = _load_text(compare_path)

    deep_ok = "_find_family_runs" in compare_text and "for family_dir in _find_family_runs" in compare_text
    if deep_ok:
        _add_result(results, "PASS", "Leaderboard ingestion (deep)", "compare_best scans family run folders")
    else:
        _add_result(
            results,
            "FAIL",
            "Leaderboard ingestion (deep)",
            "compare_best does not enumerate deep family runs",
            [str(compare_path)],
        )

    classical_ok = "classical_ml" in compare_text and "_load_classical_outputs" in compare_text
    if classical_ok:
        _add_result(results, "PASS", "Leaderboard ingestion (classical)", "classical_ml ingestion present")
    else:
        _add_result(
            results,
            "FAIL",
            "Leaderboard ingestion (classical)",
            "classical_ml ingestion missing in compare_best",
            [str(compare_path)],
        )

    threshold_ok = "threshold_baseline" in compare_text or "threshold_baselines" in compare_text
    if threshold_ok:
        _add_result(results, "PASS", "Leaderboard ingestion (threshold baseline)", "threshold baseline ingestion present")
    else:
        _add_result(
            results,
            "FAIL",
            "Leaderboard ingestion (threshold baseline)",
            "threshold baseline ingestion missing",
            [str(compare_path)],
        )

    summary = {
        "pass": sum(1 for r in results if r.status == "PASS"),
        "fail": sum(1 for r in results if r.status == "FAIL"),
        "warn": sum(1 for r in results if r.status == "WARN"),
    }

    payload = {
        "registry": registry_summary,
        "tunable_families": tunable_families,
        "config_families": config_families,
        "missing_tuning": missing_tuning,
        "checks": [
            {
                "name": r.name,
                "status": r.status,
                "rationale": r.rationale,
                "files": r.files,
            }
            for r in results
        ],
        "summary": summary,
    }

    out_dir = PROJECT_ROOT / "reports"
    _write_reports(payload, out_dir)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
