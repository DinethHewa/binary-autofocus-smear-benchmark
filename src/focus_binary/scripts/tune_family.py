from __future__ import annotations

import argparse
from pathlib import Path

from focus_binary import paths
from focus_binary.robust.leakage import assert_no_leakage_manifest
from focus_binary.tuning.tuner import run_tuning
from focus_binary.utils.logging import get_logger

logger = get_logger(__name__)


def parse_args(argv: list[str] | None = None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Run hyper-parameter search for a model family.")
    parser.add_argument(
        "--family",
        required=True,
        choices=[
            "cnn",
            "cnn_attention",
            "transfer",
            "vit",
            "hybrid_vit",
            "focus_dnn",
            "cnn_focus_hybrid",
            "convnext",
            "swin",
            "classical_ml",
        ],
    )
    parser.add_argument("--manifest", required=True, help="Manifest CSV with splits")
    parser.add_argument("--config", default=str(paths.CONFIG_DIR / "tuning.yaml"), help="Tuning config YAML path")
    parser.add_argument("--out", default=str(paths.ARTIFACT_DIR / "runs"), help="Output directory root")
    parser.add_argument("--seed", type=int, default=42, help="Random seed")
    parser.add_argument("--tuner", default=None, choices=["hyperband", "bayesian", "random"])
    parser.add_argument("--max-trials", type=int, default=None, help="Max trials for random/bayesian")
    parser.add_argument("--executions-per-trial", type=int, default=1, help="Executions per trial")
    parser.add_argument("--epochs", type=int, default=None, help="Epochs per trial")
    parser.add_argument("--batch-size", type=int, default=None, help="Batch size")
    parser.add_argument("--input-size", type=int, default=None, help="Square input size")
    parser.add_argument("--backbone-set", default=None, choices=["light", "all"], help="Transfer backbone set")
    parser.add_argument("--light-mode", action="store_true", help="Enable light-mode constraints")
    return parser.parse_args(argv)


def main(argv: list[str] | None = None):
    args = parse_args(argv)
    if args.family == "classical_ml":
        raise SystemExit(
            "classical_ml is not a tuner family. Use: "
            "python -m focus_binary.scripts.run_classical_ml --manifest <path> --out-dir ./runs/classical_ml"
        )
    cfg = {}
    try:
        from focus_binary.utils.io import load_yaml
        cfg = load_yaml(Path(args.config))
    except FileNotFoundError:
        logger.warning("Tuning config not found; using CLI defaults", extra={"path": args.config})

    families_cfg = cfg.get("families", {})
    family_cfg = families_cfg.get(args.family, {})

    base_cfg = {}
    try:
        from focus_binary.utils.io import load_yaml as _load_yaml
        base_cfg = _load_yaml(paths.CONFIG_DIR / "default.yaml")
    except FileNotFoundError:
        base_cfg = {}

    output_dir = Path(args.out)
    if not output_dir.is_absolute():
        output_dir = paths.PROJECT_ROOT / output_dir
    family_dir = output_dir / args.family

    input_size = args.input_size or int(cfg.get("input_size", 224))
    batch_size = args.batch_size or int(cfg.get("batch_size", 16))
    epochs = args.epochs or int(cfg.get("epochs", 8))
    tuner_type = args.tuner or cfg.get("tuner", "hyperband")
    max_trials = args.max_trials or int(family_cfg.get("max_trials", 15))
    backbone_set = args.backbone_set or cfg.get("transfer", {}).get("backbone_set", "light")
    light_mode = args.light_mode or bool(cfg.get("light_mode", False))
    early_stop_patience = int(cfg.get("early_stop_patience", 8))
    leakage_check = bool(cfg.get("leakage_check", True))
    leakage_sha1 = bool(cfg.get("leakage_sha1", True))
    leakage_phash = bool(cfg.get("leakage_phash", True))
    leakage_max_list = int(cfg.get("leakage_max_list", 50))
    enabled_measures = base_cfg.get("enabled_focus_measures", ["lapvar", "tenengrad", "brenner", "sml"])
    focus_from_augmented = bool(base_cfg.get("focus_vector_from_augmented", False))

    if leakage_check:
        assert_no_leakage_manifest(
            args.manifest,
            check_sha1=leakage_sha1,
            check_phash=leakage_phash,
            max_list=leakage_max_list,
        )

    result = run_tuning(
        family=args.family,
        manifest_csv=args.manifest,
        output_dir=family_dir,
        seed=args.seed,
        max_trials=max_trials,
        executions_per_trial=args.executions_per_trial,
        tuner_type=tuner_type,
        epochs=epochs,
        batch_size=batch_size,
        input_size=input_size,
        backbone_set=backbone_set,
        early_stop_patience=early_stop_patience,
        light_mode=light_mode,
        leakage_check=leakage_check,
        leakage_sha1=leakage_sha1,
        leakage_phash=leakage_phash,
        leakage_max_list=leakage_max_list,
        enabled_measures=enabled_measures,
        focus_from_augmented=focus_from_augmented,
    )

    logger.info("Tuning complete", extra={"path": str(result.best_model_path)})
    return result


if __name__ == "__main__":
    main()
