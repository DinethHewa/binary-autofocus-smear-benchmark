from __future__ import annotations

from dataclasses import dataclass
import time
from pathlib import Path
from typing import Any, Dict, List, Optional

import pandas as pd

from focus_binary import paths
from focus_binary.data.tfdata import build_datasets
from focus_binary.data.tfdata_features import build_feature_datasets
from focus_binary.data.balance import compute_class_weights, report_and_check_imbalance
from focus_binary.data.splits import assert_no_leak
from focus_binary.models.cnn_baseline import build_cnn_baseline
from focus_binary.models.cnn_attention import build_cnn_attention
from focus_binary.models.focus_dnn import build_focus_dnn
from focus_binary.models.convnext import build_convnext
from focus_binary.models.cnn_focus_hybrid import build_cnn_focus_hybrid
from focus_binary.models.hybrid import build_hybrid_vit
from focus_binary.models.transfer import (
    TRANSFER_BACKBONES_ALL,
    TRANSFER_BACKBONES_LIGHT,
    build_transfer_model,
    compile_transfer_model,
)
from focus_binary.models.vit import build_vit
from focus_binary.models.swin_tiny import build_swin_tiny
from focus_binary.tuning.spaces import build_hyperparameters, get_search_space
from focus_binary.utils.io import save_json, save_model
from focus_binary.utils.logging import get_logger
from focus_binary.utils.efficiency import count_params, hardware_string, measure_latency
from focus_binary.utils.seed import set_global_seed

logger = get_logger(__name__)

try:
    import keras_tuner as kt
except Exception:  # pragma: no cover
    kt = None

try:
    import tensorflow as tf
except Exception:  # pragma: no cover
    tf = None


def _load_model_with_custom_objects(model_path: Path):
    if tf is None:
        raise ImportError("TensorFlow is required to load models.")
    from focus_binary.models.vit import _CLSToken, _PositionalEmbedding  # type: ignore
    from focus_binary.models.swin_tiny import WindowPartition, WindowReverse  # type: ignore
    from focus_binary.models.convnext import ConvNeXtPreprocess  # type: ignore

    custom_objects = {}
    if _CLSToken is not None:
        custom_objects["_CLSToken"] = _CLSToken
    if _PositionalEmbedding is not None:
        custom_objects["_PositionalEmbedding"] = _PositionalEmbedding
    if WindowPartition is not None:
        custom_objects["WindowPartition"] = WindowPartition
    if WindowReverse is not None:
        custom_objects["WindowReverse"] = WindowReverse
    if ConvNeXtPreprocess is not None:
        custom_objects["ConvNeXtPreprocess"] = ConvNeXtPreprocess
    if custom_objects:
        return tf.keras.models.load_model(model_path, custom_objects=custom_objects)
    return tf.keras.models.load_model(model_path)


@dataclass
class TuningResult:
    family: str
    best_model_path: Path
    summary_path: Path
    trials: List[Dict[str, Any]]


if kt is not None and tf is not None:
    class _TrialMetricsLogger(tf.keras.callbacks.Callback):
        def __init__(self, trial_id: str, rows: List[Dict[str, Any]]):
            super().__init__()
            self.trial_id = trial_id
            self.rows = rows

        def on_epoch_end(self, epoch, logs=None):
            logs = logs or {}
            row = {"trial_id": self.trial_id, "epoch": epoch}
            for key, value in logs.items():
                if value is not None:
                    row[key] = float(value)
            self.rows.append(row)


    class _FocusHyperModel(kt.HyperModel):
        def __init__(
            self,
            family: str,
            input_size: int,
            backbone_choices: List[str] | None = None,
            focus_dim: int = 0,
            light_mode: bool = False,
        ):
            self.family = family
            self.input_size = input_size
            self.backbone_choices = backbone_choices or []
            self.focus_dim = focus_dim
            self.light_mode = light_mode

        def build(self, hp):
            params = build_hyperparameters(
                self.family,
                hp,
                backbone_choices=self.backbone_choices,
                light_mode=self.light_mode,
            )
            input_shape = (self.input_size, self.input_size, 3)

            if self.family == "cnn":
                model = build_cnn_baseline(input_shape=input_shape, **params)
                return {"model": model}

            if self.family == "cnn_attention":
                attention_type = params["attention_type"]
                model = build_cnn_attention(hp, input_shape=input_shape, attention_type=attention_type)
                return {"model": model}

            if self.family == "vit":
                model = build_vit(input_shape=input_shape, **params)
                return {"model": model}

            if self.family == "hybrid_vit":
                model = build_hybrid_vit(input_shape=input_shape, **params)
                return {"model": model}

            if self.family == "transfer":
                spec = build_transfer_model(
                    backbone=params["backbone"],
                    input_size=self.input_size,
                    pooling=params["pooling"],
                    head_units=params["head_units"],
                    dropout=params["dropout"],
                    base_trainable=params["base_trainable_blocks"],
                    lr=params["lr"],
                    label_smoothing=params["label_smoothing"],
                )
                model = spec["model"]
                compile_transfer_model(model, lr=params["lr"], label_smoothing=params["label_smoothing"])
                spec["params"] = params
                return spec

            if self.family == "focus_dnn":
                model = build_focus_dnn(hp, input_dim=self.focus_dim)
                return {"model": model}

            if self.family == "cnn_focus_hybrid":
                model = build_cnn_focus_hybrid(hp, input_shape=input_shape, focus_dim=self.focus_dim)
                return {"model": model}

            if self.family == "convnext":
                model = build_convnext(input_shape=input_shape, **params)
                return {"model": model}

            if self.family == "swin":
                model = build_swin_tiny(input_shape=input_shape, **params)
                return {"model": model}

            raise KeyError(f"Unsupported family: {self.family}")


    class _FocusTuner(kt.Tuner):
        def __init__(
            self,
            *args,
            family: str,
            manifest_csv: str | Path,
            input_size: int,
            batch_size: int,
            seed: int,
            epochs: int,
            class_weight: Dict[int, float] | None,
            light_mode: bool,
            early_stop_patience: int,
            enabled_measures: List[str],
            focus_from_augmented: bool,
            **kwargs,
        ):
            super().__init__(*args, **kwargs)
            self.family = family
            manifest_path = Path(manifest_csv)
            if not manifest_path.is_absolute():
                manifest_path = paths.PROJECT_ROOT / manifest_path
            self.manifest_csv = manifest_path
            self.input_size = input_size
            self.batch_size = batch_size
            self.seed = seed
            self.epochs = epochs
            self.class_weight = class_weight
            self.light_mode = light_mode
            self._early_stop_patience = early_stop_patience
            self.val_rows: List[Dict[str, Any]] = []
            self.enabled_measures = enabled_measures
            self.focus_from_augmented = focus_from_augmented

        def load_model(self, trial):
            trial_dir = Path(self.project_dir) / f"trial_{trial.trial_id}"
            direct_path = trial_dir / "model_step_0.keras"
            if direct_path.exists():
                return _load_model_with_custom_objects(direct_path)
            best_path = self._find_best_checkpoint(trial_dir)
            if best_path:
                return _load_model_with_custom_objects(best_path)
            return super().load_model(trial)

        def _find_best_checkpoint(self, trial_dir: Path) -> Optional[Path]:
            candidates = list(trial_dir.glob("**/best.keras"))
            if not candidates:
                return None
            return max(candidates, key=lambda path: path.stat().st_mtime)

        def save_model(self, trial_id, model, step=0):
            if model is None:
                return
            trial_dir = Path(self.project_dir) / f"trial_{trial_id}"
            trial_dir.mkdir(parents=True, exist_ok=True)
            model_path = trial_dir / f"model_step_{step}.keras"
            model.save(model_path)

        def run_trial(self, trial, *args, **kwargs):
            hp = trial.hyperparameters
            best_score: Optional[float] = None
            best_model: Optional[Any] = None

            for execution in range(self.executions_per_trial):
                model_bundle = self.hypermodel.build(hp)
                model = model_bundle["model"] if isinstance(model_bundle, dict) else model_bundle

                if self.light_mode and model is not None:
                    params_count = int(model.count_params())
                    if params_count > 10_000_000:
                        logger.warning(
                            "Model exceeds light-mode parameter cap",
                            extra={"trial": trial.trial_id, "params_count": params_count},
                        )

                preprocess_fn = None
                preprocess_in_model = False
                if isinstance(model_bundle, dict):
                    preprocess_fn = model_bundle.get("preprocess_input")
                    preprocess_in_model = bool(model_bundle.get("preprocess_in_model", False))

                if self.family in {"focus_dnn", "cnn_focus_hybrid"}:
                    train_ds = build_feature_datasets(
                        manifest_csv=self.manifest_csv,
                        split="train",
                        batch_size=self.batch_size,
                        input_size=self.input_size,
                        image_mode="rgb",
                        enabled_measures=self.enabled_measures,
                        augment_images=True,
                        shuffle=True,
                        seed=self.seed,
                        compute_from_augmented=self.focus_from_augmented,
                    )
                    val_ds = build_feature_datasets(
                        manifest_csv=self.manifest_csv,
                        split="val",
                        batch_size=self.batch_size,
                        input_size=self.input_size,
                        image_mode="rgb",
                        enabled_measures=self.enabled_measures,
                        augment_images=False,
                        shuffle=False,
                        seed=self.seed,
                        compute_from_augmented=self.focus_from_augmented,
                    )
                    if self.family == "focus_dnn":
                        train_ds = train_ds.map(lambda x, y: (x[1], y), num_parallel_calls=tf.data.AUTOTUNE)
                        val_ds = val_ds.map(lambda x, y: (x[1], y), num_parallel_calls=tf.data.AUTOTUNE)
                else:
                    train_ds = build_datasets(
                        manifest_csv=self.manifest_csv,
                        split="train",
                        batch_size=self.batch_size,
                        input_size=self.input_size,
                        image_mode="rgb",
                        augment=True,
                        shuffle=True,
                        seed=self.seed,
                        force_rgb=True,
                    )
                    val_ds = build_datasets(
                        manifest_csv=self.manifest_csv,
                        split="val",
                        batch_size=self.batch_size,
                        input_size=self.input_size,
                        image_mode="rgb",
                        augment=False,
                        shuffle=False,
                        seed=self.seed,
                        force_rgb=True,
                    )

                if preprocess_fn is not None and not preprocess_in_model:
                    def _apply_preprocess(img, label):
                        return preprocess_fn(img * 255.0), label

                    train_ds = train_ds.map(_apply_preprocess, num_parallel_calls=tf.data.AUTOTUNE)
                    val_ds = val_ds.map(_apply_preprocess, num_parallel_calls=tf.data.AUTOTUNE)

                trial_dir = Path(self.project_dir) / f"trial_{trial.trial_id}" / f"exec_{execution}"
                trial_dir.mkdir(parents=True, exist_ok=True)

                callbacks = [
                    tf.keras.callbacks.EarlyStopping(
                        monitor="val_auc",
                        mode="max",
                        patience=self._early_stop_patience,
                        restore_best_weights=True,
                    ),
                    tf.keras.callbacks.ReduceLROnPlateau(
                        monitor="val_auc",
                        mode="max",
                        factor=0.5,
                        patience=4,
                        min_lr=1e-6,
                    ),
                    tf.keras.callbacks.ModelCheckpoint(
                        filepath=str(trial_dir / "best.keras"),
                        monitor="val_auc",
                        mode="max",
                        save_best_only=True,
                    ),
                    tf.keras.callbacks.CSVLogger(str(trial_dir / "history.csv")),
                    _TrialMetricsLogger(trial.trial_id, self.val_rows),
                ]

                history = model.fit(
                    train_ds,
                    validation_data=val_ds,
                    epochs=self.epochs,
                    callbacks=callbacks,
                    class_weight=self.class_weight,
                    verbose=0,
                )

                metric_name = self.oracle.objective.name
                scores = history.history.get(metric_name, [])
                if not scores:
                    scores = history.history.get("val_auc", [])
                score = max(scores) if scores else None

                if score is not None and (best_score is None or score > best_score):
                    best_score = score
                    best_model = model

            if best_score is None:
                best_score = float("-inf")

            self.oracle.update_trial(trial.trial_id, metrics={self.oracle.objective.name: best_score})
            if best_model is not None:
                self.save_model(trial.trial_id, best_model)
else:  # pragma: no cover
    _TrialMetricsLogger = None
    _FocusHyperModel = None
    _FocusTuner = None


def _select_backbones(backbone_set: str) -> List[str]:
    if backbone_set == "all":
        return list(TRANSFER_BACKBONES_ALL)
    return list(TRANSFER_BACKBONES_LIGHT)


def _tuner_for_type(
    tuner_type: str,
    hypermodel: kt.HyperModel,
    objective: kt.Objective,
    max_trials: int,
    executions_per_trial: int,
    seed: int,
    project_dir: Path,
    overwrite: bool,
    family: str,
    manifest_csv: str | Path,
    input_size: int,
    batch_size: int,
    epochs: int,
    class_weight: Dict[int, float] | None,
    light_mode: bool,
    early_stop_patience: int,
    enabled_measures: List[str],
    focus_from_augmented: bool,
) -> _FocusTuner:
    def _oracle_cls(primary: str, fallback: str):
        return getattr(kt.oracles, primary, getattr(kt.oracles, fallback, None))

    if tuner_type == "bayesian":
        oracle_cls = _oracle_cls("BayesianOptimizationOracle", "BayesianOptimization")
        oracle = oracle_cls(
            objective=objective,
            max_trials=max_trials,
            seed=seed,
        )
    elif tuner_type == "hyperband":
        oracle_cls = _oracle_cls("HyperbandOracle", "Hyperband")
        oracle = oracle_cls(
            objective=objective,
            max_epochs=epochs,
            factor=3,
            seed=seed,
        )
    else:
        oracle_cls = _oracle_cls("RandomSearchOracle", "RandomSearch")
        oracle = oracle_cls(
            objective=objective,
            max_trials=max_trials,
            seed=seed,
        )

    return _FocusTuner(
        oracle=oracle,
        hypermodel=hypermodel,
        executions_per_trial=executions_per_trial,
        overwrite=overwrite,
        directory=str(project_dir),
        project_name="kt",
        family=family,
        manifest_csv=manifest_csv,
        input_size=input_size,
        batch_size=batch_size,
        seed=seed,
        epochs=epochs,
        class_weight=class_weight,
        light_mode=light_mode,
        early_stop_patience=early_stop_patience,
        enabled_measures=enabled_measures,
        focus_from_augmented=focus_from_augmented,
    )


def run_tuning(
    family: str,
    manifest_csv: str | Path,
    output_dir: Path,
    seed: int,
    max_trials: int,
    executions_per_trial: int,
    tuner_type: str,
    objective: str = "val_auc",
    epochs: int = 8,
    batch_size: int = 16,
    input_size: int = 224,
    backbone_set: str = "light",
    early_stop_patience: int = 8,
    light_mode: bool = False,
    leakage_check: bool = True,
    leakage_sha1: bool = True,
    leakage_phash: bool = True,
    leakage_max_list: int = 50,
    enabled_measures: List[str] | None = None,
    focus_from_augmented: bool = False,
) -> TuningResult:
    if kt is None or tf is None:
        raise ImportError("keras-tuner and tensorflow are required for tuning.")

    wall_start = time.perf_counter()
    set_global_seed(seed)
    output_dir.mkdir(parents=True, exist_ok=True)

    if enabled_measures is None:
        enabled_measures = ["lapvar", "tenengrad", "brenner", "sml"]

    if light_mode:
        backbone_set = "light"

    backbone_choices = _select_backbones(backbone_set) if family == "transfer" else None
    if family == "transfer" and not backbone_choices:
        raise ValueError("No backbones available for transfer tuning.")

    hypermodel = _FocusHyperModel(
        family,
        input_size=input_size,
        backbone_choices=backbone_choices,
        focus_dim=len(enabled_measures),
        light_mode=light_mode,
    )
    objective_obj = kt.Objective(objective, direction="max")

    manifest_path = Path(manifest_csv)
    if not manifest_path.is_absolute():
        manifest_path = paths.PROJECT_ROOT / manifest_path
    manifest_csv = manifest_path.as_posix()
    df = pd.read_csv(manifest_path)
    assert_no_leak(df, group_col="stack_id", split_col="split")
    if leakage_check:
        from focus_binary.robust.leakage import assert_no_leakage_manifest
        assert_no_leakage_manifest(
            manifest_path,
            check_sha1=leakage_sha1,
            check_phash=leakage_phash,
            max_list=leakage_max_list,
        )
    _, extreme = report_and_check_imbalance(df)
    class_weight = compute_class_weights(df) if extreme else None
    tuner_state_dir = output_dir / "kt"
    overwrite = not ((tuner_state_dir / "tuner.json").exists() or (tuner_state_dir / "oracle.json").exists())
    if not overwrite:
        logger.info("resuming tuner state", extra={"path": str(tuner_state_dir)})

    tuner = _tuner_for_type(
        tuner_type=tuner_type,
        hypermodel=hypermodel,
        objective=objective_obj,
        max_trials=max_trials,
        executions_per_trial=executions_per_trial,
        seed=seed,
        project_dir=output_dir,
        overwrite=overwrite,
        family=family,
        manifest_csv=manifest_csv,
        input_size=input_size,
        batch_size=batch_size,
        epochs=epochs,
        class_weight=class_weight,
        light_mode=light_mode,
        early_stop_patience=early_stop_patience,
        enabled_measures=enabled_measures,
        focus_from_augmented=focus_from_augmented,
    )

    search_space = get_search_space(family)
    logger.info("search space loaded", extra={"family": family, "keys": list(search_space.keys())})

    tuner.search()
    tuning_walltime_s = time.perf_counter() - wall_start
    logger.info(
        "tuning completed",
        extra={
            "family": family,
            "max_trials": max_trials,
            "executions_per_trial": executions_per_trial,
            "epochs": epochs,
            "walltime_s": round(tuning_walltime_s, 2),
        },
    )

    best_models = tuner.get_best_models(num_models=1)
    if not best_models:
        raise RuntimeError("No tuned models were produced.")
    best_model = best_models[0]

    best_hp = tuner.get_best_hyperparameters(num_trials=1)[0]
    best_hparams = dict(best_hp.values)

    best_model_path = output_dir / "best_model.keras"
    save_model(best_model, best_model_path)

    best_hparams_path = output_dir / "best_hparams.json"
    save_json(best_hparams, best_hparams_path)

    trials_rows = []
    for trial in tuner.oracle.trials.values():
        row = {
            "trial_id": trial.trial_id,
            "score": trial.score,
            "status": trial.status,
        }
        row.update(trial.hyperparameters.values)
        trials_rows.append(row)
    trials_df = pd.DataFrame(trials_rows)
    tuning_results_path = output_dir / "tuning_results.csv"
    trials_df.to_csv(tuning_results_path, index=False)

    val_curves_path = output_dir / "val_curves.csv"
    if tuner.val_rows:
        pd.DataFrame(tuner.val_rows).to_csv(val_curves_path, index=False)
    else:
        pd.DataFrame([]).to_csv(val_curves_path, index=False)

    params_count = count_params(best_model)
    latency_mean = None
    latency_p95 = None
    try:
        latency_mean, latency_p95 = measure_latency(best_model, input_size=input_size, batch_size=1)
    except Exception as exc:
        logger.warning("latency measurement failed", extra={"error": str(exc)})

    if family in {"focus_dnn", "cnn_focus_hybrid"}:
        val_ds = build_feature_datasets(
            manifest_csv=manifest_csv,
            split="val",
            batch_size=batch_size,
            input_size=input_size,
            image_mode="rgb",
            enabled_measures=enabled_measures,
            augment_images=False,
            shuffle=False,
            seed=seed,
            compute_from_augmented=focus_from_augmented,
        )
        if family == "focus_dnn":
            val_ds = val_ds.map(lambda x, y: (x[1], y), num_parallel_calls=tf.data.AUTOTUNE)
    else:
        val_ds = build_datasets(
            manifest_csv=manifest_csv,
            split="val",
            batch_size=batch_size,
            input_size=input_size,
            image_mode="rgb",
            augment=False,
            shuffle=False,
            seed=seed,
            force_rgb=True,
        )
    if family == "transfer":
        spec = hypermodel.build(best_hp)
        preprocess_fn = spec.get("preprocess_input") if isinstance(spec, dict) else None
        preprocess_in_model = bool(spec.get("preprocess_in_model", False)) if isinstance(spec, dict) else False
        if preprocess_fn is not None and not preprocess_in_model:
            def _apply_preprocess(img, label):
                return preprocess_fn(img * 255.0), label

            val_ds = val_ds.map(_apply_preprocess, num_parallel_calls=tf.data.AUTOTUNE)

    eval_results = best_model.evaluate(val_ds, verbose=0)
    metrics = dict(zip(best_model.metrics_names, [float(v) for v in eval_results]))

    summary = {
        "family": family,
        "params_count": params_count,
        "latency_ms_mean": latency_mean,
        "latency_ms_p95": latency_p95,
        "tuning_walltime_s": float(tuning_walltime_s),
        "training_walltime_s": 0.0,
        "hardware": hardware_string(),
        "best_val_metrics": metrics,
        "best_hparams": best_hparams,
        "objective": objective,
        "tuner_type": tuner_type,
        "max_trials": max_trials,
        "executions_per_trial": executions_per_trial,
        "epochs": epochs,
        "batch_size": batch_size,
        "input_size": input_size,
        "backbone_set": backbone_set,
        "light_mode": light_mode,
    }
    summary_path = output_dir / "summary.json"
    save_json(summary, summary_path)

    logger.info(
        "tuning complete",
        extra={"family": family, "best_model": str(best_model_path), "summary": str(summary_path)},
    )
    return TuningResult(
        family=family,
        best_model_path=best_model_path,
        summary_path=summary_path,
        trials=trials_rows,
    )
