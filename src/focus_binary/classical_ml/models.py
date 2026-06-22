from __future__ import annotations

from typing import Dict, Iterable, List, Tuple

import hashlib
from pathlib import Path

import numpy as np
import pandas as pd

from focus_binary import paths
from focus_binary.features import focus_measures_np

try:
    import tensorflow as tf
except Exception:  # pragma: no cover
    tf = None

try:
    import tensorflow_io as tfio
except Exception:  # pragma: no cover
    tfio = None

try:
    from sklearn.calibration import CalibratedClassifierCV
    from sklearn.ensemble import GradientBoostingClassifier, HistGradientBoostingClassifier, RandomForestClassifier
    from sklearn.linear_model import LogisticRegression
    from sklearn.pipeline import Pipeline
    from sklearn.preprocessing import StandardScaler
    from sklearn.svm import LinearSVC
except Exception:  # pragma: no cover
    CalibratedClassifierCV = None
    GradientBoostingClassifier = None
    HistGradientBoostingClassifier = None
    RandomForestClassifier = None
    LogisticRegression = None
    Pipeline = None
    StandardScaler = None
    LinearSVC = None

try:
    from PIL import Image
except Exception:  # pragma: no cover
    Image = None


_MEASURE_MAP_NP = {
    "lapvar": focus_measures_np.variance_of_laplacian,
    "tenengrad": focus_measures_np.tenengrad,
    "brenner": focus_measures_np.brenner,
    "sml": focus_measures_np.sml,
}


def _require_tf() -> None:
    if tf is None:
        raise ImportError("TensorFlow is required for focus vector extraction.")


def _require_sklearn() -> None:
    if LogisticRegression is None or RandomForestClassifier is None:
        raise ImportError("scikit-learn is required for classical ML baselines.")


def _require_pil() -> None:
    if Image is None:
        raise ImportError("Pillow is required for numpy focus vector extraction.")


def _calibrated_classifier(estimator, method: str = "sigmoid", cv: int = 3):
    """Compat wrapper for scikit-learn param rename (base_estimator -> estimator)."""
    try:
        return CalibratedClassifierCV(estimator=estimator, method=method, cv=cv)
    except TypeError:
        return CalibratedClassifierCV(base_estimator=estimator, method=method, cv=cv)


def build_classical_models(seed: int = 42) -> Dict[str, object]:
    _require_sklearn()

    lr = Pipeline(
        [
            ("scaler", StandardScaler()),
            (
                "clf",
                LogisticRegression(
                    solver="liblinear",
                    class_weight="balanced",
                    max_iter=1000,
                    random_state=seed,
                ),
            ),
        ]
    )

    svc = Pipeline(
        [
            ("scaler", StandardScaler()),
            (
                "clf",
                _calibrated_classifier(
                    LinearSVC(class_weight="balanced", random_state=seed),
                    method="sigmoid",
                    cv=3,
                ),
            ),
        ]
    )

    rf = RandomForestClassifier(
        n_estimators=300,
        max_depth=None,
        min_samples_split=2,
        class_weight="balanced",
        random_state=seed,
        n_jobs=-1,
    )

    if HistGradientBoostingClassifier is not None:
        gb = HistGradientBoostingClassifier(
            max_depth=6,
            learning_rate=0.1,
            max_iter=200,
            random_state=seed,
        )
    else:
        gb = GradientBoostingClassifier(random_state=seed)

    return {
        "logistic_regression": lr,
        "linear_svc_calibrated": svc,
        "random_forest": rf,
        "gradient_boosting": gb,
    }


def _load_image(path: str, input_size: int) -> "tf.Tensor":
    _require_tf()
    contents = tf.io.read_file(path)
    lower = tf.strings.lower(tf.convert_to_tensor(path))
    is_tif = tf.strings.regex_full_match(lower, ".*\\.tiff?$")

    def _decode_tif():
        if tfio is not None:
            return tfio.experimental.image.decode_tiff(contents)
        return tf.image.decode_image(contents, channels=0, expand_animations=False)

    def _decode_any():
        return tf.image.decode_image(contents, channels=3, expand_animations=False)

    image = tf.cond(is_tif, _decode_tif, _decode_any)
    channels = tf.shape(image)[-1]

    def _to_rgb():
        return tf.image.grayscale_to_rgb(image)

    def _from_rgba():
        return image[..., :3]

    image = tf.cond(
        tf.equal(channels, 1),
        _to_rgb,
        lambda: tf.cond(tf.equal(channels, 3), lambda: image, _from_rgba),
    )
    image = tf.image.resize(image, (input_size, input_size))
    image = tf.image.convert_image_dtype(image, tf.float32)
    return image


def _load_image_np(path: str, input_size: int) -> np.ndarray:
    _require_pil()
    with Image.open(path) as img:
        img = img.convert("RGB")
        img = img.resize((input_size, input_size), resample=Image.BILINEAR)
        arr = np.asarray(img, dtype=np.float32) / 255.0
    return arr


def _compute_vector_np(image: np.ndarray, measures: List[str]) -> np.ndarray:
    values = []
    for name in measures:
        fn = _MEASURE_MAP_NP.get(name)
        if fn is None:
            raise KeyError(f"Unknown focus measure '{name}'")
        values.append(fn(image))
    return np.asarray(values, dtype=np.float32)


def _manifest_hash(manifest_path: Path, measures: List[str], input_size: int) -> str:
    hasher = hashlib.sha1()
    hasher.update(manifest_path.read_bytes())
    hasher.update(str(input_size).encode("utf-8"))
    hasher.update(",".join(measures).encode("utf-8"))
    return hasher.hexdigest()


def _cache_path(manifest_hash: str, cache_dir: Path | None = None) -> Path:
    if cache_dir is None:
        cache_dir = paths.PROJECT_ROOT / "runs" / "classical_ml" / "cache"
    cache_dir.mkdir(parents=True, exist_ok=True)
    return cache_dir / f"focus_vectors_{manifest_hash}.npz"


def compute_focus_vectors(
    paths: Iterable[str],
    input_size: int,
    enabled_measures: List[str],
    batch_size: int = 64,
    use_tensorflow_loader: bool = False,
    manifest_path: str | Path | None = None,
    cache_dir: str | Path | None = None,
) -> np.ndarray:
    measures = [m.strip().lower() for m in enabled_measures]
    path_list = [str(p) for p in paths]
    if not path_list:
        return np.zeros((0, len(measures)), dtype=np.float32)

    manifest_paths: List[str] | None = None
    cache_path: Path | None = None
    if manifest_path is not None:
        manifest_path = Path(manifest_path)
        if not manifest_path.is_absolute():
            manifest_path = paths.PROJECT_ROOT / manifest_path
        manifest_hash = _manifest_hash(manifest_path, measures, input_size)
        cache_path = _cache_path(manifest_hash, Path(cache_dir) if cache_dir else None)
        if cache_path.exists():
            cached = np.load(cache_path, allow_pickle=True)
            cached_names = [str(x) for x in cached.get("feature_names", [])]
            if cached_names == measures:
                cached_paths = [str(x) for x in cached.get("image_paths", [])]
                vectors = cached.get("vectors")
                if vectors is not None and cached_paths:
                    index = {p: i for i, p in enumerate(cached_paths)}
                    try:
                        return np.stack([vectors[index[p]] for p in path_list], axis=0)
                    except KeyError:
                        pass
        df = pd.read_csv(manifest_path)
        manifest_paths = df["image_path"].astype(str).tolist()

    target_paths = manifest_paths if manifest_paths is not None else path_list

    if use_tensorflow_loader:
        _require_tf()
        from focus_binary.features.vectorize import compute_focus_vector
        ds = tf.data.Dataset.from_tensor_slices(target_paths)
        ds = ds.map(lambda p: _load_image(p, input_size=input_size), num_parallel_calls=tf.data.AUTOTUNE)
        ds = ds.batch(batch_size)
        vectors = []
        for batch in ds:
            vecs = tf.map_fn(
                lambda img: compute_focus_vector(img, list(measures)),
                batch,
                fn_output_signature=tf.float32,
            )
            vectors.append(vecs.numpy())
        vectors_arr = np.concatenate(vectors, axis=0)
    else:
        vectors_arr = np.zeros((len(target_paths), len(measures)), dtype=np.float32)
        for i, path in enumerate(target_paths):
            image = _load_image_np(path, input_size=input_size)
            vectors_arr[i] = _compute_vector_np(image, measures)

    if cache_path is not None and manifest_paths is not None and not cache_path.exists():
        np.savez_compressed(
            cache_path,
            vectors=vectors_arr,
            image_paths=np.asarray(manifest_paths),
            feature_names=np.asarray(measures),
        )

    if manifest_paths is None:
        return vectors_arr

    index = {p: i for i, p in enumerate(manifest_paths)}
    return np.stack([vectors_arr[index[p]] for p in path_list], axis=0)


def predict_probabilities(model, X: np.ndarray) -> np.ndarray:
    if hasattr(model, "predict_proba"):
        proba = model.predict_proba(X)
        if proba.ndim == 2 and proba.shape[1] >= 2:
            return proba[:, 1]
        return proba.reshape(-1)
    if hasattr(model, "decision_function"):
        scores = model.decision_function(X)
        scores = np.asarray(scores, dtype=float).reshape(-1)
        return 1.0 / (1.0 + np.exp(-scores))
    preds = model.predict(X)
    return np.asarray(preds, dtype=float).reshape(-1)
