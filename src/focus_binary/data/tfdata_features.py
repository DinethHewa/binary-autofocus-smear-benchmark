from __future__ import annotations

from pathlib import Path
from typing import Callable, Optional, Sequence, Tuple

import pandas as pd

from focus_binary import paths
from focus_binary.data.augment import build_augmenter
from focus_binary.features.vectorize import compute_focus_vector
from focus_binary.utils.logging import get_logger

logger = get_logger(__name__)

try:
    import tensorflow as tf
except Exception:  # pragma: no cover
    tf = None

try:
    import tensorflow_io as tfio
except Exception:  # pragma: no cover
    tfio = None


def _require_tf():
    if tf is None:
        raise ImportError("TensorFlow is not installed; install dependencies to enable tf.data pipelines.")


def _parse_input_size(input_size: int | Sequence[int]) -> Tuple[int, int]:
    if isinstance(input_size, int):
        return (input_size, input_size)
    if isinstance(input_size, (list, tuple)) and len(input_size) == 2:
        return (int(input_size[0]), int(input_size[1]))
    raise ValueError("input_size must be an int or (height, width)")


def _decode_tiff(contents):
    if tfio is not None:
        return tfio.experimental.image.decode_tiff(contents)
    return tf.image.decode_image(contents, channels=0, expand_animations=False)


def _decode_by_ext(contents, path: tf.Tensor, channels: int) -> tf.Tensor:
    lower = tf.strings.lower(path)
    is_png = tf.strings.regex_full_match(lower, ".*\\.png$")
    is_jpg = tf.strings.regex_full_match(lower, ".*\\.jpe?g$")
    is_bmp = tf.strings.regex_full_match(lower, ".*\\.bmp$")
    is_tif = tf.strings.regex_full_match(lower, ".*\\.tiff?$")

    def _decode_png():
        return tf.image.decode_png(contents, channels=channels)

    def _decode_jpeg():
        return tf.image.decode_jpeg(contents, channels=channels)

    def _decode_bmp():
        return tf.image.decode_bmp(contents, channels=channels)

    def _decode_tif():
        return _decode_tiff(contents)

    return tf.case(
        [(is_png, _decode_png), (is_jpg, _decode_jpeg), (is_bmp, _decode_bmp), (is_tif, _decode_tif)],
        default=_decode_jpeg,
        exclusive=False,
    )


def _ensure_channels(image: tf.Tensor, channels: int) -> tf.Tensor:
    current = tf.shape(image)[-1]

    if channels == 1:
        def _to_gray():
            rgb = image[..., :3]
            return tf.image.rgb_to_grayscale(rgb)

        return tf.cond(tf.equal(current, 1), lambda: image, _to_gray)

    if channels == 3:
        def _from_gray():
            return tf.image.grayscale_to_rgb(image)

        def _from_rgba():
            return image[..., :3]

        return tf.cond(
            tf.equal(current, 1),
            _from_gray,
            lambda: tf.cond(tf.equal(current, 3), lambda: image, _from_rgba),
        )

    raise ValueError("channels must be 1 or 3")


def _load_image(
    path: tf.Tensor,
    input_size: Tuple[int, int],
    image_mode: str,
) -> tf.Tensor:
    contents = tf.io.read_file(path)
    channels = 1 if image_mode == "grayscale" else 3
    image = _decode_by_ext(contents, path, channels=channels)
    image = _ensure_channels(image, channels=channels)
    image = tf.image.convert_image_dtype(image, tf.float32)
    image = tf.image.resize(image, input_size)
    return image


def build_feature_datasets(
    manifest_csv: str | Path,
    split: str,
    batch_size: int,
    input_size: int | Sequence[int],
    image_mode: str,
    enabled_measures: Sequence[str],
    augment_images: bool = False,
    shuffle: bool = False,
    seed: int = 42,
    compute_from_augmented: bool = False,
) -> "tf.data.Dataset":
    """Build tf.data.Dataset yielding ((image, focus_vector), label)."""
    _require_tf()

    image_mode = image_mode.lower().strip()
    if image_mode in {"gray", "greyscale"}:
        image_mode = "grayscale"
    if image_mode not in {"grayscale", "rgb"}:
        raise ValueError("image_mode must be 'grayscale' or 'rgb'")

    input_size = _parse_input_size(input_size)

    manifest_path = Path(manifest_csv)
    if not manifest_path.is_absolute():
        manifest_path = paths.PROJECT_ROOT / manifest_path
    df = pd.read_csv(manifest_path)
    if split:
        if "split" not in df.columns:
            raise KeyError("manifest is missing 'split' column")
        df = df[df["split"] == split]

    if df.empty:
        raise ValueError("No rows found for the requested split")

    paths = df["image_path"].astype(str).tolist()
    labels = df["label"].astype(int).tolist()

    augment_fn: Optional[Callable] = None
    if augment_images and split == "train":
        augment_fn = build_augmenter(enabled=True, seed=seed)

    ds = tf.data.Dataset.from_tensor_slices((paths, labels))
    if shuffle:
        ds = ds.shuffle(buffer_size=len(paths), seed=seed, reshuffle_each_iteration=True)

    def _map_fn(path, label):
        img = _load_image(path, input_size=input_size, image_mode=image_mode)
        img_for_features = img
        if augment_fn is not None:
            aug_img = augment_fn(img)
            img = aug_img
            if compute_from_augmented:
                img_for_features = aug_img
        focus_vec = compute_focus_vector(img_for_features, list(enabled_measures))
        return (img, focus_vec), tf.cast(label, tf.int32)

    ds = ds.map(_map_fn, num_parallel_calls=tf.data.AUTOTUNE)
    ds = ds.cache()
    ds = ds.batch(batch_size).prefetch(tf.data.AUTOTUNE)
    return ds
