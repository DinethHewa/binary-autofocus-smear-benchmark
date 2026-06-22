from __future__ import annotations

import math
from typing import Optional

try:
    import tensorflow as tf
except Exception:  # pragma: no cover
    tf = None


def _require_tf():
    if tf is None:
        raise ImportError("TensorFlow is required for perturbations.")


def _clip(image):
    return tf.clip_by_value(image, 0.0, 1.0)


def gaussian_noise(image, stddev: float = 0.05, clip: bool = True):
    _require_tf()
    noise = tf.random.normal(tf.shape(image), stddev=stddev, dtype=image.dtype)
    out = image + noise
    return _clip(out) if clip else out


def random_erasing(
    image,
    erase_fraction: float = 0.1,
    fill_value: float = 0.0,
    seed: Optional[int] = None,
):
    _require_tf()
    rng = tf.random.Generator.from_seed(seed or 0)
    h = tf.shape(image)[0]
    w = tf.shape(image)[1]
    c = tf.shape(image)[-1]
    erase_h = tf.maximum(1, tf.cast(tf.cast(h, tf.float32) * erase_fraction, tf.int32))
    erase_w = tf.maximum(1, tf.cast(tf.cast(w, tf.float32) * erase_fraction, tf.int32))
    y0 = rng.uniform([], minval=0, maxval=h - erase_h + 1, dtype=tf.int32)
    x0 = rng.uniform([], minval=0, maxval=w - erase_w + 1, dtype=tf.int32)
    paddings = [[y0, h - y0 - erase_h], [x0, w - x0 - erase_w], [0, 0]]
    patch = tf.ones((erase_h, erase_w, c), dtype=image.dtype) * fill_value
    patch = tf.pad(patch, paddings)
    mask = tf.pad(tf.ones((erase_h, erase_w, c), dtype=image.dtype), paddings)
    return image * (1.0 - mask) + patch


def _gaussian_kernel1d(sigma: float, radius: int) -> tf.Tensor:
    x = tf.range(-radius, radius + 1, dtype=tf.float32)
    g = tf.exp(-(x ** 2) / (2 * sigma ** 2))
    return g / tf.reduce_sum(g)


def _depthwise_separable_blur(image: tf.Tensor, sigma: float, radius: int) -> tf.Tensor:
    kernel_1d = _gaussian_kernel1d(sigma, radius)
    channels = tf.shape(image)[-1]
    kernel_x = tf.reshape(kernel_1d, [2 * radius + 1, 1, 1, 1])
    kernel_y = tf.reshape(kernel_1d, [1, 2 * radius + 1, 1, 1])
    kernel_x = tf.tile(kernel_x, [1, 1, channels, 1])
    kernel_y = tf.tile(kernel_y, [1, 1, channels, 1])

    x = tf.expand_dims(image, axis=0)
    x = tf.nn.depthwise_conv2d(x, kernel_x, strides=[1, 1, 1, 1], padding="SAME")
    x = tf.nn.depthwise_conv2d(x, kernel_y, strides=[1, 1, 1, 1], padding="SAME")
    return tf.squeeze(x, axis=0)


def gaussian_blur(image, kernel_size: int = 3, sigma: float = 1.0):
    _require_tf()
    radius = max(1, kernel_size // 2)
    return _clip(_depthwise_separable_blur(image, sigma=float(sigma), radius=radius))


def slight_blur(image, sigma: float = 0.5):
    _require_tf()
    radius = max(1, int(math.ceil(3.0 * sigma)))
    return _clip(_depthwise_separable_blur(image, sigma=float(sigma), radius=radius))


def _ensure_jpeg_compatible(image: tf.Tensor) -> tf.Tensor:
    channels = tf.shape(image)[-1]

    def _to_rgb():
        return tf.image.grayscale_to_rgb(image)

    def _identity():
        return image

    def _first_three():
        return image[..., :3]

    return tf.case(
        [
            (tf.equal(channels, 1), _to_rgb),
            (tf.equal(channels, 3), _identity),
        ],
        default=_first_three,
        exclusive=True,
    )


def jpeg_compression(image, quality: int = 95):
    _require_tf()
    image = tf.clip_by_value(image, 0.0, 1.0)
    image = _ensure_jpeg_compatible(image)
    image_uint8 = tf.cast(tf.round(image * 255.0), tf.uint8)
    encoded = tf.io.encode_jpeg(image_uint8, quality=int(quality))
    decoded = tf.io.decode_jpeg(encoded, channels=3)
    return tf.image.convert_image_dtype(decoded, tf.float32)


def brightness(image, delta: float = 0.05):
    _require_tf()
    out = tf.image.adjust_brightness(image, delta=delta)
    return _clip(out)


def contrast(image, factor: float = 0.8):
    _require_tf()
    out = tf.image.adjust_contrast(image, contrast_factor=factor)
    return _clip(out)


def feature_gaussian_noise(vector: tf.Tensor, stddev: float = 0.03) -> tf.Tensor:
    _require_tf()
    noise = tf.random.normal(tf.shape(vector), stddev=stddev, dtype=vector.dtype)
    return vector + noise


def feature_dropout(vector: tf.Tensor, p: float = 0.1) -> tf.Tensor:
    _require_tf()
    keep_prob = 1.0 - p
    mask = tf.cast(tf.random.uniform(tf.shape(vector)) < keep_prob, vector.dtype)
    return vector * mask


def apply_perturbation(ds, kind: str, **kwargs):
    """Apply a perturbation to a tf.data dataset of (image, label)."""
    _require_tf()

    kind = kind.lower()
    if kind == "gaussian_noise":
        fn = lambda img: gaussian_noise(img, **kwargs)
    elif kind == "jpeg_compression":
        fn = lambda img: jpeg_compression(img, **kwargs)
    elif kind == "brightness":
        fn = lambda img: brightness(img, **kwargs)
    elif kind == "contrast":
        fn = lambda img: contrast(img, **kwargs)
    elif kind == "slight_blur":
        fn = lambda img: slight_blur(img, **kwargs)
    elif kind == "gaussian_blur":
        fn = lambda img: gaussian_blur(img, **kwargs)
    elif kind == "random_erasing":
        fn = lambda img: random_erasing(img, **kwargs)
    else:
        raise ValueError(f"Unknown perturbation: {kind}")

    def _map(inputs, label):
        if isinstance(inputs, (tuple, list)):
            img = inputs[0]
            rest = inputs[1:]
            if len(rest) == 1:
                return (fn(img), rest[0]), label
            return (fn(img), *rest), label
        return fn(inputs), label

    return ds.map(_map, num_parallel_calls=tf.data.AUTOTUNE)
