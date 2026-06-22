from __future__ import annotations

import tensorflow as tf


_LAPLACIAN_KERNEL = tf.constant(
    [[0.0, 1.0, 0.0], [1.0, -4.0, 1.0], [0.0, 1.0, 0.0]],
    dtype=tf.float32,
)


def _to_grayscale(img: tf.Tensor) -> tf.Tensor:
    img = tf.convert_to_tensor(img, dtype=tf.float32)
    rank = tf.rank(img)

    def _rank2():
        return img[..., None]

    def _rank3():
        channels = tf.shape(img)[-1]
        return tf.cond(
            tf.equal(channels, 1),
            lambda: img,
            lambda: tf.image.rgb_to_grayscale(img[..., :3]),
        )

    def _rank4():
        channels = tf.shape(img)[-1]
        return tf.cond(
            tf.equal(channels, 1),
            lambda: img,
            lambda: tf.image.rgb_to_grayscale(img[..., :3]),
        )

    return tf.case(
        [
            (tf.equal(rank, 2), _rank2),
            (tf.equal(rank, 3), _rank3),
            (tf.equal(rank, 4), _rank4),
        ],
        default=lambda: img,
    )


def _ensure_rank4(img: tf.Tensor) -> tf.Tensor:
    rank = tf.rank(img)
    return tf.cond(tf.equal(rank, 3), lambda: img[None, ...], lambda: img)


def variance_of_laplacian(img: tf.Tensor) -> tf.Tensor:
    gray = _to_grayscale(img)
    kernel = _LAPLACIAN_KERNEL[:, :, None, None]
    lap = tf.nn.conv2d(_ensure_rank4(gray), kernel, strides=1, padding="SAME")
    return tf.math.reduce_variance(lap)


def tenengrad(img: tf.Tensor) -> tf.Tensor:
    gray = _to_grayscale(img)
    sobel = tf.image.sobel_edges(_ensure_rank4(gray))
    gx = sobel[..., 0]
    gy = sobel[..., 1]
    energy = tf.reduce_mean(tf.square(gx) + tf.square(gy))
    return energy


def brenner(img: tf.Tensor) -> tf.Tensor:
    gray = _to_grayscale(img)
    shift_x = gray[:, 2:, :] - gray[:, :-2, :]
    shift_y = gray[2:, :, :] - gray[:-2, :, :]
    return tf.reduce_mean(tf.square(shift_x)) + tf.reduce_mean(tf.square(shift_y))


def sml(img: tf.Tensor) -> tf.Tensor:
    gray = _to_grayscale(img)
    kernel = _LAPLACIAN_KERNEL[:, :, None, None]
    lap = tf.nn.conv2d(_ensure_rank4(gray), kernel, strides=1, padding="SAME")
    return tf.reduce_mean(tf.abs(lap))
