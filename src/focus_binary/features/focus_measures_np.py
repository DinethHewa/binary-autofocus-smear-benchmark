from __future__ import annotations

import numpy as np


def _to_grayscale(img: np.ndarray) -> np.ndarray:
    img = np.asarray(img, dtype=np.float32)
    if img.ndim == 2:
        return img
    if img.ndim == 3 and img.shape[-1] >= 3:
        r, g, b = img[..., 0], img[..., 1], img[..., 2]
        return 0.2989 * r + 0.5870 * g + 0.1140 * b
    if img.ndim == 3 and img.shape[-1] == 1:
        return img[..., 0]
    raise ValueError("Unsupported image shape for grayscale conversion")


def _laplacian(gray: np.ndarray) -> np.ndarray:
    gray = np.asarray(gray, dtype=np.float32)
    padded = np.pad(gray, 1, mode="edge")
    center = padded[1:-1, 1:-1]
    up = padded[:-2, 1:-1]
    down = padded[2:, 1:-1]
    left = padded[1:-1, :-2]
    right = padded[1:-1, 2:]
    return -4.0 * center + up + down + left + right


def variance_of_laplacian(img: np.ndarray) -> float:
    gray = _to_grayscale(img)
    lap = _laplacian(gray)
    return float(np.var(lap))


def tenengrad(img: np.ndarray) -> float:
    gray = _to_grayscale(img)
    padded = np.pad(gray, 1, mode="edge")
    gx = (
        padded[:-2, 2:] + 2 * padded[1:-1, 2:] + padded[2:, 2:]
        - padded[:-2, :-2] - 2 * padded[1:-1, :-2] - padded[2:, :-2]
    )
    gy = (
        padded[2:, :-2] + 2 * padded[2:, 1:-1] + padded[2:, 2:]
        - padded[:-2, :-2] - 2 * padded[:-2, 1:-1] - padded[:-2, 2:]
    )
    energy = np.mean(gx ** 2 + gy ** 2)
    return float(energy)


def brenner(img: np.ndarray) -> float:
    gray = _to_grayscale(img)
    dx = gray[:, 2:] - gray[:, :-2]
    dy = gray[2:, :] - gray[:-2, :]
    return float(np.mean(dx ** 2) + np.mean(dy ** 2))


def sml(img: np.ndarray) -> float:
    gray = _to_grayscale(img)
    lap = _laplacian(gray)
    return float(np.mean(np.abs(lap)))
