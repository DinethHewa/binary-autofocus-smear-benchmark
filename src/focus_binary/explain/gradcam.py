from __future__ import annotations

from typing import Optional, Sequence

import numpy as np

try:
    import tensorflow as tf
except Exception:  # pragma: no cover
    tf = None

try:
    from PIL import Image, ImageOps
except Exception:  # pragma: no cover
    Image = None
    ImageOps = None


def _require_tf():
    if tf is None:
        raise ImportError("TensorFlow is required for Grad-CAM.")


def find_last_conv_layer(model) -> Optional[str]:
    """Return the name of the last Conv2D-like layer."""
    _require_tf()
    for layer in reversed(model.layers):
        if isinstance(layer, (tf.keras.layers.Conv2D, tf.keras.layers.DepthwiseConv2D)):
            return layer.name
    return None


def gradcam_heatmap(
    model,
    image_tensor: "tf.Tensor",
    class_index: Optional[int] = None,
    layer_name: Optional[str] = None,
    extra_inputs: Optional[Sequence["tf.Tensor"]] = None,
) -> np.ndarray:
    """Compute a normalized Grad-CAM heatmap for a single image tensor."""
    _require_tf()

    if layer_name is None:
        layer_name = find_last_conv_layer(model)
        if layer_name is None:
            raise ValueError("No Conv2D layer found for Grad-CAM.")

    target_layer = model.get_layer(layer_name)
    grad_model = tf.keras.Model([model.inputs], [target_layer.output, model.output])

    image_tensor = tf.convert_to_tensor(image_tensor, dtype=tf.float32)
    if image_tensor.ndim == 3:
        image_tensor = tf.expand_dims(image_tensor, axis=0)

    model_inputs = image_tensor
    if extra_inputs is not None:
        batched = []
        for extra in extra_inputs:
            extra_tensor = tf.convert_to_tensor(extra, dtype=tf.float32)
            if extra_tensor.ndim == 1:
                extra_tensor = tf.expand_dims(extra_tensor, axis=0)
            batched.append(extra_tensor)
        model_inputs = [image_tensor] + batched

    with tf.GradientTape() as tape:
        conv_outputs, preds = grad_model(model_inputs)
        if preds.shape[-1] == 1:
            target = preds[:, 0]
        else:
            if class_index is None:
                class_index = int(tf.argmax(preds, axis=-1)[0])
            target = preds[:, class_index]

    grads = tape.gradient(target, conv_outputs)
    pooled = tf.reduce_mean(grads, axis=(1, 2))
    heatmap = tf.reduce_sum(conv_outputs * pooled[:, None, None, :], axis=-1)
    heatmap = tf.nn.relu(heatmap)
    max_val = tf.reduce_max(heatmap, axis=(1, 2), keepdims=True)
    heatmap = tf.where(max_val > 0, heatmap / max_val, heatmap)
    return heatmap[0].numpy()


def overlay_heatmap(original_image, heatmap: np.ndarray, alpha: float = 0.4) -> np.ndarray:
    """Overlay a heatmap on top of an image using PIL."""
    if Image is None or ImageOps is None:
        raise ImportError("Pillow is required for heatmap overlays.")

    image = np.asarray(original_image)
    if image.ndim == 2:
        image = np.stack([image] * 3, axis=-1)
    if image.dtype != np.uint8:
        image = np.clip(image * 255.0 if image.max() <= 1.0 else image, 0, 255).astype(np.uint8)

    heat = np.asarray(heatmap)
    heat = np.clip(heat, 0.0, 1.0)
    heat = (heat * 255.0).astype(np.uint8)

    heat_img = Image.fromarray(heat, mode="L").resize((image.shape[1], image.shape[0]))
    heat_color = ImageOps.colorize(heat_img, black="black", white="red")
    base = Image.fromarray(image, mode="RGB")
    overlay = Image.blend(base, heat_color, alpha=alpha)
    return np.asarray(overlay)


def compute_gradcam(
    model,
    images: np.ndarray | "tf.Tensor",
    layer_name: Optional[str] = None,
    class_index: Optional[int] = None,
    normalize: bool = True,
    extra_inputs: Optional[Sequence["tf.Tensor"]] = None,
) -> np.ndarray:
    """Compute Grad-CAM heatmap for a batch of images."""
    _require_tf()

    if layer_name is None:
        layer_name = find_last_conv_layer(model)
        if layer_name is None:
            raise ValueError("No Conv2D layer found for Grad-CAM.")

    target_layer = model.get_layer(layer_name)
    grad_model = tf.keras.Model([model.inputs], [target_layer.output, model.output])

    images = tf.convert_to_tensor(images, dtype=tf.float32)
    if images.ndim == 3:
        images = tf.expand_dims(images, axis=0)

    model_inputs = images
    if extra_inputs is not None:
        batched = []
        for extra in extra_inputs:
            extra_tensor = tf.convert_to_tensor(extra, dtype=tf.float32)
            if extra_tensor.ndim == 1:
                extra_tensor = tf.expand_dims(extra_tensor, axis=0)
            batched.append(extra_tensor)
        model_inputs = [images] + batched

    with tf.GradientTape() as tape:
        conv_outputs, preds = grad_model(model_inputs)
        if preds.shape[-1] == 1:
            target = preds[:, 0]
        else:
            if class_index is None:
                class_index = tf.argmax(preds, axis=-1)
            target = tf.gather(preds, class_index, axis=1, batch_dims=1)

    grads = tape.gradient(target, conv_outputs)
    pooled = tf.reduce_mean(grads, axis=(1, 2))
    heatmap = tf.reduce_sum(conv_outputs * pooled[:, None, None, :], axis=-1)
    heatmap = tf.nn.relu(heatmap)

    if normalize:
        max_val = tf.reduce_max(heatmap, axis=(1, 2), keepdims=True)
        heatmap = tf.where(max_val > 0, heatmap / max_val, heatmap)

    return heatmap.numpy()
