from __future__ import annotations

import numpy as np
from PIL import Image


def get_intermediate_feature_maps(
    model,
    image: np.ndarray,
    layer_names: list[str],
) -> dict[str, np.ndarray]:
    import tensorflow as tf
    from tensorflow import keras

    outputs = [model.get_layer(name).output for name in layer_names]
    extractor = keras.Model(inputs=model.inputs, outputs=outputs)

    activations = extractor(image[np.newaxis], training=False)

    # When there's only one output, keras returns a tensor directly
    if len(layer_names) == 1:
        activations = [activations]

    return {name: act.numpy()[0] for name, act in zip(layer_names, activations)}


def grad_cam(
    model,
    image: np.ndarray,
    class_idx: int,
    last_conv_layer_name: str,
) -> np.ndarray:
    import tensorflow as tf
    from tensorflow import keras

    grad_model = keras.Model(
        inputs=model.inputs,
        outputs=[
            model.get_layer(last_conv_layer_name).output,
            model.output,
        ],
    )

    with tf.GradientTape() as tape:
        inputs = tf.cast(image[np.newaxis], tf.float32)
        conv_outputs, predictions = grad_model(inputs, training=False)
        loss = predictions[:, class_idx]

    # Gradients of class score w.r.t. conv feature maps: (1, fH, fW, C_out)
    grads = tape.gradient(loss, conv_outputs)
    # Pool gradients over spatial dims to importance weight per feature map
    pooled_grads = tf.reduce_mean(grads, axis=(0, 1, 2))  # (C_out,)

    conv_out = conv_outputs[0]                                    # (fH, fW, C_out)
    heatmap = conv_out @ pooled_grads[..., tf.newaxis]            # (fH, fW, 1)
    heatmap = tf.squeeze(heatmap).numpy()                         # (fH, fW)

    # ReLU
    heatmap = np.maximum(heatmap, 0)
    heatmap = heatmap / (heatmap.max() + 1e-8)

    # Upsample to original image spatial size using PIL (no cv2 dependency)
    H, W = image.shape[:2]
    heatmap_pil = Image.fromarray((heatmap * 255).astype(np.uint8))
    heatmap_resized = np.array(
        heatmap_pil.resize((W, H), Image.Resampling.BILINEAR), dtype=np.float32
    ) / 255.0

    # Expand to RGB and alpha-blend over original image
    heatmap_rgb = np.stack([heatmap_resized] * 3, axis=-1)
    overlay = 0.4 * heatmap_rgb + 0.6 * image.astype(np.float32)
    return np.clip(overlay, 0.0, 1.0)
