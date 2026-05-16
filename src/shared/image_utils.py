from __future__ import annotations

from pathlib import Path
from typing import Iterable

import numpy as np
from PIL import Image


def load_image(path: str | Path, target_size: tuple[int, int]) -> np.ndarray:
    image_path = Path(path)
    with Image.open(image_path) as image:
        image = image.convert("RGB")
        image = image.resize((target_size[1], target_size[0]), Image.Resampling.BILINEAR)
        array = np.asarray(image, dtype=np.float32) / 255.0
    return array


def load_batch(paths: Iterable[str | Path], target_size: tuple[int, int]) -> np.ndarray:
    images = [load_image(path, target_size) for path in paths]
    if not images:
        return np.empty((0, target_size[0], target_size[1], 3), dtype=np.float32)
    return np.stack(images, axis=0)


def _infer_target_size(keras_encoder) -> tuple[int, int]:
    input_shape = getattr(keras_encoder, "input_shape", None)
    if input_shape is None and hasattr(keras_encoder, "inputs"):
        inputs = keras_encoder.inputs
        if inputs:
            input_shape = getattr(inputs[0], "shape", None)

    if not input_shape:
        raise ValueError("Unable to infer target_size from keras_encoder; pass an encoder with a defined input shape.")

    if isinstance(input_shape, (list, tuple)) and input_shape and isinstance(input_shape[0], (list, tuple)):
        input_shape = input_shape[0]

    if len(input_shape) < 3:
        raise ValueError(f"Unsupported encoder input shape: {input_shape}")

    height = input_shape[-3]
    width = input_shape[-2]
    if height is None or width is None:
        raise ValueError(f"Encoder input shape must define height and width: {input_shape}")

    return int(height), int(width)


def extract_and_save_features(
    paths,
    keras_encoder,
    out_path: str | Path,
    index_map_path: str | Path | None = None,
    skip_if_exists: bool = True,
    batch_size: int = 32,
    progress: bool = True,
) -> np.ndarray:
    """Extract CNN features with a frozen Keras encoder and persist them.

    Parameters
    ----------
    paths:
        Ordered iterable of image file paths.
    keras_encoder:
        Frozen Keras model used as feature extractor.
    out_path:
        Destination ``.npy`` file for the feature array ``(N, feature_dim)``.
    index_map_path:
        Optional path for a ``{filename: index}`` JSON mapping.
    skip_if_exists:
        When *True* (default) and ``out_path`` already exists, load and return
        the cached array instead of re-running extraction.
    batch_size:
        Number of images per forward pass to keep memory bounded.
    progress:
        When *True* (default), print periodic progress updates.
    """
    import json

    out_path = Path(out_path)
    paths = list(paths)

    if skip_if_exists and out_path.exists():
        return np.load(out_path)

    target_size = _infer_target_size(keras_encoder)

    has_predict = hasattr(keras_encoder, "predict")
    all_features = []
    n = len(paths)
    for start in range(0, n, batch_size):
        chunk = paths[start : start + batch_size]
        batch = load_batch(chunk, target_size)
        if has_predict:
            feats = np.asarray(keras_encoder.predict(batch, verbose=0), dtype=np.float32)
        else:
            feats = np.asarray(keras_encoder(batch), dtype=np.float32)
        all_features.append(feats)
        if progress and (start // batch_size) % 10 == 0:
            print(f"  feature extraction: {min(start + batch_size, n)}/{n}", flush=True)

    features = np.concatenate(all_features, axis=0) if all_features else np.empty((0, 0), dtype=np.float32)

    out_path.parent.mkdir(parents=True, exist_ok=True)
    np.save(out_path, features)

    if index_map_path is not None:
        index_map = {Path(p).name: idx for idx, p in enumerate(paths)}
        index_map_path = Path(index_map_path)
        index_map_path.parent.mkdir(parents=True, exist_ok=True)
        with open(index_map_path, "w") as f:
            json.dump(index_map, f, indent=2)

    return features