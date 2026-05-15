from __future__ import annotations

import numpy as np
from typing import Callable


def _pool2d(
    x: np.ndarray,
    pool_size: tuple[int, int],
    strides: tuple[int, int],
    padding: str,
    reduce_fn: Callable,
) -> np.ndarray:
    batched = x.ndim == 4
    if not batched:
        x = x[np.newaxis]  # (1, H, W, C)
    N, H, W, C = x.shape
    pH, pW = pool_size
    sH, sW = strides

    if padding == "same":
        out_H = int(np.ceil(H / sH))
        out_W = int(np.ceil(W / sW))
        pad_H = max((out_H - 1) * sH + pH - H, 0)
        pad_W = max((out_W - 1) * sW + pW - W, 0)
        pad_top = pad_H // 2
        pad_bottom = pad_H - pad_top
        pad_left = pad_W // 2
        pad_right = pad_W - pad_left
        fill = -np.inf if reduce_fn is np.max else 0.0
        x = np.pad(
            x,
            ((0, 0), (pad_top, pad_bottom), (pad_left, pad_right), (0, 0)),
            mode="constant",
            constant_values=fill,
        )
        N, H, W, C = x.shape
    else:
        out_H = (H - pH) // sH + 1
        out_W = (W - pW) // sW + 1

    out = np.zeros((N, out_H, out_W, C), dtype=np.float32)
    for i in range(out_H):
        for j in range(out_W):
            patch = x[:, i * sH : i * sH + pH, j * sW : j * sW + pW, :]
            out[:, i, j, :] = reduce_fn(patch, axis=(1, 2))

    return out[0] if not batched else out


class MaxPooling2DLayer:
    def __init__(
        self,
        pool_size: tuple[int, int] = (2, 2),
        strides: tuple[int, int] | None = None,
        padding: str = "valid",
    ) -> None:
        self.pool_size = pool_size
        self.strides = strides if strides is not None else pool_size
        self.padding = padding.lower()

    def forward(self, x: np.ndarray) -> np.ndarray:
        return _pool2d(x, self.pool_size, self.strides, self.padding, np.max)

    @classmethod
    def from_keras_layer(cls, keras_layer) -> "MaxPooling2DLayer":
        return cls(
            pool_size=tuple(keras_layer.pool_size),
            strides=tuple(keras_layer.strides),
            padding=keras_layer.padding,
        )

    def __call__(self, x: np.ndarray) -> np.ndarray:
        return self.forward(x)

    def __repr__(self) -> str:
        return (
            f"MaxPooling2DLayer(pool_size={self.pool_size}, "
            f"strides={self.strides}, padding={self.padding!r})"
        )


class AveragePooling2DLayer:
    def __init__(
        self,
        pool_size: tuple[int, int] = (2, 2),
        strides: tuple[int, int] | None = None,
        padding: str = "valid",
    ) -> None:
        self.pool_size = pool_size
        self.strides = strides if strides is not None else pool_size
        self.padding = padding.lower()

    def forward(self, x: np.ndarray) -> np.ndarray:
        return _pool2d(x, self.pool_size, self.strides, self.padding, np.mean)

    @classmethod
    def from_keras_layer(cls, keras_layer) -> "AveragePooling2DLayer":
        return cls(
            pool_size=tuple(keras_layer.pool_size),
            strides=tuple(keras_layer.strides),
            padding=keras_layer.padding,
        )

    def __call__(self, x: np.ndarray) -> np.ndarray:
        return self.forward(x)

    def __repr__(self) -> str:
        return (
            f"AveragePooling2DLayer(pool_size={self.pool_size}, "
            f"strides={self.strides}, padding={self.padding!r})"
        )


class GlobalAveragePooling2DLayer:
    def forward(self, x: np.ndarray) -> np.ndarray:
        if x.ndim == 4:
            return np.mean(x, axis=(1, 2))
        return np.mean(x, axis=(0, 1))

    @classmethod
    def from_keras_layer(cls, keras_layer) -> "GlobalAveragePooling2DLayer":
        return cls()

    def __call__(self, x: np.ndarray) -> np.ndarray:
        return self.forward(x)

    def __repr__(self) -> str:
        return "GlobalAveragePooling2DLayer()"


class GlobalMaxPooling2DLayer:
    def forward(self, x: np.ndarray) -> np.ndarray:
        if x.ndim == 4:
            return np.max(x, axis=(1, 2))
        return np.max(x, axis=(0, 1))

    @classmethod
    def from_keras_layer(cls, keras_layer) -> "GlobalMaxPooling2DLayer":
        return cls()

    def __call__(self, x: np.ndarray) -> np.ndarray:
        return self.forward(x)

    def __repr__(self) -> str:
        return "GlobalMaxPooling2DLayer()"
