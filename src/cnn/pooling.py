from __future__ import annotations

import numpy as np
from typing import Callable


#Shared forward helper

def _pool2d(
    x: np.ndarray,
    pool_size: tuple[int, int],
    strides: tuple[int, int],
    padding: str,
    reduce_fn: Callable,
) -> tuple[np.ndarray, np.ndarray, tuple[int,int,int,int]]:
    batched = x.ndim == 4
    if not batched:
        x = x[np.newaxis]
    N, H, W, C = x.shape
    pH, pW = pool_size
    sH, sW = strides

    if padding == "same":
        out_H = int(np.ceil(H / sH))
        out_W = int(np.ceil(W / sW))
        pad_H = max((out_H - 1) * sH + pH - H, 0)
        pad_W = max((out_W - 1) * sW + pW - W, 0)
        pad_top    = pad_H // 2
        pad_bottom = pad_H - pad_top
        pad_left   = pad_W // 2
        pad_right  = pad_W - pad_left
        fill = -np.inf if reduce_fn is np.max else 0.0
        x = np.pad(
            x,
            ((0, 0), (pad_top, pad_bottom), (pad_left, pad_right), (0, 0)),
            mode="constant",
            constant_values=fill,
        )
        N, H, W, C = x.shape
        pad_info = (pad_top, pad_bottom, pad_left, pad_right)
    else:
        out_H = (H - pH) // sH + 1
        out_W = (W - pW) // sW + 1
        pad_info = (0, 0, 0, 0)

    out = np.zeros((N, out_H, out_W, C), dtype=np.float32)
    for i in range(out_H):
        for j in range(out_W):
            patch = x[:, i*sH : i*sH+pH, j*sW : j*sW+pW, :]
            out[:, i, j, :] = reduce_fn(patch, axis=(1, 2))

    result = out[0] if not batched else out
    return result, x, pad_info


def _crop_pad(g: np.ndarray, Hp: int, Wp: int, pad_info: tuple) -> np.ndarray:
    pt, pb, pl, pr = pad_info
    return g[
        :,
        pt : Hp - pb if pb > 0 else Hp,
        pl : Wp - pr if pr > 0 else Wp,
        :,
    ]


#MaxPooling2DLayer

class MaxPooling2DLayer:
    def __init__(
        self,
        pool_size: tuple[int, int] = (2, 2),
        strides: tuple[int, int] | None = None,
        padding: str = "valid",
    ) -> None:
        self.pool_size = pool_size
        self.strides   = strides if strides is not None else pool_size
        self.padding   = padding.lower()
        self._cache: dict = {}

    @classmethod
    def from_keras_layer(cls, keras_layer) -> "MaxPooling2DLayer":
        return cls(
            pool_size=tuple(keras_layer.pool_size),
            strides=tuple(keras_layer.strides),
            padding=keras_layer.padding,
        )

    def forward(self, x: np.ndarray) -> np.ndarray:
        out, x_padded, pad_info = _pool2d(
            x, self.pool_size, self.strides, self.padding, np.max
        )
        batched = x.ndim == 4
        self._cache = {
            "x_padded": x_padded, "out": out,
            "batched": batched, "pad_info": pad_info,
        }
        return out

    def backward(self, grad_out: np.ndarray) -> np.ndarray:
        assert self._cache, "Call forward() before backward()."
        x_padded = self._cache["x_padded"]
        out      = self._cache["out"]
        batched  = self._cache["batched"]
        pad_info = self._cache["pad_info"]

        if grad_out.ndim == 3:
            grad_out = grad_out[np.newaxis]
        if x_padded.ndim == 3:
            x_padded = x_padded[np.newaxis]
        if isinstance(out, np.ndarray) and out.ndim == 3:
            out = out[np.newaxis]

        _, Hp, Wp, _ = x_padded.shape
        pH, pW = self.pool_size
        sH, sW = self.strides
        out_H, out_W = grad_out.shape[1], grad_out.shape[2]

        grad_x_padded = np.zeros_like(x_padded, dtype=np.float32)

        for i in range(out_H):
            for j in range(out_W):
                patch    = x_padded[:, i*sH:i*sH+pH, j*sW:j*sW+pW, :]  # (N,pH,pW,C)
                max_val  = out[:, i, j, :][:, None, None, :]             # (N,1,1,C)
                mask     = (patch == max_val).astype(np.float32)
                mask    /= (mask.sum(axis=(1, 2), keepdims=True) + 1e-8)
                g        = grad_out[:, i, j, :][:, None, None, :]        # (N,1,1,C)
                grad_x_padded[:, i*sH:i*sH+pH, j*sW:j*sW+pW, :] += mask * g

        grad_x = _crop_pad(grad_x_padded, Hp, Wp, pad_info)
        return grad_x[0] if not batched else grad_x

    def __call__(self, x: np.ndarray) -> np.ndarray:
        return self.forward(x)

    def __repr__(self) -> str:
        return (
            f"MaxPooling2DLayer(pool_size={self.pool_size}, "
            f"strides={self.strides}, padding={self.padding!r})"
        )


#AveragePooling2DLayer

class AveragePooling2DLayer:
    def __init__(
        self,
        pool_size: tuple[int, int] = (2, 2),
        strides: tuple[int, int] | None = None,
        padding: str = "valid",
    ) -> None:
        self.pool_size = pool_size
        self.strides   = strides if strides is not None else pool_size
        self.padding   = padding.lower()
        self._cache: dict = {}

    @classmethod
    def from_keras_layer(cls, keras_layer) -> "AveragePooling2DLayer":
        return cls(
            pool_size=tuple(keras_layer.pool_size),
            strides=tuple(keras_layer.strides),
            padding=keras_layer.padding,
        )

    def forward(self, x: np.ndarray) -> np.ndarray:
        out, x_padded, pad_info = _pool2d(
            x, self.pool_size, self.strides, self.padding, np.mean
        )
        batched = x.ndim == 4
        self._cache = {
            "x_padded": x_padded, "batched": batched, "pad_info": pad_info,
        }
        return out

    def backward(self, grad_out: np.ndarray) -> np.ndarray:
        assert self._cache, "Call forward() before backward()."
        x_padded = self._cache["x_padded"]
        batched  = self._cache["batched"]
        pad_info = self._cache["pad_info"]

        if grad_out.ndim == 3:
            grad_out = grad_out[np.newaxis]
        if x_padded.ndim == 3:
            x_padded = x_padded[np.newaxis]

        _, Hp, Wp, _ = x_padded.shape
        pH, pW = self.pool_size
        sH, sW = self.strides
        out_H, out_W = grad_out.shape[1], grad_out.shape[2]

        grad_x_padded = np.zeros_like(x_padded, dtype=np.float32)
        scale = 1.0 / (pH * pW)

        for i in range(out_H):
            for j in range(out_W):
                g = grad_out[:, i, j, :][:, None, None, :] * scale  # (N,1,1,C)
                grad_x_padded[:, i*sH:i*sH+pH, j*sW:j*sW+pW, :] += g

        grad_x = _crop_pad(grad_x_padded, Hp, Wp, pad_info)
        return grad_x[0] if not batched else grad_x

    def __call__(self, x: np.ndarray) -> np.ndarray:
        return self.forward(x)

    def __repr__(self) -> str:
        return (
            f"AveragePooling2DLayer(pool_size={self.pool_size}, "
            f"strides={self.strides}, padding={self.padding!r})"
        )


#GlobalAveragePooling2DLayer

class GlobalAveragePooling2DLayer:
    def __init__(self) -> None:
        self._cache: dict = {}

    def forward(self, x: np.ndarray) -> np.ndarray:
        batched = x.ndim == 4
        self._cache = {"x_orig": x, "batched": batched}
        if batched:
            return np.mean(x, axis=(1, 2))
        return np.mean(x, axis=(0, 1))

    def backward(self, grad_out: np.ndarray) -> np.ndarray:
        assert self._cache, "Call forward() before backward()."
        x_orig  = self._cache["x_orig"]
        batched = self._cache["batched"]

        if batched:
            N, H, W, C = x_orig.shape
            if grad_out.ndim == 1:
                grad_out = grad_out[np.newaxis]
            return np.broadcast_to(
                grad_out[:, np.newaxis, np.newaxis, :] / (H * W), (N, H, W, C)
            ).copy()
        else:
            H, W, C = x_orig.shape
            return np.broadcast_to(
                grad_out[np.newaxis, np.newaxis, :] / (H * W), (H, W, C)
            ).copy()

    @classmethod
    def from_keras_layer(cls, keras_layer) -> "GlobalAveragePooling2DLayer":
        return cls()

    def __call__(self, x: np.ndarray) -> np.ndarray:
        return self.forward(x)

    def __repr__(self) -> str:
        return "GlobalAveragePooling2DLayer()"

#GlobalMaxPooling2DLayer

class GlobalMaxPooling2DLayer:
    def __init__(self) -> None:
        self._cache: dict = {}

    def forward(self, x: np.ndarray) -> np.ndarray:
        batched = x.ndim == 4
        self._cache = {"x_orig": x, "batched": batched}
        if batched:
            return np.max(x, axis=(1, 2))
        return np.max(x, axis=(0, 1))

    def backward(self, grad_out: np.ndarray) -> np.ndarray:
        assert self._cache, "Call forward() before backward()."
        x_orig  = self._cache["x_orig"]
        batched = self._cache["batched"]

        if batched:
            if grad_out.ndim == 1:
                grad_out = grad_out[np.newaxis]
            max_val = np.max(x_orig, axis=(1, 2), keepdims=True)    # (N,1,1,C)
            mask    = (x_orig == max_val).astype(np.float32)
            mask   /= (mask.sum(axis=(1, 2), keepdims=True) + 1e-8)
            return mask * grad_out[:, np.newaxis, np.newaxis, :]
        else:
            max_val = np.max(x_orig, axis=(0, 1), keepdims=True)    # (1,1,C)
            mask    = (x_orig == max_val).astype(np.float32)
            mask   /= (mask.sum(axis=(0, 1), keepdims=True) + 1e-8)
            return mask * grad_out[np.newaxis, np.newaxis, :]

    @classmethod
    def from_keras_layer(cls, keras_layer) -> "GlobalMaxPooling2DLayer":
        return cls()

    def __call__(self, x: np.ndarray) -> np.ndarray:
        return self.forward(x)

    def __repr__(self) -> str:
        return "GlobalMaxPooling2DLayer()"
