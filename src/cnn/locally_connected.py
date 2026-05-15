from __future__ import annotations

import math
from typing import Callable

import numpy as np

from src.shared.activations import get_activation


def _act_grad(activation_fn, pre_act: np.ndarray, out: np.ndarray) -> np.ndarray:
    name = getattr(activation_fn, "__name__", "")
    if name == "relu":
        return (pre_act > 0).astype(np.float32)
    if name == "sigmoid":
        return (out * (1.0 - out)).astype(np.float32)
    if name == "tanh":
        return (1.0 - out ** 2).astype(np.float32)
    return np.ones_like(pre_act, dtype=np.float32)


class LocallyConnected2DLayer:
    def __init__(
        self,
        kernel: np.ndarray,
        bias: np.ndarray,
        out_rows: int,
        out_cols: int,
        kH: int,
        kW: int,
        strides: tuple[int, int] = (1, 1),
        padding: str = "valid",
        activation: str | Callable | None = None,
    ) -> None:
        self.kernel = kernel.astype(np.float32)   # (n_pos, kH*kW*C_in, C_out)
        self.bias   = bias.astype(np.float32)     # (n_pos, C_out)
        self.out_rows = out_rows
        self.out_cols = out_cols
        self.kH = kH
        self.kW = kW
        self.strides  = strides
        self.padding  = padding.lower()
        self.activation_fn = get_activation(activation)
        self._cache: dict = {}

    @classmethod
    def from_keras_layer(cls, keras_layer) -> "LocallyConnected2DLayer":
        weights = keras_layer.get_weights()
        raw_kernel = weights[0]
        raw_bias   = weights[1] if len(weights) > 1 else None

        kernel_size = keras_layer.kernel_size
        strides     = tuple(keras_layer.strides)
        filters     = keras_layer.filters
        padding     = getattr(keras_layer, "padding", "valid")
        act_name    = getattr(getattr(keras_layer, "activation", None), "__name__", None)

        out_rows = keras_layer.output_shape[1]
        out_cols = keras_layer.output_shape[2]
        kH, kW   = kernel_size
        n_pos    = out_rows * out_cols

        C_in   = raw_kernel.size // (n_pos * kH * kW * filters)
        kernel = raw_kernel.reshape(n_pos, kH * kW * C_in, filters)

        if raw_bias is not None:
            bias = raw_bias.reshape(n_pos, filters)
        else:
            bias = np.zeros((n_pos, filters), dtype=np.float32)

        return cls(kernel, bias, out_rows, out_cols, kH, kW, strides, padding, act_name)

    #Padding helpers

    def _pad_dims(self, H: int, W: int) -> tuple[int, int, int, int]:
        """Return (pad_top, pad_bottom, pad_left, pad_right) for same padding."""
        sH, sW = self.strides
        pad_H = max((self.out_rows - 1) * sH + self.kH - H, 0)
        pad_W = max((self.out_cols - 1) * sW + self.kW - W, 0)
        pad_top    = pad_H // 2
        pad_bottom = pad_H - pad_top
        pad_left   = pad_W // 2
        pad_right  = pad_W - pad_left
        return pad_top, pad_bottom, pad_left, pad_right

    def _pad_batch(self, x: np.ndarray) -> tuple[np.ndarray, tuple[int, int, int, int]]:
        H, W = x.shape[1], x.shape[2]
        pt, pb, pl, pr = self._pad_dims(H, W)
        return np.pad(x, ((0, 0), (pt, pb), (pl, pr), (0, 0)), mode="constant"), (pt, pb, pl, pr)

    #Forward

    def forward(self, x: np.ndarray) -> np.ndarray:
        batched = x.ndim == 4
        if not batched:
            x = x[np.newaxis]

        pad_info = (0, 0, 0, 0)
        if self.padding == "same":
            x, pad_info = self._pad_batch(x)

        N    = x.shape[0]
        sH, sW = self.strides
        C_out  = self.kernel.shape[2]

        pre_act = np.zeros((N, self.out_rows, self.out_cols, C_out), dtype=np.float32)
        for i in range(self.out_rows):
            for j in range(self.out_cols):
                pos   = i * self.out_cols + j
                patch = x[:, i*sH : i*sH+self.kH, j*sW : j*sW+self.kW, :]
                flat  = patch.reshape(N, -1)                   # (N, kH*kW*C_in)
                pre_act[:, i, j, :] = flat @ self.kernel[pos] + self.bias[pos]

        out = self.activation_fn(pre_act)
        self._cache = {
            "x_padded": x, "pre_act": pre_act, "out": out,
            "batched": batched, "pad_info": pad_info,
        }
        return out[0] if not batched else out

    #Backward

    def backward(
        self, grad_out: np.ndarray
    ) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
        assert self._cache, "Call forward() before backward()."
        x_padded = self._cache["x_padded"]   # (N, Hp, Wp, C_in)
        pre_act  = self._cache["pre_act"]    # (N, out_rows, out_cols, C_out)
        out      = self._cache["out"]
        batched  = self._cache["batched"]

        if grad_out.ndim == 3:
            grad_out = grad_out[np.newaxis]

        N, Hp, Wp, C_in = x_padded.shape
        C_out = self.kernel.shape[2]

        # Chain rule through activation
        act_d    = _act_grad(self.activation_fn, pre_act, out)  # (N, or, oc, C_out)
        grad_pre = grad_out * act_d                              # (N, or, oc, C_out)

        grad_kernel   = np.zeros_like(self.kernel)              # (n_pos, kH*kW*C_in, C_out)
        grad_bias     = np.zeros_like(self.bias)                # (n_pos, C_out)
        grad_x_padded = np.zeros_like(x_padded, dtype=np.float32)

        sH, sW = self.strides
        for i in range(self.out_rows):
            for j in range(self.out_cols):
                pos   = i * self.out_cols + j
                patch = x_padded[:, i*sH:i*sH+self.kH, j*sW:j*sW+self.kW, :]
                flat  = patch.reshape(N, -1)                    # (N, kH*kW*C_in)
                g     = grad_pre[:, i, j, :]                    # (N, C_out)

                grad_kernel[pos] += flat.T @ g                  # (kH*kW*C_in, C_out)
                grad_bias[pos]   += g.sum(axis=0)               # (C_out,)

                grad_flat = g @ self.kernel[pos].T              # (N, kH*kW*C_in)
                grad_x_padded[:, i*sH:i*sH+self.kH, j*sW:j*sW+self.kW, :] += (
                    grad_flat.reshape(N, self.kH, self.kW, C_in)
                )

        # Strip padding to recover grad_x in original input space
        pt, pb, pl, pr = self._cache["pad_info"]
        if pt > 0 or pb > 0 or pl > 0 or pr > 0:
            grad_x = grad_x_padded[
                :,
                pt : Hp - pb if pb > 0 else Hp,
                pl : Wp - pr if pr > 0 else Wp,
                :,
            ]
        else:
            grad_x = grad_x_padded

        if not batched:
            grad_x = grad_x[0]

        return grad_x, grad_kernel, grad_bias

    def __call__(self, x: np.ndarray) -> np.ndarray:
        return self.forward(x)

    def __repr__(self) -> str:
        return (
            f"LocallyConnected2DLayer("
            f"out=({self.out_rows},{self.out_cols}), "
            f"kernel=({self.kH},{self.kW}), "
            f"strides={self.strides}, padding={self.padding!r}, "
            f"activation={getattr(self.activation_fn, '__name__', repr(self.activation_fn))})"
        )
