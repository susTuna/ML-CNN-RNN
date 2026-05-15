from __future__ import annotations

from typing import Callable

import numpy as np

from src.shared.activations import get_activation


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
        self.kernel = kernel.astype(np.float32)
        self.bias = bias.astype(np.float32)
        self.out_rows = out_rows
        self.out_cols = out_cols
        self.kH = kH
        self.kW = kW
        self.strides = strides
        self.padding = padding.lower()
        self.activation_fn = get_activation(activation)

    @classmethod
    def from_keras_layer(cls, keras_layer) -> "LocallyConnected2DLayer":
        weights = keras_layer.get_weights()
        raw_kernel = weights[0]
        raw_bias = weights[1] if len(weights) > 1 else None

        kernel_size = keras_layer.kernel_size  # (kH, kW)
        strides = tuple(keras_layer.strides)
        filters = keras_layer.filters
        padding = getattr(keras_layer, "padding", "valid")
        act_name = getattr(getattr(keras_layer, "activation", None), "__name__", None)

        # output_shape: (batch, out_rows, out_cols, filters)
        out_rows = keras_layer.output_shape[1]
        out_cols = keras_layer.output_shape[2]
        kH, kW = kernel_size
        n_pos = out_rows * out_cols

        C_in = raw_kernel.size // (n_pos * kH * kW * filters)
        kernel = raw_kernel.reshape(n_pos, kH * kW * C_in, filters)

        if raw_bias is not None:
            bias = raw_bias.reshape(n_pos, filters)
        else:
            bias = np.zeros((n_pos, filters), dtype=np.float32)

        return cls(kernel, bias, out_rows, out_cols, kH, kW, strides, padding, act_name)

    def _pad_input(self, x: np.ndarray) -> np.ndarray:
        H, W, _ = x.shape
        sH, sW = self.strides
        pad_H = max((self.out_rows - 1) * sH + self.kH - H, 0)
        pad_W = max((self.out_cols - 1) * sW + self.kW - W, 0)
        pad_top = pad_H // 2
        pad_bottom = pad_H - pad_top
        pad_left = pad_W // 2
        pad_right = pad_W - pad_left
        return np.pad(
            x,
            ((pad_top, pad_bottom), (pad_left, pad_right), (0, 0)),
            mode="constant",
        )

    def forward(self, x: np.ndarray) -> np.ndarray:
        if self.padding == "same":
            x = self._pad_input(x)

        sH, sW = self.strides
        C_out = self.kernel.shape[2]

        out = np.zeros((self.out_rows, self.out_cols, C_out), dtype=np.float32)
        for i in range(self.out_rows):
            for j in range(self.out_cols):
                pos = i * self.out_cols + j
                patch = x[i * sH : i * sH + self.kH, j * sW : j * sW + self.kW, :]
                flat = patch.flatten(order="C")           # (kH*kW*C_in,)
                out[i, j, :] = flat @ self.kernel[pos] + self.bias[pos]

        return self.activation_fn(out)

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
