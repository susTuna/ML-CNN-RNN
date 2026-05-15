from __future__ import annotations

from typing import Callable

import numpy as np

from src.shared.activations import get_activation


class Conv2DLayer:
    """2D convolution with shared parameters (NumPy only).

    Loads kernel and bias from a trained Keras Conv2D layer and reproduces
    the same forward pass using sliding-window dot products.
    """

    def __init__(
        self,
        kernel: np.ndarray,
        bias: np.ndarray,
        strides: tuple[int, int] = (1, 1),
        padding: str = "valid",
        activation: str | Callable | None = None,
    ) -> None:
        self.kernel = kernel.astype(np.float32)  # (kH, kW, C_in, C_out)
        self.bias = bias.astype(np.float32)       # (C_out,)
        self.strides = strides
        self.padding = padding.lower()
        self.activation_fn = get_activation(activation)

        self.kH, self.kW, self.C_in, self.C_out = kernel.shape
        # Pre-reshape for efficient matmul: (kH*kW*C_in, C_out)
        self._kernel_mat = self.kernel.reshape(-1, self.C_out)

    @classmethod
    def from_keras_layer(cls, keras_layer) -> "Conv2DLayer":
        weights = keras_layer.get_weights()
        kernel = weights[0]
        bias = weights[1] if len(weights) > 1 else np.zeros(kernel.shape[-1], dtype=np.float32)
        strides = tuple(keras_layer.strides)
        padding = keras_layer.padding
        act_name = getattr(getattr(keras_layer, "activation", None), "__name__", None)
        return cls(kernel, bias, strides, padding, act_name)

    def forward(self, x: np.ndarray) -> np.ndarray:
        """Compute 2D convolution.

        Parameters
        ----------
        x : np.ndarray
            Shape ``(H, W, C_in)`` or ``(N, H, W, C_in)``.

        Returns
        -------
        np.ndarray
            Shape ``(out_H, out_W, C_out)`` or ``(N, out_H, out_W, C_out)``.
        """
        batched = x.ndim == 4
        if not batched:
            x = x[np.newaxis]  # (1, H, W, C_in)
        N, H, W, C = x.shape
        sH, sW = self.strides

        if self.padding == "same":
            out_H = int(np.ceil(H / sH))
            out_W = int(np.ceil(W / sW))
            pad_H = max((out_H - 1) * sH + self.kH - H, 0)
            pad_W = max((out_W - 1) * sW + self.kW - W, 0)
            pad_top = pad_H // 2
            pad_bottom = pad_H - pad_top
            pad_left = pad_W // 2
            pad_right = pad_W - pad_left
            x = np.pad(
                x,
                ((0, 0), (pad_top, pad_bottom), (pad_left, pad_right), (0, 0)),
                mode="constant",
            )
        else:  # 'valid'
            out_H = (H - self.kH) // sH + 1
            out_W = (W - self.kW) // sW + 1

        out = np.zeros((N, out_H, out_W, self.C_out), dtype=np.float32)
        for i in range(out_H):
            for j in range(out_W):
                patch = x[:, i * sH : i * sH + self.kH, j * sW : j * sW + self.kW, :]
                patch_flat = patch.reshape(N, -1)           # (N, kH*kW*C_in)
                out[:, i, j, :] = patch_flat @ self._kernel_mat + self.bias

        out = self.activation_fn(out)
        return out[0] if not batched else out

    def __call__(self, x: np.ndarray) -> np.ndarray:
        return self.forward(x)

    def __repr__(self) -> str:
        return (
            f"Conv2DLayer(kernel={self.kernel.shape}, "
            f"strides={self.strides}, padding={self.padding!r}, "
            f"activation={getattr(self.activation_fn, '__name__', repr(self.activation_fn))})"
        )
