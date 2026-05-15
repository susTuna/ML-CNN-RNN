from __future__ import annotations

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
    # linear / softmax: return ones (softmax gradient handled externally via loss)
    return np.ones_like(pre_act, dtype=np.float32)


class Conv2DLayer:
    def __init__(
        self,
        kernel: np.ndarray,
        bias: np.ndarray,
        strides: tuple[int, int] = (1, 1),
        padding: str = "valid",
        activation: str | Callable | None = None,
    ) -> None:
        self.kernel = kernel.astype(np.float32)   # (kH, kW, C_in, C_out)
        self.bias   = bias.astype(np.float32)     # (C_out,)
        self.strides = strides
        self.padding = padding.lower()
        self.activation_fn = get_activation(activation)

        self.kH, self.kW, self.C_in, self.C_out = kernel.shape
        self._kernel_mat = self.kernel.reshape(-1, self.C_out)  # (kH*kW*C_in, C_out)
        self._cache: dict = {}

    @classmethod
    def from_keras_layer(cls, keras_layer) -> "Conv2DLayer":
        weights = keras_layer.get_weights()
        kernel  = weights[0]
        bias    = weights[1] if len(weights) > 1 else np.zeros(kernel.shape[-1], dtype=np.float32)
        strides = tuple(keras_layer.strides)
        padding = keras_layer.padding
        act_name = getattr(getattr(keras_layer, "activation", None), "__name__", None)
        return cls(kernel, bias, strides, padding, act_name)

    #Forward
    def forward(self, x: np.ndarray) -> np.ndarray:
        batched = x.ndim == 4
        if not batched:
            x = x[np.newaxis]
        N, H, W, C = x.shape
        sH, sW = self.strides

        if self.padding == "same":
            out_H = int(np.ceil(H / sH))
            out_W = int(np.ceil(W / sW))
            pad_H = max((out_H - 1) * sH + self.kH - H, 0)
            pad_W = max((out_W - 1) * sW + self.kW - W, 0)
            pad_top    = pad_H // 2
            pad_bottom = pad_H - pad_top
            pad_left   = pad_W // 2
            pad_right  = pad_W - pad_left
            x = np.pad(x, ((0,0),(pad_top,pad_bottom),(pad_left,pad_right),(0,0)))
            pad_info = (pad_top, pad_bottom, pad_left, pad_right)
        else:
            out_H = (H - self.kH) // sH + 1
            out_W = (W - self.kW) // sW + 1
            pad_info = (0, 0, 0, 0)

        pre_act = np.zeros((N, out_H, out_W, self.C_out), dtype=np.float32)
        for i in range(out_H):
            for j in range(out_W):
                patch = x[:, i*sH : i*sH+self.kH, j*sW : j*sW+self.kW, :]
                pre_act[:, i, j, :] = patch.reshape(N, -1) @ self._kernel_mat + self.bias

        out = self.activation_fn(pre_act)
        self._cache = {
            "x_padded": x, "pre_act": pre_act, "out": out,
            "batched": batched, "pad_info": pad_info, "orig_HW": (H, W),
        }
        return out[0] if not batched else out

    #Backward

    def backward(
        self, grad_out: np.ndarray
    ) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
        assert self._cache, "Call forward() before backward()."
        x_padded = self._cache["x_padded"]
        pre_act  = self._cache["pre_act"]
        out      = self._cache["out"]
        batched  = self._cache["batched"]
        pad_info = self._cache["pad_info"]

        if grad_out.ndim == 3:
            grad_out = grad_out[np.newaxis]

        N, Hp, Wp, _ = x_padded.shape
        sH, sW = self.strides
        _, out_H, out_W, _ = grad_out.shape

        # Chain rule through activation: dL/d(pre_act) = dL/d(out) * act'(pre_act)
        grad_pre = grad_out * _act_grad(self.activation_fn, pre_act, out)

        grad_bias     = grad_pre.sum(axis=(0, 1, 2))            # (C_out,)
        grad_kernel   = np.zeros_like(self.kernel)              # (kH, kW, C_in, C_out)
        grad_x_padded = np.zeros_like(x_padded, dtype=np.float32)

        for i in range(out_H):
            for j in range(out_W):
                patch = x_padded[:, i*sH:i*sH+self.kH, j*sW:j*sW+self.kW, :]
                flat  = patch.reshape(N, -1)                    # (N, kH*kW*C_in)
                g     = grad_pre[:, i, j, :]                    # (N, C_out)

                # Accumulate kernel gradient
                grad_kernel += (flat.T @ g).reshape(self.kH, self.kW, self.C_in, self.C_out)

                # Scatter input gradient back through patch
                grad_flat = g @ self._kernel_mat.T              # (N, kH*kW*C_in)
                grad_x_padded[:, i*sH:i*sH+self.kH, j*sW:j*sW+self.kW, :] += (
                    grad_flat.reshape(N, self.kH, self.kW, self.C_in)
                )

        pt, pb, pl, pr = pad_info
        grad_x = grad_x_padded[
            :,
            pt : Hp - pb if pb > 0 else Hp,
            pl : Wp - pr if pr > 0 else Wp,
            :,
        ]

        if not batched:
            grad_x = grad_x[0]

        return grad_x, grad_kernel, grad_bias

    def __call__(self, x: np.ndarray) -> np.ndarray:
        return self.forward(x)

    def __repr__(self) -> str:
        return (
            f"Conv2DLayer(kernel={self.kernel.shape}, "
            f"strides={self.strides}, padding={self.padding!r}, "
            f"activation={getattr(self.activation_fn, '__name__', repr(self.activation_fn))})"
        )
