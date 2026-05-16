from __future__ import annotations

import math
import warnings
from typing import List

import numpy as np

from .conv2d import Conv2DLayer
from .flatten import FlattenLayer
from .locally_connected import LocallyConnected2DLayer
from .pooling import (
    AveragePooling2DLayer,
    GlobalAveragePooling2DLayer,
    GlobalMaxPooling2DLayer,
    MaxPooling2DLayer,
)
from src.shared.dense_layer import DenseLayer

_LAYER_REGISTRY: dict[str, object] = {
    "Conv2D": Conv2DLayer.from_keras_layer,
    "LocallyConnected2D": LocallyConnected2DLayer.from_keras_layer,
    "MaxPooling2D": MaxPooling2DLayer.from_keras_layer,
    "AveragePooling2D": AveragePooling2DLayer.from_keras_layer,
    "GlobalAveragePooling2D": GlobalAveragePooling2DLayer.from_keras_layer,
    "GlobalMaxPooling2D": GlobalMaxPooling2DLayer.from_keras_layer,
    "Flatten": FlattenLayer.from_keras_layer,
    "Dense": DenseLayer.from_keras_layer,
}

_SKIP_CLASSES = frozenset({"InputLayer", "Dropout", "BatchNormalization"})


class CNNScratchModel:
    def __init__(self) -> None:
        self.layers: List = []

    @classmethod
    def load_from_keras(cls, keras_model) -> "CNNScratchModel":
        model = cls()
        for keras_layer in keras_model.layers:
            class_name = type(keras_layer).__name__
            if class_name in _SKIP_CLASSES:
                continue
            factory = _LAYER_REGISTRY.get(class_name)
            if factory is None:
                warnings.warn(
                    f"CNNScratchModel: unknown layer type '{class_name}', skipping. "
                    "Add it to _LAYER_REGISTRY if needed.",
                    stacklevel=2,
                )
                continue
            model.layers.append(factory(keras_layer))
        return model

    def forward(self, x: np.ndarray) -> np.ndarray:
        for layer in self.layers:
            x = layer.forward(x)
        return x

    def as_locally_connected(self, input_shape: tuple[int, int, int] | None = None) -> "CNNScratchModel":
        new_model = CNNScratchModel()
        current_shape: tuple[int, int] | None = (
            (input_shape[0], input_shape[1]) if input_shape is not None else None
        )

        for layer in self.layers:
            if not isinstance(layer, Conv2DLayer):
                # Track spatial shape through non-conv layers for static build
                if current_shape is not None and isinstance(
                    layer, (MaxPooling2DLayer, AveragePooling2DLayer)
                ):
                    H, W = current_shape
                    sH, sW = layer.strides
                    pH, pW = layer.pool_size
                    if layer.padding == "same":
                        current_shape = (math.ceil(H / sH), math.ceil(W / sW))
                    else:
                        current_shape = ((H - pH) // sH + 1, (W - pW) // sW + 1)
                new_model.layers.append(layer)
                continue

            kH, kW, C_in, C_out = layer.kernel.shape

            if current_shape is not None:
                # Build layer immediately with known spatial dims
                H, W = current_shape
                lc = _expand_conv_to_lc(layer, H, W)
                sH, sW = layer.strides
                if layer.padding == "same":
                    current_shape = (math.ceil(H / sH), math.ceil(W / sW))
                else:
                    current_shape = ((H - kH) // sH + 1, (W - kW) // sW + 1)
                new_model.layers.append(lc)
            else:
                # Build lazily - spatial dims determined at first forward call
                new_model.layers.append(
                    _LazyLocallyConnectedLayer(
                        conv_layer=layer,
                        kH=kH, kW=kW, C_in=C_in, C_out=C_out,
                        strides=layer.strides,
                        padding=layer.padding,
                        activation_fn=layer.activation_fn,
                    )
                )
        return new_model

    def count_parameters(self) -> int:
        total = 0
        for layer in self.layers:
            if isinstance(layer, Conv2DLayer):
                total += layer.kernel.size + layer.bias.size
            elif isinstance(layer, LocallyConnected2DLayer):
                total += layer.kernel.size + layer.bias.size
            elif isinstance(layer, DenseLayer):
                total += layer.weights.size + layer.biases.size
        return total

    def __call__(self, x: np.ndarray) -> np.ndarray:
        return self.forward(x)

    def __repr__(self) -> str:
        layer_strs = "\n  ".join(repr(l) for l in self.layers)
        return f"CNNScratchModel(\n  {layer_strs}\n)"


def _expand_conv_to_lc(conv: Conv2DLayer, H: int, W: int) -> LocallyConnected2DLayer:
    kH, kW, C_in, C_out = conv.kernel.shape
    sH, sW = conv.strides

    if conv.padding == "same":
        out_rows = math.ceil(H / sH)
        out_cols = math.ceil(W / sW)
    else:
        out_rows = (H - kH) // sH + 1
        out_cols = (W - kW) // sW + 1

    n_pos = out_rows * out_cols
    # Broadcast the shared kernel to every position: (kH*kW*C_in, C_out) → (n_pos, ...)
    flat_kernel = conv.kernel.reshape(kH * kW * C_in, C_out)
    kernel_expanded = np.broadcast_to(flat_kernel, (n_pos, kH * kW * C_in, C_out)).copy()
    # Broadcast bias to every position
    bias_expanded = np.broadcast_to(conv.bias, (n_pos, C_out)).copy()

    lc = LocallyConnected2DLayer(
        kernel_expanded, bias_expanded,
        out_rows, out_cols, kH, kW,
        conv.strides, conv.padding, None,
    )
    lc.activation_fn = conv.activation_fn
    return lc


class _LazyLocallyConnectedLayer:
    def __init__(self, conv_layer, kH, kW, C_in, C_out, strides, padding, activation_fn):
        self._conv = conv_layer
        self.kH = kH
        self.kW = kW
        self.C_in = C_in
        self.C_out = C_out
        self.strides = strides
        self.padding = padding
        self.activation_fn = activation_fn
        self._built: LocallyConnected2DLayer | None = None

    def forward(self, x: np.ndarray) -> np.ndarray:
        H = x.shape[-3]
        W = x.shape[-2]
        if self._built is None:
            self._built = _expand_conv_to_lc(self._conv, H, W)
        return self._built.forward(x)

    def __call__(self, x: np.ndarray) -> np.ndarray:
        return self.forward(x)

    def __repr__(self) -> str:
        if self._built is not None:
            return repr(self._built)
        return (
            f"_LazyLocallyConnectedLayer(kernel=({self.kH},{self.kW}), "
            f"strides={self.strides}, padding={self.padding!r}, unbuilt)"
        )
