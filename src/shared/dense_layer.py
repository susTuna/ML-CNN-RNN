from __future__ import annotations

from pathlib import Path
from typing import Callable, Iterable

import numpy as np

from .activations import get_activation


class DenseLayer:
    def __init__(
        self,
        input_dim: int,
        output_dim: int,
        activation: str | Callable[[np.ndarray], np.ndarray] | None = "linear",
        weights: np.ndarray | None = None,
        bias: np.ndarray | None = None,
    ) -> None:
        self.input_dim = int(input_dim)
        self.output_dim = int(output_dim)
        self.weights = self._validate_weights(weights)
        self.bias = self._validate_bias(bias)
        self.activation = get_activation(activation) if isinstance(activation, str) or activation is None else activation

    def _validate_weights(self, weights: np.ndarray | None) -> np.ndarray:
        if weights is None:
            return np.zeros((self.input_dim, self.output_dim), dtype=np.float32)

        array = np.asarray(weights, dtype=np.float32)
        if array.shape != (self.input_dim, self.output_dim):
            raise ValueError(f"Expected weights shape {(self.input_dim, self.output_dim)}, got {array.shape}")
        return array

    def _validate_bias(self, bias: np.ndarray | None) -> np.ndarray:
        if bias is None:
            return np.zeros((self.output_dim,), dtype=np.float32)

        array = np.asarray(bias, dtype=np.float32)
        if array.shape not in {(self.output_dim,), (1, self.output_dim)}:
            raise ValueError(f"Expected bias shape {(self.output_dim,)} or {(1, self.output_dim)}, got {array.shape}")
        return array.reshape(self.output_dim)

    def load_weights(self, weights: np.ndarray, bias: np.ndarray | None = None) -> None:
        self.weights = self._validate_weights(weights)
        if bias is not None:
            self.bias = self._validate_bias(bias)

    def load_keras_weights(self, keras_weights: Iterable[np.ndarray]) -> None:
        keras_weights = list(keras_weights)
        if not keras_weights:
            raise ValueError("keras_weights must contain at least the kernel matrix")

        kernel = np.asarray(keras_weights[0], dtype=np.float32)
        if kernel.shape != (self.input_dim, self.output_dim):
            raise ValueError(f"Expected kernel shape {(self.input_dim, self.output_dim)}, got {kernel.shape}")

        self.weights = kernel
        if len(keras_weights) > 1:
            self.bias = self._validate_bias(keras_weights[1])

    @classmethod
    def from_keras_layer(cls, keras_layer, activation: str | Callable[[np.ndarray], np.ndarray] | None = None):
        weights = keras_layer.get_weights()
        if not weights:
            raise ValueError("keras_layer does not have weights")

        kernel = np.asarray(weights[0], dtype=np.float32)
        bias = np.asarray(weights[1], dtype=np.float32) if len(weights) > 1 else None
        input_dim, output_dim = kernel.shape
        layer_activation = activation if activation is not None else getattr(keras_layer, "activation", None)
        if callable(layer_activation) and activation is None:
            layer_activation = getattr(layer_activation, "__name__", None)
        return cls(input_dim=input_dim, output_dim=output_dim, activation=layer_activation, weights=kernel, bias=bias)

    @classmethod
    def from_weights_file(
        cls,
        input_dim: int,
        output_dim: int,
        weights_path: str | Path,
        bias_path: str | Path | None = None,
        activation: str | Callable[[np.ndarray], np.ndarray] | None = "linear",
    ):
        weights = np.load(weights_path)
        bias = np.load(bias_path) if bias_path is not None else None
        return cls(input_dim=input_dim, output_dim=output_dim, activation=activation, weights=weights, bias=bias)

    def forward(self, x: np.ndarray) -> np.ndarray:
        array = np.asarray(x, dtype=np.float32)
        output = array @ self.weights + self.bias
        return self.activation(output) if self.activation is not None else output

    def __call__(self, x: np.ndarray) -> np.ndarray:
        return self.forward(x)

    def __repr__(self) -> str:
        return (
            f"DenseLayer(input_dim={self.input_dim}, output_dim={self.output_dim}, "
            f"activation={getattr(self.activation, '__name__', repr(self.activation))})"
        )