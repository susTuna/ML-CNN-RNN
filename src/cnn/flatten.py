from __future__ import annotations

import numpy as np


class FlattenLayer:
    def __init__(self) -> None:
        self._cache: dict = {}

    def forward(self, x: np.ndarray) -> np.ndarray:
        self._cache = {"input_shape": x.shape}
        if x.ndim == 4:
            return x.reshape(x.shape[0], -1)
        return x.flatten(order="C")

    def backward(self, grad_out: np.ndarray) -> np.ndarray:
        assert self._cache, "Call forward() before backward()."
        return grad_out.reshape(self._cache["input_shape"])

    @classmethod
    def from_keras_layer(cls, keras_layer) -> "FlattenLayer":
        return cls()

    def __call__(self, x: np.ndarray) -> np.ndarray:
        return self.forward(x)

    def __repr__(self) -> str:
        return "FlattenLayer()"
