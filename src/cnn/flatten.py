from __future__ import annotations

import numpy as np


class FlattenLayer:
    def forward(self, x: np.ndarray) -> np.ndarray:
        if x.ndim == 4:
            return x.reshape(x.shape[0], -1)
        return x.flatten(order="C")

    @classmethod
    def from_keras_layer(cls, keras_layer) -> "FlattenLayer":
        return cls()

    def __call__(self, x: np.ndarray) -> np.ndarray:
        return self.forward(x)

    def __repr__(self) -> str:
        return "FlattenLayer()"
