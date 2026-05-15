from __future__ import annotations

import numpy as np


class FlattenLayer:
    """Row-major flatten, equivalent to Keras Flatten layer."""

    def forward(self, x: np.ndarray) -> np.ndarray:
        """Flatten spatial dimensions to 1D.

        Parameters
        ----------
        x : np.ndarray
            Shape ``(H, W, C)`` for a single image or ``(N, H, W, C)`` for a batch.

        Returns
        -------
        np.ndarray
            Shape ``(H*W*C,)`` or ``(N, H*W*C)``.
        """
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
