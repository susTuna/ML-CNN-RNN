import numpy as np
from typing import Callable

from .activations import get_activation


class DenseLayer:
    def __init__(
        self,
        input_size: int,
        output_size: int,
        activation: str | Callable | None = None,
    ):
        self.weights = np.random.randn(input_size, output_size) * 0.01
        self.biases = np.zeros((1, output_size))

        # Dependency injection: resolve once, call directly
        self.activation_fn: Callable = get_activation(activation)

    @classmethod
    def from_keras_layer(cls, keras_layer) -> "DenseLayer":
        """Build a DenseLayer from a Keras Dense layer, copying its weights."""
        weights = keras_layer.get_weights()
        kernel = weights[0]
        bias   = weights[1] if len(weights) > 1 else np.zeros(kernel.shape[1])
        act_name = getattr(getattr(keras_layer, "activation", None), "__name__", None)
        layer = cls(kernel.shape[0], kernel.shape[1], activation=act_name)
        layer.weights = kernel.copy()
        layer.biases  = bias.reshape(1, -1).copy()
        return layer

    def load_keras_weights(self, weights: np.ndarray, biases: np.ndarray) -> None:
        if self.weights.shape != weights.shape:
            raise ValueError(
                f"Expected weights shape {self.weights.shape}, got {weights.shape}"
            )

        biases = np.array(biases)
        if biases.ndim == 1:
            biases = biases.reshape(1, -1)

        if self.biases.shape != biases.shape:
            raise ValueError(
                f"Expected biases shape {self.biases.shape}, got {biases.shape}"
            )

        self.weights = weights.copy()
        self.biases  = biases.copy()

    def forward(self, x: np.ndarray) -> np.ndarray:
        z = np.dot(x, self.weights) + self.biases
        return self.activation_fn(z)

    def __call__(self, x: np.ndarray) -> np.ndarray:
        return self.forward(x)

    def __repr__(self) -> str:
        return (
            f"DenseLayer(input={self.weights.shape[0]}, "
            f"output={self.weights.shape[1]}, "
            f"activation={getattr(self.activation_fn, '__name__', repr(self.activation_fn))})"
        )
