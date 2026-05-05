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
        """Initialize a Dense layer.

        Parameters
        ----------
        input_size:
            Number of input features.
        output_size:
            Number of output units.
        activation:
            Activation function — accepts:
            - A string name (``'relu'``, ``'softmax'``, ``'sigmoid'``, …)
            - A callable (dependency injection — pass any function directly)
            - ``None`` or ``'linear'`` for identity (no activation)

        The activation is resolved once at construction via ``get_activation``
        and stored as a plain callable, so ``forward`` never branches.
        """
        self.weights = np.random.randn(input_size, output_size) * 0.01
        self.biases = np.zeros((1, output_size))

        # Dependency injection: resolve once, call directly — no if/else
        self.activation_fn: Callable = get_activation(activation)

    @classmethod
    def from_keras_layer(cls, keras_layer) -> "DenseLayer":
        """Build a DenseLayer from a Keras Dense layer, copying its weights."""
        weights = keras_layer.get_weights()
        kernel = weights[0]                          # (input_size, output_size)
        bias   = weights[1] if len(weights) > 1 else np.zeros(kernel.shape[1])
        act_name = getattr(getattr(keras_layer, "activation", None), "__name__", None)
        layer = cls(kernel.shape[0], kernel.shape[1], activation=act_name)
        layer.weights = kernel.copy()
        layer.biases  = bias.reshape(1, -1).copy()
        return layer

    def load_keras_weights(self, weights: np.ndarray, biases: np.ndarray) -> None:
        """Load weights from a Keras layer (kernel + bias arrays).

        Parameters
        ----------
        weights:
            Shape ``(input_size, output_size)``.
        biases:
            Shape ``(output_size,)`` or ``(1, output_size)``.
        """
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
        """Forward pass: ``activation(x @ W + b)``.

        Parameters
        ----------
        x:
            Shape ``(batch_size, input_size)`` or ``(input_size,)``.
        """
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
