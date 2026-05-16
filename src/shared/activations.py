import numpy as np
from typing import Callable


def linear(x):
    """Linear (identity) activation"""
    return x


def relu(x):
    """Rectified Linear Unit"""
    return np.maximum(0, x)


def leaky_relu(x, alpha=0.01):
    """Leaky Rectified Linear Unit"""
    return np.maximum(x, 0) + alpha * np.minimum(x, 0)


def elu(x, alpha=1.0):
    """Exponential Linear Unit"""
    return np.where(x > 0, x, alpha * (np.exp(x) - 1))


def sigmoid(x):
    """Sigmoid activation"""
    return np.where(
        x >= 0,
        1 / (1 + np.exp(-np.clip(x, -np.inf, 88.7))),
        np.exp(np.clip(x, -88.7, np.inf)) / (1 + np.exp(np.clip(x, -88.7, np.inf)))
    )


def tanh(x):
    """Hyperbolic tangent"""
    return np.tanh(x)


def softmax(x):
    """Numerically stable softmax for multi-class classification"""
    shifted = x - np.max(x, axis=-1, keepdims=True)
    exp_x = np.exp(shifted)
    return exp_x / np.sum(exp_x, axis=-1, keepdims=True)

ACTIVATIONS: dict[str, Callable] = {
    "linear":     linear,
    "identity":   linear,
    "none":       linear,
    "relu":       relu,
    "leaky_relu": leaky_relu,
    "elu":        elu,
    "sigmoid":    sigmoid,
    "tanh":       tanh,
    "softmax":    softmax,
}


def get_activation(name: str | Callable | None) -> Callable:
    if name is None:
        return linear
    if callable(name):
        return name
    key = name.lower()
    if key not in ACTIVATIONS:
        raise ValueError(
            f"Unknown activation: {name!r}. "
            f"Available: {sorted(ACTIVATIONS)}"
        )
    return ACTIVATIONS[key]
