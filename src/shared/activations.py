from __future__ import annotations

from typing import Callable

import numpy as np


def relu(x: np.ndarray) -> np.ndarray:
    return np.maximum(0.0, x)


def sigmoid(x: np.ndarray) -> np.ndarray:
    return 1.0 / (1.0 + np.exp(-x))


def tanh(x: np.ndarray) -> np.ndarray:
    return np.tanh(x)


def softmax(x: np.ndarray) -> np.ndarray:
    x = np.asarray(x, dtype=np.float64)
    shifted = x - np.max(x, axis=-1, keepdims=True)
    exp_x = np.exp(shifted)
    return exp_x / np.sum(exp_x, axis=-1, keepdims=True)


ACTIVATIONS: dict[str, Callable[[np.ndarray], np.ndarray]] = {
    "relu": relu,
    "sigmoid": sigmoid,
    "tanh": tanh,
    "softmax": softmax,
    "linear": lambda x: x,
    "identity": lambda x: x,
    "none": lambda x: x,
}


def get_activation(name: str | None) -> Callable[[np.ndarray], np.ndarray]:
    if name is None:
        return ACTIVATIONS["linear"]

    key = name.lower()
    if key not in ACTIVATIONS:
        raise ValueError(f"Unknown activation: {name}. Available: {sorted(ACTIVATIONS)}")
    return ACTIVATIONS[key]