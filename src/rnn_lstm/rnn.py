from __future__ import annotations

from typing import List, Optional, Tuple

import numpy as np

from src.shared.activations import tanh


class SimpleRNNCell:

    def __init__(
        self,
        W_x: np.ndarray,
        W_h: np.ndarray,
        b: np.ndarray,
    ) -> None:
        self.W_x = W_x.astype(np.float32)
        self.W_h = W_h.astype(np.float32)
        self.b = b.astype(np.float32)
        self.input_dim: int = W_x.shape[0]
        self.hidden_dim: int = W_x.shape[1]

    def forward(self, x_t: np.ndarray, h_prev: np.ndarray) -> np.ndarray:
        return tanh(x_t @ self.W_x + h_prev @ self.W_h + self.b)

    @classmethod
    def from_keras_weights(
        cls,
        keras_weights: List[np.ndarray],
    ) -> "SimpleRNNCell":
        if len(keras_weights) != 3:
            raise ValueError(
                f"SimpleRNNCell expects 3 weight arrays, got {len(keras_weights)}."
            )
        W_x, W_h, b = keras_weights
        return cls(W_x, W_h, b)

    @classmethod
    def from_keras_layer(cls, keras_layer) -> "SimpleRNNCell":
        return cls.from_keras_weights(keras_layer.get_weights())

    def backward(
        self,
        x_t: np.ndarray,
        h_prev: np.ndarray,
        h_t: np.ndarray,
        grad_h: np.ndarray,
    ) -> tuple:
        # Derivative of tanh: d/dz tanh(z) = 1 - tanh(z)^2
        d_z = grad_h * (1.0 - h_t ** 2)

        if d_z.ndim == 1:
            grad_W_x = np.outer(x_t, d_z)
            grad_W_h = np.outer(h_prev, d_z)
        else:
            grad_W_x = x_t.T @ d_z
            grad_W_h = h_prev.T @ d_z

        grad_b = d_z.sum(axis=tuple(range(d_z.ndim - 1))) if d_z.ndim > 1 else d_z

        grad_x = d_z @ self.W_x.T
        grad_h_prev = d_z @ self.W_h.T

        return grad_x, grad_h_prev, grad_W_x, grad_W_h, grad_b

    def __repr__(self) -> str:
        return (
            f"SimpleRNNCell(input_dim={self.input_dim}, "
            f"hidden_dim={self.hidden_dim})"
        )

class SimpleRNNLayer:

    def __init__(
        self,
        cells: List[SimpleRNNCell],
        return_sequences: bool = False,
    ) -> None:
        if not cells:
            raise ValueError("cells list must not be empty.")
        self.cells = cells
        self.return_sequences = return_sequences
        self.hidden_dim: int = cells[-1].hidden_dim

    def forward(
        self,
        x_seq: np.ndarray,
        h0: Optional[np.ndarray] = None,
    ) -> np.ndarray:
        if x_seq.ndim == 3:
            batch_size, seq_len, _ = x_seq.shape
            batched = True
        elif x_seq.ndim == 2:
            seq_len, _ = x_seq.shape
            batch_size = None
            batched = False
        else:
            raise ValueError(
                f"x_seq must be 2-D (seq_len, input_dim) or "
                f"3-D (batch, seq_len, input_dim), got shape {x_seq.shape}."
            )

        outputs = []

        current_input = x_seq
        for cell in self.cells:
            h_dim = cell.hidden_dim
            if batched:
                h = (
                    np.zeros((batch_size, h_dim), dtype=np.float32)
                    if h0 is None
                    else h0.astype(np.float32)
                )
            else:
                h = (
                    np.zeros(h_dim, dtype=np.float32)
                    if h0 is None
                    else h0.astype(np.float32)
                )

            step_outputs = []
            for t in range(seq_len):
                x_t = current_input[:, t, :] if batched else current_input[t]
                h = cell.forward(x_t, h)
                step_outputs.append(h)

            if batched:
                stacked = np.stack(step_outputs, axis=1)
            else:
                stacked = np.stack(step_outputs, axis=0)

            current_input = stacked
            outputs = step_outputs

        if self.return_sequences:
            return stacked
        else:
            return outputs[-1]

    @classmethod
    def from_keras_layer(
        cls,
        keras_layer,
        return_sequences: Optional[bool] = None,
    ) -> "SimpleRNNLayer":
        cell = SimpleRNNCell.from_keras_layer(keras_layer)
        rs = (
            return_sequences
            if return_sequences is not None
            else bool(keras_layer.return_sequences)
        )
        return cls([cell], return_sequences=rs)

    @classmethod
    def from_keras_layers(
        cls,
        keras_layers: list,
        return_sequences: Optional[bool] = None,
    ) -> "SimpleRNNLayer":
        cells = [SimpleRNNCell.from_keras_layer(lyr) for lyr in keras_layers]
        rs = (
            return_sequences
            if return_sequences is not None
            else bool(keras_layers[-1].return_sequences)
        )
        return cls(cells, return_sequences=rs)

    def __repr__(self) -> str:
        return (
            f"SimpleRNNLayer(num_cells={len(self.cells)}, "
            f"hidden_dim={self.hidden_dim}, "
            f"return_sequences={self.return_sequences})"
        )
