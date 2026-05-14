from __future__ import annotations

from typing import List, Optional, Tuple

import numpy as np

from src.shared.activations import tanh


class SimpleRNNCell:
    """
    Single-step vanilla RNN cell.

    Parameters
    ----------
    W_x : np.ndarray, shape (input_dim, hidden_dim)
    W_h : np.ndarray, shape (hidden_dim, hidden_dim)
    b   : np.ndarray, shape (hidden_dim,)
    """

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
        """
        Compute one RNN timestep.

        Parameters
        ----------
        x_t    : np.ndarray, shape (input_dim,) or (batch, input_dim)
        h_prev : np.ndarray, shape (hidden_dim,) or (batch, hidden_dim)

        Returns
        -------
        h_t : np.ndarray — same leading shape as input, trailing (hidden_dim,)
        """
        return tanh(x_t @ self.W_x + h_prev @ self.W_h + self.b)

    @classmethod
    def from_keras_weights(
        cls,
        keras_weights: List[np.ndarray],
    ) -> "SimpleRNNCell":
        """
        Build from the list returned by keras_layer.get_weights().

        Parameters
        ----------
        keras_weights : list of 3 arrays — [kernel, recurrent_kernel, bias]
        """
        if len(keras_weights) != 3:
            raise ValueError(
                f"SimpleRNNCell expects 3 weight arrays, got {len(keras_weights)}."
            )
        W_x, W_h, b = keras_weights
        return cls(W_x, W_h, b)

    @classmethod
    def from_keras_layer(cls, keras_layer) -> "SimpleRNNCell":
        """Build from a tf.keras.layers.SimpleRNN layer object."""
        return cls.from_keras_weights(keras_layer.get_weights())

    def backward(
        self,
        x_t: np.ndarray,
        h_prev: np.ndarray,
        h_t: np.ndarray,
        grad_h: np.ndarray,
    ) -> tuple:
        """
        Backprop through one RNN timestep.

        Forward:  h_t = tanh(x_t @ W_x + h_prev @ W_h + b)

        Parameters
        ----------
        x_t    : np.ndarray, shape (..., input_dim)  — cached forward input
        h_prev : np.ndarray, shape (..., hidden_dim) — cached previous state
        h_t    : np.ndarray, shape (..., hidden_dim) — cached output (= tanh(z))
        grad_h : np.ndarray, shape (..., hidden_dim) — upstream gradient

        Returns
        -------
        grad_x      : np.ndarray, shape (..., input_dim)
        grad_h_prev : np.ndarray, shape (..., hidden_dim)
        grad_W_x    : np.ndarray, shape (input_dim,  hidden_dim)
        grad_W_h    : np.ndarray, shape (hidden_dim, hidden_dim)
        grad_b      : np.ndarray, shape (hidden_dim,)
        """
        # Derivative of tanh: d/dz tanh(z) = 1 - tanh(z)^2
        d_z = grad_h * (1.0 - h_t ** 2)  # (... , hidden_dim)

        # Parameter gradients — sum over batch / seq dims if present
        if d_z.ndim == 1:
            grad_W_x = np.outer(x_t, d_z)      # (input_dim,  hidden_dim)
            grad_W_h = np.outer(h_prev, d_z)   # (hidden_dim, hidden_dim)
        else:
            grad_W_x = x_t.T @ d_z             # (input_dim,  hidden_dim)
            grad_W_h = h_prev.T @ d_z           # (hidden_dim, hidden_dim)

        grad_b = d_z.sum(axis=tuple(range(d_z.ndim - 1))) if d_z.ndim > 1 else d_z

        # Input / state gradients
        grad_x = d_z @ self.W_x.T              # (..., input_dim)
        grad_h_prev = d_z @ self.W_h.T         # (..., hidden_dim)

        return grad_x, grad_h_prev, grad_W_x, grad_W_h, grad_b

    def __repr__(self) -> str:
        return (
            f"SimpleRNNCell(input_dim={self.input_dim}, "
            f"hidden_dim={self.hidden_dim})"
        )

class SimpleRNNLayer:
    """
    Multi-timestep, optionally stacked SimpleRNN layer.

    Wraps one or more SimpleRNNCell instances and iterates over the
    time dimension.

    Parameters
    ----------
    cells : list of SimpleRNNCell
        Pass a single-element list for a standard (non-deep) RNN.
    return_sequences : bool
        If True, return every hidden state (seq_len, hidden_dim).
        If False (default), return only the last hidden state
        (hidden_dim,).
    """

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
        """
        Run the full sequence through all stacked cells.

        Parameters
        ----------
        x_seq : np.ndarray
            Shape (seq_len, input_dim) — single sample.
            Batch mode (seq_len, input_dim) is used; batched
            (batch, seq_len, input_dim) is handled by transposing
            so the loop is over time.
        h0 : np.ndarray or None
            Initial hidden state.  Defaults to zeros matching the last
            cell's hidden_dim.

        Returns
        -------
        np.ndarray
            (seq_len, hidden_dim) if return_sequences else
            (hidden_dim,).
        """
        # Determine batch mode
        if x_seq.ndim == 3:
            # (batch, seq_len, input_dim)
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

        # Propagate through each stacked cell
        current_input = x_seq  # (seq_len, dim) or (batch, seq_len, dim)

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

            # Stack: (seq_len, batch, h_dim) → (batch, seq_len, h_dim) or
            #        (seq_len, h_dim)
            if batched:
                stacked = np.stack(step_outputs, axis=1)  # (batch, seq_len, h_dim)
            else:
                stacked = np.stack(step_outputs, axis=0)  # (seq_len, h_dim)

            current_input = stacked
            outputs = step_outputs  # keep for final hidden state

        if self.return_sequences:
            return stacked
        else:
            # Last timestep
            return outputs[-1]

    @classmethod
    def from_keras_layer(
        cls,
        keras_layer,
        return_sequences: Optional[bool] = None,
    ) -> "SimpleRNNLayer":
        """
        Build from a single tf.keras.layers.SimpleRNN layer.

        Parameters
        ----------
        keras_layer : Keras SimpleRNN layer
        return_sequences : bool or None
            Overrides the layer's return_sequences attribute if given.
        """
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
        """
        Build a stacked (deep) SimpleRNNLayer from multiple Keras layers.

        Parameters
        ----------
        keras_layers : list of Keras SimpleRNN layers
            Ordered from bottom (input-side) to top.
        return_sequences : bool or None
            If None, the last layer's return_sequences attribute is used.
        """
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
