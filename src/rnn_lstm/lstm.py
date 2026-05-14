from __future__ import annotations

from typing import List, Optional, Tuple

import numpy as np

from src.shared.activations import sigmoid, tanh

class LSTMCell:
    """
    Single-step LSTM cell.

    Parameters
    ----------
    W_x : np.ndarray, shape (input_dim, 4 * hidden_dim)
    W_h : np.ndarray, shape (hidden_dim, 4 * hidden_dim)
    b   : np.ndarray, shape (4 * hidden_dim,)
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
        self.hidden_dim: int = W_x.shape[1] // 4

    def forward(
        self,
        x_t: np.ndarray,
        h_prev: np.ndarray,
        c_prev: np.ndarray,
    ) -> Tuple[np.ndarray, np.ndarray]:
        """
        Compute one LSTM timestep.

        Parameters
        ----------
        x_t    : np.ndarray, shape (input_dim,) or (batch, input_dim)
        h_prev : np.ndarray, shape (hidden_dim,) or (batch, hidden_dim)
        c_prev : np.ndarray, shape (hidden_dim,) or (batch, hidden_dim)

        Returns
        -------
        (h_t, c_t) — same leading batch shape, trailing (hidden_dim,)
        """
        H = self.hidden_dim
        # Linear projection — (... , 4*H)
        z = x_t @ self.W_x + h_prev @ self.W_h + self.b

        i = sigmoid(z[..., 0 * H : 1 * H])   # input gate
        f = sigmoid(z[..., 1 * H : 2 * H])   # forget gate
        g = tanh(z[..., 2 * H : 3 * H])    # cell gate  (candidate)
        o = sigmoid(z[..., 3 * H : 4 * H])   # output gate

        c_t = f * c_prev + i * g
        h_t = o * tanh(c_t)

        return h_t, c_t

    @classmethod
    def from_keras_weights(
        cls,
        keras_weights: List[np.ndarray],
    ) -> "LSTMCell":
        """
        Build from the list returned by keras_layer.get_weights().

        Parameters
        ----------
        keras_weights : list of 3 arrays — [kernel, recurrent_kernel, bias]
        """
        if len(keras_weights) != 3:
            raise ValueError(
                f"LSTMCell expects 3 weight arrays, got {len(keras_weights)}."
            )
        W_x, W_h, b = keras_weights
        return cls(W_x, W_h, b)

    @classmethod
    def from_keras_layer(cls, keras_layer) -> "LSTMCell":
        """Build from a tf.keras.layers.LSTM layer object."""
        return cls.from_keras_weights(keras_layer.get_weights())

    def backward(
        self,
        x_t: np.ndarray,
        h_prev: np.ndarray,
        c_prev: np.ndarray,
        c_t: np.ndarray,
        h_t: np.ndarray,
        grad_h: np.ndarray,
        grad_c: np.ndarray,
    ) -> tuple:
        """
        Backprop through one LSTM timestep via chain rule through all gates.

        Keras gate ordering in the concatenated weight matrix: i, f, g, o.

        Parameters
        ----------
        x_t    : np.ndarray, (..., input_dim)   — cached forward input
        h_prev : np.ndarray, (..., hidden_dim)  — cached previous hidden state
        c_prev : np.ndarray, (..., hidden_dim)  — cached previous cell state
        c_t    : np.ndarray, (..., hidden_dim)  — cached cell state output
        h_t    : np.ndarray, (..., hidden_dim)  — cached hidden state output
        grad_h : np.ndarray, (..., hidden_dim)  — upstream grad w.r.t. h_t
        grad_c : np.ndarray, (..., hidden_dim)  — upstream grad w.r.t. c_t
                                                  (pass zeros if not needed)

        Returns
        -------
        grad_x      : np.ndarray, (..., input_dim)
        grad_h_prev : np.ndarray, (..., hidden_dim)
        grad_c_prev : np.ndarray, (..., hidden_dim)
        grad_W_x    : np.ndarray, (input_dim,  4*hidden_dim)
        grad_W_h    : np.ndarray, (hidden_dim, 4*hidden_dim)
        grad_b      : np.ndarray, (4*hidden_dim,)
        """
        H = self.hidden_dim

        # Re-compute gate pre-activations from cached values
        z = x_t @ self.W_x + h_prev @ self.W_h + self.b  # (..., 4H)
        i_pre = z[..., 0 * H : 1 * H]
        f_pre = z[..., 1 * H : 2 * H]
        g_pre = z[..., 2 * H : 3 * H]
        o_pre = z[..., 3 * H : 4 * H]

        # Gate activations (recomputed — avoids storing them)
        from src.shared.activations import sigmoid, tanh as _tanh
        i_val = sigmoid(i_pre)
        f_val = sigmoid(f_pre)
        g_val = _tanh(g_pre)
        o_val = sigmoid(o_pre)
        tanh_ct = _tanh(c_t)

        # Gradient through h_t = o * tanh(c_t)
        d_o   = grad_h * tanh_ct                           # (..., H)
        d_c_t = grad_h * o_val * (1.0 - tanh_ct ** 2)    # (..., H)

        # Accumulate gradient on c_t from the upstream cell-state path
        d_c_t = d_c_t + grad_c

        # Gradient through c_t = f*c_prev + i*g
        d_f      = d_c_t * c_prev
        d_i      = d_c_t * g_val
        d_g      = d_c_t * i_val
        grad_c_prev = d_c_t * f_val                        # (..., H)

        # Backprop through gate activations
        # sigmoid'(x) = s(x)*(1-s(x));  tanh'(x) = 1 - tanh(x)^2
        d_i_pre = d_i * (i_val * (1.0 - i_val))
        d_f_pre = d_f * (f_val * (1.0 - f_val))
        d_g_pre = d_g * (1.0 - g_val ** 2)
        d_o_pre = d_o * (o_val * (1.0 - o_val))

        # Concatenate in Keras order: i, f, g, o
        import numpy as _np
        d_z = _np.concatenate([d_i_pre, d_f_pre, d_g_pre, d_o_pre], axis=-1)  # (..., 4H)

        # Parameter gradients
        if d_z.ndim == 1:
            grad_W_x = _np.outer(x_t, d_z)      # (input_dim,  4H)
            grad_W_h = _np.outer(h_prev, d_z)   # (hidden_dim, 4H)
        else:
            grad_W_x = x_t.T @ d_z              # (input_dim,  4H)
            grad_W_h = h_prev.T @ d_z            # (hidden_dim, 4H)

        grad_b = d_z.sum(axis=tuple(range(d_z.ndim - 1))) if d_z.ndim > 1 else d_z

        # Input / previous-state gradients
        grad_x      = d_z @ self.W_x.T           # (..., input_dim)
        grad_h_prev = d_z @ self.W_h.T           # (..., hidden_dim)

        return grad_x, grad_h_prev, grad_c_prev, grad_W_x, grad_W_h, grad_b

    def __repr__(self) -> str:
        return (
            f"LSTMCell(input_dim={self.input_dim}, "
            f"hidden_dim={self.hidden_dim})"
        )

class LSTMLayer:
    """
    Multi-timestep, optionally stacked LSTM layer.

    Wraps one or more LSTMCell instances and iterates over the
    time dimension.

    Parameters
    ----------
    cells : list of LSTMCell
        Pass a single-element list for a standard (non-deep) LSTM.
    return_sequences : bool
        If True, return every hidden state (seq_len, hidden_dim).
        If False (default), return only the last hidden state
        (hidden_dim,).
    """

    def __init__(
        self,
        cells: List[LSTMCell],
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
        c0: Optional[np.ndarray] = None,
    ) -> np.ndarray:
        """
        Run the full sequence through all stacked cells.

        Parameters
        ----------
        x_seq : np.ndarray
            Shape (seq_len, input_dim) — single sample, or
            (batch, seq_len, input_dim) — batched.
        h0 : np.ndarray or None
            Initial hidden state.  Defaults to zeros.
        c0 : np.ndarray or None
            Initial cell state.  Defaults to zeros.

        Returns
        -------
        np.ndarray
            (seq_len, hidden_dim) if return_sequences else
            (hidden_dim,).
            In batched mode: (batch, seq_len, hidden_dim) or
            (batch, hidden_dim).
        """
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

        current_input = x_seq
        stacked = None
        step_outputs: list = []

        for cell in self.cells:
            H = cell.hidden_dim

            if batched:
                h = (
                    np.zeros((batch_size, H), dtype=np.float32)
                    if h0 is None
                    else h0.astype(np.float32)
                )
                c = (
                    np.zeros((batch_size, H), dtype=np.float32)
                    if c0 is None
                    else c0.astype(np.float32)
                )
            else:
                h = (
                    np.zeros(H, dtype=np.float32)
                    if h0 is None
                    else h0.astype(np.float32)
                )
                c = (
                    np.zeros(H, dtype=np.float32)
                    if c0 is None
                    else c0.astype(np.float32)
                )

            step_outputs = []
            for t in range(seq_len):
                x_t = current_input[:, t, :] if batched else current_input[t]
                h, c = cell.forward(x_t, h, c)
                step_outputs.append(h)

            if batched:
                stacked = np.stack(step_outputs, axis=1)  # (batch, seq_len, H)
            else:
                stacked = np.stack(step_outputs, axis=0)  # (seq_len, H)

            current_input = stacked

        if self.return_sequences:
            return stacked
        else:
            return step_outputs[-1]

    @classmethod
    def from_keras_layer(
        cls,
        keras_layer,
        return_sequences: Optional[bool] = None,
    ) -> "LSTMLayer":
        """
        Build from a single tf.keras.layers.LSTM layer.

        Parameters
        ----------
        keras_layer : Keras LSTM layer
        return_sequences : bool or None
            Overrides the layer's return_sequences attribute if given.
        """
        cell = LSTMCell.from_keras_layer(keras_layer)
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
    ) -> "LSTMLayer":
        """
        Build a stacked (deep) LSTMLayer from multiple Keras layers.

        Parameters
        ----------
        keras_layers : list of Keras LSTM layers
            Ordered from bottom (input-side) to top.
        return_sequences : bool or None
            If None, the last layer's return_sequences attribute is used.
        """
        cells = [LSTMCell.from_keras_layer(lyr) for lyr in keras_layers]
        rs = (
            return_sequences
            if return_sequences is not None
            else bool(keras_layers[-1].return_sequences)
        )
        return cls(cells, return_sequences=rs)

    def __repr__(self) -> str:
        return (
            f"LSTMLayer(num_cells={len(self.cells)}, "
            f"hidden_dim={self.hidden_dim}, "
            f"return_sequences={self.return_sequences})"
        )
