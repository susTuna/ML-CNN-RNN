from __future__ import annotations

from typing import List, Optional, Tuple

import numpy as np

from src.shared.activations import sigmoid, tanh

class LSTMCell:
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
        H = self.hidden_dim
        z = x_t @ self.W_x + h_prev @ self.W_h + self.b

        i = sigmoid(z[..., 0 * H : 1 * H])
        f = sigmoid(z[..., 1 * H : 2 * H])
        g = tanh(z[..., 2 * H : 3 * H])
        o = sigmoid(z[..., 3 * H : 4 * H])

        c_t = f * c_prev + i * g
        h_t = o * tanh(c_t)

        return h_t, c_t

    @classmethod
    def from_keras_weights(
        cls,
        keras_weights: List[np.ndarray],
    ) -> "LSTMCell":
        if len(keras_weights) != 3:
            raise ValueError(
                f"LSTMCell expects 3 weight arrays, got {len(keras_weights)}."
            )
        W_x, W_h, b = keras_weights
        return cls(W_x, W_h, b)

    @classmethod
    def from_keras_layer(cls, keras_layer) -> "LSTMCell":
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
        H = self.hidden_dim

        z = x_t @ self.W_x + h_prev @ self.W_h + self.b
        i_pre = z[..., 0 * H : 1 * H]
        f_pre = z[..., 1 * H : 2 * H]
        g_pre = z[..., 2 * H : 3 * H]
        o_pre = z[..., 3 * H : 4 * H]

        from src.shared.activations import sigmoid, tanh as _tanh
        i_val = sigmoid(i_pre)
        f_val = sigmoid(f_pre)
        g_val = _tanh(g_pre)
        o_val = sigmoid(o_pre)
        tanh_ct = _tanh(c_t)

        d_o   = grad_h * tanh_ct
        d_c_t = grad_h * o_val * (1.0 - tanh_ct ** 2)

        d_c_t = d_c_t + grad_c

        # Gradient through c_t = f*c_prev + i*g
        d_f      = d_c_t * c_prev
        d_i      = d_c_t * g_val
        d_g      = d_c_t * i_val
        grad_c_prev = d_c_t * f_val

        # Backprop
        # sigmoid'(x) = s(x)*(1-s(x));  tanh'(x) = 1 - tanh(x)^2
        d_i_pre = d_i * (i_val * (1.0 - i_val))
        d_f_pre = d_f * (f_val * (1.0 - f_val))
        d_g_pre = d_g * (1.0 - g_val ** 2)
        d_o_pre = d_o * (o_val * (1.0 - o_val))

        import numpy as _np
        d_z = _np.concatenate([d_i_pre, d_f_pre, d_g_pre, d_o_pre], axis=-1)  # (..., 4H)

        if d_z.ndim == 1:
            grad_W_x = _np.outer(x_t, d_z)
            grad_W_h = _np.outer(h_prev, d_z)
        else:
            grad_W_x = x_t.T @ d_z
            grad_W_h = h_prev.T @ d_z

        grad_b = d_z.sum(axis=tuple(range(d_z.ndim - 1))) if d_z.ndim > 1 else d_z

        grad_x      = d_z @ self.W_x.T
        grad_h_prev = d_z @ self.W_h.T

        return grad_x, grad_h_prev, grad_c_prev, grad_W_x, grad_W_h, grad_b

    def __repr__(self) -> str:
        return (
            f"LSTMCell(input_dim={self.input_dim}, "
            f"hidden_dim={self.hidden_dim})"
        )

class LSTMLayer:

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
                stacked = np.stack(step_outputs, axis=1)
            else:
                stacked = np.stack(step_outputs, axis=0)

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
