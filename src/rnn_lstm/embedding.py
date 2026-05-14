import numpy as np
from typing import Optional


class EmbeddingLayer:
    """
    Token-embedding lookup table.

    Attributes
    ----------
    embedding_matrix : np.ndarray, shape (vocab_size, embed_dim)
        The learned embedding weights.
    vocab_size : int
    embed_dim : int
    """

    def __init__(self, embedding_matrix: np.ndarray) -> None:
        """
        Parameters
        ----------
        embedding_matrix : np.ndarray, shape (vocab_size, embed_dim)
        """
        self.embedding_matrix = embedding_matrix.astype(np.float32)
        self.vocab_size, self.embed_dim = embedding_matrix.shape

    def forward(self, token_ids: np.ndarray) -> np.ndarray:
        """
        Look up embeddings for one or more token ids.

        Parameters
        ----------
        token_ids : np.ndarray
            Shape (seq_len,) or (batch, seq_len) — integer token ids.

        Returns
        -------
        np.ndarray
            Shape (seq_len, embed_dim) or (batch, seq_len, embed_dim).
        """
        return self.embedding_matrix[token_ids]

    @classmethod
    def from_keras_layer(cls, keras_layer) -> "EmbeddingLayer":
        """
        Build an EmbeddingLayer from a tf.keras.layers.Embedding layer.

        Parameters
        ----------
        keras_layer : keras Embedding layer
            Must expose get_weights() returning [embedding_matrix].

        Returns
        -------
        EmbeddingLayer
        """
        weights = keras_layer.get_weights()
        if len(weights) != 1:
            raise ValueError(
                f"Expected 1 weight tensor from Embedding layer, got {len(weights)}."
            )
        return cls(weights[0])


    def backward(
        self,
        grad_out: np.ndarray,
        token_ids: np.ndarray,
    ) -> np.ndarray:
        """
        Accumulate gradients into the embedding matrix.

        Parameters
        ----------
        grad_out  : np.ndarray
            Gradient of the loss w.r.t. the embedding output.
            Shape (seq_len, embed_dim) or (batch, seq_len, embed_dim).
        token_ids : np.ndarray
            Integer token ids used in the forward pass.
            Shape (seq_len,) or (batch, seq_len).

        Returns
        -------
        np.ndarray
            Gradient matrix w.r.t. embedding_matrix,
            shape (vocab_size, embed_dim).
            Repeated token ids have their gradients *summed* (scatter-add).
        """
        grad_embedding = np.zeros_like(self.embedding_matrix)  # (V, D)
        np.add.at(grad_embedding, token_ids.ravel(), grad_out.reshape(-1, self.embed_dim))
        return grad_embedding

    def __repr__(self) -> str:
        return (
            f"EmbeddingLayer(vocab_size={self.vocab_size}, embed_dim={self.embed_dim})"
        )
