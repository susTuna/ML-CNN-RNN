import numpy as np


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
            Shape ``(seq_len,)`` or ``(batch, seq_len)`` — integer token ids.

        Returns
        -------
        np.ndarray
            Shape ``(seq_len, embed_dim)`` or ``(batch, seq_len, embed_dim)``.
        """
        return self.embedding_matrix[token_ids]

    @classmethod
    def from_keras_layer(cls, keras_layer) -> "EmbeddingLayer":
        """
        Build an EmbeddingLayer from a ``tf.keras.layers.Embedding`` layer.

        Parameters
        ----------
        keras_layer : keras Embedding layer
            Must expose ``get_weights()`` returning ``[embedding_matrix]``.

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

    def __repr__(self) -> str:
        return (
            f"EmbeddingLayer(vocab_size={self.vocab_size}, embed_dim={self.embed_dim})"
        )
