import numpy as np
from typing import Optional


class EmbeddingLayer:

    def __init__(self, embedding_matrix: np.ndarray) -> None:
        self.embedding_matrix = embedding_matrix.astype(np.float32)
        self.vocab_size, self.embed_dim = embedding_matrix.shape

    def forward(self, token_ids: np.ndarray) -> np.ndarray:
        return self.embedding_matrix[token_ids]

    @classmethod
    def from_keras_layer(cls, keras_layer) -> "EmbeddingLayer":
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
        grad_embedding = np.zeros_like(self.embedding_matrix)  # (V, D)
        np.add.at(grad_embedding, token_ids.ravel(), grad_out.reshape(-1, self.embed_dim))
        return grad_embedding

    def __repr__(self) -> str:
        return (
            f"EmbeddingLayer(vocab_size={self.vocab_size}, embed_dim={self.embed_dim})"
        )
