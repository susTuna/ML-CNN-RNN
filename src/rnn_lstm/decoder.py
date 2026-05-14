from __future__ import annotations

from typing import Dict, List, Optional

import numpy as np

from src.rnn_lstm.embedding import EmbeddingLayer
from src.rnn_lstm.lstm import LSTMCell
from src.rnn_lstm.rnn import SimpleRNNCell
from src.shared.activations import softmax
from src.shared.dense_layer import DenseLayer
from src.shared.image_utils import load_image


class ImageCaptionerScratch:
    """
    Inference-only image captioning decoder using NumPy-only scratch layers.

    Supports both SimpleRNN and LSTM decoder architectures with the
    **pre-inject** strategy: the CNN feature is projected to embed_dim
    and fed as the very first timestep of the RNN/LSTM, then the model
    generates tokens autoregressively.

    Parameters
    ----------
    embedding : EmbeddingLayer
    rnn_cells : list of SimpleRNNCell or LSTMCell
        One entry per stacked layer, ordered bottom (input-side) to top.
    dense_out : DenseLayer
        Maps hidden state -> vocab logits.
    dense_proj : DenseLayer or None
        Maps CNN feature vector -> embed_dim.  Pass None if the model
        does not project CNN features (rare).
    word2idx : dict[str, int]
    idx2word : dict[int, str]
    rnn_type : 'lstm' or 'rnn'
    cnn_encoder : Keras model or None
        Frozen CNN encoder used to extract image features at inference.
    """

    def __init__(
        self,
        embedding: EmbeddingLayer,
        rnn_cells: list,
        dense_out: DenseLayer,
        dense_proj: Optional[DenseLayer],
        word2idx: Dict[str, int],
        idx2word: Dict[int, str],
        rnn_type: str = "lstm",
        cnn_encoder=None,
    ) -> None:
        self.embedding = embedding
        self.rnn_cells = rnn_cells
        self.dense_out = dense_out
        self.dense_proj = dense_proj
        self.word2idx = word2idx
        self.idx2word = idx2word
        self.rnn_type = rnn_type.lower()
        self.cnn_encoder = cnn_encoder

        # Special token IDs (use common defaults if missing from vocab)
        self.start_id: int = word2idx.get("<start>", 1)
        self.end_id: int = word2idx.get("<end>", 2)
        self.pad_id: int = word2idx.get("<pad>", 0)

    @classmethod
    def load_from_keras(
        cls,
        keras_decoder_model,
        cnn_encoder,
        word2idx: Dict[str, int],
        idx2word: Dict[int, str],
    ) -> "ImageCaptionerScratch":
        """
        Auto-load all weights from a compiled Keras decoder model.

        The method inspects every layer and classifies it as:
        - Embedding  -> EmbeddingLayer
        - LSTM       -> LSTMCell (stacking supported)
        - SimpleRNN  -> SimpleRNNCell (stacking supported)
        - Dense      -> projection layer *or* output layer
          (distinguished by output size: projection ≤ embed_dim < output)

        Parameters
        ----------
        keras_decoder_model : compiled Keras model
        cnn_encoder : Keras frozen encoder (e.g. InceptionV3 without top)
        word2idx : dict
        idx2word : dict
        """
        try:
            import tensorflow as tf
        except ImportError as exc:
            raise ImportError("TensorFlow/Keras is required for load_from_keras.") from exc

        embedding_layer: Optional[EmbeddingLayer] = None
        rnn_cells: list = []
        dense_layers: list = []
        rnn_type: str = "lstm"

        for layer in keras_decoder_model.layers:
            ltype = type(layer).__name__

            if isinstance(layer, tf.keras.layers.Embedding):
                embedding_layer = EmbeddingLayer.from_keras_layer(layer)

            elif isinstance(layer, tf.keras.layers.LSTM):
                rnn_cells.append(LSTMCell.from_keras_layer(layer))
                rnn_type = "lstm"

            elif isinstance(layer, tf.keras.layers.SimpleRNN):
                rnn_cells.append(SimpleRNNCell.from_keras_layer(layer))
                rnn_type = "rnn"

            elif isinstance(layer, tf.keras.layers.Dense):
                weights = layer.get_weights()
                if weights:  # skip unbuilt Dense layers
                    dense_layers.append(layer)

            elif ltype == "TimeDistributed":
                # Unwrap TimeDistributed(Dense)
                inner = getattr(layer, "layer", None)
                if inner is not None and isinstance(inner, tf.keras.layers.Dense):
                    dense_layers.append(inner)

        if embedding_layer is None:
            raise ValueError(
                "Could not find an Embedding layer in keras_decoder_model. "
                "Ensure the model contains tf.keras.layers.Embedding."
            )

        # Identify projection vs output Dense by output dimensionality.
        # Projection: smaller output dim = embed_dim
        # Output:     larger output dim  = vocab_size
        dense_proj: Optional[DenseLayer] = None
        dense_out: Optional[DenseLayer] = None

        if len(dense_layers) == 0:
            raise ValueError("No Dense layers found in keras_decoder_model.")
        elif len(dense_layers) == 1:
            dense_out = DenseLayer.from_keras_layer(dense_layers[0])
        else:
            # Sort by output size ascending
            dense_layers_sorted = sorted(
                dense_layers,
                key=lambda l: l.get_weights()[0].shape[1],
            )
            dense_proj = DenseLayer.from_keras_layer(dense_layers_sorted[0])
            dense_out = DenseLayer.from_keras_layer(dense_layers_sorted[-1])

        return cls(
            embedding=embedding_layer,
            rnn_cells=rnn_cells,
            dense_out=dense_out,
            dense_proj=dense_proj,
            word2idx=word2idx,
            idx2word=idx2word,
            rnn_type=rnn_type,
            cnn_encoder=cnn_encoder,
        )

    def _extract_cnn_feature(self, image_path: str) -> np.ndarray:
        """Load image and run through frozen CNN encoder -> feature vector."""
        if self.cnn_encoder is None:
            raise RuntimeError(
                "cnn_encoder is None. Provide a Keras encoder at construction."
            )
        img = load_image(image_path)          # (H, W, C), float32 [0,1]
        img_batch = img[np.newaxis, ...]      # (1, H, W, C)
        feat = self.cnn_encoder.predict(img_batch, verbose=0)  # (1, feat_dim)
        return feat[0].astype(np.float32)     # (feat_dim,)

    def _init_hidden(self) -> list:
        """
        Return zero-initialised hidden state(s) for all stacked cells.

        For LSTM: list of (h, c) tuples.
        For RNN:  list of h arrays.
        """
        states: list = []
        for cell in self.rnn_cells:
            H = cell.hidden_dim
            if self.rnn_type == "lstm":
                states.append(
                    (
                        np.zeros(H, dtype=np.float32),
                        np.zeros(H, dtype=np.float32),
                    )
                )
            else:
                states.append(np.zeros(H, dtype=np.float32))
        return states

    def _step(self, x_t: np.ndarray, states: list):
        """
        Run x_t through all stacked cells for one timestep.

        Parameters
        ----------
        x_t    : np.ndarray, shape (input_dim,)
        states : list — current hidden states (one entry per cell)

        Returns
        -------
        h_out   : np.ndarray — output of the top cell
        new_states : list — updated hidden states
        """
        new_states: list = []
        inp = x_t

        for i, cell in enumerate(self.rnn_cells):
            if self.rnn_type == "lstm":
                h_prev, c_prev = states[i]
                h_new, c_new = cell.forward(inp, h_prev, c_prev)
                new_states.append((h_new, c_new))
            else:
                h_prev = states[i]
                h_new = cell.forward(inp, h_prev)
                new_states.append(h_new)
            inp = h_new  # feed hidden state as input to next stacked cell

        return inp, new_states  # inp == h_out of the top cell

    def generate_caption_greedy(
        self,
        image_path: str,
        max_len: int = 20,
    ) -> str:
        """
        Generate a caption for a single image using greedy decoding.

        Parameters
        ----------
        image_path : str
        max_len    : int — maximum number of tokens to generate

        Returns
        -------
        str — decoded caption (words joined by spaces, no special tokens)
        """
        # Step 1-2: extract CNN feature
        cnn_feat = self._extract_cnn_feature(image_path)

        # Step 3: project CNN feature to embed_dim
        if self.dense_proj is not None:
            x_proj = self.dense_proj.forward(cnn_feat).ravel()  # (embed_dim,)
        else:
            x_proj = cnn_feat  # assume feature_dim == embed_dim

        # Step 4-5: initialise hidden state and run projection step
        states = self._init_hidden()
        _, states = self._step(x_proj, states)

        # Step 6-7: autoregressive generation
        token: int = self.start_id
        generated: List[int] = []

        for _ in range(max_len):
            x_t = self.embedding.forward(np.array(token, dtype=np.int32))  # (embed_dim,)
            h_out, states = self._step(x_t, states)

            logits = self.dense_out.forward(h_out).ravel()  # (vocab_size,)
            token = int(np.argmax(logits))

            if token == self.end_id:
                break
            if token != self.pad_id:
                generated.append(token)

        # Step 8: decode
        words = [self.idx2word.get(str(t), self.idx2word.get(t, "<unk>")) for t in generated]
        return " ".join(words)

    def generate_captions(
        self,
        image_paths: List[str],
        max_len: int = 20,
    ) -> List[str]:
        """
        Generate captions for multiple images (sequential loop).

        Parameters
        ----------
        image_paths : list of str
        max_len     : int

        Returns
        -------
        list of str — one caption per image, in the same order
        """
        return [self.generate_caption_greedy(p, max_len) for p in image_paths]

    def __repr__(self) -> str:
        proj_info = (
            f"{self.dense_proj.weights.shape[0]}->{self.dense_proj.weights.shape[1]}"
            if self.dense_proj is not None
            else "None"
        )
        return (
            f"ImageCaptionerScratch("
            f"rnn_type={self.rnn_type}, "
            f"num_layers={len(self.rnn_cells)}, "
            f"vocab_size={self.embedding.vocab_size}, "
            f"dense_proj={proj_info})"
        )
