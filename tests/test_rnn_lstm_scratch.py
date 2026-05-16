import sys
import os
import types
import numpy as np
import pytest

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
if ROOT not in sys.path:
    sys.path.insert(0, ROOT)

# Tiny vocab (20 words) used across all tests
VOCAB_WORDS = ["<pad>", "<start>", "<end>", "<unk>",
               "a", "dog", "cat", "sits", "on", "the",
               "mat", "red", "blue", "large", "small",
               "fluffy", "cute", "happy", "brown", "white"]
WORD2IDX = {w: i for i, w in enumerate(VOCAB_WORDS)}
IDX2WORD = {i: w for w, i in WORD2IDX.items()}
VOCAB_SIZE  = len(VOCAB_WORDS)
EMBED_DIM   = 16
HIDDEN_DIM  = 32
FEAT_DIM    = 64   # fake CNN feature dimension

# Helpers to build fake Keras-like objects (no real TF needed for unit tests)

class _FakeWeights:
    """Minimal stand-in for a Keras weight tensor."""
    def __init__(self, arr):
        self._arr = arr
    def numpy(self):
        return self._arr


def _make_fake_embedding():
    """Returns a fake Keras Embedding layer object."""
    W = np.random.randn(VOCAB_SIZE, EMBED_DIM).astype("f")
    layer = types.SimpleNamespace(
        __class__=type("Embedding", (), {}),
        get_weights=lambda: [W],
        return_sequences=False,
    )
    return layer, W


def _make_fake_lstm():
    """Returns a fake Keras LSTM layer object (single layer, no stacking)."""
    W_x = np.random.randn(EMBED_DIM,  4 * HIDDEN_DIM).astype("f") * 0.1
    W_h = np.random.randn(HIDDEN_DIM, 4 * HIDDEN_DIM).astype("f") * 0.1
    b   = np.zeros(4 * HIDDEN_DIM, dtype="f")
    layer = types.SimpleNamespace(
        __class__=type("LSTM", (), {}),
        get_weights=lambda: [W_x, W_h, b],
        return_sequences=False,
    )
    return layer, (W_x, W_h, b)


def _make_fake_rnn():
    """Returns a fake Keras SimpleRNN layer object."""
    W_x = np.random.randn(EMBED_DIM,  HIDDEN_DIM).astype("f") * 0.1
    W_h = np.random.randn(HIDDEN_DIM, HIDDEN_DIM).astype("f") * 0.1
    b   = np.zeros(HIDDEN_DIM, dtype="f")
    layer = types.SimpleNamespace(
        __class__=type("SimpleRNN", (), {}),
        get_weights=lambda: [W_x, W_h, b],
        return_sequences=False,
    )
    return layer, (W_x, W_h, b)


def _make_fake_dense(in_dim, out_dim, activation="linear"):
    """Returns a fake Keras Dense layer object."""
    W = np.random.randn(in_dim, out_dim).astype("f") * 0.1
    b = np.zeros(out_dim, dtype="f")
    layer = types.SimpleNamespace(
        __class__=type("Dense", (), {}),
        activation=types.SimpleNamespace(__name__=activation),
        get_weights=lambda: [W, b],
    )
    return layer, (W, b)


class _FakeCNNEncoder:
    """Mimics a frozen Keras CNN encoder: predict(img_batch) -> (N, FEAT_DIM)."""
    def predict(self, img_batch, verbose=0):
        N = img_batch.shape[0]
        # Deterministic output so we can reproduce results
        rng = np.random.default_rng(seed=42)
        return rng.random((N, FEAT_DIM), dtype=np.float32)


class _FakeImage:
    """
    Patches load_image so no real image file is needed.
    Returns a (224, 224, 3) white float32 array.
    """
    @staticmethod
    def load(path):
        return np.ones((224, 224, 3), dtype=np.float32)

# Patch load_image before importing decoder
import unittest.mock as mock

_PATCH_TARGET = "src.rnn_lstm.decoder.load_image"

# Part 1 - Unit tests for individual layers

class TestEmbeddingLayer:
    def setup_method(self):
        from src.rnn_lstm.embedding import EmbeddingLayer
        self.E = EmbeddingLayer(np.eye(VOCAB_SIZE, EMBED_DIM, dtype="f"))

    def test_forward_1d(self):
        out = self.E.forward(np.array([1, 5, 10]))
        assert out.shape == (3, EMBED_DIM)

    def test_forward_2d(self):
        out = self.E.forward(np.array([[1, 2], [3, 4]]))
        assert out.shape == (2, 2, EMBED_DIM)

    def test_from_keras_layer(self):
        from src.rnn_lstm.embedding import EmbeddingLayer
        fake_layer, W = _make_fake_embedding()
        E2 = EmbeddingLayer.from_keras_layer(fake_layer)
        np.testing.assert_array_equal(E2.embedding_matrix, W.astype("f"))

    def test_backward_shape(self):
        ids = np.array([0, 1, 2])
        grad_out = np.ones((3, EMBED_DIM), dtype="f")
        g = self.E.backward(grad_out, ids)
        assert g.shape == (VOCAB_SIZE, EMBED_DIM)

    def test_backward_accumulates(self):
        # token 0 appears twice -> gradient should be summed
        ids = np.array([0, 0])
        grad_out = np.ones((2, EMBED_DIM), dtype="f")
        g = self.E.backward(grad_out, ids)
        np.testing.assert_allclose(g[0], np.full(EMBED_DIM, 2.0))
        np.testing.assert_allclose(g[1], np.zeros(EMBED_DIM))


class TestSimpleRNNCell:
    def setup_method(self):
        from src.rnn_lstm.rnn import SimpleRNNCell
        self.cell = SimpleRNNCell(
            np.random.randn(EMBED_DIM, HIDDEN_DIM).astype("f"),
            np.random.randn(HIDDEN_DIM, HIDDEN_DIM).astype("f"),
            np.zeros(HIDDEN_DIM, "f"),
        )

    def test_forward_shape(self):
        x = np.random.randn(EMBED_DIM).astype("f")
        h = np.zeros(HIDDEN_DIM, "f")
        h_t = self.cell.forward(x, h)
        assert h_t.shape == (HIDDEN_DIM,)

    def test_forward_range(self):
        x = np.random.randn(EMBED_DIM).astype("f")
        h = np.zeros(HIDDEN_DIM, "f")
        h_t = self.cell.forward(x, h)
        assert np.all(h_t >= -1) and np.all(h_t <= 1)  # tanh output

    def test_backward_shapes(self):
        x = np.random.randn(EMBED_DIM).astype("f")
        h = np.zeros(HIDDEN_DIM, "f")
        h_t = self.cell.forward(x, h)
        grad_h = np.ones(HIDDEN_DIM, "f")
        gx, gh_p, gWx, gWh, gb = self.cell.backward(x, h, h_t, grad_h)
        assert gx.shape  == (EMBED_DIM,)
        assert gh_p.shape == (HIDDEN_DIM,)
        assert gWx.shape == (EMBED_DIM, HIDDEN_DIM)
        assert gWh.shape == (HIDDEN_DIM, HIDDEN_DIM)
        assert gb.shape  == (HIDDEN_DIM,)

    def test_from_keras_layer(self):
        from src.rnn_lstm.rnn import SimpleRNNCell
        fake_layer, _ = _make_fake_rnn()
        cell2 = SimpleRNNCell.from_keras_layer(fake_layer)
        assert cell2.hidden_dim == HIDDEN_DIM


class TestLSTMCell:
    def setup_method(self):
        from src.rnn_lstm.lstm import LSTMCell
        self.cell = LSTMCell(
            np.random.randn(EMBED_DIM,  4 * HIDDEN_DIM).astype("f") * 0.1,
            np.random.randn(HIDDEN_DIM, 4 * HIDDEN_DIM).astype("f") * 0.1,
            np.zeros(4 * HIDDEN_DIM, "f"),
        )

    def test_forward_shapes(self):
        x = np.random.randn(EMBED_DIM).astype("f")
        h = np.zeros(HIDDEN_DIM, "f")
        c = np.zeros(HIDDEN_DIM, "f")
        h_t, c_t = self.cell.forward(x, h, c)
        assert h_t.shape == (HIDDEN_DIM,)
        assert c_t.shape == (HIDDEN_DIM,)

    def test_backward_shapes(self):
        x = np.random.randn(EMBED_DIM).astype("f")
        h = np.zeros(HIDDEN_DIM, "f")
        c = np.zeros(HIDDEN_DIM, "f")
        h_t, c_t = self.cell.forward(x, h, c)
        grad_h = np.ones(HIDDEN_DIM, "f")
        grad_c = np.zeros(HIDDEN_DIM, "f")
        gx, gh_p, gc_p, gWx, gWh, gb = self.cell.backward(x, h, c, c_t, h_t, grad_h, grad_c)
        assert gx.shape  == (EMBED_DIM,)
        assert gh_p.shape == (HIDDEN_DIM,)
        assert gc_p.shape == (HIDDEN_DIM,)
        assert gWx.shape == (EMBED_DIM,  4 * HIDDEN_DIM)
        assert gWh.shape == (HIDDEN_DIM, 4 * HIDDEN_DIM)
        assert gb.shape  == (4 * HIDDEN_DIM,)


class TestSimpleRNNLayer:
    def setup_method(self):
        from src.rnn_lstm.rnn import SimpleRNNCell, SimpleRNNLayer
        cell = SimpleRNNCell(
            np.random.randn(EMBED_DIM, HIDDEN_DIM).astype("f"),
            np.random.randn(HIDDEN_DIM, HIDDEN_DIM).astype("f"),
            np.zeros(HIDDEN_DIM, "f"),
        )
        self.layer_seq = SimpleRNNLayer([cell], return_sequences=True)
        self.layer_last = SimpleRNNLayer([cell], return_sequences=False)

    def test_return_sequences_true(self):
        seq = np.random.randn(10, EMBED_DIM).astype("f")
        out = self.layer_seq.forward(seq)
        assert out.shape == (10, HIDDEN_DIM)

    def test_return_sequences_false(self):
        seq = np.random.randn(10, EMBED_DIM).astype("f")
        out = self.layer_last.forward(seq)
        assert out.shape == (HIDDEN_DIM,)

    def test_batch_input(self):
        seq = np.random.randn(4, 10, EMBED_DIM).astype("f")
        out = self.layer_last.forward(seq)
        assert out.shape == (4, HIDDEN_DIM)


class TestLSTMLayer:
    def setup_method(self):
        from src.rnn_lstm.lstm import LSTMCell, LSTMLayer
        cell = LSTMCell(
            np.random.randn(EMBED_DIM,  4 * HIDDEN_DIM).astype("f") * 0.1,
            np.random.randn(HIDDEN_DIM, 4 * HIDDEN_DIM).astype("f") * 0.1,
            np.zeros(4 * HIDDEN_DIM, "f"),
        )
        self.layer_seq  = LSTMLayer([cell], return_sequences=True)
        self.layer_last = LSTMLayer([cell], return_sequences=False)

    def test_return_sequences_true(self):
        seq = np.random.randn(10, EMBED_DIM).astype("f")
        out = self.layer_seq.forward(seq)
        assert out.shape == (10, HIDDEN_DIM)

    def test_return_sequences_false(self):
        seq = np.random.randn(10, EMBED_DIM).astype("f")
        out = self.layer_last.forward(seq)
        assert out.shape == (HIDDEN_DIM,)

    def test_batch_input(self):
        seq = np.random.randn(4, 10, EMBED_DIM).astype("f")
        out = self.layer_last.forward(seq)
        assert out.shape == (4, HIDDEN_DIM)

# Part 2 - ImageCaptionerScratch (manual construction, no Keras)

def _build_captioner(rnn_type="lstm"):
    """Build an ImageCaptionerScratch with random weights, fake CNN encoder."""
    from src.rnn_lstm.embedding import EmbeddingLayer
    from src.rnn_lstm.lstm import LSTMCell
    from src.rnn_lstm.rnn import SimpleRNNCell
    from src.rnn_lstm.decoder import ImageCaptionerScratch
    from src.shared.dense_layer import DenseLayer

    embedding = EmbeddingLayer(
        np.random.randn(VOCAB_SIZE, EMBED_DIM).astype("f") * 0.1
    )

    if rnn_type == "lstm":
        cell = LSTMCell(
            np.random.randn(EMBED_DIM,  4 * HIDDEN_DIM).astype("f") * 0.1,
            np.random.randn(HIDDEN_DIM, 4 * HIDDEN_DIM).astype("f") * 0.1,
            np.zeros(4 * HIDDEN_DIM, "f"),
        )
    else:
        cell = SimpleRNNCell(
            np.random.randn(EMBED_DIM, HIDDEN_DIM).astype("f") * 0.1,
            np.random.randn(HIDDEN_DIM, HIDDEN_DIM).astype("f") * 0.1,
            np.zeros(HIDDEN_DIM, "f"),
        )

    # dense_proj: FEAT_DIM -> EMBED_DIM
    dense_proj = DenseLayer(FEAT_DIM, EMBED_DIM)
    # dense_out:  HIDDEN_DIM -> VOCAB_SIZE  (softmax)
    dense_out = DenseLayer(HIDDEN_DIM, VOCAB_SIZE, activation="softmax")

    return ImageCaptionerScratch(
        embedding=embedding,
        rnn_cells=[cell],
        dense_out=dense_out,
        dense_proj=dense_proj,
        word2idx=WORD2IDX,
        idx2word=IDX2WORD,
        rnn_type=rnn_type,
        cnn_encoder=_FakeCNNEncoder(),
    )


class TestImageCaptionerScratch:
    @pytest.mark.parametrize("rnn_type", ["lstm", "rnn"])
    def test_greedy_returns_string(self, rnn_type):
        captioner = _build_captioner(rnn_type)
        with mock.patch(_PATCH_TARGET, _FakeImage.load):
            caption = captioner.generate_caption_greedy("fake_image.jpg", max_len=10)
        assert isinstance(caption, str)

    @pytest.mark.parametrize("rnn_type", ["lstm", "rnn"])
    def test_greedy_no_special_tokens(self, rnn_type):
        captioner = _build_captioner(rnn_type)
        with mock.patch(_PATCH_TARGET, _FakeImage.load):
            caption = captioner.generate_caption_greedy("fake_image.jpg", max_len=15)
        for special in ["<start>", "<end>", "<pad>"]:
            assert special not in caption, f"Special token {special!r} leaked into caption"

    @pytest.mark.parametrize("rnn_type", ["lstm", "rnn"])
    def test_greedy_max_len_respected(self, rnn_type):
        captioner = _build_captioner(rnn_type)
        with mock.patch(_PATCH_TARGET, _FakeImage.load):
            caption = captioner.generate_caption_greedy("fake_image.jpg", max_len=5)
        words = caption.split()
        assert len(words) <= 5

    def test_generate_captions_batch(self):
        captioner = _build_captioner("lstm")
        with mock.patch(_PATCH_TARGET, _FakeImage.load):
            captions = captioner.generate_captions(
                ["a.jpg", "b.jpg", "c.jpg"], max_len=8
            )
        assert len(captions) == 3
        assert all(isinstance(c, str) for c in captions)

    def test_repr(self):
        captioner = _build_captioner("lstm")
        r = repr(captioner)
        assert "lstm" in r
        assert "vocab_size" in r

# Part 3 - Beam Search

class TestBeamSearch:
    @pytest.mark.parametrize("k", [1, 3, 5])
    def test_beam_returns_string(self, k):
        from src.rnn_lstm.beam_search import beam_search_decode
        captioner = _build_captioner("lstm")
        with mock.patch(_PATCH_TARGET, _FakeImage.load):
            caption = beam_search_decode(captioner, "fake.jpg", k=k, max_len=10)
        assert isinstance(caption, str)

    def test_beam_no_special_tokens(self):
        from src.rnn_lstm.beam_search import beam_search_decode
        captioner = _build_captioner("lstm")
        with mock.patch(_PATCH_TARGET, _FakeImage.load):
            caption = beam_search_decode(captioner, "fake.jpg", k=3, max_len=12)
        for special in ["<start>", "<end>", "<pad>"]:
            assert special not in caption

    def test_beam_batch(self):
        from src.rnn_lstm.beam_search import beam_search_batch
        captioner = _build_captioner("lstm")
        with mock.patch(_PATCH_TARGET, _FakeImage.load):
            captions = beam_search_batch(captioner, ["a.jpg", "b.jpg"], k=3, max_len=8)
        assert len(captions) == 2

    def test_rnn_beam(self):
        from src.rnn_lstm.beam_search import beam_search_decode
        captioner = _build_captioner("rnn")
        with mock.patch(_PATCH_TARGET, _FakeImage.load):
            caption = beam_search_decode(captioner, "fake.jpg", k=3, max_len=10)
        assert isinstance(caption, str)

# Part 4 - Numerical gradient check (finite differences vs backward)

class TestNumericalGradients:
    """
    Verify backward() implementations against finite-difference approximations.
    Tests run with small random weights so activations stay in a reasonable range.
    """
    EPS = 1e-4
    ATOL = 1e-3

    def _finite_diff(self, f, param, idx):
        """Compute central finite-difference for scalar parameter param[idx]."""
        orig = param.flat[idx]
        param.flat[idx] = orig + self.EPS
        f_plus = f()
        param.flat[idx] = orig - self.EPS
        f_minus = f()
        param.flat[idx] = orig
        return (f_plus - f_minus) / (2 * self.EPS)

    def test_rnn_cell_grad_x(self):
        from src.rnn_lstm.rnn import SimpleRNNCell
        np.random.seed(0)
        cell = SimpleRNNCell(
            np.random.randn(4, 8).astype("f") * 0.1,
            np.random.randn(8, 8).astype("f") * 0.1,
            np.zeros(8, "f"),
        )
        x = np.random.randn(4).astype("f") * 0.5
        h = np.random.randn(8).astype("f") * 0.5
        # Loss = sum(h_t)
        h_t = cell.forward(x, h)
        grad_h = np.ones_like(h_t)
        gx, _, _, _, _ = cell.backward(x, h, h_t, grad_h)

        for i in range(len(x)):
            fd = self._finite_diff(lambda: cell.forward(x, h).sum(), x, i)
            assert abs(gx[i] - fd) < self.ATOL, \
                f"RNN grad_x[{i}]: analytical={gx[i]:.6f}, fd={fd:.6f}"

    def test_lstm_cell_grad_x(self):
        from src.rnn_lstm.lstm import LSTMCell
        np.random.seed(1)
        cell = LSTMCell(
            np.random.randn(4, 32).astype("f") * 0.1,
            np.random.randn(8, 32).astype("f") * 0.1,
            np.zeros(32, "f"),
        )
        x = np.random.randn(4).astype("f") * 0.5
        h = np.random.randn(8).astype("f") * 0.5
        c = np.random.randn(8).astype("f") * 0.5
        h_t, c_t = cell.forward(x, h, c)
        grad_h = np.ones_like(h_t)
        grad_c = np.zeros_like(c_t)
        gx, _, _, _, _, _ = cell.backward(x, h, c, c_t, h_t, grad_h, grad_c)

        for i in range(len(x)):
            fd = self._finite_diff(lambda: cell.forward(x, h, c)[0].sum(), x, i)
            assert abs(gx[i] - fd) < self.ATOL, \
                f"LSTM grad_x[{i}]: analytical={gx[i]:.6f}, fd={fd:.6f}"

if __name__ == "__main__":
    import pytest
    sys.exit(pytest.main([__file__, "-v"]))
