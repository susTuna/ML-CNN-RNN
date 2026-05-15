import json
import os
import sys
from types import SimpleNamespace

import numpy as np

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
if ROOT not in sys.path:
    sys.path.insert(0, ROOT)

from src.rnn_lstm.caption_preprocessing import (
    END_ID,
    PAD_ID,
    START_ID,
    preprocess_captions,
    tokenize_and_pad,
)
from src.rnn_lstm.train import train_model


class FakeModel:
    name = "fake decoder"

    def __init__(self):
        self.fit_kwargs = None
        self.saved_weights_path = None

    def fit(self, train_data, **kwargs):
        self.train_data = train_data
        self.fit_kwargs = kwargs
        return SimpleNamespace(history={"loss": [np.float32(1.5)], "val_loss": [2]})

    def save_weights(self, path):
        self.saved_weights_path = path
        path.write_text("weights", encoding="utf-8")


def test_preprocess_captions_accepts_flickr8k_file(tmp_path):
    captions_file = tmp_path / "Flickr8k.token.txt"
    captions_file.write_text(
        "img1.jpg#0\tA Dog, sits!\n"
        "img1.jpg#1\tThe CAT runs.\n",
        encoding="utf-8",
    )

    processed = preprocess_captions(captions_file)

    assert processed == {
        "img1.jpg": [
            "<start> a dog sits <end>",
            "<start> the cat runs <end>",
        ]
    }


def test_tokenize_and_pad_defaults_to_length_35():
    word2idx = {"<pad>": PAD_ID, "<start>": START_ID, "<end>": END_ID, "dog": 4}

    tokens = tokenize_and_pad(["<start> dog <end>"], word2idx)

    assert tokens.shape == (1, 35)
    assert tokens[0, :3].tolist() == [START_ID, 4, END_ID]
    assert np.all(tokens[0, 3:] == PAD_ID)


def test_train_model_returns_pure_history_and_can_persist_outputs(tmp_path):
    model = FakeModel()
    train_data = ("x_train", "y_train")
    val_data = ("x_val", "y_val")

    history = train_model(
        model,
        train_data,
        val_data,
        epochs=3,
        batch_size=2,
        output_dir=tmp_path,
        verbose=0,
    )

    assert history == {"loss": [1.5], "val_loss": [2.0]}
    assert model.train_data == train_data
    assert model.fit_kwargs["validation_data"] == val_data
    assert model.fit_kwargs["epochs"] == 3
    assert model.fit_kwargs["batch_size"] == 2

    history_path = tmp_path / "fake_decoder_history.json"
    weights_path = tmp_path / "fake_decoder.weights.h5"
    assert json.loads(history_path.read_text(encoding="utf-8")) == history
    assert weights_path.read_text(encoding="utf-8") == "weights"
