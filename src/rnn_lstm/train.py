"""Training pipeline for the image-caption decoder.

Trains a compiled Keras decoder on (feature, decoder_input, decoder_target)
arrays and saves weights + per-epoch history + an architecture config so the
from-scratch implementation can load the same weights later.
"""

import json
import time
from pathlib import Path

from .model import build_decoder_pre_inject, build_decoder_pre_inject_summary


class DecoderConfig:
    """Hyperparameter bundle for one decoder training run."""

    def __init__(
        self,
        rnn_type="lstm",
        num_layers=1,
        hidden_dim=256,
        embed_dim=256,
        architecture="pre_inject",
        learning_rate=1e-3,
        dropout=0.0,
        recurrent_dropout=0.0,
        batch_size=64,
        epochs=20,
        extra=None,
    ):
        self.rnn_type = rnn_type
        self.num_layers = num_layers
        self.hidden_dim = hidden_dim
        self.embed_dim = embed_dim
        self.architecture = architecture
        self.learning_rate = learning_rate
        self.dropout = dropout
        self.recurrent_dropout = recurrent_dropout
        self.batch_size = batch_size
        self.epochs = epochs
        self.extra = extra if extra is not None else {}

    def variant_name(self):
        return (
            self.architecture + "_" + self.rnn_type
            + "_l" + str(self.num_layers)
            + "_h" + str(self.hidden_dim)
            + "_e" + str(self.embed_dim)
        )

    def to_dict(self):
        return {
            "rnn_type": self.rnn_type,
            "num_layers": self.num_layers,
            "hidden_dim": self.hidden_dim,
            "embed_dim": self.embed_dim,
            "architecture": self.architecture,
            "learning_rate": self.learning_rate,
            "dropout": self.dropout,
            "recurrent_dropout": self.recurrent_dropout,
            "batch_size": self.batch_size,
            "epochs": self.epochs,
            "extra": dict(self.extra),
        }


def _build_model(config, vocab_size, feat_dim, seq_len):
    """Pick the right builder based on config.architecture."""
    if config.architecture == "pre_inject":
        return build_decoder_pre_inject(
            vocab_size=vocab_size,
            embed_dim=config.embed_dim,
            hidden_dim=config.hidden_dim,
            feat_dim=feat_dim,
            seq_len=seq_len,
            num_layers=config.num_layers,
            rnn_type=config.rnn_type,
            learning_rate=config.learning_rate,
            dropout=config.dropout,
            recurrent_dropout=config.recurrent_dropout,
        )
    if config.architecture == "init_inject":
        from .model_init_inject import build_decoder_init_inject

        return build_decoder_init_inject(
            vocab_size=vocab_size,
            embed_dim=config.embed_dim,
            hidden_dim=config.hidden_dim,
            feat_dim=feat_dim,
            seq_len=seq_len,
            num_layers=config.num_layers,
            rnn_type=config.rnn_type,
            learning_rate=config.learning_rate,
            dropout=config.dropout,
            recurrent_dropout=config.recurrent_dropout,
        )
    raise ValueError("Unknown architecture: " + repr(config.architecture))


def train_model(
    model,
    train_data,
    val_data=None,
    epochs=20,
    batch_size=None,
    output_dir=None,
    weights_path=None,
    history_path=None,
    verbose=None,
    callbacks=None,
    **fit_kwargs,
):
    """Train a compiled Keras model and return the history as a plain dict.

    train_data / val_data follow the usual Keras shapes — typically
    ([features, decoder_input], decoder_target). If output_dir is given,
    weights and history JSON are persisted there using the model name.
    """
    fit_options = {"epochs": epochs}
    if val_data is not None:
        fit_options["validation_data"] = val_data
    if verbose is not None:
        fit_options["verbose"] = verbose
    if callbacks is not None:
        fit_options["callbacks"] = list(callbacks)
    if batch_size is not None:
        fit_options["batch_size"] = batch_size
    fit_options.update(fit_kwargs)

    history = model.fit(train_data, **fit_options)
    history_dict = _history_to_dict(history)

    # If only output_dir is given, derive both filenames from the model name.
    if output_dir is not None:
        output_dir = Path(output_dir)
        output_dir.mkdir(parents=True, exist_ok=True)
        stem = _safe_model_stem(model)
        if weights_path is None:
            weights_path = output_dir / (stem + ".weights.h5")
        if history_path is None:
            history_path = output_dir / (stem + "_history.json")

    if weights_path is not None:
        weights_path = Path(weights_path)
        weights_path.parent.mkdir(parents=True, exist_ok=True)
        model.save_weights(weights_path)

    if history_path is not None:
        history_path = Path(history_path)
        history_path.parent.mkdir(parents=True, exist_ok=True)
        with open(history_path, "w", encoding="utf-8") as f:
            json.dump(history_dict, f, indent=2)

    return history_dict


def _history_to_dict(history):
    """Convert a Keras History (or already-dict) into plain Python floats."""
    raw = getattr(history, "history", history)
    if not isinstance(raw, dict):
        raise TypeError("model.fit must return a Keras History or a history dict.")

    out = {}
    for metric, values in raw.items():
        # values is usually a list of floats; sometimes scalar
        try:
            out[str(metric)] = [float(v) for v in values]
        except TypeError:
            out[str(metric)] = [float(values)]
    return out


def _safe_model_stem(model):
    """Filesystem-friendly stem for a Keras model name."""
    raw_name = str(getattr(model, "name", "") or "model")
    stem = ""
    for ch in raw_name:
        if ch.isalnum() or ch in ("-", "_", "."):
            stem += ch
        else:
            stem += "_"
    if stem == "":
        stem = "model"
    return stem


def train_decoder_config(
    config,
    train_features,
    train_decoder_input,
    train_decoder_target,
    val_features=None,
    val_decoder_input=None,
    val_decoder_target=None,
    vocab_size=None,
    output_dir=None,
    verbose=1,
    callbacks=None,
):
    """Train one decoder configuration end-to-end.

    Returns a dict with the trained model, history, elapsed time, and the
    architecture summary used for the config JSON. When output_dir is set,
    weights + history + config files are written alongside.
    """
    if vocab_size is None:
        vocab_size = int(train_decoder_target.max()) + 1

    feat_dim = train_features.shape[1]
    seq_len = train_decoder_input.shape[1]

    model = _build_model(config, vocab_size=vocab_size, feat_dim=feat_dim, seq_len=seq_len)

    val_data = None
    if val_features is not None and val_decoder_input is not None and val_decoder_target is not None:
        val_data = ([val_features, val_decoder_input], val_decoder_target)

    start = time.time()
    history_dict = train_model(
        model,
        train_data=([train_features, train_decoder_input], train_decoder_target),
        val_data=val_data,
        epochs=config.epochs,
        batch_size=config.batch_size,
        verbose=verbose,
        callbacks=callbacks,
    )
    elapsed = time.time() - start

    summary = build_decoder_pre_inject_summary(
        vocab_size=vocab_size,
        embed_dim=config.embed_dim,
        hidden_dim=config.hidden_dim,
        feat_dim=feat_dim,
        seq_len=seq_len,
        num_layers=config.num_layers,
        rnn_type=config.rnn_type,
    )
    # the summary helper always tags pre_inject — fix it for init-inject runs
    summary["architecture"] = config.architecture

    artefacts = {
        "variant": config.variant_name(),
        "history": history_dict,
        "elapsed_seconds": elapsed,
        "model": model,
        "config": config.to_dict(),
        "summary": summary,
    }

    if output_dir is not None:
        save_artefacts(artefacts, output_dir)

    return artefacts


def save_artefacts(artefacts, output_dir):
    """Write weights, history JSON, and architecture config to output_dir."""
    output_dir = Path(output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    variant = artefacts["variant"]
    weights_path = output_dir / (variant + ".weights.h5")
    history_path = output_dir / (variant + "_history.json")
    config_path = output_dir / (variant + "_config.json")

    model = artefacts.get("model")
    if model is not None:
        model.save_weights(weights_path)

    history_payload = {
        "history": artefacts["history"],
        "elapsed_seconds": artefacts["elapsed_seconds"],
        "config": artefacts["config"],
    }
    with open(history_path, "w", encoding="utf-8") as f:
        json.dump(history_payload, f, indent=2)

    with open(config_path, "w", encoding="utf-8") as f:
        json.dump(artefacts["summary"], f, indent=2)

    return {
        "weights": weights_path,
        "history": history_path,
        "config": config_path,
    }


def hyperparameter_grid(
    rnn_types=None,
    num_layers_options=None,
    hidden_dim_options=None,
    embed_dim=256,
    architecture="pre_inject",
    epochs=20,
    batch_size=64,
):
    """Build the spec's required hyperparameter sweep.

    Default = 3 layer counts (1/2/3) x 2 hidden sizes (128/512) x 2 decoders
    (rnn + lstm) = 12 runs.
    """
    if rnn_types is None:
        rnn_types = ["lstm", "rnn"]
    if num_layers_options is None:
        num_layers_options = [1, 2, 3]
    if hidden_dim_options is None:
        hidden_dim_options = [128, 512]

    configs = []
    for rnn_type in rnn_types:
        for n in num_layers_options:
            for h in hidden_dim_options:
                configs.append(
                    DecoderConfig(
                        rnn_type=rnn_type,
                        num_layers=n,
                        hidden_dim=h,
                        embed_dim=embed_dim,
                        architecture=architecture,
                        epochs=epochs,
                        batch_size=batch_size,
                    )
                )
    return configs


def train_grid(
    configs,
    train_features,
    train_decoder_input,
    train_decoder_target,
    val_features=None,
    val_decoder_input=None,
    val_decoder_target=None,
    vocab_size=None,
    output_root=None,
    verbose=1,
):
    """Train every config in ``configs`` and return {variant_name: artefacts}.

    When output_root is given, each run's files are written under
    output_root/{lstm or rnn}/ so the layout matches models/lstm/ and
    models/rnn/ from the spec.
    """
    results = {}
    for config in configs:
        out_dir = None
        if output_root is not None:
            if config.rnn_type == "lstm":
                subdir = "lstm"
            else:
                subdir = "rnn"
            out_dir = Path(output_root) / subdir

        artefacts = train_decoder_config(
            config,
            train_features=train_features,
            train_decoder_input=train_decoder_input,
            train_decoder_target=train_decoder_target,
            val_features=val_features,
            val_decoder_input=val_decoder_input,
            val_decoder_target=val_decoder_target,
            vocab_size=vocab_size,
            output_dir=out_dir,
            verbose=verbose,
        )
        results[config.variant_name()] = artefacts

    return results
