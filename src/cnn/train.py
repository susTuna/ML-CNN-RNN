"""Training pipeline for the CNN image classifier (Intel Image Classification).

Drives the spec's 16-architecture hyperparameter sweep:
    2 (num_conv_layers) x 2 (filter combos) x 2 (kernel sizes) x 2 (pooling)

Saves weights, per-epoch history, and macro F1 per variant so the notebook
can build comparison tables. Uses the Keras builder from cnn/model.py and
the MacroF1Callback from there.
"""

import json
import time
from pathlib import Path

import numpy as np
from sklearn.metrics import f1_score

from .model import MacroF1Callback, build_cnn_model


class CNNConfig:
    """Hyperparameter bundle for one CNN training run."""

    def __init__(
        self,
        num_conv_layers=2,
        filters_per_layer=None,
        kernel_size=3,
        pooling_type="max",
        learning_rate=1e-3,
        batch_size=32,
        epochs=20,
        input_shape=(150, 150, 3),
        num_classes=6,
    ):
        self.num_conv_layers = num_conv_layers
        self.filters_per_layer = filters_per_layer if filters_per_layer is not None else [32, 64]
        self.kernel_size = kernel_size
        self.pooling_type = pooling_type
        self.learning_rate = learning_rate
        self.batch_size = batch_size
        self.epochs = epochs
        self.input_shape = tuple(input_shape)
        self.num_classes = num_classes

    def variant_name(self):
        # e.g. "l2_f32-64_k3_max"
        filt_str = "-".join(str(f) for f in self.filters_per_layer[: self.num_conv_layers])
        return (
            "l" + str(self.num_conv_layers)
            + "_f" + filt_str
            + "_k" + str(self.kernel_size)
            + "_" + self.pooling_type
        )

    def to_dict(self):
        return {
            "num_conv_layers": self.num_conv_layers,
            "filters_per_layer": list(self.filters_per_layer),
            "kernel_size": self.kernel_size,
            "pooling_type": self.pooling_type,
            "learning_rate": self.learning_rate,
            "batch_size": self.batch_size,
            "epochs": self.epochs,
            "input_shape": list(self.input_shape),
            "num_classes": self.num_classes,
        }


def hyperparameter_grid(
    num_layers_options=None,
    filter_options=None,
    kernel_options=None,
    pooling_options=None,
    epochs=20,
    batch_size=32,
    input_shape=(150, 150, 3),
    num_classes=6,
):
    """Spec's 16-architecture sweep — 2 x 2 x 2 x 2 by default."""
    if num_layers_options is None:
        num_layers_options = [2, 4]
    if filter_options is None:
        # each entry is a list of per-layer filters (must be long enough for max layers)
        filter_options = [
            [32, 64, 128, 256],
            [16, 32, 64, 128],
        ]
    if kernel_options is None:
        kernel_options = [3, 5]
    if pooling_options is None:
        pooling_options = ["max", "avg"]

    configs = []
    for n in num_layers_options:
        for filters in filter_options:
            for k in kernel_options:
                for pool in pooling_options:
                    configs.append(
                        CNNConfig(
                            num_conv_layers=n,
                            filters_per_layer=list(filters),
                            kernel_size=k,
                            pooling_type=pool,
                            epochs=epochs,
                            batch_size=batch_size,
                            input_shape=input_shape,
                            num_classes=num_classes,
                        )
                    )
    return configs


def _evaluate_macro_f1(model, test_ds):
    """Run the model on test_ds and return macro F1 + (y_true, y_pred)."""
    y_true = []
    y_pred = []
    for x_batch, y_batch in test_ds:
        preds = np.argmax(model.predict(x_batch, verbose=0), axis=1)
        y_pred.extend(preds.tolist())
        if hasattr(y_batch, "numpy"):
            y_true.extend(y_batch.numpy().tolist())
        else:
            y_true.extend(list(y_batch))
    macro_f1 = f1_score(y_true, y_pred, average="macro", zero_division=0)
    return float(macro_f1), y_true, y_pred


def train_cnn_config(
    config,
    train_ds,
    val_ds,
    test_ds=None,
    output_dir=None,
    verbose=1,
    extra_callbacks=None,
):
    """Train one CNN config and return artefacts (model, history, F1, etc.)."""
    # Translate pooling_type to what build_cnn_model expects
    pooling_arg = "max" if config.pooling_type == "max" else "average"

    model = build_cnn_model(
        num_conv_layers=config.num_conv_layers,
        filters_per_layer=config.filters_per_layer,
        kernel_size=config.kernel_size,
        pooling_type=pooling_arg,
        input_shape=config.input_shape,
        num_classes=config.num_classes,
    )

    callbacks = [MacroF1Callback(val_ds)]
    if extra_callbacks is not None:
        callbacks.extend(extra_callbacks)

    start = time.time()
    history = model.fit(
        train_ds,
        validation_data=val_ds,
        epochs=config.epochs,
        verbose=verbose,
        callbacks=callbacks,
    )
    elapsed = time.time() - start

    # Convert history values to plain floats so the JSON is clean
    history_dict = {}
    raw_history = getattr(history, "history", history)
    for key, values in raw_history.items():
        try:
            history_dict[str(key)] = [float(v) for v in values]
        except TypeError:
            history_dict[str(key)] = [float(values)]

    test_macro_f1 = None
    if test_ds is not None:
        test_macro_f1, _, _ = _evaluate_macro_f1(model, test_ds)

    artefacts = {
        "variant": config.variant_name(),
        "config": config.to_dict(),
        "history": history_dict,
        "elapsed_seconds": elapsed,
        "test_macro_f1": test_macro_f1,
        "model": model,
    }

    if output_dir is not None:
        save_artefacts(artefacts, output_dir)

    return artefacts


def save_artefacts(artefacts, output_dir):
    """Persist weights, history JSON, and config JSON to output_dir."""
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
        "test_macro_f1": artefacts["test_macro_f1"],
        "config": artefacts["config"],
    }
    with open(history_path, "w", encoding="utf-8") as f:
        json.dump(history_payload, f, indent=2)

    with open(config_path, "w", encoding="utf-8") as f:
        json.dump(artefacts["config"], f, indent=2)

    return {
        "weights": weights_path,
        "history": history_path,
        "config": config_path,
    }


def train_grid(
    configs,
    train_ds,
    val_ds,
    test_ds=None,
    output_root=None,
    verbose=1,
):
    """Train every config and return {variant_name: artefacts}.

    When output_root is given, each variant's artefacts are saved under
    output_root/{variant_name}/ (well — just inside output_root, the variant
    name is already unique).
    """
    results = {}
    for config in configs:
        out_dir = None
        if output_root is not None:
            out_dir = Path(output_root)
        artefacts = train_cnn_config(
            config,
            train_ds=train_ds,
            val_ds=val_ds,
            test_ds=test_ds,
            output_dir=out_dir,
            verbose=verbose,
        )
        results[config.variant_name()] = artefacts
    return results
