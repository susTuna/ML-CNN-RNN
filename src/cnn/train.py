from __future__ import annotations

import json
from itertools import product
from pathlib import Path

import numpy as np


SWEEP_GRID = {
    "conv_layers": [2, 4],
    "filters": [[32, 64, 64, 64], [64, 128, 128, 128]],
    "kernel_size": [3, 5],
    "pooling": ["max", "average"],
}


def train_model(
    model,
    train_ds,
    val_ds,
    epochs: int = 20,
    model_save_path: str | Path = "models/cnn/model",
) -> dict:
    import tensorflow as tf
    from tensorflow import keras

    from .model import MacroF1Callback

    save_path = Path(model_save_path)
    save_path.parent.mkdir(parents=True, exist_ok=True)

    callbacks = [
        MacroF1Callback(val_ds),
        keras.callbacks.ModelCheckpoint(
            str(save_path) + ".weights.h5",
            save_weights_only=True,
            monitor="val_accuracy",
            save_best_only=True,
            verbose=0,
        ),
        keras.callbacks.EarlyStopping(
            monitor="val_accuracy",
            patience=5,
            restore_best_weights=True,
            verbose=1,
        ),
    ]

    history = model.fit(
        train_ds,
        validation_data=val_ds,
        epochs=epochs,
        callbacks=callbacks,
        verbose=1,
    )

    history_dict = {k: [float(v) for v in vals] for k, vals in history.history.items()}
    with open(str(save_path) + "_history.json", "w") as f:
        json.dump(history_dict, f, indent=2)

    return history_dict


def run_hyperparameter_sweep(
    train_ds,
    val_ds,
    save_dir: str | Path = "models/cnn",
    epochs: int = 20,
) -> list[dict]:
    import tensorflow as tf

    from .model import build_cnn_model

    save_dir = Path(save_dir)
    results = []

    combos = list(
        product(
            SWEEP_GRID["conv_layers"],
            SWEEP_GRID["filters"],
            SWEEP_GRID["kernel_size"],
            SWEEP_GRID["pooling"],
        )
    )
    print(f"Starting hyperparameter sweep: {len(combos)} combinations")

    for idx, (conv_layers, filters, ksize, pooling) in enumerate(combos, 1):
        config = {
            "conv_layers": conv_layers,
            "filters": filters,
            "kernel_size": ksize,
            "pooling": pooling,
        }
        name = f"cnn_conv{conv_layers}_f{filters[0]}_k{ksize}_{pooling}pool"
        print(f"\n[{idx}/{len(combos)}] {name}")

        model = build_cnn_model(
            num_conv_layers=conv_layers,
            filters_per_layer=filters,
            kernel_size=ksize,
            pooling_type=pooling,
        )

        save_path = save_dir / name
        history = train_model(
            model,
            train_ds,
            val_ds,
            epochs=epochs,
            model_save_path=save_path,
        )

        results.append(
            {"config": config, "history": history, "save_path": str(save_path)}
        )

        tf.keras.backend.clear_session()

    # Save combined sweep results
    save_dir.mkdir(parents=True, exist_ok=True)
    with open(save_dir / "sweep_results.json", "w") as f:
        json.dump(results, f, indent=2)

    return results
