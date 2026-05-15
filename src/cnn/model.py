from __future__ import annotations

import numpy as np
from sklearn.metrics import f1_score


def build_cnn_model(
    num_conv_layers: int,
    filters_per_layer: list[int],
    kernel_size: int | tuple[int, int],
    pooling_type: str,
    input_shape: tuple = (224, 224, 3),
    num_classes: int = 6,
):
    import tensorflow as tf
    from tensorflow import keras
    from tensorflow.keras import layers

    if len(filters_per_layer) < num_conv_layers:
        raise ValueError(
            f"filters_per_layer has {len(filters_per_layer)} entries but "
            f"num_conv_layers={num_conv_layers} requires at least that many."
        )

    pool_cls = layers.MaxPooling2D if pooling_type == "max" else layers.AveragePooling2D

    inputs = keras.Input(shape=input_shape)
    x = inputs
    for i in range(num_conv_layers):
        x = layers.Conv2D(
            filters=filters_per_layer[i],
            kernel_size=kernel_size,
            padding="same",
            activation="relu",
        )(x)
        x = pool_cls(pool_size=(2, 2))(x)

    x = layers.GlobalAveragePooling2D()(x)
    x = layers.Dense(128, activation="relu")(x)
    outputs = layers.Dense(num_classes, activation="softmax")(x)

    model = keras.Model(inputs=inputs, outputs=outputs)
    model.compile(
        optimizer=keras.optimizers.Adam(learning_rate=1e-3),
        loss=keras.losses.SparseCategoricalCrossentropy(),
        metrics=["accuracy"],
    )
    return model


class MacroF1Callback:
    def __new__(cls, val_ds):
        import tensorflow as tf
        from tensorflow import keras

        class _MacroF1(keras.callbacks.Callback):
            def __init__(self, val_ds):
                super().__init__()
                self._val_ds = val_ds

            def on_epoch_end(self, epoch, logs=None):
                y_true, y_pred = [], []
                for x_batch, y_batch in self._val_ds:
                    preds = np.argmax(
                        self.model.predict(x_batch, verbose=0), axis=1
                    )
                    y_pred.extend(preds.tolist())
                    y_true.extend(
                        y_batch.numpy().tolist()
                        if hasattr(y_batch, "numpy")
                        else list(y_batch)
                    )
                macro_f1 = f1_score(y_true, y_pred, average="macro", zero_division=0)
                if logs is not None:
                    logs["val_macro_f1"] = macro_f1
                print(f"  val_macro_f1: {macro_f1:.4f}")

        return _MacroF1(val_ds)
