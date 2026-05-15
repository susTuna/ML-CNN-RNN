"""[BONUS] Init-inject decoder (Tanti et al., 2017).

Same captioning task as the pre-inject decoder, but the projected CNN
feature is fed as the initial hidden state of the recurrent stack instead
of as a sequence element. For LSTM we also project to the initial cell
state. Each layer gets its own projection so hidden_dim can differ from
feat_dim.
"""

from .model import (
    masked_sparse_categorical_accuracy,
    masked_sparse_categorical_crossentropy,
)


def build_decoder_init_inject(
    vocab_size,
    embed_dim,
    hidden_dim,
    feat_dim,
    seq_len,
    num_layers=1,
    rnn_type="lstm",
    learning_rate=1e-3,
    dropout=0.0,
    recurrent_dropout=0.0,
):
    """Build and compile a Keras init-inject captioning decoder.

    Output length equals seq_len directly (no image timestep to drop).
    """
    from tensorflow import keras
    from tensorflow.keras import layers

    rnn_type = rnn_type.lower()
    if rnn_type not in ("lstm", "rnn", "simplernn"):
        raise ValueError("rnn_type must be 'lstm' or 'rnn', got " + repr(rnn_type))

    if rnn_type == "lstm":
        rnn_layer_cls = layers.LSTM
    else:
        rnn_layer_cls = layers.SimpleRNN

    image_input = keras.Input(shape=(feat_dim,), name="image_feature")
    caption_input = keras.Input(shape=(seq_len,), dtype="int32", name="decoder_input")

    embedded = layers.Embedding(
        input_dim=vocab_size,
        output_dim=embed_dim,
        mask_zero=False,
        name="token_embedding",
    )(caption_input)

    x = embedded
    for i in range(num_layers):
        # one h0 projection per layer (and one c0 if LSTM)
        h0 = layers.Dense(
            hidden_dim, activation="tanh", name="init_h_" + str(i + 1)
        )(image_input)
        if rnn_type == "lstm":
            c0 = layers.Dense(
                hidden_dim, activation="tanh", name="init_c_" + str(i + 1)
            )(image_input)
            x = rnn_layer_cls(
                hidden_dim,
                return_sequences=True,
                dropout=dropout,
                recurrent_dropout=recurrent_dropout,
                name=rnn_type + "_" + str(i + 1),
            )(x, initial_state=[h0, c0])
        else:
            x = rnn_layer_cls(
                hidden_dim,
                return_sequences=True,
                dropout=dropout,
                recurrent_dropout=recurrent_dropout,
                name=rnn_type + "_" + str(i + 1),
            )(x, initial_state=h0)

    outputs = layers.Dense(vocab_size, activation="softmax", name="vocab_output")(x)

    model = keras.Model(
        inputs=[image_input, caption_input],
        outputs=outputs,
        name="caption_decoder_" + rnn_type + "_init_inject",
    )

    model.compile(
        optimizer=keras.optimizers.Adam(learning_rate=learning_rate),
        loss=masked_sparse_categorical_crossentropy(),
        metrics=[masked_sparse_categorical_accuracy()],
    )
    return model


def build_decoder_init_inject_summary(
    vocab_size,
    embed_dim,
    hidden_dim,
    feat_dim,
    seq_len,
    num_layers,
    rnn_type,
):
    return {
        "architecture": "init_inject",
        "rnn_type": rnn_type.lower(),
        "vocab_size": int(vocab_size),
        "embed_dim": int(embed_dim),
        "hidden_dim": int(hidden_dim),
        "feat_dim": int(feat_dim),
        "seq_len": int(seq_len),
        "num_layers": int(num_layers),
    }
