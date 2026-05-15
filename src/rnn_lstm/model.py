"""Keras pre-inject decoder for image captioning (Show and Tell, Vinyals 2015).

The CNN feature is projected to embed_dim and inserted as the very first
timestep of the recurrent decoder, before the embedded <start> token.
Hidden state is zero-initialised. Training uses teacher forcing with masked
sparse categorical crossentropy so <pad> positions don't count toward loss.
"""


def build_decoder_pre_inject(
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
    """Build and compile a Keras pre-inject captioning decoder.

    seq_len is the length of the decoder input sequence (= max_len - 1 if the
    raw caption was padded to max_len). The recurrent stack actually sees
    seq_len + 1 timesteps (projected image + tokens); the first output is
    dropped so the output shape matches the target.
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

    # Project CNN feature to embed_dim and reshape into a length-1 sequence
    projected = layers.Dense(embed_dim, name="image_projection")(image_input)
    projected = layers.Reshape((1, embed_dim), name="image_proj_reshape")(projected)

    # Token embeddings for the caption prefix
    embedded = layers.Embedding(
        input_dim=vocab_size,
        output_dim=embed_dim,
        mask_zero=False,
        name="token_embedding",
    )(caption_input)

    # Concatenate: [projected_image, emb(<start>), emb(S_0), ...]
    merged = layers.Concatenate(axis=1, name="prepend_image")([projected, embedded])

    x = merged
    for i in range(num_layers):
        x = rnn_layer_cls(
            hidden_dim,
            return_sequences=True,
            dropout=dropout,
            recurrent_dropout=recurrent_dropout,
            name=rnn_type + "_" + str(i + 1),
        )(x)

    # Drop the image-timestep output so output length matches target length
    x = layers.Lambda(lambda t: t[:, 1:, :], name="drop_image_step")(x)
    outputs = layers.Dense(vocab_size, activation="softmax", name="vocab_output")(x)

    model = keras.Model(
        inputs=[image_input, caption_input],
        outputs=outputs,
        name="caption_decoder_" + rnn_type + "_pre_inject",
    )

    model.compile(
        optimizer=keras.optimizers.Adam(learning_rate=learning_rate),
        loss=masked_sparse_categorical_crossentropy(),
        metrics=[masked_sparse_categorical_accuracy()],
    )
    return model


def masked_sparse_categorical_crossentropy(pad_id=0):
    """Sparse categorical CE that ignores <pad> positions."""
    import tensorflow as tf

    base = tf.keras.losses.SparseCategoricalCrossentropy(
        from_logits=False,
        reduction=tf.keras.losses.Reduction.NONE,
    )

    def loss_fn(y_true, y_pred):
        y_true = tf.cast(y_true, tf.int32)
        mask = tf.cast(tf.not_equal(y_true, pad_id), tf.float32)
        per_token = base(y_true, y_pred)
        per_token = per_token * mask
        # avoid divide-by-zero when a batch happens to be all <pad>
        denom = tf.reduce_sum(mask) + 1e-8
        return tf.reduce_sum(per_token) / denom

    loss_fn.__name__ = "masked_sparse_categorical_crossentropy"
    return loss_fn


def masked_sparse_categorical_accuracy(pad_id=0):
    """Token-level accuracy ignoring <pad> positions."""
    import tensorflow as tf

    def metric_fn(y_true, y_pred):
        y_true = tf.cast(y_true, tf.int32)
        preds = tf.cast(tf.argmax(y_pred, axis=-1), tf.int32)
        mask = tf.cast(tf.not_equal(y_true, pad_id), tf.float32)
        correct = tf.cast(tf.equal(preds, y_true), tf.float32) * mask
        return tf.reduce_sum(correct) / (tf.reduce_sum(mask) + 1e-8)

    metric_fn.__name__ = "masked_accuracy"
    return metric_fn


def build_decoder_pre_inject_summary(
    vocab_size,
    embed_dim,
    hidden_dim,
    feat_dim,
    seq_len,
    num_layers,
    rnn_type,
):
    """Architecture descriptor dict — used for the config JSON dump."""
    return {
        "architecture": "pre_inject",
        "rnn_type": rnn_type.lower(),
        "vocab_size": int(vocab_size),
        "embed_dim": int(embed_dim),
        "hidden_dim": int(hidden_dim),
        "feat_dim": int(feat_dim),
        "seq_len": int(seq_len),
        "num_layers": int(num_layers),
    }
