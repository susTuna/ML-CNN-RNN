import tensorflow as tf
from tensorflow import keras


class LocallyConnected2D(keras.layers.Layer):
    def __init__(
        self,
        filters,
        kernel_size,
        strides=(1, 1),
        padding="valid",
        activation=None,
        **kwargs,
    ):
        super().__init__(**kwargs)
        if isinstance(kernel_size, int):
            kernel_size = (kernel_size, kernel_size)
        if isinstance(strides, int):
            strides = (strides, strides)
        self.filters = filters
        self.kernel_size = tuple(kernel_size)
        self.strides = tuple(strides)
        self.padding = padding.lower()
        if self.padding != "valid":
            # LC2D in legacy Keras only ever supported 'valid' for the
            # output-shape math we rely on below.
            raise ValueError("LocallyConnected2D only supports padding='valid'.")
        self.activation = keras.activations.get(activation)

        # set after build()
        self.out_rows = None
        self.out_cols = None
        self.kernel = None
        self.bias = None

    def build(self, input_shape):
        # input_shape: (batch, H, W, C_in)
        if len(input_shape) != 4:
            raise ValueError(
                "LocallyConnected2D expects a 4-D input (batch, H, W, C). Got: "
                + str(input_shape)
            )
        _, H, W, C_in = input_shape
        kH, kW = self.kernel_size
        sH, sW = self.strides

        self.out_rows = (H - kH) // sH + 1
        self.out_cols = (W - kW) // sW + 1
        n_pos = self.out_rows * self.out_cols

        self.kernel = self.add_weight(
            name="kernel",
            shape=(n_pos, kH * kW * C_in, self.filters),
            initializer="glorot_uniform",
            trainable=True,
        )
        self.bias = self.add_weight(
            name="bias",
            shape=(n_pos, self.filters),
            initializer="zeros",
            trainable=True,
        )
        self._c_in = C_in
        super().build(input_shape)

    def call(self, inputs):
        kH, kW = self.kernel_size
        sH, sW = self.strides

        # extract non-overlapping (kH, kW) patches at each position
        # -> shape (batch, out_rows, out_cols, kH*kW*C_in)
        patches = tf.image.extract_patches(
            inputs,
            sizes=[1, kH, kW, 1],
            strides=[1, sH, sW, 1],
            rates=[1, 1, 1, 1],
            padding="VALID",
        )

        # flatten the spatial axis so we can do a per-position matmul
        # patches: (batch, n_pos, kH*kW*C_in)
        n_pos = self.out_rows * self.out_cols
        patches = tf.reshape(patches, [-1, n_pos, kH * kW * self._c_in])

        # out[b, p, c] = sum_k patches[b, p, k] * kernel[p, k, c]
        out = tf.einsum("bpk,pkc->bpc", patches, self.kernel) + self.bias
        out = tf.reshape(out, [-1, self.out_rows, self.out_cols, self.filters])

        if self.activation is not None:
            out = self.activation(out)
        return out

    def compute_output_shape(self, input_shape):
        return (input_shape[0], self.out_rows, self.out_cols, self.filters)

    @property
    def output_shape(self):
        # mirrors the attribute the from-scratch loader reads
        return (None, self.out_rows, self.out_cols, self.filters)

    def get_config(self):
        cfg = super().get_config()
        cfg.update(
            {
                "filters": self.filters,
                "kernel_size": self.kernel_size,
                "strides": self.strides,
                "padding": self.padding,
                "activation": keras.activations.serialize(self.activation),
            }
        )
        return cfg
