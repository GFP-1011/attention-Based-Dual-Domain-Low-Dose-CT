import numpy as np
import tensorflow as tf


class Gelu(tf.keras.layers.Layer):
    def __init__(self, **kwargs):
        super(Gelu, self).__init__(**kwargs)
        self.supports_masking = True

    def call(self, inputs):
        return 0.5 * inputs * (1 + tf.tanh(tf.sqrt(2 / np.math.pi) * (inputs + 0.044715 * tf.pow(inputs, 3))))

    def get_config(self):
        config = super(Gelu, self).get_config()
        return config

    def compute_output_shape(self, input_shape):
        return input_shape