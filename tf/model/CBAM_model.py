import numpy as np
import tensorflow as tf


class Spatial_attention(tf.keras.Model):
    def __init__(self, **kwargs):
        super(Spatial_attention, self).__init__(**kwargs)
        self.conv = tf.keras.layers.Conv2D(
            filters=1,
            kernel_size=3,
            strides=(1, 1),
            padding="same",
            activation="sigmoid",
        )

    def call(self, inputs):
        avgout = tf.expand_dims(tf.reduce_mean(inputs, 3), 3)
        maxout = tf.expand_dims(tf.reduce_max(inputs, 3), 3)
        spiatial_weight = self.conv(tf.concat([avgout, maxout], 3))
        # spiatial_weight = self.conv(inputs)
        spiatial_out = tf.multiply(inputs, spiatial_weight)
        return spiatial_out


class CBAM_model(tf.keras.Model):
    def __init__(self, input_channel, reduction, **kwargs):
        super(CBAM_model, self).__init__(**kwargs)
        self.max_pool = tf.keras.layers.MaxPool2D(pool_size=256, strides=(1, 1))
        self.avg_pool = tf.keras.layers.AvgPool2D(pool_size=256, strides=(1, 1))

        self.channel_attention = tf.keras.models.Sequential(
            [
                tf.keras.layers.Dense(input_channel // reduction),
                tf.keras.layers.Conv2D(
                    input_channel // reduction, 1, padding="same", activation="relu"
                ),
                tf.keras.layers.Dense(input_channel),
            ]
        )

        self.conv = tf.keras.layers.Conv2D(
            filters=1,
            kernel_size=3,
            strides=(1, 1),
            padding="same",
            activation="sigmoid",
        )
        # self.sigmoid = tf.nn.sigmoid()

    def call(self, inputs):
        max_out = self.channel_attention(self.max_pool(inputs))
        avg_out = self.channel_attention(self.avg_pool(inputs))
        channel_out = tf.nn.sigmoid(max_out + avg_out)
        inputs = channel_out * inputs

        avgout = tf.expand_dims(tf.reduce_mean(inputs, 3), 3)
        maxout = tf.expand_dims(tf.reduce_max(inputs, 3), 3)
        spiatial_weight = self.conv(tf.concat([avgout, maxout], 3))
        # spiatial_weight = self.conv(inputs)
        spiatial_out = tf.multiply(inputs, spiatial_weight)
        return spiatial_out


if __name__ == "__main__":
    sp = CBAM_model(32, 8)
    sp.build((2, 256, 256, 32))
    sp.summary()
