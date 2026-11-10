import numpy as np
import tensorflow as tf


class ChannelAttention(tf.keras.Model):
    def __init__(self, channel, reduction=16):
        super(ChannelAttention, self).__init__()

        self.avg_pool = tf.keras.layers.AvgPool2D(pool_size=256, strides=(1, 1))

        self.channel_attention = tf.keras.models.Sequential(
            [
                tf.keras.layers.Dense(channel // reduction, activation="relu"),
                tf.keras.layers.Dense(channel),
            ]
        )

    def call(self, x):
        avg = tf.reshape(self.avg_pool(x), [x.shape[0], -1])
        # out = self.channel_attention(avg).unsqueeze(2).unsqueeze(3).expand_as(x)
        out = tf.expand_dims(tf.expand_dims(self.channel_attention(avg), 2), 3)
        out = tf.tile(tf.transpose(out, [0, 2, 3, 1]), [1, x.shape[1], x.shape[2], 1])
        return out


class SpatialAttention(tf.keras.Model):
    def __init__(self, channel, reduction=16, dilation_conv_num=2, dilation_rate=4):
        super(SpatialAttention, self).__init__()
        mid_channel = channel // reduction
        self.reduce_conv = tf.keras.models.Sequential(
            [
                tf.keras.layers.Conv2D(mid_channel, kernel_size=1),
                tf.keras.layers.BatchNormalization(),
                tf.keras.layers.ReLU(),
            ]
        )
        # dilation_convs_list = []
        # for i in range(dilation_conv_num):
        #     dilation_convs_list.append(
        #         tf.keras.layers.Conv2D(mid_channel,  kernel_size=3, padding=dilation_rate, dilation_rate=dilation_rate))
        #     dilation_convs_list.append(tf.keras.layers.BatchNormalization())
        #     dilation_convs_list.append( tf.keras.layers.ReLU())
        # self.dilation_convs =tf.keras.models.Sequential(*dilation_convs_list)
        self.final_conv = tf.keras.layers.Conv2D(1, kernel_size=1)

    def call(self, x):
        y = self.reduce_conv(x)
        # x = self.dilation_convs(y)
        out = self.final_conv(y)
        out = tf.tile(out, [1, 1, 1, x.shape[3]])  # .expand_as(x)
        return out


class BAM(tf.keras.Model):
    """
    BAM: Bottleneck Attention Module
    """

    def __init__(self, channel):
        super(BAM, self).__init__()
        self.channel_attention = ChannelAttention(channel)
        self.spatial_attention = SpatialAttention(channel)
        # self.sigmoid = tf.keras.activations.sigmoid()

    def call(self, x):
        att = 1 + tf.keras.activations.sigmoid(
            self.channel_attention(x) * self.spatial_attention(x)
        )
        return att


if __name__ == "__main__":
    sp = BAM(32)
    sp.build((2, 256, 256, 32))
    sp.summary()
