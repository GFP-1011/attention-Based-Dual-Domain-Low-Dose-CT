import numpy as np
import tensorflow as tf


class ARFB_block(tf.keras.layers.Layer):
    def __init__(self, input_channel, **kwargs):
        super(ARFB_block, self).__init__(**kwargs)
        self.RU_layer = []
        self.RU_layer.append(
            tf.keras.layers.Conv2D(input_channel // 2, 3, padding="same")
        )
        self.RU_layer.append(tf.keras.layers.Conv2D(input_channel, 3, padding="same"))
        self.RU_layer.append(
            tf.keras.layers.Conv2D(input_channel // 2, 3, padding="same")
        )
        self.RU_layer.append(tf.keras.layers.Conv2D(input_channel, 3, padding="same"))

        self.conv_layer = []
        self.conv_layer.append(
            tf.keras.layers.Conv2D(input_channel * 2, 1, padding="same")
        )
        self.conv_layer.append(tf.keras.layers.Conv2D(input_channel, 3, padding="same"))

    def call(self, input):
        ru_1 = self.RU_layer[0](input)
        ru_1 = self.RU_layer[1](ru_1)
        ru_1 = ru_1 + input

        ru_2 = self.RU_layer[2](ru_1)
        ru_2 = self.RU_layer[3](ru_2)
        ru_2 = ru_2 + ru_1

        ru = tf.concat([ru_1, ru_2], 3)
        conv = self.conv_layer[0](ru)
        conv = self.conv_layer[1](conv)
        out = conv + input
        return out


class HFM_block(tf.keras.layers.Layer):
    def __init__(self, input_channel, **kwargs):
        super(HFM_block, self).__init__(**kwargs)

        self.avg = tf.keras.layers.AvgPool2D((2, 2))
        self.up = tf.keras.layers.Conv2DTranspose(
            input_channel, (3, 3), strides=(2, 2), padding="same"
        )

    def call(self, input):
        x = self.avg(input)
        x = self.up(x)
        x = x + input
        return x


class Funsion_model(tf.keras.Model):
    def __init__(self, input_channel, **kwargs):
        super(Funsion_model, self).__init__(**kwargs)
        self.ARFB = []
        self.num_ARFB = 2
        for i in range(self.num_ARFB):
            self.ARFB.append(ARFB_block(input_channel))

        self.HFM = HFM_block(input_channel)

        self.Conv = tf.keras.layers.Conv2D(1, 1, padding="same", activation=None)

        self.fusionModule = []
        self.fusionModule.append(
            tf.keras.layers.Conv2D(
                32, 5, name="fusion", padding="same", activation=tf.nn.relu
            )
        )
        self.fusionModule.append(
            tf.keras.layers.Conv2D(1, kernel_size=1, name="fusion", padding="same")
        )

    def call(self, input, training=False):
        # if training:
        #     input = train_batch[0]
        # else:
        #     input = train_batch
        # x = tf.keras.layers.Conv2D(32,3,padding='same',activation='relu')(input)

        fusion_out = self.fusionModule[0](input)
        fusion_out = self.fusionModule[1](fusion_out)
        # x = self.ARFB[0](x)
        # x = self.HFM(x)
        # x = self.ARFB[1](x)
        # x = self.ARFB[2](x)
        # x = self.ARFB[3](x)
        # y = self.Conv(x)

        return fusion_out


if __name__ == "__main__":
    CT = Funsion_model(2)
    CT.build((1, 256, 256, 2))
    CT.summary()
