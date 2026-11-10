import numpy as np
import tensorflow as tf


class UnetModel(tf.keras.Model):
    def __init__(self):
        super(UnetModel, self).__init__()
        # initialize  the sinLayers
        self.Inputs_num = 2
        self.Contracting_num = 12
        self.Expansive_num = 13
        # initialize  the unet_Layers
        self.inputlayer = []
        self.inputlayer.append(
            tf.keras.layers.Conv2D(
                32, 1, padding="same", name="inputlayer1", activation=tf.nn.leaky_relu
            )
        )
        # self.inputlayer.append(tf.keras.layers.Conv2D(64, 3, padding='same',name='inputlayer2', activation=tf.nn.leaky_relu))

        self.ContractingPathLayer = []

        self.ContractingPathLayer.append(
            tf.keras.layers.Conv2D(
                64, 3, padding="same", strides=2, activation=tf.nn.leaky_relu
            )
        )
        self.ContractingPathLayer.append(
            tf.keras.layers.Conv2D(64, 3, padding="same", activation=tf.nn.leaky_relu)
        )

        self.ContractingPathLayer.append(
            tf.keras.layers.Conv2D(
                128, 3, padding="same", strides=2, activation=tf.nn.leaky_relu
            )
        )
        self.ContractingPathLayer.append(
            tf.keras.layers.Conv2D(128, 3, padding="same", activation=tf.nn.leaky_relu)
        )

        self.ContractingPathLayer.append(
            tf.keras.layers.Conv2D(
                256, 3, padding="same", strides=2, activation=tf.nn.leaky_relu
            )
        )
        self.ContractingPathLayer.append(
            tf.keras.layers.Conv2D(256, 3, padding="same", activation=tf.nn.leaky_relu)
        )

        self.ExpansivePathLayer = []

        self.ExpansivePathLayer.append(
            tf.keras.layers.Conv2DTranspose(128, (3, 3), strides=(2, 2), padding="same")
        )  # 上采样层
        self.ExpansivePathLayer.append(
            tf.keras.layers.Conv2D(128, 3, padding="same", activation=tf.nn.leaky_relu)
        )

        self.ExpansivePathLayer.append(
            tf.keras.layers.Conv2DTranspose(64, (3, 3), strides=(2, 2), padding="same")
        )  # 上采样层
        self.ExpansivePathLayer.append(
            tf.keras.layers.Conv2D(64, 3, padding="same", activation=tf.nn.leaky_relu)
        )

        self.ExpansivePathLayer.append(
            tf.keras.layers.Conv2DTranspose(32, (3, 3), strides=(2, 2), padding="same")
        )  # 上采样层
        self.ExpansivePathLayer.append(
            tf.keras.layers.Conv2D(32, 3, padding="same", activation=tf.nn.leaky_relu)
        )

        self.ExpansivePathLayer.append(
            tf.keras.layers.Conv2D(1, 1, padding="same", activation=tf.nn.leaky_relu)
        )

    def call(self, train_batch, training=False):  # 定义正向传播过程
        # extend input for sinLayer
        if training:
            ct_in = train_batch[0]
        else:
            ct_in = train_batch
        # input block
        input_block = self.inputlayer[0](ct_in)

        conv1 = self.ContractingPathLayer[0](input_block)
        conv1 = self.ContractingPathLayer[1](conv1)

        conv2 = self.ContractingPathLayer[2](conv1)
        conv2 = self.ContractingPathLayer[3](conv2)

        conv3 = self.ContractingPathLayer[4](conv2)
        conv3 = self.ContractingPathLayer[5](conv3)

        exp2 = self.ExpansivePathLayer[0](conv3)
        exp2 = tf.concat([exp2, conv2], axis=3)
        exp2 = self.ExpansivePathLayer[1](exp2)

        exp3 = self.ExpansivePathLayer[2](exp2)
        exp3 = tf.concat([exp3, conv1], axis=3)
        exp3 = self.ExpansivePathLayer[3](exp3)

        exp4 = self.ExpansivePathLayer[4](exp3)
        exp4 = tf.concat([exp4, input_block], axis=3)
        exp4 = self.ExpansivePathLayer[5](exp4)

        ct_out = ct_in - self.ExpansivePathLayer[6](exp4)  # 最终输出

        model_out = [ct_out, ct_in]
        if training:
            return model_out, self.loss(model_out, train_batch)
        else:
            return model_out

    def loss(self, model_out, train_batch):
        # model_out = [sin_out, fbp_out, ct_out, sin_interp]
        loss = tf.reduce_mean(tf.math.square(model_out[0] - train_batch[2]))
        return loss


if __name__ == "__main__":
    CT = UnetModel()
    CT.build((1, 256, 256, 1))
    CT.summary()
