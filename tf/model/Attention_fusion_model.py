import numpy as np
import tensorflow as tf
from model.CBAM_model import CBAM_model
from model.Bam_model import BAM
from model.GAM_model import GAM_Attention


class Attention_Funsion_model(tf.keras.Model):
    def __init__(self, expand_channel, **kwargs):
        super(Attention_Funsion_model, self).__init__(**kwargs)

        # self.attentionModule = CBAM_model(expand_channel,8)
        # self.attentionModule = BAM(expand_channel)
        self.attentionModule = GAM_Attention(expand_channel)
        self.fusionModule = []
        self.fusionModule.append(
            tf.keras.layers.Conv2D(
                32, 5, name="fusion", padding="same", activation=tf.nn.relu
            )
        )
        self.fusionModule.append(
            tf.keras.layers.Conv2D(1, kernel_size=1, name="fusion", padding="same")
        )

    def call(self, train_batch, training=False):
        if training:
            input = train_batch[0]
        else:
            input = train_batch
        fusion_in = self.fusionModule[0](input)
        attention_out = self.attentionModule(fusion_in)
        fusion_out = self.fusionModule[1](attention_out)
        if training:
            return fusion_out, self.loss(train_batch[2], fusion_out)
        else:
            return fusion_out

    def loss(self, ct_label, model_out):
        loss = tf.reduce_mean(tf.math.square(model_out - ct_label))
        return loss


if __name__ == "__main__":
    CT = Attention_Funsion_model(32)
    CT.build((1, 256, 256, 2))
    CT.summary()
