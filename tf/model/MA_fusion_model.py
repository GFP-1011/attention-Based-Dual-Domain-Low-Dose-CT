import numpy as np
import tensorflow as tf
from model.Gelu import Gelu


class Patches(tf.keras.layers.Layer):
    """Gets some images and returns the patches for each image"""

    def __init__(self, patch_size):
        super(Patches, self).__init__()
        self.patch_size = patch_size

    def call(self, images):
        batch_size = tf.shape(images)[0]
        patches = tf.image.extract_patches(
            images=images,
            sizes=[1, self.patch_size, self.patch_size, 1],
            strides=[1, self.patch_size, self.patch_size, 1],
            rates=[1, 1, 1, 1],
            padding="VALID",
        )
        patch_dims = patches.shape[-1]
        patches = tf.reshape(patches, [batch_size, -1, patch_dims])
        return patches


class PatchEncoder(tf.keras.layers.Layer):
    """Adding (learnable) positional encoding to input patches"""

    def __init__(self, num_patches, projection_dim):
        super(PatchEncoder, self).__init__()
        self.num_patches = num_patches
        self.projection = tf.keras.layers.Dense(units=projection_dim)
        self.position_embedding = tf.keras.layers.Embedding(
            input_dim=num_patches, output_dim=projection_dim
        )

    def call(self, patch):
        positions = tf.range(start=0, limit=self.num_patches, delta=1)
        encoded = self.projection(patch) + self.position_embedding(positions)
        return encoded


class Attention(tf.keras.layers.Layer):
    def __init__(self, num_features, num_heads, **kwargs):
        super(Attention, self).__init__(**kwargs)
        self.num_features = num_features
        self.num_heads = num_heads
        self.projection_dim = num_features // num_heads

    def get_config(self):
        base_config = super(Attention, self).get_config()
        return dict(list(base_config.items()))

    def compute_output_shape(self, input_shape):
        return (input_shape[0], input_shape[1], input_shape[2] // 3)

    def call(self, inputs):
        # -----------------------------------------------#
        #   获得batch_size
        # -----------------------------------------------#
        bs = tf.shape(inputs)[0]
        # -----------------------------------------------#
        #   b, 197, 3 * 768 -> b, 197, 3, 12, 64
        # -----------------------------------------------#
        inputs = tf.reshape(inputs, [bs, -1, 3, self.num_heads, self.projection_dim])
        # -----------------------------------------------#
        #   b, 197, 3, 12, 64 -> 3, b, 12, 197, 64
        # -----------------------------------------------#
        inputs = tf.transpose(inputs, [2, 0, 3, 1, 4])
        # -----------------------------------------------#
        #   将query, key, value划分开
        #   query     b, 12, 197, 64
        #   key       b, 12, 197, 64
        #   value     b, 12, 197, 64
        # -----------------------------------------------#
        query, key, value = inputs[0], inputs[1], inputs[2]
        # -----------------------------------------------#
        #   b, 12, 197, 64 @ b, 12, 197, 64 = b, 12, 197, 197
        # -----------------------------------------------#
        score = tf.matmul(query, key, transpose_b=True)
        # -----------------------------------------------#
        #   进行数量级的缩放
        # -----------------------------------------------#
        scaled_score = score / tf.math.sqrt(tf.cast(self.projection_dim, score.dtype))
        # -----------------------------------------------#
        #   b, 12, 197, 197 -> b, 12, 197, 197
        # -----------------------------------------------#
        weights = tf.nn.softmax(scaled_score, axis=-1)
        # -----------------------------------------------#
        #   b, 12, 197, 197 @ b, 12, 197, 64 = b, 12, 197, 64
        # -----------------------------------------------#
        value = tf.matmul(weights, value)

        # -----------------------------------------------#
        #   b, 12, 197, 64 -> b, 197, 12, 64
        # -----------------------------------------------#
        value = tf.transpose(value, perm=[0, 2, 1, 3])
        # -----------------------------------------------#
        #   b, 197, 12, 64 -> b, 197, 768
        # -----------------------------------------------#
        output = tf.reshape(value, (bs, -1, self.num_features))
        return output


class MultiHeadSelfAttention(tf.keras.layers.Layer):
    def __init__(self, num_features, num_heads, dropout, name, **kwargs):
        super(MultiHeadSelfAttention, self).__init__(**kwargs)
        self.supports_masking = True
        self.qkv_layer = tf.keras.layers.Dense(int(num_features * 3), name=name + "qkv")
        # self.att_layer = Attention(num_features, num_heads)
        self.att_layer = Attention(num_features, num_heads)
        self.dense = []
        self.dense.append(tf.keras.layers.Dense(num_features, name=name + "proj"))
        self.dense.append(tf.keras.layers.Dropout(dropout))

    def call(self, inputs):
        qkv = self.qkv_layer(inputs)
        x = self.att_layer(qkv)
        x = self.dense[0](x)
        x = self.dense[1](x)
        return x


def MLP(d_model, diff, drop_out):
    return tf.keras.Sequential(
        [
            tf.keras.layers.Dense(diff, activation=None),
            Gelu(),
            tf.keras.layers.Dropout(drop_out),
            tf.keras.layers.Dense(d_model),
        ]
    )


# def MLP(y, num_features, mlp_dim, dropout, name):
#     y = tf.keras.layers.Dense(mlp_dim, name = name + "fc1")(y)
#     y = Gelu()(y)
#     y = tf.keras.layers.Dropout(dropout)(y)
#     y = tf.keras.layers.Dense(num_features, name = name + "fc2")(y)
#     return y


class TransformerBlock(tf.keras.layers.Layer):
    def __init__(self, num_features, num_heads, mlp_dim, dropout, name, **kwargs):
        super(TransformerBlock, self).__init__(**kwargs)

        self.layernorm = tf.keras.layers.LayerNormalization(
            epsilon=1e-6, name=name + "norm1"
        )
        self.mul_att = MultiHeadSelfAttention(
            num_features, num_heads, dropout, name=name + "attn."
        )
        # self.dense = []
        # self.dense.append(tf.keras.layers.Dropout(dropout))
        # self.dense.append(tf.keras.layers.Add())
        # self.dense.append(tf.keras.layers.LayerNormalization(epsilon=1e-6, name = name + "norm2"))
        # # self.dense.append(MLP(num_features, mlp_dim, dropout))
        # self.dense.append( tf.keras.layers.Dense(mlp_dim,activation=None))
        # self.dense.append(Gelu())
        # self.dense.append(tf.keras.layers.Dropout(dropout))
        # self.dense.append(tf.keras.layers.Dense(num_features))
        # self.dense.append(tf.keras.layers.Dropout(dropout))
        # self.dense.append(tf.keras.layers.Add())
        # self.dense.append(tf.keras.layers.LayerNormalization(epsilon=1e-6, name=name + "norm3"))

    def call(self, inputs):

        x = self.layernorm(inputs)
        y = self.mul_att(x)

        # x = self.dense[0](x)
        # x = self.dense[1]([x, inputs])
        # y = x
        # for i in range(6):
        #     y = self.dense[i+2](y)
        # y = self.dense[8]([x,y])
        # y = self.dense[9](y)
        return y


class TransformerModel(tf.keras.Model):
    def __init__(
        self,
        transformer_layers,
        num_features,
        num_heads=8,
        expend_units=4,
        patch_size=16,
        num_patches=256,
        projection_dim=360,
    ):
        super(TransformerModel, self).__init__()
        self.patches = Patches(patch_size)
        self.patchesencoder = PatchEncoder(num_patches, projection_dim)
        self.transformer_layers = transformer_layers
        self.transformer_block = []
        self.num_features = num_features
        for i in range(transformer_layers):
            self.transformer_block.append(
                TransformerBlock(
                    num_features=self.num_features,
                    num_heads=num_heads,
                    mlp_dim=self.num_features * expend_units,
                    dropout=0.1,
                    name="mytrans" + str(i),
                )
            )
        self.cut_sz = 6  # (5-1)/2 + (5-1)/2 + (5-1)/2 + (1-1)/2
        # initialize training mode and trainable variables
        self.trainMode = None
        self.trainable_var = []

    def call(self, train_batch, training=False):  # 定义正向传播过程
        if training:
            sin_in = train_batch[0]
        else:
            sin_in = train_batch

        patches = self.patches(sin_in)
        encoded_patches = self.patchesencoder(patches)
        x = encoded_patches
        for i in range(self.transformer_layers):
            x = self.transformer_block[i](x)
        # x = encoded_patches+x

        y = tf.keras.layers.Conv2D(
            32, 5, name="fusion", padding="same", activation=tf.nn.relu
        )(x)
        y = tf.keras.layers.Conv2D(1, kernel_size=1, name="fusion", padding="same")(y)
        model_out = y

        if training:
            return model_out, self.loss(train_batch[1], model_out)
        else:
            return model_out
        return x

    def setTrainMode(self, sinLayers=True, fbpLayer=False, ctLayers=False):
        self.trainMode = (sinLayers, fbpLayer, ctLayers)
        # update my trainable variables
        self.trainable_var = []
        if sinLayers:
            self.trainable_var.extend(
                self.trainable_variables[0 : len(self.trainable_variables) - 3]
            )
        if fbpLayer:
            self.trainable_var.extend(
                self.trainable_variables[
                    len(self.trainable_variables) - 3 : len(self.trainable_variables)
                ]
            )
        # if ctLayers:
        #     self.trainable_var.extend(self.trainable_variables[11:30])
        print("Training mode:", self.trainMode)

    def loss(self, sin_label, model_out):
        loss = tf.reduce_mean(tf.math.square(model_out[0] - sin_label))
        return loss


if __name__ == "__main__":
    vit = TransformerModel(1, 360)
    vit.build((1, 256, 256, 3))
    vit.summary()
