import numpy as np
import tensorflow as tf


class Gelu(tf.keras.layers.Layer):
    def __init__(self, **kwargs):
        super(Gelu, self).__init__(**kwargs)
        self.supports_masking = True

    def call(self, inputs):
        return (
            0.5
            * inputs
            * (
                1
                + tf.tanh(
                    tf.sqrt(2 / np.math.pi) * (inputs + 0.044715 * tf.pow(inputs, 3))
                )
            )
        )

    def get_config(self):
        config = super(Gelu, self).get_config()
        return config

    def compute_output_shape(self, input_shape):
        return input_shape


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
        # encoded = self.projection(patch)
        return encoded


class fusion_MA_model(tf.keras.Model):
    def __init__(
        self,
        num_features,
        num_heads=8,
        expend_units=4,
        patch_size=16,
        num_patches=256,
        projection_dim=360,
        dropout=0.1,
        name="MA",
        **kwargs
    ):
        super(fusion_MA_model, self).__init__(**kwargs)
        self.patches = Patches(patch_size)
        self.patchesencoder = PatchEncoder(num_patches, projection_dim)
        # self.layernorm = tf.keras.layers.LayerNormalization(epsilon=1e-6, name = name + "norm1")
        # self.mul_att = MultiHeadSelfAttention(num_features, num_heads, dropout, name = name + "attn.")
        self.mul_att = []
        self.mul_att.append(
            MultiHeadSelfAttention(
                projection_dim, num_heads, dropout, name=name + "attn."
            )
        )
        # self.mul_att.append(MultiHeadSelfAttention(projection_dim, num_heads, dropout, name=name + "attn."))

    def call(self, train_batch, training=False):
        if training:
            sin_in = train_batch[0]
        else:
            sin_in = train_batch
        # sin_in = tf.keras.layers.Conv2D(8,3,padding='same')(sin_in)
        patches = self.patches(sin_in)
        encoded_patches = self.patchesencoder(patches)
        x = encoded_patches
        # x = self.layernorm(x)
        x = self.mul_att[0](x)
        # x = self.mul_att[1](x)
        # x = self.dense[0](x)
        # x = self.dense[1]([x, encoded_patches])
        x = tf.keras.layers.Dense(360)(x)
        x = tf.keras.layers.Dense(256)(x)

        y = self.fold(x, 16)

        # x = tf.expand_dims(x,3)

        # y = self.fold(y,16)
        return y

    def fold(self, input_tensor, patches):
        # b, h, w, c = input_tensor.shape()
        b = 2
        h = 256
        input_tensor = tf.reshape(input_tensor, [b, h, patches, patches])
        z = tf.transpose(
            tf.reshape(tf.transpose(input_tensor, [0, 2, 3, 1]), [b, h, h, 1]),
            [0, 2, 1, 3],
        )
        h = tf.image.extract_patches(
            z,
            [1, patches, patches, 1],
            [1, patches, patches, 1],
            [1, 1, 1, 1],
            padding="VALID",
        )
        p = tf.reshape(h, [b, -1, patches * patches])
        p = tf.expand_dims(p, 3)
        return p


if __name__ == "__main__":
    MA = fusion_MA_model(360)
    MA.build((2, 256, 256, 2))
    MA.summary()
