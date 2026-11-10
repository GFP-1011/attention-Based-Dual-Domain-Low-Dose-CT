import tensorflow as tf


class GAM_Attention(tf.keras.Model):
    def __init__(self, in_channels, reduction=4):
        super(GAM_Attention, self).__init__()

        self.channel_attention = tf.keras.models.Sequential(
            [
                tf.keras.layers.Dense(in_channels // reduction, activation="relu"),
                tf.keras.layers.Dense(in_channels),
            ]
        )

        self.spatial_attention = tf.keras.models.Sequential(
            [
                tf.keras.layers.Conv2D(
                    in_channels / reduction, kernel_size=7, padding="same"
                ),
                tf.keras.layers.BatchNormalization(),
                # nn.BatchNorm2d(int(in_channels / rate)),
                tf.keras.layers.ReLU(),
                # nn.ReLU(inplace=True),
                tf.keras.layers.Conv2D(in_channels, kernel_size=7, padding="same"),
                # nn.Conv2d(int(in_channels / rate), out_channels, kernel_size=7, padding=3),
                tf.keras.layers.BatchNormalization(),
            ]
        )

    def call(self, x):
        b, h, w, c = x.shape
        x_permute = tf.reshape(tf.transpose(x, [0, 3, 1, 2]), [b, -1, c])
        # x.permute(0, 2, 3, 1).view(b, -1, c)
        x_att_permute = tf.reshape(
            self.channel_attention(x_permute),
            [
                b,
                c,
                h,
                w,
            ],
        )
        x_channel_att = tf.transpose(x_att_permute, [0, 2, 3, 1])

        x = x * x_channel_att
        x_spatial_att = tf.keras.activations.sigmoid(self.spatial_attention(x))
        out = x * x_spatial_att
        return out


if __name__ == "__main__":
    sp = GAM_Attention(32)
    sp.build((2, 256, 256, 32))
    sp.summary()
