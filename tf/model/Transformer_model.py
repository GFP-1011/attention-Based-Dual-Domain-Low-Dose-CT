
import numpy as np
import tensorflow as tf
from model.Gelu import Gelu


class Attention(tf.keras.layers.Layer):
    def __init__(self, num_features, num_heads, **kwargs):
        super(Attention, self).__init__(**kwargs)
        self.num_features   = num_features
        self.num_heads      = num_heads
        self.projection_dim = num_features // num_heads

    def get_config(self):
        base_config = super(Attention, self).get_config()
        return dict(list(base_config.items()))

    def compute_output_shape(self, input_shape):
        return (input_shape[0], input_shape[1], input_shape[2] // 3)

    def call(self, inputs):
        #-----------------------------------------------#
        #   获得batch_size
        #-----------------------------------------------#
        bs      = tf.shape(inputs)[0]
        #-----------------------------------------------#
        #   b, 197, 3 * 768 -> b, 197, 3, 12, 64
        #-----------------------------------------------#
        inputs  = tf.reshape(inputs, [bs, -1, 3, self.num_heads, self.projection_dim])
        #-----------------------------------------------#
        #   b, 197, 3, 12, 64 -> 3, b, 12, 197, 64
        #-----------------------------------------------#
        inputs  = tf.transpose(inputs, [2, 0, 3, 1, 4])
        #-----------------------------------------------#
        #   将query, key, value划分开
        #   query     b, 12, 197, 64
        #   key       b, 12, 197, 64
        #   value     b, 12, 197, 64
        #-----------------------------------------------#
        query, key, value = inputs[0], inputs[1], inputs[2]
        #-----------------------------------------------#
        #   b, 12, 197, 64 @ b, 12, 197, 64 = b, 12, 197, 197
        #-----------------------------------------------#
        score           = tf.matmul(query, key, transpose_b=True)
        #-----------------------------------------------#
        #   进行数量级的缩放
        #-----------------------------------------------#
        scaled_score    = score / tf.math.sqrt(tf.cast(self.projection_dim, score.dtype))
        #-----------------------------------------------#
        #   b, 12, 197, 197 -> b, 12, 197, 197
        #-----------------------------------------------#
        weights         = tf.nn.softmax(scaled_score, axis=-1)
        #-----------------------------------------------#
        #   b, 12, 197, 197 @ b, 12, 197, 64 = b, 12, 197, 64
        #-----------------------------------------------#
        value          = tf.matmul(weights, value)

        #-----------------------------------------------#
        #   b, 12, 197, 64 -> b, 197, 12, 64
        #-----------------------------------------------#
        value = tf.transpose(value, perm=[0, 2, 1, 3])
        #-----------------------------------------------#
        #   b, 197, 12, 64 -> b, 197, 768
        #-----------------------------------------------#
        output = tf.reshape(value, (bs, -1, self.num_features))
        return output


class MultiHeadSelfAttention(tf.keras.layers.Layer):
    def __init__(self, num_features, num_heads, dropout, name,**kwargs):
        super(MultiHeadSelfAttention, self).__init__(**kwargs)
        self.supports_masking = True
        self.qkv_layer = tf.keras.layers.Dense(int(num_features * 3), name = name + "qkv")
        # self.att_layer = Attention(num_features, num_heads)
        self.att_layer = Attention(num_features , num_heads)
        self.dense = []
        self.dense.append(tf.keras.layers.Dense(num_features, name=name + "proj"))
        self.dense.append(tf.keras.layers.Dropout(dropout))

    def call(self, inputs):
        qkv = self.qkv_layer(inputs)
        x = self.att_layer(qkv)
        x = self.dense[0](x)
        x = self.dense[1](x)
        return x

def MLP( d_model, diff, drop_out):
    return tf.keras.Sequential([
        tf.keras.layers.Dense(diff,activation=None),
        Gelu(),
        tf.keras.layers.Dropout(drop_out),
        tf.keras.layers.Dense(d_model),
    ])

# def MLP(y, num_features, mlp_dim, dropout, name):
#     y = tf.keras.layers.Dense(mlp_dim, name = name + "fc1")(y)
#     y = Gelu()(y)
#     y = tf.keras.layers.Dropout(dropout)(y)
#     y = tf.keras.layers.Dense(num_features, name = name + "fc2")(y)
#     return y

class TransformerBlock(tf.keras.layers.Layer):
    def __init__(self,num_features, num_heads, mlp_dim, dropout, name, **kwargs):
        super(TransformerBlock, self).__init__(**kwargs)
        self.layernorm = tf.keras.layers.LayerNormalization(epsilon=1e-6, name = name + "norm1")
        self.mul_att = MultiHeadSelfAttention(num_features, num_heads, dropout, name = name + "attn.")
        self.dense = []
        self.dense.append(tf.keras.layers.Dropout(dropout))
        self.dense.append(tf.keras.layers.Add())
        self.dense.append(tf.keras.layers.LayerNormalization(epsilon=1e-6, name = name + "norm2"))
        # self.dense.append(MLP(num_features, mlp_dim, dropout))
        self.dense.append( tf.keras.layers.Dense(mlp_dim,activation=None))
        self.dense.append(Gelu())
        self.dense.append(tf.keras.layers.Dropout(dropout))
        self.dense.append(tf.keras.layers.Dense(num_features))
        self.dense.append(tf.keras.layers.Dropout(dropout))
        self.dense.append(tf.keras.layers.Add())
        # self.dense.append(tf.keras.layers.LayerNormalization(epsilon=1e-6, name=name + "norm3"))

    def call(self, inputs):
        x = self.layernorm(inputs)
        x = self.mul_att(x)
        x = self.dense[0](x)
        x = self.dense[1]([x, inputs])
        y = x
        for i in range(6):
            y = self.dense[i+2](y)
        y = self.dense[8]([x,y])
        # y = self.dense[9](y)
        return y


class TransformerModel(tf.keras.Model):
    def __init__(self, transformer_layers,num_features ,num_heads = 8 ,expend_units=4):
        super(TransformerModel, self).__init__()
        self.transformer_layers = transformer_layers
        self.transformer_block = []
        self.num_features = num_features
        for i in range( transformer_layers):
            self.transformer_block.append(TransformerBlock(num_features=self.num_features ,num_heads=num_heads,mlp_dim=self.num_features * expend_units,dropout=0.1,name="mytrans"+str(i)))
        self.cut_sz = 6  # (5-1)/2 + (5-1)/2 + (5-1)/2 + (1-1)/2
        self.fbpModule = FbpLayer(name='fbp_layer')
        # initialize training mode and trainable variables
        self.trainMode = None
        self.trainable_var = []

    def call(self, train_batch, training=False):  # 定义正向传播过程
        if training:
            sin_in = train_batch[0]
        else:
            sin_in = train_batch
        sin_in_ex, sin_interp = self.inputModule(sin_in)
        x = sin_in-sin_interp
        for i in range(self.transformer_layers):
            x = self.transformer_block[i](x)
        x = sin_interp+x
        sin_map = tf.reshape(tf.concat([sin_in, x], 2), [sin_in.shape[0], -1, sin_in.shape[2]])
        sin_map1 =  tf.expand_dims(sin_map,3)
        fbp_out = self.fbpModule(sin_map1[:,:,0:self.num_features-3,:])
        model_out = [x, fbp_out, sin_interp]
        if training:
            return model_out, self.loss(train_batch[1],  model_out)
        else:
            return model_out
        return x

    def inputModule(self, sin_in):
        # extend input for sinLayer
        sin_in = tf.expand_dims(sin_in,3)
        axis1_ext_end = tf.gather(sin_in, range(0, self.cut_sz), axis=1)
        axis1_ext_start = tf.gather(sin_in, range(sin_in.shape[1] - self.cut_sz, sin_in.shape[1]), axis=1)
        sin_in_ex = tf.concat([axis1_ext_start, sin_in, axis1_ext_end], 1)
        # linear interpolation
        sin_interp = (sin_in + tf.gather(sin_in_ex, range(self.cut_sz + 1, self.cut_sz
                                                          + sin_in.shape[1] + 1), axis=1)) / 2

        axis2_ext = tf.zeros([sin_in_ex.shape[0], sin_in_ex.shape[1], self.cut_sz, 1], tf.float32)
        sin_ex = tf.concat([axis2_ext, sin_in_ex, axis2_ext], 2)
        sin_ex = tf.squeeze(sin_ex,3)
        sin_interp = tf.squeeze(sin_interp,3)
        return sin_ex, sin_interp
    def setTrainMode(self, sinLayers=True, fbpLayer=False, ctLayers=False):
        self.trainMode = (sinLayers, fbpLayer, ctLayers)
        # update my trainable variables
        self.trainable_var = []
        if sinLayers:
            self.trainable_var.extend(self.trainable_variables[0:len(self.trainable_variables)-3])
        if fbpLayer:
            self.trainable_var.extend(self.trainable_variables[len(self.trainable_variables)-3:len(self.trainable_variables)])
        # if ctLayers:
        #     self.trainable_var.extend(self.trainable_variables[11:30])
        print('Training mode:', self.trainMode)

    def loss(self,sin_label,model_out):
        loss =  tf.reduce_mean(tf.math.square(model_out[0] - sin_label))
        return loss


class FbpLayer(tf.keras.layers.Layer):
    def __init__(self, **kwargs):
        super(FbpLayer, self).__init__(**kwargs)
        # load AT, fbp_filter
        # _rawAT = np.load('Data/My_AT_512.npz')
        _rawAT = np.load('./Data/My_AT_256.npz')
        _AT = tf.sparse.SparseTensor(_rawAT['arr_0'].astype('int32'), _rawAT['arr_1'].astype('float32'),
                                     _rawAT['arr_2'])  # 使用index,val,shape构建稀疏反投影矩阵 #!!!!!!!!!!!!!!!!!!!!!!1
        # self.A_Matrix = tf.sparse.transpose(_AT)
        self.A_Matrix = _AT
        _out_sz = round(np.sqrt(float(self.A_Matrix.shape[1])))
        self.out_shape = (_out_sz, _out_sz)
        # FBP时使用的滤波器
        self.fbp_filter = tf.Variable(_rawAT['arr_3'].astype('float32').reshape(-1, 1, 1),
                                      name=self.name + '/fbp_filter')
        self.scale = tf.Variable([10.0], name=self.name + '/scale')  # scale for CT image
        self.bias = tf.Variable([0.0], name=self.name + '/bias')

    def call(self, sin_fan):
        sin_sz = sin_fan.shape[1] * sin_fan.shape[2] * sin_fan.shape[3]
        sin_fan_flt = tf.nn.conv1d(sin_fan, self.fbp_filter, stride=1, padding='SAME')
        # print(tf.shape(sin_fan_flt))
        fbpOut = tf.sparse.sparse_dense_matmul(tf.reshape(sin_fan_flt, [-1, sin_sz]), self.A_Matrix)
        fbpOut = tf.reshape(fbpOut, [-1, self.out_shape[0], self.out_shape[1], 1])

        output = fbpOut * self.scale + self.bias
        return output

if __name__ == '__main__':
    vit = TransformerModel(1,360)
    vit.build((1,180,360))
    vit.summary()





