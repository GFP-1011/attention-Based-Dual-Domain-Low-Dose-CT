from model.Transformer_CNN_model import TransformerModel
from model.Unet_CT_model import UnetModel
from model.GAM_model import GAM_Attention
from skimage.metrics import structural_similarity as s_ssim
import tensorflow as tf
import numpy as np
import matplotlib.pyplot as plt
import cv2


class dual_parallel_model(tf.keras.Model):
    def __init__(self, transformer_layers,num_features ,sparse_scale= 2,ckpt_sine = None,ckpt_ct=None):
        super(dual_parallel_model, self).__init__()
        self.num_features = num_features
        self.SineModel = TransformerModel(transformer_layers,num_features,sparse_scale)
        if(ckpt_sine!=None):
            self.SineModel.load_weights(ckpt_sine)
        self.CtModel = UnetModel()
        if(ckpt_ct!=None):
            self.CtModel.load_weights(ckpt_ct)
        self.Fbp = FbpLayer()
        # self.Fusion_model = CBAM_model(32,4)
        self.Fusion_model =  GAM_Attention(32)
        # self.Fusion_model = fusion_MA_model(360)
        # self.Fusion_layer = []
        # self.Fusion_layer.append(tf.keras.layers.Conv2D(32,3,padding='same'))
        # self.Fusion_layer.append(tf.keras.layers.Conv2D(1, 1,padding='same'))

        self.trainMode = None
        self.trainable_var = []

    def call(self,train_batch, training=False):
        if training:
            sin_in = train_batch[0]
        else:
            sin_in = train_batch

        sin_out, fbp_out, sin_interp = self.SineModel(sin_in, training=False)
        sin_interp_map = concatSino_2(sin_interp,sin_in)[:,:,0:self.num_features-3,:]
        # sin_interp_map = sin_interp_map
        ct_in = self.Fbp(sin_interp_map)
        ct_out, ct_in = self.CtModel(ct_in, training=False)
        # fusion_in = self.Fusion_layer[0](tf.concat([fbp_out,ct_out],3))
        # fusion_out = self.Fusion_model(fusion_in)
        # fusion_out = self.Fusion_layer[1](fusion_in)+ct_out
        fusion_in = tf.concat([fbp_out,ct_out],3)
        fusion_in = tf.keras.layers.Conv2D(32,5,padding='same')(fusion_in)

        fusion_out =  self.Fusion_model(fusion_in, training=False)
        fusion_out = tf.keras.layers.Conv2D( 32, 5, padding='same')(fusion_out)
        fusion_out = tf.keras.layers.Conv2D(1, 3,padding='same')(fusion_out)+ct_out
        model_out = [sin_out, fbp_out, ct_in, ct_out, fusion_out,sin_interp]
        if training:
            return model_out,self.loss(train_batch[1], train_batch[2], model_out)
        else:
            return model_out

    def setTrainMode(self, sinLayers=True,fbpLayer=False, ctLayers=False,fusionLayers = False):
        self.trainMode = (sinLayers, fbpLayer, ctLayers,fusionLayers)
        # update my trainable variables
        self.trainable_var = []
        if sinLayers:
            self.trainable_var.extend(self.trainable_variables[0:32])
        if ctLayers:
            self.trainable_var.extend(self.trainable_variables[35:63])
        if fbpLayer:
            self.trainable_var.extend(self.trainable_variables[63:66])
        if fusionLayers:
            self.trainable_var.extend(self.trainable_variables[66:len(self.trainable_variables)])

        print('Training mode:', self.trainMode)

    def loss(self,sin_label, ct_label, model_out, weights=(1.0, 1.0, 1.0,1.0)):
        loss = 0
        # model_out = [sin_out, fbp_out, ct_in, ct_out, fusion_out,sin_interp]
        if self.trainMode[0]:  # sine loss
            loss = weights[0] * tf.reduce_mean(tf.math.square(model_out[0] - sin_label))
        if self.trainMode[2]:  # ct loss
            loss += weights[2] * tf.reduce_mean(tf.math.square(model_out[3] - ct_label))
        if self.trainMode[3]:  # fusion loss
            loss = weights[3] * tf.reduce_mean(tf.math.square(model_out[4] - ct_label))
        return loss

def concatSino_2(sin_interp, sin_in):
    channel = sin_interp.shape[3]
    sin_in = tf.expand_dims(sin_in,3)
    sin_map = sin_in
    for i in range(channel):
        sin_map = tf.concat([sin_map, tf.expand_dims(sin_interp[:, :, :, i], 3)], 2)
    sin_map = tf.reshape(sin_map, [sin_in.shape[0], -1, sin_in.shape[2], 1])
    return sin_map

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

def load_testdata(trainDataDir="./Data/mytest/My_data_512_200.npz", c=2):
    # data = np.load(trainDataDir)
    train_data = np.load(trainDataDir)
    f_img = train_data['f_img'].astype('float32')  # 正弦域input
    ct_label = train_data['ct_label'].astype('float32')  # 正弦域label

    exp = np.zeros([f_img.shape[0], f_img.shape[1], 3])
    f_img = np.concatenate((f_img, exp), axis=2)

    sin_input = np.expand_dims(f_img[:, 0::c, :], 3)
    sin_label = np.zeros([f_img.shape[0], int(f_img.shape[1] / c), f_img.shape[2], c - 1])
    for i in range(c - 1):
        sin_label[:, :, :, i] = f_img[:, i + 1::c, :]

    sin_label = concatSino(sin_label)
    sin_input = np.squeeze(sin_input)
    sin_label = np.squeeze(sin_label)
    # ct_label = np.squeeze(ct_label)
    sin_label = sin_label.astype('float32')
    sin_input = sin_input.astype('float32')
    print('shapes of ct_label, sin_label, :', ct_label.shape)
    print('shape of sin_label:', sin_label.shape)
    print('shape of sin_input:', sin_input.shape)

    train_data = tf.data.Dataset.from_tensor_slices((sin_input, sin_label, ct_label))
    return train_data


def save_data(data_batch, test_out, i, c):
    sin_out, fbp_out, ct_in, ct_out, fusion_out,sin_interp = test_out
    sin_in = data_batch[0]
    # sin_map = concatSino(sin_out, sin_in).numpy()[0, :, :, 0]
    #     sin_map = tf.reshape(tf.concat([sin_in, sin_out], 2), [sin_in.shape[0], -1, sin_in.shape[2], 1]).numpy()[0,:,:,0]
    # fbp_intp = fbp_intp.numpy()[0, :, :, 0] * 255
    fbp_out = fbp_out.numpy()[0, :, :, 0] * 255
    fusion_out = fusion_out.numpy()[0, :, :, 0] * 255
    cv2.imwrite('./Result/public/' + str(c) + 'x/parallel/parallel_' + str(i) +'_'+str(c)+'x'+ '_fusion_out.bmp', fusion_out)


def concatSino(sin_interp):
    channel = sin_interp.shape[3]
    sin_map = tf.expand_dims(sin_interp[:, :, :, 0], 3)
    for i in range(channel - 1):
        sin_map = tf.concat([sin_map, tf.expand_dims(sin_interp[:, :, :, i + 1], 3)], 2)
    sin_map = tf.reshape(sin_map, [sin_interp.shape[0], -1, sin_interp.shape[2], 1])
    return sin_map

def ShuffleSino( sin_interp, sin_in):
    channel = 360//sin_in.shape[1]-1
    sin_map = sin_in
    for i in range(channel):
        sin_map = tf.concat([sin_map, sin_interp[:, i::channel, :]], 2)
    sin_map = tf.reshape(sin_map, [sin_in.shape[0], -1, sin_in.shape[2], 1])
    return sin_map

def compute_psnr(data_batch, model_out):
    def psnr(ref, out):
        return tf.reduce_mean(tf.image.psnr(ref, out, max_val=tf.reduce_max(ref) - tf.reduce_min(ref))).numpy()
    # [sin_out, fbp_out, ct_in, ct_out, fusion_out,sin_interp]
    sin_out, fbp_out, ct_in, ct_out, fusion_out,sin_interp= model_out
    sin_in, sin_label, ct_label = data_batch

    sin_label_large = ShuffleSino(sin_label,sin_in)
    sin_out_large = ShuffleSino(sin_out,sin_in)
    sin_interp = tf.squeeze(concatSino(sin_interp))
    sin_interp_large = ShuffleSino(sin_interp,sin_in)

    # psnr for sin_in vs sin_label, sin_out vs sin_label, fbp_out vs ct_label, ct_out vs ct_label
    return [psnr(sin_label_large, sin_interp_large),
            psnr(sin_label_large, sin_out_large),
            psnr(ct_label,fbp_out),
            psnr(ct_label, ct_in),
            psnr(ct_label, ct_out), 
            psnr(ct_label,fusion_out)]


def compute_ssim(data_batch, model_out):
    def ssim(ref, out):
        ref = tf.squeeze(ref).numpy()
        out = tf.squeeze(out).numpy()
        temp = []
        for i in range(np.shape(ref)[0]):
            temp.append(s_ssim(ref[i, :, :], out[i, :, :]))
        return np.mean(temp)

    sin_out, fbp_out, ct_in, ct_out, fusion_out, sin_interp = model_out
    sin_in, sin_label, ct_label = data_batch

    # sin_label_large = ShuffleSino(sin_label, sin_in)
    # sin_out_large = ShuffleSino(sin_out, sin_in)
    # sin_interp = tf.squeeze(concatSino(sin_interp))
    # sin_interp_large = ShuffleSino(sin_interp, sin_in)

    return [ssim(ct_label, fbp_out), ssim(ct_label, ct_out)]

def show_image(train_data, model_out):
    sin_out, fbp_out, ct_out, sin_interp = model_out
    sin_in, sin_label, ct_label = train_data
    sin_label_large = sin_label
    sin_out_large = sin_out
    # sin_map = tf.reshape(tf.concat([sin_in, sin_out], 2), [sin_in.shape[0], -1, sin_in.shape[2], 1])
    for i in range(sin_out.shape[0]):
        plt.cla()
        plt.subplot(3, 2, 1)
        plt.title("sine label %d" % i, loc='center')
        plt.imshow(sin_label_large[i, :, :, :])
        plt.subplot(3, 2, 2)
        plt.title("sine output %d" % i, loc='center')
        plt.imshow(sin_out_large[i, :, :, :])

        plt.subplot(3, 2, 3)
        plt.title("CT label %d" % i, loc='center')
        plt.imshow(ct_label[i, :, :, :])
        plt.subplot(3, 2, 4)
        plt.title("CT output %d" % i, loc='center')
        plt.imshow(ct_out[i, :, :, :])

        plt.subplot(3, 2, 5)
        plt.title("CT label %d" % i, loc='center')
        plt.imshow(ct_label[i, :, :, :])
        plt.subplot(3, 2, 6)
        plt.title("FBP output %d" % i, loc='center')
        plt.imshow(fbp_out[i, :, :, :])
        plt.pause(0.5)
def test_mymodel(ckpt='./256x256/weights/new_model_lambda=0.5',testDataDir = '',test_sz =2 ,c=2):
    ct_model = dual_parallel_model(2, 360, c)
    ct_model.build((2, 360 // c, 360))
    ct_model.summary()
    # ct_model.build((1, 180, 360, 1))  # Manually build the model and initialize the weights
    ct_model.load_weights(ckpt)
    test_data  = load_testdata(testDataDir,c)
    total_psnr = [0, 0, 0, 0, 0, 0]
    total_ssim = [0, 0, 0, 0, 0, 0]
    result = []
    for iterNo, data_batch in enumerate(test_data.batch(test_sz)):
        test_out = ct_model(data_batch[0], training=0)
        psnr_test = compute_psnr(data_batch, test_out)  # psnr for training dataset
        ssim_test = compute_ssim(data_batch, test_out)
        total_psnr=list(map(lambda x: x[0] + x[1], zip(total_psnr, psnr_test)))
        total_ssim=list(map(lambda x: x[0] + x[1], zip(total_ssim, ssim_test)))
        save_data(data_batch,test_out,iterNo,c)
        # print("iterNo: ", iterNo, ":", "psnr_valid", psnr_test)
        # print("iterNo: ", iterNo, ":", "ssim_valid", ssim_test)
        # if iterNo% 2==0:
        #     show_image(data_batch, test_out)

    print("the average psnr :" ,total_psnr/(test_data.cardinality().numpy())*test_sz)
    print("the average ssim :", total_ssim / (test_data.cardinality().numpy()) * test_sz)

if __name__ == '__main__':
    testDataDir = './Data/mytest/My_data_public_256_200.npz'
    c = 15
    i = 1
    iterNo = 500
    ckpt = './256x256_parallel/' + str(c) + 'x_weights_public_' + str(i) + '_epoch' + str(iterNo) + 'iter' + '/ckpt/'
    test_sz = 2
    test_mymodel(ckpt,testDataDir,test_sz,c)