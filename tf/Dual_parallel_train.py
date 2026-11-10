from model.Transformer_CNN_model import TransformerModel
from model.Unet_CT_model import UnetModel
from model.CBAM_model import CBAM_model
from model.Bam_model import BAM
from model.GAM_model import GAM_Attention
from model.fusion_muiltA_model import fusion_MA_model
from skimage.metrics import structural_similarity as s_ssim
import tensorflow as tf
import numpy as np
import matplotlib.pyplot as plt
import cv2


class dual_parallel_model(tf.keras.Model):
    def __init__(
        self,
        transformer_layers,
        num_features,
        sparse_scale=2,
        ckpt_sine=None,
        ckpt_ct=None,
    ):
        super(dual_parallel_model, self).__init__()
        self.num_features = num_features
        self.SineModel = TransformerModel(
            transformer_layers, num_features, sparse_scale
        )
        if ckpt_sine != None:
            self.SineModel.load_weights(ckpt_sine)
        self.CtModel = UnetModel()
        if ckpt_ct != None:
            self.CtModel.load_weights(ckpt_ct)
        self.Fbp = FbpLayer()
        # self.Fusion_model = CBAM_model(32,4)
        self.Fusion_model = GAM_Attention(32)
        # self.Fusion_model = fusion_MA_model(360)
        # self.Fusion_layer = []
        # self.Fusion_layer.append(tf.keras.layers.Conv2D(32,3,padding='same'))
        # self.Fusion_layer.append(tf.keras.layers.Conv2D(1, 1,padding='same'))

        self.trainMode = None
        self.trainable_var = []

    def call(self, train_batch, training=False):
        if training:
            sin_in = train_batch[0]
        else:
            sin_in = train_batch

        sin_out, fbp_out, sin_interp = self.SineModel(sin_in, training=False)
        sin_interp_map = concatSino_2(sin_interp, sin_in)[
            :, :, 0 : self.num_features - 3, :
        ]
        # sin_interp_map = sin_interp_map
        ct_in = self.Fbp(sin_interp_map)
        ct_out, ct_in = self.CtModel(ct_in, training=False)
        # fusion_in = self.Fusion_layer[0](tf.concat([fbp_out,ct_out],3))
        # fusion_out = self.Fusion_model(fusion_in)
        # fusion_out = self.Fusion_layer[1](fusion_in)+ct_out
        fusion_in = tf.concat([fbp_out, ct_out], 3)
        fusion_in = tf.keras.layers.Conv2D(32, 5, padding="same")(fusion_in)

        fusion_out = self.Fusion_model(fusion_in, training=False)
        fusion_out = tf.keras.layers.Conv2D(32, 5, padding="same")(fusion_out)
        fusion_out = tf.keras.layers.Conv2D(1, 3, padding="same")(fusion_out) + ct_out
        model_out = [sin_out, fbp_out, ct_in, ct_out, fusion_out, sin_interp]
        if training:
            return model_out, self.loss(train_batch[1], train_batch[2], model_out)
        else:
            return model_out

    def setTrainMode(
        self, sinLayers=True, fbpLayer=False, ctLayers=False, fusionLayers=False
    ):
        self.trainMode = (sinLayers, fbpLayer, ctLayers, fusionLayers)
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

        print("Training mode:", self.trainMode)

    def loss(self, sin_label, ct_label, model_out, weights=(1.0, 1.0, 1.0, 1.0)):
        loss = 0
        # model_out = [sin_out, fbp_out, ct_in, ct_out, fusion_out,sin_interp]
        if self.trainMode[0]:  # sine loss
            loss = weights[0] * tf.reduce_mean(tf.math.square(model_out[0] - sin_label))
        if self.trainMode[2]:  # ct loss
            loss += weights[2] * tf.reduce_mean(tf.math.square(model_out[3] - ct_label))
        if self.trainMode[3]:  # fusion loss
            loss = weights[3] * tf.reduce_mean(tf.math.square(model_out[4] - ct_label))
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
        sin_fan_flt = tf.nn.conv1d(sin_fan, self.fbp_filter, stride=1, padding="SAME")
        # print(tf.shape(sin_fan_flt))
        fbpOut = tf.sparse.sparse_dense_matmul(tf.reshape(sin_fan_flt, [-1, sin_sz]), self.A_Matrix)
        fbpOut = tf.reshape(fbpOut, [-1, self.out_shape[0], self.out_shape[1], 1])

        output = fbpOut * self.scale + self.bias
        return output


def concatSino(sin_interp):
    channel = sin_interp.shape[3]
    sin_map = tf.expand_dims(sin_interp[:, :, :, 0], 3)
    for i in range(channel - 1):
        sin_map = tf.concat([sin_map, tf.expand_dims(sin_interp[:, :, :, i + 1], 3)], 2)
    sin_map = tf.reshape(sin_map, [sin_interp.shape[0], -1, sin_interp.shape[2], 1])
    return sin_map


def concatSino_2(sin_interp, sin_in):
    channel = sin_interp.shape[3]
    sin_in = tf.expand_dims(sin_in, 3)
    sin_map = sin_in
    for i in range(channel):
        sin_map = tf.concat([sin_map, tf.expand_dims(sin_interp[:, :, :, i], 3)], 2)
    sin_map = tf.reshape(sin_map, [sin_in.shape[0], -1, sin_in.shape[2], 1])
    return sin_map


def ShuffleSino(sin_interp, sin_in):
    channel = 360 // sin_in.shape[1] - 1
    sin_map = sin_in
    for i in range(channel):
        sin_map = tf.concat([sin_map, sin_interp[:, i::channel, :]], 2)
    sin_map = tf.reshape(sin_map, [sin_in.shape[0], -1, sin_in.shape[2], 1])
    return sin_map


def load_traindata(trainDataDir="./Data/mymodel/My_data_512_1800.npz", val_sz=2, c=2):
    # data = np.load(trainDataDir)
    train_data = np.load(trainDataDir)
    f_img = train_data["f_img"].astype("float32")  # 正弦域input
    ct_label = train_data["ct_label"].astype("float32")  # 正弦域label

    if c == 15:
        f_img = f_img[0:1200, :, :]
        ct_label = ct_label[0:1200, :, :]

    exp = np.zeros([f_img.shape[0], f_img.shape[1], 3])
    f_img = np.concatenate((f_img, exp), axis=2)

    sin_input = np.expand_dims(f_img[:, 0::c, :], 3)
    sin_label = np.zeros(
        [f_img.shape[0], int(f_img.shape[1] / c), f_img.shape[2], c - 1]
    )
    for i in range(c - 1):
        sin_label[:, :, :, i] = f_img[:, i + 1 :: c, :]

    sin_label = concatSino(sin_label)
    sin_input = np.squeeze(sin_input)
    sin_label = np.squeeze(sin_label)
    # ct_label = np.squeeze(ct_label)
    sin_label = sin_label.astype("float32")
    sin_input = sin_input.astype("float32")

    print("shapes of ct_label, sin_label, :", ct_label.shape)
    print("shape of sin_label:", sin_label.shape)
    print("shape of sin_input:", sin_input.shape)

    # 处理数据集，随机选取前dataset_sz-val_sz个作为数据并shuffle，剩下的val_sz个作为验证集
    dataset_sz = sin_input.shape[0]
    train_sz = dataset_sz - val_sz
    ids = np.random.permutation(dataset_sz)
    train_ids = ids[0:train_sz]
    val_ids = ids[train_sz:dataset_sz]
    val_data = [sin_input[val_ids], sin_label[val_ids], ct_label[val_ids]]

    train_data = tf.data.Dataset.from_tensor_slices(
        (sin_input[train_ids], sin_label[train_ids], ct_label[train_ids])
    ).shuffle(dataset_sz - val_sz)
    print("shape of val_data:", val_data[0].shape, val_data[1].shape, val_data[2].shape)
    return train_data, val_data


def train_step(data_batch, model, optimizer):
    # 开启上下文管理，参数watch_accessed_variables=False表示手动设置可训练参数
    with tf.GradientTape(watch_accessed_variables=False) as tape:
        tape.watch(model.trainable_var)
        model_out, model_loss = model(data_batch, training=1)

    grads = tape.gradient(model_loss, model.trainable_var)  # 使用loss对后者进行求导
    optimizer.apply_gradients(zip(grads, model.trainable_var))
    return model_out, model_loss


def compute_psnr(data_batch, model_out):
    def psnr(ref, out):
        return tf.reduce_mean(
            tf.image.psnr(ref, out, max_val=tf.reduce_max(ref) - tf.reduce_min(ref))
        ).numpy()

    # [sin_out, fbp_out, ct_in, ct_out, fusion_out,sin_interp]
    sin_out, fbp_out, ct_in, ct_out, fusion_out, sin_interp = model_out
    sin_in, sin_label, ct_label = data_batch

    sin_label_large = ShuffleSino(sin_label, sin_in)
    sin_out_large = ShuffleSino(sin_out, sin_in)
    sin_interp = tf.squeeze(concatSino(sin_interp))
    sin_interp_large = ShuffleSino(sin_interp, sin_in)

    # psnr for sin_in vs sin_label, sin_out vs sin_label, fbp_out vs ct_label, ct_out vs ct_label
    return [
        psnr(sin_label_large, sin_interp_large),
        psnr(sin_label_large, sin_out_large),
        psnr(ct_label, fbp_out),
        psnr(ct_label, ct_in),
        psnr(ct_label, ct_out),
        psnr(ct_label, fusion_out),
    ]


def show_image(train_data, model_out):
    sin_out, fbp_out, ct_in, ct_out, fusion_out, sin_interp = model_out
    sin_in, sin_label, ct_label = train_data
    # sin_label_large = ShuffleSino(sin_label,sin_in)
    # sin_out_large = ShuffleSino(sin_out,sin_in)

    for i in range(sin_out.shape[0]):
        plt.cla()

        plt.subplot(2, 2, 1)
        plt.title("CT label %d" % i, loc="center")
        plt.imshow(ct_label[i, :, :])
        plt.subplot(2, 2, 2)
        plt.title("sinogram output %d" % i, loc="center")
        plt.imshow(fbp_out[i, :, :])
        plt.subplot(2, 2, 3)
        plt.title("CT out %d" % i, loc="center")
        plt.imshow(ct_out[i, :, :])
        plt.subplot(2, 2, 4)
        plt.title("fusion output %d" % i, loc="center")
        plt.imshow(fusion_out[i, :, :])
        plt.pause(0.5)


def train(epoch=3, batch_sz=1, c=2, ckpt_sine=None, ckpt_ct=None):
    # ckpt = './512x512/weights/new_model_lambda=0.5'
    # load data
    tf.keras.backend.clear_session()
    train_data, val_data = load_traindata(
        "./Data/mymodel/My_data_public_256_1800.npz", c=c
    )

    ct_model = dual_parallel_model(2, 360, c, ckpt_sine, ckpt_ct)
    ct_model.build((2, 360 // c, 360))
    ct_model.summary()

    # create optimizer
    optimizer_sine = tf.keras.optimizers.Adam(learning_rate=0.001)
    optimizer_CT = tf.keras.optimizers.Adam(learning_rate=0.001)
    optimizer_fusion = tf.keras.optimizers.Adam(learning_rate=0.001)

    loss = []
    for i in range(epoch):
        if 0 <= i < 1:
            ct_model.setTrainMode(False, False, False, True)
            Optimizer = optimizer_fusion
        if 1 <= i < 4:
            ct_model.setTrainMode(False, True, True, True)
            Optimizer = optimizer_CT

        loss_epoch = 0
        for iterNo, data_batch in enumerate(train_data.batch(batch_sz)):
            model_out, model_loss = train_step(data_batch, ct_model, Optimizer)
            loss_epoch += model_loss
            if iterNo % 100 == 0:  # & (i%2==0) :
                psnr_train = compute_psnr(
                    data_batch, model_out
                )  # psnr for training dataset
                print(
                    iterNo, "/", i, ":", model_loss.numpy(), "; psnr_train:", psnr_train
                )
                val_out = ct_model(val_data[0], training=0)
                psnr_valid = compute_psnr(val_data, val_out)
                # show_image(val_data, val_out)
                print("epoch: ", i, ":", "psnr_valid", psnr_valid)
                ckpt = (
                    "./256x256_parallel/"
                    + str(c)
                    + "x_weights_public_"
                    + str(i)
                    + "_epoch"
                    + str(iterNo)
                    + "iter"
                    + "/ckpt/"
                )
                ct_model.save_weights(ckpt)
                loss.append(loss_epoch)
                loss_epoch = 0

    plt.plot(loss, label="Training Loss")
    plt.title("Training and Validation Loss")
    plt.legend()
    plt.show()


if __name__ == "__main__":
    c = 15
    ct_i = 4

    ct_iterNo = 800
    # ckpt_sine = r'E:\A.Study\Uestc\毕业设计\模型参数\参数\正弦域单独实验参数\public\8_0x_weights_4_epoch\ckpt/'
    ckpt_sine = r"E:\A.Study\Uestc\毕业设计\模型参数\参数\正弦域单独实验参数\public\15_500x_weights_14_epoch\ckpt/"
    # ckpt_sine = r'E:\A.Study\Uestc\毕业设计\模型参数\参数\正弦域单独实验参数\our\4_0x_weights_7_epoch\ckpt/'
    # ckpt_sine = r'E:\A.Study\Uestc\毕业设计\模型参数\参数\正弦域单独实验参数\our\8_800x_weights_9_epoch\ckpt/'
    # ckpt_sine = r'E:\A.Study\Uestc\毕业设计\模型参数\参数\正弦域单独实验参数\our\15_500x_weights_12_epoch\ckpt/'
    ckpt_ct = (
        "./256x256_unet/"
        + str(c)
        + "x_weights_public_"
        + str(ct_i)
        + "_epoch"
        + str(ct_iterNo)
        + "iter"
        + "/ckpt/"
    )
    # ckpt_ct  = './256x256_unet/' + str(c) + 'x_weights_public_' + str(ct_i) + '_epoch' + str(ct_iterNo) + 'iter' + '/ckpt/'
    train(epoch=2, batch_sz=2, c=c, ckpt_sine=ckpt_sine, ckpt_ct=ckpt_ct)

    # ckpt = './256x256/weights/new_model_lambda=0.5'
    # testDataDir = './Data/mytest/My_data_our_256_200.npz'
    # c = 2
    # i = 3
    # ckpt = './256x256/' + str(c) + 'x_weights_' + str(i) + '_epoch/ckpt/'
    # test_sz = 2
    # test(ckpt, testDataDir, test_sz)
