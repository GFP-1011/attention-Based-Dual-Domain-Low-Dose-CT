from model.Transformer_CNN_model import TransformerModel
from model.Unet_CT_model import UnetModel
from model.funsion_model import Funsion_model
import tensorflow as tf
import numpy as np
import matplotlib.pyplot as plt


class dual_seq_model(tf.keras.Model):
    def __init__(self, transformer_layers,num_features ,sparse_scale= 2,num_heads = 8 ,expend_units=4):
        super(dual_seq_model, self).__init__()
        self.SineModel = TransformerModel(transformer_layers,num_features,sparse_scale)
        self.CtModel = UnetModel()
        self.Fusion_model = Funsion_model(32)

        self.trainMode = None
        self.trainable_var = []

    def call(self,train_batch, training=False):
        if training:
            sin_in = train_batch[0]
            sin_out, fbp_out, sin_interp = self.SineModel(sin_in, training=False)
            ct_out, ct_in = self.CtModel(fbp_out, training=False)
            fusion_in = tf.concat([fbp_out,ct_out],3)
            fusion_out = self.Fusion_model(fusion_in )
            model_out = [sin_out, fbp_out, ct_out, fusion_out, sin_interp]
            return model_out,self.loss(train_batch[1], train_batch[2], model_out)
        else:
            sin_in = train_batch
            sin_out, fbp_out, sin_interp = self.SineModel(sin_in)
            ct_out, ct_in = self.CtModel(fbp_out)
            fusion_in = tf.concat([fbp_out, ct_out],3)
            fusion_out = self.Fusion_model(fusion_in )
            return [sin_out ,fbp_out,ct_out, fusion_out,sin_interp]

    def setTrainMode(self, sinLayers=True,fbpLayer=False, ctLayers=False, fusionLayers=False):
        self.trainMode = (sinLayers, fbpLayer, ctLayers,fusionLayers)
        # update my trainable variables
        self.trainable_var = []
        if sinLayers:
            self.trainable_var.extend(self.trainable_variables[0:20])
        if ctLayers:
            self.trainable_var.extend(self.trainable_variables[23:51])
        if fusionLayers:
            self.trainable_var.extend(
                self.trainable_variables[51:len(self.trainable_variables)])
        print('Training mode:', self.trainMode)

    def loss(self,sin_label, ct_label, model_out, weights=(1.0, 1.0, 1.0)):
        loss = 0
        if self.trainMode[0]:  # sine loss
            loss = weights[0] * tf.reduce_mean(tf.math.square(model_out[0] - sin_label))
        if self.trainMode[2]:  # ct loss
            loss += weights[2] * tf.reduce_mean(tf.math.square(model_out[2] - ct_label))
        if self.trainMode[3]:  # ct loss
            loss += weights[2] * tf.reduce_mean(tf.math.square(model_out[3] - ct_label))
        return loss



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



def load_traindata(trainDataDir="./Data/mymodel/My_data_512_1800.npz", val_sz=2, c=2):
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

    # 处理数据集，随机选取前dataset_sz-val_sz个作为数据并shuffle，剩下的val_sz个作为验证集
    dataset_sz = sin_input.shape[0]
    train_sz = dataset_sz - val_sz
    ids = np.random.permutation(dataset_sz)
    train_ids = ids[0:train_sz]
    val_ids = ids[train_sz:dataset_sz]
    val_data = [sin_input[val_ids], sin_label[val_ids], ct_label[val_ids]]

    train_data = tf.data.Dataset.from_tensor_slices((sin_input[train_ids], sin_label[train_ids], ct_label[train_ids])). \
        shuffle(dataset_sz - val_sz)
    print('shape of val_data:', val_data[0].shape, val_data[1].shape, val_data[2].shape)
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
        return tf.reduce_mean(tf.image.psnr(ref, out, max_val=tf.reduce_max(ref) - tf.reduce_min(ref))).numpy()
    sin_out, fbp_out, ct_out, fusion_out,sin_interp= model_out
    sin_in, sin_label, ct_label = data_batch
    sin_label_large = ShuffleSino(sin_label,sin_in)
    sin_out_large = ShuffleSino(sin_out,sin_in)
    sin_interp = tf.squeeze(concatSino(sin_interp))
    sin_interp_large = ShuffleSino(sin_interp,sin_in)
    # psnr for sin_in vs sin_label, sin_out vs sin_label, fbp_out vs ct_label, ct_out vs ct_label
    return [  psnr(sin_label_large, sin_interp_large),psnr(sin_label_large, sin_out_large),psnr(ct_label, fbp_out),psnr(ct_label, ct_out),psnr(ct_label,fusion_out)]

def show_image(train_data, model_out):
    sin_out, fbp_out, ct_out, fusion_out,sin_interp= model_out
    sin_in, sin_label, ct_label = train_data
    sin_label_large = ShuffleSino(sin_label,sin_in)
    sin_out_large = ShuffleSino(sin_out,sin_in)

    for i in range(sin_out.shape[0]):
        plt.cla()
        plt.subplot(2, 2, 1)
        plt.title("sine label %d" % i, loc='center')
        plt.imshow(sin_label_large[i, :, :])
        plt.subplot(2, 2, 2)
        plt.title("sine output %d" % i, loc='center')
        plt.imshow(sin_out_large[i, :, :])
        plt.subplot(2, 2, 3)
        plt.title("CT label %d" % i, loc='center')
        plt.imshow(ct_label[i, :, :])
        plt.subplot(2, 2, 4)
        plt.title("CT output %d" % i, loc='center')
        plt.imshow(fbp_out[i, :, :])
        plt.pause(0.5)

def train(epoch=3, batch_sz=1, c=2):
    # ckpt = './512x512/weights/new_model_lambda=0.5'
    # load data
    tf.keras.backend.clear_session()
    train_data, val_data= load_traindata("./Data/mymodel/My_data_our_256_1800.npz", c=c)
    ct_model = dual_seq_model(1,360,c)
    ct_model.build((1,360//c,360))
    ct_model.summary()

    # create optimizer
    optimizer_sine = tf.keras.optimizers.Adam(learning_rate=0.001)
    optimizer_CT = tf.keras.optimizers.Adam(learning_rate=0.001)
    optimizer_fusion = tf.keras.optimizers.Adam(learning_rate=0.001)

    loss = []
    for i in range(epoch):
        if i <= 20:
            ct_model.setTrainMode(True, False, False, False)
            loss_epoch = 0
            for iterNo, data_batch in enumerate(train_data.batch(batch_sz)):
                model_out, model_loss = train_step(data_batch, ct_model, optimizer_sine)
                loss_epoch+=model_loss
                if iterNo % 100 == 0:
                    psnr_train = compute_psnr(data_batch, model_out)  # psnr for training dataset
                    print(iterNo, "/", i, ":", model_loss.numpy(), "; psnr_train:", psnr_train)
            loss.append(loss_epoch)
            if i % 25 ==0:
                val_out = ct_model(val_data[0], training=0)
                psnr_valid = compute_psnr(val_data, val_out)
                show_image(val_data, val_out)
                print("epoch: ", i, ":", "psnr_valid", psnr_valid)
                ckpt = './256x256/' + str(c) + 'x_weights_' + str(i) + '_epoch/ckpt/'
                ct_model.save_weights(ckpt)

        if 20 < i <=40:
            ct_model.setTrainMode(False, False, True, False)
            loss_epoch = 0
            for iterNo, data_batch in enumerate(train_data.batch(batch_sz)):
                model_out, model_loss = train_step(data_batch, ct_model, optimizer_CT)
                loss_epoch += model_loss
                if iterNo % 100 == 0:
                    psnr_train = compute_psnr(data_batch, model_out)  # psnr for training dataset
                    print(iterNo, "/", i, ":", model_loss.numpy(), "; psnr_train:", psnr_train)
        if 40 < i <=600:
            ct_model.setTrainMode(False, False, False, True)
            loss_epoch = 0
            for iterNo, data_batch in enumerate(train_data.batch(batch_sz)):
                model_out, model_loss = train_step(data_batch, ct_model, optimizer_fusion)
                loss_epoch += model_loss
                if iterNo % 100 == 0:
                    psnr_train = compute_psnr(data_batch, model_out)  # psnr for training dataset
                    print(iterNo, "/", i, ":", model_loss.numpy(), "; psnr_train:", psnr_train)
        # loss.append(loss_epoch)
        # if i % 25 ==0:
        #     val_out = ct_model(val_data[0], training=0)
        #     psnr_valid = compute_psnr(val_data, val_out)
        #     show_image(val_data, val_out)
        #     print("epoch: ", i, ":", "psnr_valid", psnr_valid)
        #     ckpt = './512x512/' + str(c) + 'x_weights_' + str(i) + '_epoch/ckpt/'
        #     ct_model.save_weights(ckpt)
    plt.plot(loss, label='Training Loss')
    plt.title('Training and Validation Loss')
    plt.legend()
    plt.show()

if __name__ == '__main__':
    train(epoch=600, batch_sz = 2, c=2)

