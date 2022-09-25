import collections
import json
import math
from statistics import mode
import os
import random
from PIL import Image, ImageEnhance
import tensorflow as tf
import numpy as np
import time
import shutil
import logging
import inspect
import nltk
import albumentations as alb
from matplotlib import pyplot as plt
from tensorflow.python import keras
from tqdm import tqdm
from keras_preprocessing.text import tokenizer_from_json
from keras.layers import Dense
from keras.layers import Conv2D
from keras.layers import MaxPooling2D
from utils.images_preprocessing import make_fix_size
from keras.applications.vgg16 import VGG16

# RESIZED_IMG_H = 175
# RESIZED_IMG_W = 1500
RESIZED_IMG_H = 110
RESIZED_IMG_W = 940
DATA_SPLIT = 1
global DATASET_PATH, ANNOTATION_FILE, tokenizer_path, checkpoint_path, meta_path


transforms = alb.Compose(
    [
        # alb.Compose(
        #     [alb.ShiftScaleRotate(shift_limit=0, scale_limit=(-.15, 0), rotate_limit=1, border_mode=0, interpolation=3,
        #                           value=[255, 255, 255], p=1),
        #      alb.GridDistortion(distort_limit=0.1, border_mode=0, interpolation=3, value=[255, 255, 255], p=.5)], p=.15),
        # alb.RGBShift(r_shift_limit=15, g_shift_limit=15,
        #              b_shift_limit=15, p=0.3),
        # alb.GaussNoise(25, p=.5),
        # alb.RandomBrightnessContrast(.05, (-.2, 0), True, p=0.2),
        alb.GaussianBlur(blur_limit=(1, 3),  sigma_limit=1, p=0.5)
        # alb.ImageCompression(quality_lower=50, quality_upper=85, p=.5),
        # alb.ToGray(always_apply=True),
        # alb.Normalize((0.7931, 0.7931, 0.7931), (0.1738, 0.1738, 0.1738)),
    ]
)

def log_init(path, name):
    logger = logging.getLogger(__name__)
    logger.setLevel(logging.INFO)

    formatter = logging.Formatter("%(asctime)s:    %(message)s")

    file_handler = logging.FileHandler(f"{path}/{name}.txt", mode="w")
    file_handler.setLevel(logging.INFO)
    file_handler.setFormatter(formatter)

    stream_handler = logging.StreamHandler()
    stream_handler.setFormatter(formatter)

    logger.addHandler(file_handler)
    logger.addHandler(stream_handler)
    return logger


def enable_gpu(turn, gb: int = 4):  # enable of disable GPU backend
    if turn:
        os.environ['TF_FORCE_GPU_ALLOW_GROWTH'] = 'true'
        gpus = tf.config.experimental.list_physical_devices('GPU')
        if gpus:
            # Restrict TensorFlow to only allocate 1*X GB of memory on the first GPU
            try:
                tf.config.experimental.set_virtual_device_configuration(
                    gpus[0],
                    [tf.config.experimental.VirtualDeviceConfiguration(memory_limit=(1024 * gb))])
                logical_gpus = tf.config.experimental.list_logical_devices(
                    'GPU')
                print(len(gpus), "Physical GPUs,", len(
                    logical_gpus), "Logical GPUs")
            except RuntimeError as e:
                # Virtual devices must be set before GPUs have been initialized
                print(e)
    else:
        os.environ["CUDA_VISIBLE_DEVICES"] = "-1"
        try:
            # Disable all GPUS
            tf.config.set_visible_devices([], 'GPU')
            visible_devices = tf.config.get_visible_devices()
            for device in visible_devices:
                assert device.device_type != 'GPU'
        except:
            # Invalid device or cannot modify virtual devices once initialized.
            pass

def get_cnn_model():
    base_model = VGG16(
        input_shape=(*(RESIZED_IMG_H, RESIZED_IMG_W), 3), include_top=False, weights=None,
    )
    # We freeze our feature extractor
    base_model.trainable = True
    base_model_out = base_model.output
    # print(base_model_out.shape)
    base_model_out = keras.layers.Reshape((-1, base_model_out.shape[-1]))(base_model_out)
    cnn_model = keras.models.Model(base_model.input, base_model_out)
    return cnn_model

class BahdanauAttention(tf.keras.Model):
    def __init__(self, units):
        super(BahdanauAttention, self).__init__()
        self.W1 = tf.keras.layers.Dense(units)
        self.W2 = tf.keras.layers.Dense(units)
        self.V = tf.keras.layers.Dense(1)

    def call(self, features, hidden):
        # features(CNN_encoder output) shape == (batch_size, H*W, embedding_dim)

        # hidden shape == (batch_size, hidden_size)
        # hidden_with_time_axis shape == (batch_size, 1, hidden_size)
        hidden_with_time_axis = tf.expand_dims(hidden, 1)

        # attention_hidden_layer shape == (batch_size, H*W, units)
        attention_hidden_layer = (tf.nn.tanh(self.W1(features) +
                                             self.W2(hidden_with_time_axis)))
 
        # score shape == (batch_size, H*W, 1)
        # This gives you an unnormalized score for each image feature.
        score = self.V(attention_hidden_layer)

        # attention_weights shape == (batch_size, H*W, 1)
        attention_weights = tf.nn.softmax(score, axis=1)

        # context_vector shape after sum == (batch_size, hidden_size)
        context_vector = attention_weights * features
        context_vector = tf.reduce_sum(context_vector, axis=1)

        return context_vector, attention_weights


class CNN_Encoder(tf.keras.Model):
    def __init__(self, embedding_dim):
        super(CNN_Encoder, self).__init__()

        self.rescale = tf.keras.layers.experimental.preprocessing.Rescaling(scale=1. / 127.5, offset=-1)
        # self.resnet_50 = tf.keras.applications.resnet50.ResNet50(include_top=False, weights=None, input_shape=(RESIZED_IMG_H, RESIZED_IMG_W, 3))
        # self.cnn = tf.keras.Model(self.resnet_50.input, self.resnet_50.layers[-1].output)

        # self.rescale = tf.keras.layers.experimental.preprocessing.Rescaling(
        #     scale=1. / 127.5, offset=-1)

        # self.conv_i = tf.keras.layers.Conv2D(64, kernel_size=7, strides=2, padding='same')
        # self.bn_i = tf.keras.layers.BatchNormalization()
        # self.act_i = tf.keras.layers.Activation('relu')
        # self.mp_i = tf.keras.layers.MaxPool2D(pool_size=3, strides=2, padding='same')

        # # Layer 1_1
        # self.conv_1_1 = tf.keras.layers.Conv2D(64, (3,3), padding = 'same')
        # self.bn_1_1 = tf.keras.layers.BatchNormalization(axis=3)
        # self.act_1_1 = tf.keras.layers.Activation('relu')
        # # Layer 1_2
        # self.conv_1_2 = tf.keras.layers.Conv2D(64, (3,3), padding = 'same')
        # self.bn_1_2 = tf.keras.layers.BatchNormalization(axis=3)
        # # Add Residue 1_3
        # self.add_1_3 = tf.keras.layers.Add()
        # self.act_1_3 = tf.keras.layers.Activation('relu')

        # # Layer 2_1
        # self.conv_2_1 = tf.keras.layers.Conv2D(64, (3,3), padding = 'same')
        # self.bn_2_1 = tf.keras.layers.BatchNormalization(axis=3)
        # self.act_2_1 = tf.keras.layers.Activation('relu')
        # # Layer 2_2
        # self.conv_2_2 = tf.keras.layers.Conv2D(64, (3,3), padding = 'same')
        # self.bn_2_2 = tf.keras.layers.BatchNormalization(axis=3)
        # # Add Residue 2_3
        # self.add_2_3 = tf.keras.layers.Add()
        # self.act_2_3 = tf.keras.layers.Activation('relu')

        # # Layer 3_1
        # self.conv_3_1 = tf.keras.layers.Conv2D(64, (3,3), padding = 'same')
        # self.bn_3_1 = tf.keras.layers.BatchNormalization(axis=3)
        # self.act_3_1 = tf.keras.layers.Activation('relu')
        # # Layer 3_2
        # self.conv_3_2 = tf.keras.layers.Conv2D(64, (3,3), padding = 'same')
        # self.bn_3_2 = tf.keras.layers.BatchNormalization(axis=3)
        # # Add Residue 3_3
        # self.add_3_3 = tf.keras.layers.Add()
        # self.act_3_3 = tf.keras.layers.Activation('relu')

        # # Layer 1
        # x = tf.keras.layers.Conv2D(filter, (3,3), padding = 'same', strides = (2,2))(x)
        # x = tf.keras.layers.BatchNormalization(axis=3)(x)
        # x = tf.keras.layers.Activation('relu')(x)
        # # Layer 2
        # x = tf.keras.layers.Conv2D(filter, (3,3), padding = 'same')(x)
        # x = tf.keras.layers.BatchNormalization(axis=3)(x)
        # # Processing Residue with conv(1,1)
        # x_skip = tf.keras.layers.Conv2D(filter, (1,1), strides = (2,2))(x_skip)
        # # Add Residue
        # x = tf.keras.layers.Add()([x, x_skip])
        # x = tf.keras.layers.Activation('relu')(x)


        # self.conv1_block1 = Conv2D(filters=64, kernel_size=(3, 3), padding='same', activation='relu')
        # self.conv2_block1 = Conv2D(filters=64, kernel_size=(3, 3), padding='same', activation='relu')
        # self.pool_block1 = MaxPooling2D(pool_size=(2, 2))

        # self.conv1_block2 = Conv2D(filters=128, kernel_size=(3, 3), padding='same', activation='relu')
        # self.conv2_block2 = Conv2D(filters=128, kernel_size=(3, 3), padding='same', activation='relu')
        # self.pool_block2 = MaxPooling2D(pool_size=(2, 2))

        # self.conv1_block3 = Conv2D(filters=256, kernel_size=(3, 3), padding='same', activation='relu')
        # self.conv2_block3 = Conv2D(filters=256, kernel_size=(3, 3), padding='same', activation='relu')
        # self.conv3_block3 = Conv2D(filters=256, kernel_size=(3, 3), padding='same', activation='relu')
        # self.pool_block3 = MaxPooling2D(pool_size=(2, 2))

        # self.conv1_block4 = Conv2D(filters=512, kernel_size=(3, 3), padding='same', activation='relu')
        # self.conv2_block4 = Conv2D(filters=512, kernel_size=(3, 3), padding='same', activation='relu')
        # self.conv3_block4 = Conv2D(filters=512, kernel_size=(3, 3), padding='same', activation='relu')
        # self.pool_block4 = MaxPooling2D(pool_size=(2, 2))
        
        # self.conv1_block5 = Conv2D(filters=512, kernel_size=(3, 3), padding='same', activation='relu')
        # self.conv2_block5 = Conv2D(filters=512, kernel_size=(3, 3), padding='same', activation='relu')
        # self.conv3_block5 = Conv2D(filters=512, kernel_size=(3, 3), padding='same', activation='relu')
        # self.pool_block5 = MaxPooling2D(pool_size=(2, 2))
        
        self.conv1_block1 = Conv2D(filters=64, kernel_size=(3, 3), padding='same', activation='relu')
        self.pool_block1 = MaxPooling2D(pool_size=(2, 2))

        self.conv1_block2 = Conv2D(filters=128, kernel_size=(3, 3), padding='same', activation='relu')
        # self.pool_block2 = MaxPooling2D(pool_size=(4, 4))
        self.pool_block2 = MaxPooling2D(pool_size=(2, 2))

        self.conv1_block3 = Conv2D(filters=256, kernel_size=(3, 3), padding='same', activation='relu')
        self.conv2_block3 = Conv2D(filters=256, kernel_size=(3, 3), padding='same', activation='relu')
        self.pool_block3 = MaxPooling2D(pool_size=(2, 2))

        self.conv1_block4 = Conv2D(filters=512, kernel_size=(3, 3), padding='same', activation='relu')
        self.pool_block4 = MaxPooling2D(pool_size=(2, 2))
        # self.pool_block4 = MaxPooling2D(pool_size=(4, 4))
        self.conv2_block4 = Conv2D(filters=512, kernel_size=(3, 3), padding='same', activation='relu')
    
        # shape after fc == (batch_size, 64, embedding_dim)
        self.fc = Dense(embedding_dim, activation='relu')

    def call(self, x):
        # Rescaling image to [-1 ; 1] scale
        x = self.rescale(x)

        # Perfoming convolutions
        x = self.conv1_block1(x)
        x = self.pool_block1(x)

        x = self.conv1_block2(x)
        x = self.pool_block2(x)

        x = self.conv1_block3(x)
        x = self.conv2_block3(x)
        x = self.pool_block3(x)

        x = self.conv1_block4(x)
        x = self.pool_block4(x)
        x = self.conv2_block4(x)
        
        
        # x = self.conv1_block5(x)
        # x = self.conv2_block5(x)
        # x = self.conv3_block5(x)
        # x = self.pool_block5(x)

        # Positional embeddings
        x = self.add_timing_signal_nd(x, min_timescale=10.0)

        # Reshaping to [batch_size, H*W, 512] shape
        x = tf.reshape(x, (x.shape[0], -1, x.shape[3]))

        # Final vector of shape [batch_size, H*W, embedding_dim]
        x = self.fc(x)

        return x

    def add_timing_signal_nd(self, x, min_timescale=5.0, max_timescale=1.0e4):
        """Adds a bunch of sinusoids of different frequencies to a Tensor.
        Each channel of the input Tensor is incremented by a sinusoid of a different
        frequency and phase in one of the positional dimensions.
        This allows attention to learn to use absolute and relative positions.
        Timing signals should be added to some precursors of both the query and the
        memory inputs to attention.
        The use of relative position is possible because sin(a+b) and cos(a+b) can be
        experessed in terms of b, sin(a) and cos(a).
        x is a Tensor with n "positional" dimensions, e.g. one dimension for a
        sequence or two dimensions for an image
        We use a geometric sequence of timescales starting with
        min_timescale and ending with max_timescale.  The number of different
        timescales is equal to channels // (n * 2). For each timescale, we
        generate the two sinusoidal signals sin(timestep/timescale) and
        cos(timestep/timescale).  All of these sinusoids are concatenated in
        the channels dimension.
        Args:
            x: a Tensor with shape [batch, d1 ... dn, channels]
            min_timescale: a float
            max_timescale: a float
        Returns:
            a Tensor the same shape as x.
        """
        static_shape = x.get_shape().as_list()
        num_dims = len(static_shape) - 2
        channels = tf.shape(x)[-1]
        num_timescales = channels // (num_dims * 2)
        log_timescale_increment = (
                math.log(float(max_timescale) / float(min_timescale)) /
                (tf.cast(num_timescales, tf.float32) - 1))
        
        ddwdwe = tf.cast(tf.range(num_timescales), tf.float32) * -log_timescale_increment
        inv_timescales = min_timescale * tf.exp(
                ddwdwe)
        for dim in range(num_dims):
            length = tf.shape(x)[dim + 1]
            position = tf.cast(tf.range(length), tf.float32)
            scaled_time = tf.expand_dims(position, 1) * tf.expand_dims(
                    inv_timescales, 0)
            signal = tf.concat([tf.sin(scaled_time), tf.cos(scaled_time)], axis=1)
            prepad = dim * 2 * num_timescales
            postpad = channels - (dim + 1) * 2 * num_timescales
            signal = tf.pad(signal, [[0, 0], [prepad, postpad]])
            for _ in range(1 + dim):
                signal = tf.expand_dims(signal, 0)
            for _ in range(num_dims - 1 - dim):
                signal = tf.expand_dims(signal, -2)
            x += signal
        return x


class RNN_Decoder(tf.keras.Model):
    def __init__(self, embedding_dim, units, vocab_size):
        super(RNN_Decoder, self).__init__()
        self.units = units

        self.embedding = tf.keras.layers.Embedding(vocab_size, embedding_dim)

        self.lstm = tf.keras.layers.LSTMCell(self.units,
                                             recurrent_initializer='glorot_uniform')
        
        self.lstm_2 = tf.keras.layers.LSTM(self.units,  
                                             return_sequences=True,
                                             return_state=True,
                                             recurrent_initializer='glorot_uniform')

        self.fc1 = tf.keras.layers.Dense(self.units)
        self.fc2 = tf.keras.layers.Dense(vocab_size)

        self.attention = BahdanauAttention(self.units)

        # self.dropout = tf.keras.layers.Dropout(0.5)
        # self.b_n = tf.keras.layers.BatchNormalization()

        self.get_initial_state = self.lstm.get_initial_state

    def call(self, x, features, state_output, hidden):
        # defining attention as a separate model
        context_vector, attention_weights = self.attention(
            features, state_output)

        # x shape after passing through embedding == (batch_size, 1, embedding_dim)
        x = self.embedding(x) # 24, 1  [[12], [12], [12], ...] --> [[[0.42432432, 0.432243]], []]

        # x shape after concatenation == (batch_size, 1, embedding_dim + hidden_size)
        x = tf.expand_dims(tf.concat([context_vector, tf.squeeze(x, axis=1)], axis=-1), 1) 
        # x = tf.concat([context_vector, tf.squeeze(x, axis=1)], axis=-1)

        out, hidden_state, cell_state = self.lstm_2(x, initial_state=hidden)
        
        # passing the concatenated vector to the LSTM
        # hidden_state, cell_state = self.lstm(x, hidden)

        # shape == (batch_size, max_length, hidden_size)
        x = self.fc1(tf.squeeze(out, axis=1))
        # x = self.fc1(hidden_state)

        # x = self.dropout(x)
        # x = self.b_n(x)

        # output shape == (batch_size * max_length, vocab)
        x = self.fc2(x)

        return x, hidden_state, [hidden_state, cell_state], attention_weights
        # return x, hidden_state, cell_state, attention_weights

    def reset_state(self, batch_size):
        return tf.zeros((batch_size, self.units))


class Rnn_Global_Decoder(tf.keras.Model):  # Luong-attention decoder
    def __init__(self, embedding_dim, units, vocab_size, scoring_type="dot"):
        super(Rnn_Global_Decoder, self).__init__()

        self.units = units
        self.embedding = tf.keras.layers.Embedding(vocab_size, embedding_dim)

        self.gru = tf.keras.layers.GRU(self.units,
                                       return_sequences=True,
                                       return_state=True,
                                       recurrent_initializer='glorot_uniform')

        self.wc = tf.keras.layers.Dense(units, activation='tanh')
        self.ws = tf.keras.layers.Dense(vocab_size)

        # For Attention
        self.wa = tf.keras.layers.Dense(units)
        self.wb = tf.keras.layers.Dense(units)

        # For Score 3 i.e. Concat score
        self.Vattn = tf.keras.layers.Dense(1)
        self.wd = tf.keras.layers.Dense(units, activation='tanh')

        self.scoring_type = scoring_type

    def call(self, sequence, features, hidden):

        # features : (64,49,256)
        # hidden : (64,512)

        embed = self.embedding(sequence)
        # embed ==> (64,1,256) ==> decoder_input after embedding (embedding dim=256)

        output, state = self.gru(embed)
        # output :(64,1,512)

        score = 0

        # Dot Score as per paper(Dot score : h_t (dot) h_s') (NB:just need to tweak gru units to 256)
        '''----------------------------------------------------------'''
        if (self.scoring_type == 'dot'):
            xt = output  # (64,1,512)
            xs = features  # (256,49,64)
            score = tf.matmul(xt, xs, transpose_b=True)

            # score : (64,1,49)

        '''----------------------------------------------------------'''
        '''----------------------------------------------------------'''

        # General Score as per Paper ( General score: h_t (dot) Wa (dot) h_s')
        '''----------------------------------------------------------'''
        if (self.scoring_type == 'general'):
            score = tf.matmul(output, self.wa(features), transpose_b=True)
            # score :(64,1,49)
        '''----------------------------------------------------------'''
        '''----------------------------------------------------------'''

        # Concat score as per paper (score: VT*tanh(W[ht;hs']))
        '''----------------------------------------------------------'''
        # https://www.tensorflow.org/api_docs/python/tf/tile
        if (self.scoring_type == 'concat'):
            tiled_features = tf.tile(features, [1, 1, 2])  # (64,49,512)
            tiled_output = tf.tile(output, [1, 148, 1])  # (64,49,512)

            concating_ht_hs = tf.concat(
                [tiled_features, tiled_output], 2)  # (64,49,1024)

            tanh_activated = self.wd(concating_ht_hs)
            score = self.Vattn(tanh_activated)
            # score :(64,49,1), but we want (64,1,49)
            score = tf.squeeze(score, 2)
            # score :(64,49)
            score = tf.expand_dims(score, 1)

            # score :(64,1,49)
        '''----------------------------------------------------------'''
        '''----------------------------------------------------------'''

        # alignment vector a_t
        alignment = tf.nn.softmax(score, axis=2)
        # alignment :(64,1,49)

        # context vector c_t is the average sum of encoder output
        context = tf.matmul(alignment, features)
        # context : (64,1,256)

        # Combine the context vector and the LSTM output

        output = tf.concat([tf.squeeze(context, 1), tf.squeeze(output, 1)], 1)
        # output: concat[(64,1,256):(64,1,512)] = (64,768)

        output = self.wc(output)
        # output :(64,512)

        # Finally, it is converted back to vocabulary space: (batch_size, vocab_size)
        logits = self.ws(output)
        # logits/predictions: (64,8239) i.e. (batch_size,vocab_size))

        return logits, state, alignment

    def reset_state(self, batch_size):
        return tf.zeros((batch_size, self.units))


class Training:
    global DATASET_PATH, ANNOTATION_FILE

    manager = None

    def __init__(self, manager, **kwargs):
        global meta_path
        self.tokenizer = None
        self.img_to_cap_vector = None

        self.manager = manager
        self.top_k = kwargs["top_k"]
        self.image_count = kwargs["image_count"]
        self.BATCH_SIZE = kwargs["BATCH_SIZE"]
        self.BUFFER_SIZE = kwargs["BUFFER_SIZE"]
        self.embedding_dim = kwargs["embedding_dim"]
        self.units = kwargs["units"]
        self.vocab_size = self.top_k + 1
        self.EPOCHS = kwargs["EPOCHS"]
        self.conv_var = kwargs["conv_var"]

        os.makedirs(os.path.dirname(meta_path), exist_ok=True)
        with open(meta_path, "w") as meta:
            meta.write(
                f"{kwargs['embedding_dim']}\n{kwargs['units']}\n{kwargs['top_k'] + 1}\n")

        self.logger = log_init(manager.get_model_path(), "model_log")
        self.logger.info(
            f"Model '{manager.get_model_path()}' has been created with these params:")
        for k, v in kwargs.items():
            self.logger.info(f"{k} - {v}")

        self.logger.info(
            f"---------------MODEL AND UTILS SUMMARY---------------")
        self.logger.info(
            f"ENCODER_INIT:\n{inspect.getsource(CNN_Encoder.__init__)}")
        self.logger.info(
            f"ENCODER_CALL:\n{inspect.getsource(CNN_Encoder.call)}")
        self.logger.info(
            f"IMG_PREPROCESSING:\n{inspect.getsource(self.load_image)}")
        self.logger.info(f"RESIZE_HEIGHT: {RESIZED_IMG_H}")
        self.logger.info(f"RESIZE_WIDTH: {RESIZED_IMG_W}")
        self.logger.info(f"DATA_SPLIT_INDEX: {DATA_SPLIT}")
        self.logger.info(
            f"-----------------------------------------------------")
        
        # self.sr = cv2.dnn_superres.DnnSuperResImpl_create()
        # self.sr.readModel("C:/Users/shace/Downloads/EDSR_x2.pb")
        # self.sr.setModel("edsr", 3)

    def auto_train(self):
        self.data_preprocess()
        self.train()

    def loss_function(self, real, pred, loss_object):
        # print(f"real = {real}  ;  pred = {pred}")
        # tf.print(real)
        # tf.print(pred)

        mask = tf.math.logical_not(tf.math.equal(real, 0))

        # tf.print(mask)

        loss_ = loss_object(real, pred)

        # tf.print(loss_)

        mask = tf.cast(mask, dtype=loss_.dtype)
        loss_ *= mask
        # tf.print(loss_)   
        # tf.print(tf.reduce_mean(loss_))
        # print(f"loss = {loss_}")
        return tf.reduce_mean(loss_)

    @staticmethod
    def plot_attention(image, result, attention_plot):
        # temp_image = np.array(Image.open(image).convert('RGB'))

        # fig = plt.figure(figsize=(10, 10))

        # len_result = len(result)
        # for i in range(len_result):
        #     temp_att = np.resize(attention_plot[i], (4, 37))
        #     grid_size = max(np.ceil(len_result / 2), 2)
        #     ax = fig.add_subplot(grid_size, grid_size, i + 1)
        #     ax.set_title(result[i])
        #     img = ax.imshow(temp_image)
        #     ax.imshow(temp_att, cmap='gray', alpha=0.6, extent=img.get_extent())

        # #plt.tight_layout()
        # plt.show()

        plotting = []
        for _ in range(len(result)):
            plotting.append(plt.imread(image))  # real ; pred ; plt image

        fig, axes = plt.subplots(nrows=len(result) - 10, ncols=1)

        for ax, plot, i in zip(axes.flat, plotting, range(0, len(result) - 10)):
            temp_att = np.resize(attention_plot[i], (4, 37))
            img = ax.imshow(plot)
            ax.set_title(result[i])
            ax.axis('off')
            ax.imshow(temp_att, cmap='gray', alpha=0.6,
                      extent=img.get_extent())
        plt.show()

    # @tf.function
    def train_step(self, img_tensor, target, decoder, encoder, optimizer, loss_object):
        loss = 0

        # initializing the hidden state for each batch
        # because the captions are not related from image to image
        # hidden = decoder.reset_state(batch_size=target.shape[0])
        hidden = decoder.get_initial_state(batch_size=target.shape[0], dtype="float32")
        # hidden = decoder.get_initial_state(
        #     batch_size=target.shape[0], dtype="float32")
        state_out = hidden[0]

        # 24, 1  [[12], [12], [12], ...]
        dec_input = tf.expand_dims(
            [self.tokenizer.word_index['<start>']] * target.shape[0], 1)

        with tf.GradientTape() as tape:
            features = encoder(img_tensor)
            
            for i in range(1, target.shape[1]):
                # passing the features through the decoder
                predictions, state_out, hidden, _ = decoder(dec_input, features, state_out, hidden)
                
                loss += self.loss_function(target[:, i], predictions, loss_object)
                # b = tf.reduce_min(tf.where(tf.equal(predictions, val)))
                # a = tf.expand_dims(tf.math.reduce_max(predictions, -1), 1)
                # using teacher forcing
                dec_input = tf.expand_dims(target[:, i], 1) 

        total_loss = (loss / int(target.shape[1]))

        trainable_variables = encoder.trainable_variables + decoder.trainable_variables

        gradients = tape.gradient(loss, trainable_variables)

        optimizer.apply_gradients(zip(gradients, trainable_variables))

        return loss, total_loss

    # def save_convoluted_images(self, dataset):
    #     for img, path in tqdm(dataset):
    #         batch_features = image_features_extract_model(img)
    #         print(tf.shape(batch_features).numpy())
    #         batch_features = tf.reshape(batch_features,
    #                                     (batch_features.shape[0], -1, batch_features.shape[3]))
    #         print(tf.shape(batch_features).numpy())
    #         for bf, p in zip(batch_features, path):
    #             path_of_feature = p.numpy().decode("utf-8")
    #             np.save(path_of_feature, bf.numpy())  # сохранить извлеченные признаки в форме (16, ?, 2048)

    def save_reshaped_images(self, dataset):
        for img, path in tqdm(dataset):
            for im, p in zip(img, path):
                path_of_feature = p.numpy().decode("utf-8")
                # сохранить извлеченные признаки в форме (16, ?, 2048)
                np.save(path_of_feature, im.numpy())

    # Find the maximum length of any caption in our dataset
    def calc_max_length(self, tensor):
        return max(len(t) for t in tensor)

    # Load the numpy files
    def map_func_alb(self, img_name, cap):
        img = tf.io.read_file(img_name.decode("utf-8"))
        img = tf.image.decode_png(img, channels=3)
        # img = self.sr.upsample(img.numpy())
        # img = tf.image.resize(img, (int(img.shape[0]*0.8), int(img.shape[1]*0.8)), ResizeMethod.GAUSSIAN)
        # enhancer = ImageEnhance.Sharpness(Image.fromarray(np.uint8(img.numpy())).convert('RGB'))
        # img = enhancer.enhance(3)
        # data = {"image":img.numpy()}

        # aug_data = transforms(**data)
        # aug_img = aug_data["image"]
        img = make_fix_size(img.numpy(), RESIZED_IMG_W, RESIZED_IMG_H, False)
        # img = np.append(img, np.repeat(255, RESIZED_IMG_W*RESIZED_IMG_H).reshape(RESIZED_IMG_H, RESIZED_IMG_W, 1) ,axis=-1)  # добавление значение Alpha для совместимости        

        img = tf.image.resize(img, (RESIZED_IMG_H, RESIZED_IMG_W))

        # to_img = tf.keras.preprocessing.image.array_to_img(img)
        # to_img.show()
        # print(tf.image.convert_image_dtype(img, dtype=tf.float32))
        # img = tf.image.convert_image_dtype(img, dtype=tf.float32)
        # img_tensor = np.load(img_name.decode('utf-8') + '.npy')
        return img.numpy(), cap

    # Load the numpy files
    def map_func(self, img_name, cap):
        img = tf.io.read_file(img_name.decode("utf-8"))
        img = tf.image.decode_png(img, channels=3)
        img = tf.image.resize(img, (RESIZED_IMG_H, RESIZED_IMG_W), method=tf.image.ResizeMethod.GAUSSIAN,
                              antialias=True)

        # print(tf.image.convert_image_dtype(img, dtype=tf.float32))
        # img = tf.image.convert_image_dtype(img, dtype=tf.float32)
        # img_tensor = np.load(img_name.decode('utf-8') + '.npy')

        return img.numpy(), cap

    @staticmethod
    def load_image(image_path):
        img = tf.io.read_file(image_path)
        img = tf.image.decode_png(img, channels=3)
        # img = tf.image.resize(img, (int(img.shape[0]*0.8), int(img.shape[1]*0.8)), ResizeMethod.GAUSSIAN)
        # enhancer = ImageEnhance.Sharpness(Image.fromarray(np.uint8(img.numpy())).convert('RGB'))
        # img = enhancer.enhance(3)
        # img = tf.image.convert_image_dtype(img, dtype=tf.float32)

        img = make_fix_size(img.numpy(), RESIZED_IMG_W, RESIZED_IMG_H, False)
        img = np.append(img, np.repeat(255, RESIZED_IMG_W*RESIZED_IMG_H).reshape(RESIZED_IMG_H, RESIZED_IMG_W, 1) ,axis=-1)  # добавление значение Alpha для совместимости
        # img = tf.image.convert_image_dtype(img, dtype=tf.float32)
        img = tf.image.resize(img, (RESIZED_IMG_H, RESIZED_IMG_W))

        # to_img = tf.keras.preprocessing.image.array_to_img(img)
        # to_img.show()
        # img = tf.image.rgb_to_grayscale(img)
        # Image.fromarray(np.asarray(img.numpy().astype(np.uint8))).show()
        return img, image_path

    def get_image_to_caption(self, annotations_file):
        # словарь, в который автоматически будет добавляться лист
        image_path_to_caption = collections.defaultdict(list)
        # при попытке доступа к несущестувующему ключу
        for val in annotations_file['annotations']:
            caption = f"<start> {val['caption']} <end>"
            image_path = DATASET_PATH + val["image_id"] + ".png"
            image_path_to_caption[image_path].append(
                caption)  # словарь типа 'путь_фото': ['описание1', 'описание2', ...]

        return image_path_to_caption

    def data_preprocess(self):  # составление токенайзера
        global tokenizer_path, meta_path

        with open(ANNOTATION_FILE, 'r') as f:
            annotations = json.load(f)

        image_path_to_caption = self.get_image_to_caption(annotations)

        image_paths = list(image_path_to_caption.keys())
        random.shuffle(image_paths)

        train_image_paths = image_paths[:self.image_count]
        self.logger.info(
            f"Annotations and image paths have been successfully extracted from {ANNOTATION_FILE}\ntotal images count - {len(train_image_paths)}")

        train_captions = []  # описания для тренировки
        img_name_vector = []  # пути к фото

        for image_path in train_image_paths:
            caption_list = image_path_to_caption[image_path]
            train_captions.extend(caption_list)
            img_name_vector.extend([image_path] * len(caption_list))

        # print(img_name_vector[:5])  # 5 одинаковых путей к фото соответсвуют 5 разных описаний
        # print(train_captions[:5])

        # Получение уникальных путей к фото
        # encode_train = sorted(set(img_name_vector))

        # # Feel free to change batch_size according to your system configuration
        # image_dataset = tf.data.Dataset.from_tensor_slices(encode_train)
        # image_dataset = image_dataset.map(
        #     Training.load_image, num_parallel_calls=tf.data.AUTOTUNE).batch(64)

        # print(list(image_dataset)[:100])  # после мэппинга и препроцессинга получаются тензоры формы (16, 299, 299, 3)
        # self.logger.info(f"Images are being preprocessed and saved...")
        # start = time.time()
        # try:
        #     pass
        #     # self.save_reshaped_images(image_dataset)
        # except:
        #     self.logger.error("Something went wrong!")
        #     self.manager.remove_temp()
        #     sys.exit()

        # self.logger.info(f"Done in {time.time() - start:.2f} sec")

        os.makedirs(os.path.dirname(tokenizer_path), exist_ok=True)
        with open(tokenizer_path, 'w', encoding='utf-8') as f:
            # Choose the top top_k words from the vocabulary
            self.tokenizer = tf.keras.preprocessing.text.Tokenizer(num_words=self.top_k, split=' ', oov_token="<unk>",
                                                                   lower=False, filters='%')

            self.tokenizer.fit_on_texts(train_captions)
            self.tokenizer.word_index['<pad>'] = 0
            # создание токенайзера и сохранение его в json
            self.tokenizer.index_word[0] = '<pad>'

            tokenizer_json = self.tokenizer.to_json()
            f.write(json.dumps(tokenizer_json, ensure_ascii=False))

        self.logger.info("\nTokenizer has been created with params:")
        for k, v in self.tokenizer.get_config().items():
            self.logger.info(f"{k} - {v}")

        # Create the tokenized vectors
        train_seqs = self.tokenizer.texts_to_sequences(
            train_captions)  # преобразование текста в последовательность чисел
        # print(train_seqs[:100])

        # Pad each vector to the max_length of the captions
        # If you do not provide a max_length value, pad_sequences calculates it automatically
        cap_vector = tf.keras.preprocessing.sequence.pad_sequences(
            train_seqs, padding='post')  # приведение всех
        # последовательностей к одинаковой длине
        # print(cap_vector[:100])

        # Calculates the max_length, which is used to store the attention weights

        # os.makedirs(os.path.dirname(tokenizer_params_path), exist_ok=True)
        max_length = self.calc_max_length(train_seqs)
        # with open(tokenizer_params_path, 'w') as f:

        #     f.write(str(max_length))

        with open(meta_path, "a") as meta:
            meta.write(f"{max_length}")

        self.img_to_cap_vector = collections.defaultdict(list)
        for img, cap in zip(img_name_vector, cap_vector):
            self.img_to_cap_vector[img].append(cap)

        self.logger.info(
            f"Img_to_cap vector is compiled. Max cap length - {max_length}")
        # for k, v in self.img_to_cap_vector.items():
        #     print(f"key = {k} ; value = {v}")

    def train(self):
        global checkpoint_path
        # Create training and validation sets using an 80-20 split randomly.
        img_keys = list(self.img_to_cap_vector.keys())
        random.shuffle(img_keys)
        # print(img_keys[:100])

        slice_index = int(len(img_keys) * DATA_SPLIT)
        img_name_train_keys, img_name_val_keys = img_keys[:
                                                          slice_index], img_keys[slice_index:]

        img_name_train = []
        cap_train = []
        for imgt in img_name_train_keys:  # тут теперь идут все фотки, а не только 80%
            capt_len = len(self.img_to_cap_vector[imgt])
            img_name_train.extend([imgt] * capt_len)
            cap_train.extend(self.img_to_cap_vector[imgt])

        img_name_val = []
        cap_val = []
        for imgv in img_name_val_keys:
            capv_len = len(self.img_to_cap_vector[imgv])
            img_name_val.extend([imgv] * capv_len)
            cap_val.extend(self.img_to_cap_vector[imgv])

        # print(img_name_train[:20])
        # print(cap_train[:20])
        self.logger.info(
            f"Dataset has been splitted:\nIMG_TRAIN - {len(img_name_train)}\nCAP_TRAIN - {len(cap_train)}\nIMG_VAL - {len(img_name_val)}\nCAP_VAL - {len(cap_val)}")

        num_steps = len(img_name_train) // self.BATCH_SIZE
        dataset = tf.data.Dataset.from_tensor_slices(
            (img_name_train, cap_train))
        # Use map to load the numpy files in parallel
        dataset = dataset.map(lambda item1, item2: tf.numpy_function(
            self.map_func_alb, [item1, item2], [tf.float32, tf.int32]),
            num_parallel_calls=tf.data.AUTOTUNE)

        # Shuffle and batch
        dataset = dataset.shuffle(self.BUFFER_SIZE).batch(self.BATCH_SIZE)
        dataset = dataset.prefetch(buffer_size=tf.data.AUTOTUNE)

        self.logger.info(f"Dataset is ready")

        # инициализация параметров нейросети
        encoder = CNN_Encoder(self.embedding_dim)
        # encoder = get_cnn_model()
        decoder = RNN_Decoder(self.embedding_dim, self.units, self.vocab_size)

        optimizer = tf.keras.optimizers.Adam()
        loss_object = tf.keras.losses.SparseCategoricalCrossentropy(
            from_logits=True, reduction='none')

        ckpt = tf.train.Checkpoint(encoder=encoder,
                                   decoder=decoder,
                                   optimizer=optimizer)
        ckpt_manager = tf.train.CheckpointManager(
            ckpt, checkpoint_path, max_to_keep=10)
        # ckpt_manager = tf.train.CheckpointManager(
        #     ckpt, "C:\\Users\\shace\\Documents\\GitHub\\im2latex\\trained_models\\model_latex_x22\\checkpoints\\", max_to_keep=10)

        self.logger.info(f"OPTIMIZER SUMMARY:\n{optimizer.get_config()}")
        # start_epoch = 0
        # if ckpt_manager.latest_checkpoint:
        #     # start_epoch = int(ckpt_manager.latest_checkpoint.split('-')[-1])
        #     # restoring the latest checkpoint in checkpoint_path
        #     ckpt.restore(ckpt_manager.latest_checkpoint)

        # adding this in a separate cell because if you run the training cell
        # many times, the loss_plot array will be reset
        loss_plot = []

        for epoch in range(self.EPOCHS):
            start = time.time()
            total_loss = 0

            for (batch, (img_tensor, target)) in tqdm(enumerate(dataset)):
                # print(f"img_tensor shape = {img_tensor.shape}")
                # print(f"target shape = {target.shape}")

                # print(img_tensor.numpy())
                # print(target.numpy())
                batch_loss, t_loss = self.train_step(
                    img_tensor, target, decoder, encoder, optimizer, loss_object)
                total_loss += t_loss

                if batch % 100 == 0:
                    average_batch_loss = batch_loss.numpy() / \
                        int(target.shape[1])
                    self.logger.info(
                        f'Epoch {epoch + 1} Batch {batch} Loss = {batch_loss.numpy()} / {int(target.shape[1])} = {average_batch_loss:.4f}')
            # storing the epoch end loss value to plot later
            loss_plot.append(total_loss / num_steps)

            ckpt_manager.save()
            self.logger.info(f"Epoch {epoch} checkpoint saved!")

            self.logger.info(
                f'Epoch {epoch + 1} Loss {total_loss / num_steps:.6f}')
            self.logger.info(
                f'Time taken for 1 epoch {time.time() - start:.2f} sec\n')

        self.logger.info(f"Training is done!")

        plt.plot(loss_plot)
        plt.xlabel('Epochs')
        plt.ylabel('Loss')
        plt.title('Loss Plot')
        plt.show()

        print("Deleting temporary files, please wait...")
        self.manager.remove_temp()


class Prediction:
    def __init__(self):
        global meta_path
        self.max_length = None
        self.tokenizer = None

        try:
            with open(tokenizer_path, "r") as f:
                data = json.load(f)
                self.tokenizer = tokenizer_from_json(data)
        except:
            raise IOError("Something went wrong with initializing tokenizer")

        try:
            with open(meta_path, "r") as meta:
                for index, line in enumerate(meta):
                    if index == 0:
                        self.embedding_dim = int(line)
                    elif index == 1:
                        self.units = int(line)
                    elif index == 2:
                        self.vocab_size = int(line)
                    elif index == 3:
                        self.max_length = int(line)
        except:
            raise IOError("Meta file reading failed")

    def index(self, array, item):
        for idx, val in np.ndenumerate(array):
            if val == item:
                return idx[1]

    def find_n_best(self, array, n):
        probs = [np.partition(array[0], i)[i] for i in range(-1, -n - 1, -1)]
        ids = [self.index(array, p) for p in probs]
        return [[prob, id] for prob, id in zip(probs, ids)]

    def beam_evaluate(self, image, decoder, encoder, beam_width=10, temperature=1.0):
        global checkpoint_path
        # attention_plots = [np.zeros((self.max_length, 148)) for _ in range(beam_width)]

        hidden = decoder.get_initial_state(batch_size=1, dtype="float32")
        state_out = hidden[0]

        temp_input = tf.expand_dims(Training.load_image(image)[0], 0)

        features = encoder(temp_input)

        dec_input = tf.expand_dims([self.tokenizer.word_index['<start>']], 0)

        predictions, state_out, hidden, _ = decoder(
            dec_input, features, state_out, hidden)
        predictions = tf.nn.softmax(predictions / temperature).numpy()

        init = self.find_n_best(predictions, beam_width)
        results = [[obj[0], obj[1], hidden, self.tokenizer.index_word[int(obj[1])] + " ", state_out] for obj in
                   init]  # 0 - prob ; 1 - id ; 2 - hidden

        for i in range(self.max_length):
            tmp_res = []

            for r in results:
                tmp_preds, tmp_state_out, tmp_hidden, attention_plot = decoder(tf.expand_dims([r[1]], 0), features,
                                                                               r[4], r[2])

                for obj in self.find_n_best(tf.nn.softmax(tmp_preds / temperature).numpy(), beam_width):
                    tmp_res.append(
                        [obj[0] + r[0], obj[1], tmp_hidden, r[3] + self.tokenizer.index_word[int(obj[1])] + " ",
                         tmp_state_out, attention_plot])  # multiplied scores, curr id, hidden, prev id

            results.clear()
            tmp_res.sort(reverse=True, key=lambda x: x[0])
            # attention_plot[i] = tf.reshape(attention_weights, (-1,)).numpy()
            for el in range(beam_width):
                results.append(tmp_res[el])

            # for res, att in zip(results, attention_plots):
            #     att[i] = tf.reshape(res[5], (-1,)).numpy()

            # if any(self.tokenizer.index_word[int(results[i][1])] == '<end>' for i in range(len(results))):
            #     break

            if all(['<end>' in r[3] for r in results]):
                break

        # for el in results:
        #     tf.print(el[3] + "\n")
        # tf.print(results[0][3])
        # return [results[0][3]], None
        return [el[3] for el in results], None

    def beam_evaluate_2(self, image, decoder, encoder, beam_width=5):
        hidden = decoder.get_initial_state(batch_size=1, dtype="float32")
        state_out = hidden[0]

        temp_input = tf.expand_dims(Training.load_image(image)[0], 0)

        dec_input = tf.expand_dims([self.tokenizer.word_index['<start>']], 0)

        features = encoder(temp_input)
        result = []
        for _ in range(self.max_length):
            ids = []

            for _ in range(beam_width):
                predictions, state_out, tmp_hidden, _ = decoder(
                    dec_input, features, state_out, hidden)
                ids.append([tf.random.categorical(predictions, 1)[0][0].numpy(),
                            max(tf.nn.softmax(predictions).numpy().tolist()[0]), tmp_hidden, state_out])  # predicted_id

            md = mode([id[0] for id in ids])
            tmp = []
            for el in ids:
                if el[0] == md:
                    tmp.append(el)
            best = max(el[1] for el in ids)
            # hidden = random.choice([h[2] for h in tmp])
            for el in tmp:
                if el[1] == best:
                    hidden = el[2]
                    state_out = el[3]
                    break

            result.append(self.tokenizer.index_word[md])

            if self.tokenizer.index_word[md] == '<end>':
                return result, None

            dec_input = tf.expand_dims([md], 0)
            # tf.print(hidden)

        return result, None

    def categorical_evaluate(self, image, decoder, encoder):
        global checkpoint_path
        # attention_plot = np.zeros((self.max_length, 148))

        hidden = decoder.get_initial_state(batch_size=1, dtype="float32")
        state_out = hidden[0]

        temp_input = tf.expand_dims(Training.load_image(image)[0], 0)
        # img_tensor_val = image_features_extract_model(temp_input)
        # img_tensor_val = tf.reshape(img_tensor_val, (img_tensor_val.shape[0],
        #                                             -1,
        #                                             img_tensor_val.shape[3]))

        features = encoder(temp_input)
        # hidden = hidden_enc

        dec_input = tf.expand_dims([self.tokenizer.word_index['<start>']], 0)
        result = []

        for i in range(self.max_length):

            predictions, state_out, hidden, attention_weights = decoder(dec_input,
                                                                        features,
                                                                        state_out, hidden)

            # [print(f"{val} - {sorted(attention_weights[0].numpy().tolist(), reverse=True).index(val)}") for val in
            #  attention_weights[0].numpy().tolist()]
            # print("---------------------")

            # attention_plot[i] = tf.reshape(attention_weights, (-1,)).numpy()

            predicted_id = tf.random.categorical(predictions, 1)[0][0].numpy()
            result.append(self.tokenizer.index_word[predicted_id])

            if self.tokenizer.index_word[predicted_id] == '<end>':
                # Training.plot_attention(image, result, attention_plot)
                return result, None

            dec_input = tf.expand_dims([predicted_id], 0)
            # tf.print(hidden)

        # attention_plot = attention_plot[:len(result), :]

        return result, None

    def levenshteinDistance(s1, s2):
        if len(s1) > len(s2):
            s1, s2 = s2, s1

        distances = range(len(s1) + 1)
        for i2, c2 in enumerate(s2):
            distances_ = [i2 + 1]
            for i1, c1 in enumerate(s1):
                if c1 == c2:
                    distances_.append(distances[i1])
                else:
                    distances_.append(
                        1 + min((distances[i1], distances[i1 + 1], distances_[-1])))
            distances = distances_
        return distances[-1]

    def predict(self, decoder_type, image_path: str):
        global checkpoint_path

        encoder = CNN_Encoder(self.embedding_dim)
        decoder = RNN_Decoder(self.embedding_dim, self.units, self.vocab_size)
        optimizer = tf.keras.optimizers.Adam()

        ckpt = tf.train.Checkpoint(encoder=encoder,
                                   decoder=decoder,
                                   optimizer=optimizer)
        ckpt_manager = tf.train.CheckpointManager(
            ckpt, checkpoint_path, max_to_keep=5)
        if ckpt_manager.latest_checkpoint:
            # restoring the latest checkpoint in checkpoint_path
            ckpt.restore(ckpt_manager.latest_checkpoint).expect_partial()
            # print("Restored from {}".format(ckpt_manager.latest_checkpoint))
        else:
            print("Restored from scratch")

        if decoder_type == "categorical":
            result, attention_plot = self.categorical_evaluate(
                image_path, decoder, encoder)
        elif decoder_type == "beam":
            result, attention_plot = self.beam_evaluate(
                image_path, decoder, encoder, beam_width=5, temperature=0.33)
        else:
            raise IOError("decoder error")

        return result
        # print('Prediction Caption:', ' '.join(result))
        # Training.plot_attention(image_path, result, attention_plot)
        # print(f"len res = {len(result)}")
        # opening the image
        # Image.open(open(image_path, 'rb'))


class Image2Latex:
    loaded = False
    d_p = "images_150\\"
    c_p = "5_dataset_large.json"
    top_k = 300
    image_count = 100000
    BATCH_SIZE = 24
    BUFFER_SIZE = 100
    embedding_dim = 200
    units = 200
    EPOCHS = 100
    conv_var = 9

    def __init__(self, model_name: str, working_path: str = ""):
        self.model_path = os.path.abspath(
            ".") + "\\trained_models\\" + working_path + model_name
        self.init()

    def init(self):
        global tokenizer_path, checkpoint_path, meta_path
        tokenizer_path = self.model_path + "\\tokenizer.json"
        checkpoint_path = self.model_path + "\\checkpoints\\"
        meta_path = self.model_path + "\\meta.txt"

        if os.path.exists(checkpoint_path) and len(os.listdir(checkpoint_path)) != 0 and os.path.exists(tokenizer_path) and os.path.exists(meta_path):
            print(f"Model has been found at {self.model_path}")
            self.loaded = True
        elif not os.path.exists(self.model_path):
            print(f"Model will be created at {self.model_path}")
        else:
            if input(f"Model is corrupted. Remove? (y/AB): ") == "y":
                shutil.rmtree(self.model_path)
                self.loaded = False
            else:
                self.loaded = True

    def get_model_path(self):
        return self.model_path

    def input(self, inp, val):
        return val if inp == "" else inp

    def train(self):
        global DATASET_PATH, ANNOTATION_FILE

        print(f"Caption file must be in {os.path.abspath('.')}")
        print(f"Dataset folder must be in {os.path.abspath('.')}\\datasets\\")
        if self.loaded:
            if input("You will loose your model. Proceed? (y/AB): ") == "y":
                shutil.rmtree(self.model_path)
                self.loaded = False
            else:
                return

        if not self.loaded:
            print("\nPlease enter some data for new model: ")
            try:
                self.d_p = self.input(
                    input(f"dataset_path - Default = {self.d_p}: "), self.d_p)
                self.c_p = self.input(
                    input(f"caption file - Default = {self.c_p}: "), self.c_p)
                self.top_k = int(
                    self.input(input(f"top_k - Default = {self.top_k} (number of top used words in caps): "),
                               self.top_k))
                self.image_count = int(
                    self.input(input(f"image_count - Default = {self.image_count} (number of photos to be used): "),
                               self.image_count))
                self.BATCH_SIZE = int(self.input(
                    input(f"BATCH_SIZE - Default = {self.BATCH_SIZE}: "), self.BATCH_SIZE))
                self.BUFFER_SIZE = int(
                    self.input(input(f"BUFFER_SIZE - Default = {self.BUFFER_SIZE}: "), self.BUFFER_SIZE))
                self.embedding_dim = int(
                    self.input(input(f"embedding_dim - Default = {self.embedding_dim}: "), self.embedding_dim))
                self.units = int(self.input(
                    input(f"units - Default = {self.units}: "), self.units))
                self.EPOCHS = int(self.input(
                    input(f"EPOCHS - Default = {self.EPOCHS}: "), self.EPOCHS))
                self.conv_var = int(self.input(
                    input(f"conv_var - Default = {self.conv_var}: "), self.conv_var))
            except:
                raise TypeError("Model params initialization failed")

            self.loaded = True
            DATASET_PATH = os.path.abspath(".") + "\\datasets\\" + self.d_p
            ANNOTATION_FILE = os.path.abspath(".") + "\\" + self.c_p

            train = Training(manager=self,
                             top_k=self.top_k,
                             image_count=self.image_count,
                             BATCH_SIZE=self.BATCH_SIZE,
                             BUFFER_SIZE=self.BUFFER_SIZE,
                             embedding_dim=self.embedding_dim,
                             units=self.units,
                             EPOCHS=self.EPOCHS,
                             conv_var=self.conv_var)

            train.auto_train()

    def predict(self, decoder, image_path: str):
        if not self.loaded:
            raise ValueError("Model is not loaded!")
        pred = Prediction()
        res = pred.predict(decoder, image_path)
        if decoder == "categorical" or decoder == "beam2" or decoder == "beam3":
            print(' '.join(res))
        if decoder == "beam":
            [print(r[:r.index("<end>")] + "\n") for r in res]

    def calulate_edit_dist_metric(self, dataset_path, caption_path, number=5000):
        if not self.loaded:
            raise ValueError("Model is not loaded!")
        with open(caption_path, 'r+') as file:
            capt = json.load(file)["annotations"]

        mean = 0
        pred = Prediction()

        random.shuffle(capt)

        for v in tqdm(capt[:number]):
            mean += levenshteinDistance(v["caption"],
                                        pred.predict("beam", dataset_path + v["image_id"] + ".png")[0].replace('<end>',
                                                                                                               '')[:-2])
        print(f"edit_dist = {mean / number}")

    def calulate_bleu_metric(self, dataset_path, caption_path, number=5000):
        if not self.loaded:
            raise ValueError("Model is not loaded!")
        with open(caption_path, 'r+') as file:
            capt = json.load(file)["annotations"]

        mean = 0
        pred = Prediction()

        random.shuffle(capt)

        for x, v in tqdm(enumerate(capt[:number])):
            # mean += levenshteinDistance(v["caption"], pred.predict("beam", dataset_path + v["image_id"] + ".png")[0].replace('<end>', '')[:-2])
            try:
                mean += nltk.translate.bleu_score.sentence_bleu(
                    [list(filter(lambda a: a != " ", v["caption"].split(" ")))], list(filter(lambda a: a != " ",
                                                                                             pred.predict("beam",
                                                                                                          dataset_path +
                                                                                                          v[
                                                                                                              "image_id"] + ".png")[
                                                                                                 0].replace('<end>',
                                                                                                            '')[
                                                                                                 :-2].split(" "))))
            except:
                print(
                    f"PIZDA\nr = {v['caption']}\np = {pred.predict('beam', dataset_path + v['image_id'] + '.png')[0].replace('<end>', '')[:-2]}\nx = {x}")
        print(f"edit_dist = {mean / number}")

    def random_predict(self, dataset_path, caption_path, number=9):
        if not self.loaded:
            raise ValueError("Model is not loaded!")
        with open(caption_path, 'r+') as file:
            capt = json.load(file)["annotations"]

        parameters = {'axes.labelsize': 10,
                      'axes.titlesize': 10,
                      'figure.subplot.hspace': 0.999}
        plt.rcParams.update(parameters)
        # print(plt.rcParams.keys())

        pred = Prediction()
        images = random.choices(capt, k=number)
        plotting = []
        for im in images:
            plotting.append([im["caption"], pred.predict("beam", dataset_path + im["image_id"] + ".png"),
                             plt.imread(dataset_path + im["image_id"] + ".png")])  # real ; pred ; plt image

        fig, axes = plt.subplots(nrows=5, ncols=1)

        for ax, plot in zip(axes.flat, plotting):
            ax.imshow(plot[2])
            edit_dist = levenshteinDistance(
                plot[0], plot[1][0].replace('<end>', '')[:-2])
            bleu = nltk.translate.bleu_score.sentence_bleu([list(filter(lambda a: a != " ", plot[0].split(" ")))], list(
                filter(lambda a: a != " ", plot[1][0][:plot[1][0].index("<end>")].split(" "))))
            ax.set(
                title=f"real = {plot[0]}\npred = {plot[1][0][:plot[1][0].index('<end>')]}\nbleu = {bleu}")
            ax.axis('off')
        plt.show()


def levenshteinDistance(s1, s2):
    if len(s1) > len(s2):
        s1, s2 = s2, s1

    distances = range(len(s1) + 1)
    for i2, c2 in enumerate(s2):
        distances_ = [i2 + 1]
        for i1, c1 in enumerate(s1):
            if c1 == c2:
                distances_.append(distances[i1])
            else:
                distances_.append(
                    1 + min((distances[i1], distances[i1 + 1], distances_[-1])))
        distances = distances_
    return distances[-1]


# config = tf.compat.v1.ConfigProto()
# config.gpu_options.allow_growth = True
# session = tf.compat.v1.Session(config=config)

enable_gpu(True, gb=10)
van = Image2Latex("model_latex_x25")

# van.calulate_bleu_metric("C:\\Users\\shace\\Documents\\GitHub\\im2latex\\datasets\\formula_images_png_5_large_resized\\",
#                       "C:\\Users\\shace\\Documents\\GitHub\\im2latex\\5_dataset_large.json")

van.train()

# van.random_predict("C:\\Users\\shace\\Documents\\GitHub\\im2latex\\datasets\\images_150\\",
#                    "C:\\Users\\shace\\Documents\\GitHub\\im2latex\\5_dataset_large.json", 5)
van.predict("beam", "C:\\Users\\shace\\Desktop\\lol2\\210443.png")
