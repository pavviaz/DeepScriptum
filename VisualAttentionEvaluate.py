import collections
import json
from json import encoder
import os
import random
import sys
import tensorflow as tf
import numpy as np
import time
import re
import shutil
import logging
import inspect
from PIL import Image
from matplotlib import pyplot as plt
from tensorflow import keras
from tensorflow.python.profiler.trace import Trace
from tqdm import tqdm
from keras_preprocessing.text import tokenizer_from_json
from tensorflow.keras.layers import Dense
from tensorflow.keras.layers import Conv2D
from tensorflow.keras.layers import AveragePooling2D
from tensorflow.keras.layers import MaxPooling2D
from tensorflow.keras.layers import Dropout

RESIZED_IMG_H = 85
RESIZED_IMG_W = 602
DATA_SPLIT = 1
global DATASET_PATH, ANNOTATION_FILE, tokenizer_path, tokenizer_params_path, checkpoint_path, meta_path


def InceptionV3_convolutional_model():
    image_model = tf.keras.applications.VGG16(include_top=False, weights=None)
    # image_model.trainable = False
    new_input = image_model.input
    hidden_layer = image_model.layers[-1].output
    return tf.keras.Model(new_input, hidden_layer)


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
                logical_gpus = tf.config.experimental.list_logical_devices('GPU')
                print(len(gpus), "Physical GPUs,", len(logical_gpus), "Logical GPUs")
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


class BahdanauAttention(tf.keras.Model):
    def __init__(self, units):
        super(BahdanauAttention, self).__init__()
        self.W1 = tf.keras.layers.Dense(units)
        self.W2 = tf.keras.layers.Dense(units)
        self.V = tf.keras.layers.Dense(1)

    def call(self, features, hidden):
        # features(CNN_encoder output) shape == (batch_size, 64, embedding_dim)

        # hidden shape == (batch_size, hidden_size)
        # hidden_with_time_axis shape == (batch_size, 1, hidden_size)
        hidden_with_time_axis = tf.expand_dims(hidden, 1)

        # attention_hidden_layer shape == (batch_size, 64, units)
        attention_hidden_layer = (tf.nn.tanh(self.W1(features) +
                                             self.W2(hidden_with_time_axis)))

        # score shape == (batch_size, 64, 1)
        # This gives you an unnormalized score for each image feature.
        score = self.V(attention_hidden_layer)

        # attention_weights shape == (batch_size, 64, 1)
        attention_weights = tf.nn.softmax(score, axis=1)

        # context_vector shape after sum == (batch_size, hidden_size)
        context_vector = attention_weights * features
        context_vector = tf.reduce_sum(context_vector, axis=1)

        return context_vector, attention_weights


# class TransformerEncoderBlock(tf.keras.Model):
#     def __init__(self, embed_dim, dense_dim, num_heads, **kwargs):
#         super().__init__(**kwargs)
#         self.embed_dim = embed_dim
#         self.dense_dim = dense_dim
#         self.num_heads = num_heads
#         self.attention = keras.layers.MultiHeadAttention(
#             num_heads=num_heads, key_dim=embed_dim
#         )
#         self.dense_proj = Dense(embed_dim, activation="relu")
#         self.layernorm_1 = keras.layers.LayerNormalization()

#     def call(self, inputs):
#         inputs = self.dense_proj(inputs)
#         attention_output = self.attention(
#             query=inputs, value=inputs, key=inputs, attention_mask=None
#         )
#         proj_input = self.layernorm_1(inputs + attention_output)
#         return proj_input


class CNN_Encoder(tf.keras.Model):
    m = None

    def __init__(self, embedding_dim, m=1):
        super(CNN_Encoder, self).__init__()

        self.rescale = tf.keras.layers.experimental.preprocessing.Rescaling(1. / 127.5, offset=-1)

        self.m == m
        if m == 1:  # Epoch 41 Loss 0.332003
            self.conv1 = Conv2D(filters=16, kernel_size=(3, 3))
            self.pool1 = MaxPooling2D()
            self.conv2 = Conv2D(filters=32, kernel_size=(3, 3))
            self.pool2 = MaxPooling2D()
            self.conv3 = Conv2D(filters=64, kernel_size=(3, 3))
            self.pool3 = MaxPooling2D()
            self.conv4 = Conv2D(filters=128, kernel_size=(3, 3))
            self.pool4 = MaxPooling2D()
            self.conv5 = Conv2D(filters=256, kernel_size=(3, 3))
            self.pool5 = MaxPooling2D(pool_size=(1, 2))

        elif m == 2:  # Результат чуть хуже, чем у 1 и 3
            self.conv1 = Conv2D(filters=16, kernel_size=(3, 3))
            self.pool1 = AveragePooling2D()
            self.conv2 = Conv2D(filters=32, kernel_size=(3, 3))
            self.pool2 = AveragePooling2D()
            self.conv3 = Conv2D(filters=64, kernel_size=(3, 3))
            self.pool3 = AveragePooling2D()
            self.conv4 = Conv2D(filters=128, kernel_size=(3, 3))
            self.pool4 = AveragePooling2D()
            self.conv5 = Conv2D(filters=256, kernel_size=(3, 3))
            self.pool5 = AveragePooling2D(pool_size=(1, 2))

        elif m == 3:  # Epoch 92 Loss 0.256778
            self.conv1 = Conv2D(filters=16, kernel_size=(3, 3))
            self.pool1 = MaxPooling2D()
            self.conv2 = Conv2D(filters=32, kernel_size=(4, 4))
            self.pool2 = MaxPooling2D(pool_size=(1, 2))
            self.conv3 = Conv2D(filters=64, kernel_size=(3, 3))
            self.pool3 = MaxPooling2D()
            self.conv4 = Conv2D(filters=128, kernel_size=(4, 4))
            self.pool4 = MaxPooling2D(pool_size=(2, 1))
            self.conv5 = Conv2D(filters=256, kernel_size=(3, 3))
            self.pool5 = MaxPooling2D(pool_size=(2, 4))

        elif m == 4:  # Epoch 60 Loss 0.179976
            self.conv1 = Conv2D(filters=16, kernel_size=(3, 3))
            self.pool1 = MaxPooling2D()
            self.conv2 = Conv2D(filters=32, kernel_size=(4, 4))
            self.pool2 = MaxPooling2D(pool_size=(1, 2))
            self.conv3 = Conv2D(filters=64, kernel_size=(3, 3))
            self.pool3 = MaxPooling2D()
            self.conv4 = Conv2D(filters=128, kernel_size=(4, 4))
            self.pool4 = MaxPooling2D(pool_size=(2, 1))
            self.conv5 = Conv2D(filters=256, kernel_size=(3, 3))
            self.pool5 = MaxPooling2D(pool_size=(2, 4))
            self.conv6 = Conv2D(filters=512, kernel_size=(2, 3))

        elif m == 5:
            self.conv1_block1 = Conv2D(filters=16, kernel_size=(3, 3), padding='same', activation='relu')
            self.conv2_block1 = Conv2D(filters=16, kernel_size=(3, 3), padding='same', activation='relu')
            self.pool_block1 = MaxPooling2D()

            self.conv1_block2 = Conv2D(filters=32, kernel_size=(3, 3), padding='same', activation='relu')
            self.conv2_block2 = Conv2D(filters=32, kernel_size=(3, 3), padding='same', activation='relu')
            self.pool_block2 = MaxPooling2D(pool_size=(1, 2))

            self.conv1_block3 = Conv2D(filters=64, kernel_size=(3, 3), padding='same', activation='relu')
            self.conv2_block3 = Conv2D(filters=64, kernel_size=(3, 3), padding='same', activation='relu')
            self.pool_block3 = MaxPooling2D()

            self.conv1_block4 = Conv2D(filters=128, kernel_size=(3, 3), padding='same', activation='relu')
            self.conv2_block4 = Conv2D(filters=128, kernel_size=(3, 3), padding='same', activation='relu')
            self.pool_block4 = MaxPooling2D(pool_size=(2, 1))

            self.conv1_block5 = Conv2D(filters=256, kernel_size=(3, 3), padding='same', activation='relu')
            self.conv2_block5 = Conv2D(filters=256, kernel_size=(3, 3), padding='same', activation='relu')
            self.pool_block5 = MaxPooling2D(pool_size=(3, 5))

            self.conv1_block6 = Conv2D(filters=512, kernel_size=(3, 3), padding='same', activation='relu')
            self.conv2_block6 = Conv2D(filters=512, kernel_size=(3, 3), padding='same', activation='relu')

        elif m == 6:
            self.conv1 = Conv2D(filters=16, kernel_size=(3, 3))
            self.pool1 = MaxPooling2D()
            self.conv2 = Conv2D(filters=32, kernel_size=(4, 4))
            self.pool2 = MaxPooling2D(pool_size=(1, 2))
            self.conv3 = Conv2D(filters=64, kernel_size=(3, 3))
            self.pool3 = MaxPooling2D()
            self.conv4 = Conv2D(filters=128, kernel_size=(4, 4))
            self.pool4 = MaxPooling2D(pool_size=(2, 1))
            self.conv5 = Conv2D(filters=256, kernel_size=(3, 3))
            self.pool5 = MaxPooling2D(pool_size=(2, 4))
            self.conv6 = Conv2D(filters=512, kernel_size=(2, 3))

        elif m == 7:
            self.conv1_block1 = Conv2D(filters=16, kernel_size=(3, 3), padding='same', activation='relu')
            self.conv2_block1 = Conv2D(filters=16, kernel_size=(3, 3), padding='same', activation='relu')
            self.pool_block1 = MaxPooling2D(pool_size=(2, 3))

            self.conv1_block2 = Conv2D(filters=32, kernel_size=(3, 3), padding='same', activation='relu')
            self.conv2_block2 = Conv2D(filters=32, kernel_size=(3, 3), padding='same', activation='relu')
            self.pool_block2 = MaxPooling2D(pool_size=(2, 3))

            self.conv1_block3 = Conv2D(filters=64, kernel_size=(3, 3), padding='same', activation='relu')
            self.conv2_block3 = Conv2D(filters=64, kernel_size=(3, 3), padding='same', activation='relu')
            self.pool_block3 = MaxPooling2D()

            self.conv1_block4 = Conv2D(filters=128, kernel_size=(3, 3), padding='same', activation='relu')
            self.conv2_block4 = Conv2D(filters=128, kernel_size=(3, 3), padding='same', activation='relu')
            self.pool_block4 = MaxPooling2D(pool_size=(1, 2))

            self.conv1_block5 = Conv2D(filters=256, kernel_size=(3, 3), padding='same', activation='relu')
            self.conv2_block5 = Conv2D(filters=256, kernel_size=(3, 3), padding='same', activation='relu')
            self.pool_block5 = MaxPooling2D(pool_size=(2, 3))

            self.conv1_block6 = Conv2D(filters=512, kernel_size=(3, 3), padding='same', activation='relu')
            self.conv2_block6 = Conv2D(filters=512, kernel_size=(3, 3), padding='same', activation='relu')

        elif m == 8:
            self.conv1_block1 = Conv2D(filters=64, kernel_size=(3, 3), padding='same', activation='relu')
            # self.conv2_block1 = Conv2D(filters=64, kernel_size=(3, 3), padding='same', activation='relu')
            self.pool_block1 = MaxPooling2D(pool_size=(2, 3))

            self.conv1_block2 = Conv2D(filters=128, kernel_size=(3, 3), padding='same', activation='relu')
            # self.conv2_block2 = Conv2D(filters=32, kernel_size=(3, 3), padding='same', activation='relu')
            self.pool_block2 = MaxPooling2D(pool_size=(2, 3))

            self.conv1_block3 = Conv2D(filters=256, kernel_size=(3, 3), padding='same', activation='relu')
            self.conv2_block3 = Conv2D(filters=256, kernel_size=(3, 3), padding='same', activation='relu')
            self.pool_block3 = MaxPooling2D(pool_size=(1, 2))

            self.conv1_block4 = Conv2D(filters=512, kernel_size=(3, 3), padding='same', activation='relu')
            self.pool_block4 = MaxPooling2D(pool_size=(2, 3))
            self.conv2_block4 = Conv2D(filters=512, kernel_size=(3, 3), padding='same', activation='relu')

            # self.conv1_block5 = Conv2D(filters=256, kernel_size=(3, 3), padding='same', activation='relu')
            # self.conv2_block5 = Conv2D(filters=256, kernel_size=(3, 3), padding='same', activation='relu')
            # self.pool_block5 = MaxPooling2D(pool_size=(2, 3))

            # self.conv1_block6 = Conv2D(filters=512, kernel_size=(3, 3), padding='same', activation='relu')
            # self.conv2_block6 = Conv2D(filters=512, kernel_size=(3, 3), padding='same', activation='relu')


        elif m == 9:
            self.model = InceptionV3_convolutional_model()

        self.fc_00 = Dense(4 * embedding_dim)
        self.fc_0 = Dense(2 * embedding_dim)
        self.do_0 = Dropout(0.15)
        self.do_1 = Dropout(0.15)
        # shape after fc == (batch_size, 64, embedding_dim)
        self.fc = Dense(embedding_dim)

    def call(self, x):
        # print(f"init shape = {x.shape}")

        x = self.rescale(x)
        # print(f"rescale shape = {x.shape}")

        x = self.conv1_block1(x)
        # x = self.conv2_block1(x)
        x = self.pool_block1(x)

        x = self.conv1_block2(x)
        # x = self.conv2_block2(x)
        x = self.pool_block2(x)

        x = self.conv1_block3(x)
        x = self.conv2_block3(x)
        x = self.pool_block3(x)

        x = self.conv1_block4(x)
        x = self.pool_block4(x)
        x = self.conv2_block4(x)

        # x = self.conv1_block5(x)
        # x = self.conv2_block5(x)
        # x = self.pool_block5(x)

        # x = self.conv1_block6(x)
        # x = self.conv2_block6(x)
        # x = self.pool_block6(x)

        # x = self.conv1(x)
        # x = tf.nn.relu(x)
        # x = self.pool1(x)
        # # print(f"1_shape = {x.shape}")

        # x = self.conv2(x)
        # x = tf.nn.relu(x)
        # x = self.pool2(x)
        # # print(f"2_shape = {x.shape}")

        # x = self.conv3(x)
        # x = tf.nn.relu(x)
        # x = self.pool3(x)
        # # print(f"3_shape = {x.shape}")

        # x = self.conv4(x)
        # x = tf.nn.relu(x)
        # x = self.pool4(x)
        # # print(f"4_shape = {x.shape}")

        # x = self.conv5(x)
        # x = tf.nn.relu(x)
        # x = self.pool5(x)
        # print(f"5_shape = {x.shape}")

        # x = self.model(x)
        # print(f"model = {x.shape}")
        # x = self.conv6(x)
        # x = tf.nn.relu(x)

        # print(f"model = {x.shape}")

        x = tf.reshape(x, (x.shape[0], -1, x.shape[3]))

        # print(f"reshape_shape = {x.shape}")

        x = self.fc_00(x)
        x = tf.nn.relu(x)

        # x = self.do_0(x)

        x = self.fc_0(x)
        x = tf.nn.relu(x)

        x = self.do_1(x)

        x = self.fc(x)
        x = tf.nn.relu(x)
        print(f"final = {x.shape}")
        return x


class RNN_Decoder(tf.keras.Model):
    def __init__(self, embedding_dim, units, vocab_size):
        super(RNN_Decoder, self).__init__()
        self.units = units

        self.embedding = tf.keras.layers.Embedding(vocab_size, embedding_dim)
        self.gru = tf.keras.layers.GRU(self.units,
                                       return_sequences=True,
                                       return_state=True,
                                       recurrent_initializer='glorot_uniform')

        self.fc1 = tf.keras.layers.Dense(self.units)
        self.fc2 = tf.keras.layers.Dense(vocab_size)

        self.attention = BahdanauAttention(self.units)
        # self.encoder_2 = TransformerEncoderBlock(200, 200, 2)

    def call(self, x, features, hidden):
        # defining attention as a separate model
        context_vector, attention_weights = self.attention(features, hidden)

        # x shape after passing through embedding == (batch_size, 1, embedding_dim)
        x = self.embedding(x)

        # x shape after concatenation == (batch_size, 1, embedding_dim + hidden_size)
        x = tf.concat([tf.expand_dims(context_vector, 1), x], axis=-1)

        # passing the concatenated vector to the GRU
        output, state = self.gru(x)

        # shape == (batch_size, max_length, hidden_size)
        x = self.fc1(output)

        # x shape == (batch_size * max_length, hidden_size)
        x = tf.reshape(x, (-1, x.shape[2]))

        # output shape == (batch_size * max_length, vocab)
        x = self.fc2(x)

        return x, state, attention_weights

    def reset_state(self, batch_size):
        return tf.zeros((batch_size, self.units))


class Training:
    global DATASET_PATH, ANNOTATION_FILE

    manager = None

    def __init__(self, manager, **params):
        global meta_path
        self.tokenizer = None
        self.img_to_cap_vector = None

        self.manager = manager
        self.top_k = params["top_k"]
        self.image_count = params["image_count"]
        self.BATCH_SIZE = params["BATCH_SIZE"]
        self.BUFFER_SIZE = params["BUFFER_SIZE"]
        self.embedding_dim = params["embedding_dim"]
        self.units = params["units"]
        self.vocab_size = self.top_k + 1
        self.EPOCHS = params["EPOCHS"]
        self.conv_var = params["conv_var"]

        print("Deleting temporary files, please wait...")
        self.manager.remove_temp()

        os.makedirs(os.path.dirname(meta_path), exist_ok=True)
        with open(meta_path, "w") as meta:
            for param in params.values():
                meta.write(str(param) + ";")

        self.logger = log_init(manager.get_model_path(), "model_log")
        self.logger.info(f"Model '{manager.get_model_path()}' has been created with these params:")
        for k, v in params.items():
            self.logger.info(f"{k} - {v}")

        self.logger.info(f"---------------MODEL AND UTILS SUMMARY---------------")
        self.logger.info(f"ENCODER_INIT:\n{inspect.getsource(CNN_Encoder.__init__)}")
        self.logger.info(f"ENCODER_CALL:\n{inspect.getsource(CNN_Encoder.call)}")
        self.logger.info(f"IMG_PREPROCESSING:\n{inspect.getsource(self.load_image)}")
        self.logger.info(f"RESIZE_HEIGHT: {RESIZED_IMG_H}")
        self.logger.info(f"RESIZE_WIDTH: {RESIZED_IMG_W}")
        self.logger.info(f"DATA_SPLIT_INDEX: {DATA_SPLIT}")
        self.logger.info(f"-----------------------------------------------------")


    def auto_train(self):
        self.data_preprocess()
        self.train()


    def loss_function(self, real, pred, loss_object):
        # print(f"real = {real}  ;  pred = {pred}")
        mask = tf.math.logical_not(tf.math.equal(real, 0))
        loss_ = loss_object(real, pred)

        mask = tf.cast(mask, dtype=loss_.dtype)
        loss_ *= mask
        # print(f"loss = {loss_}")
        return tf.reduce_mean(loss_)


    @staticmethod
    def plot_attention(image, result, attention_plot):
        temp_image = np.array(Image.open(image).convert('RGB'))

        fig = plt.figure(figsize=(10, 10))

        len_result = len(result)
        for i in range(len_result):
            temp_att = np.resize(attention_plot[i], (8, 8))
            grid_size = max(np.ceil(len_result / 2), 2)
            ax = fig.add_subplot(grid_size, grid_size, i + 1)
            ax.set_title(result[i])
            img = ax.imshow(temp_image)
            ax.imshow(temp_att, cmap='gray', alpha=0.6, extent=img.get_extent())

        plt.tight_layout()
        plt.show()


    @tf.function
    def train_step(self, img_tensor, target, decoder, encoder, optimizer, loss_object):
        loss = 0

        # initializing the hidden state for each batch
        # because the captions are not related from image to image
        hidden = decoder.reset_state(batch_size=target.shape[0])

        dec_input = tf.expand_dims([self.tokenizer.word_index['<start>']] * target.shape[0], 1)

        with tf.GradientTape() as tape:
            features = encoder(img_tensor)

            for i in range(1, target.shape[1]):
                # passing the features through the decoder
                predictions, hidden, _ = decoder(dec_input, features, hidden)

                loss += self.loss_function(target[:, i], predictions, loss_object)

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
                np.save(path_of_feature, im.numpy())  # сохранить извлеченные признаки в форме (16, ?, 2048)


    # Find the maximum length of any caption in our dataset
    def calc_max_length(self, tensor):
        return max(len(t) for t in tensor)


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
        # img = tf.image.convert_image_dtype(img, dtype=tf.float32)
        img = tf.image.resize(img, (RESIZED_IMG_H, RESIZED_IMG_W), method=tf.image.ResizeMethod.GAUSSIAN,
                              antialias=True)
        # img = tf.image.rgb_to_grayscale(img)
        # Image.fromarray(np.asarray(img.numpy().astype(np.uint8))).show()
        return img, image_path


    def get_image_to_caption(self, annotations_file):
        image_path_to_caption = collections.defaultdict(list)  # словарь, в который автоматически будет добавляться лист
        # при попытке доступа к несущестувующему ключу
        for val in annotations_file['annotations']:
            caption = f"<start> {val['caption']} <end>"
            image_path = DATASET_PATH + val["image_id"] + ".png"
            image_path_to_caption[image_path].append(
                caption)  # словарь типа 'путь_фото': ['описание1', 'описание2', ...]

        return image_path_to_caption


    def data_preprocess(self):  # составление токенайзера
        global tokenizer_path, tokenizer_params_path

        with open(ANNOTATION_FILE, 'r') as f:
            annotations = json.load(f)

        image_path_to_caption = self.get_image_to_caption(annotations)

        image_paths = list(image_path_to_caption.keys())
        random.shuffle(image_paths)

        # for k, v in image_path_to_caption.items():
        #     print(f"key = {k} ; value = {v}")

        train_image_paths = image_paths[:self.image_count]  # берется только 6000 верхних фото
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
        encode_train = sorted(set(img_name_vector))

        # Feel free to change batch_size according to your system configuration
        image_dataset = tf.data.Dataset.from_tensor_slices(encode_train)
        image_dataset = image_dataset.map(
            Training.load_image, num_parallel_calls=tf.data.AUTOTUNE).batch(64)

        # print(list(image_dataset)[:100])  # после мэппинга и препроцессинга получаются тензоры формы (16, 299, 299, 3)
        self.logger.info(f"Images are being preprocessed and saved...")
        start = time.time()
        try:
            pass
            # self.save_reshaped_images(image_dataset)
        except:
            self.logger.error("Something went wrong!")
            self.manager.remove_temp()
            sys.exit()

        self.logger.info(f"Done in {time.time() - start:.2f} sec")

        os.makedirs(os.path.dirname(tokenizer_path), exist_ok=True)
        with open(tokenizer_path, 'w', encoding='utf-8') as f:
            # Choose the top top_k words from the vocabulary
            self.tokenizer = tf.keras.preprocessing.text.Tokenizer(num_words=self.top_k, split=' ', oov_token="<unk>",
                                                                   lower=False, filters='%')

            self.tokenizer.fit_on_texts(train_captions)
            self.tokenizer.word_index['<pad>'] = 0
            self.tokenizer.index_word[0] = '<pad>'  # создание токенайзера и сохранение его в json

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
        cap_vector = tf.keras.preprocessing.sequence.pad_sequences(train_seqs, padding='post')  # приведение всех
        # последовательностей к одинаковой длине
        # print(cap_vector[:100])

        # Calculates the max_length, which is used to store the attention weights

        os.makedirs(os.path.dirname(tokenizer_params_path), exist_ok=True)
        with open(tokenizer_params_path, 'w') as f:
            max_length = self.calc_max_length(train_seqs)
            f.write(str(max_length))

        self.img_to_cap_vector = collections.defaultdict(list)
        for img, cap in zip(img_name_vector, cap_vector):
            self.img_to_cap_vector[img].append(cap)

        self.logger.info(f"Img_to_cap vector is compiled. Max cap length - {max_length}")
        # for k, v in self.img_to_cap_vector.items():
        #     print(f"key = {k} ; value = {v}")


    def train(self):
        global checkpoint_path
        # Create training and validation sets using an 80-20 split randomly.
        img_keys = list(self.img_to_cap_vector.keys())
        random.shuffle(img_keys)
        # print(img_keys[:100])

        slice_index = int(len(img_keys) * DATA_SPLIT)
        img_name_train_keys, img_name_val_keys = img_keys[:slice_index], img_keys[slice_index:]

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
        dataset = tf.data.Dataset.from_tensor_slices((img_name_train, cap_train))
        # Use map to load the numpy files in parallel
        dataset = dataset.map(lambda item1, item2: tf.numpy_function(
            self.map_func, [item1, item2], [tf.float32, tf.int32]),
                              num_parallel_calls=tf.data.AUTOTUNE)

        # Shuffle and batch
        dataset = dataset.shuffle(self.BUFFER_SIZE).batch(self.BATCH_SIZE)
        dataset = dataset.prefetch(buffer_size=tf.data.AUTOTUNE)
        self.logger.info(f"Dataset is ready")

        # инициализация параметров нейросети
        encoder = CNN_Encoder(self.embedding_dim, m=self.conv_var)
        decoder = RNN_Decoder(self.embedding_dim, self.units, self.vocab_size)

        optimizer = tf.keras.optimizers.RMSprop()
        loss_object = tf.keras.losses.SparseCategoricalCrossentropy(
            from_logits=True, reduction='none')

        ckpt = tf.train.Checkpoint(encoder=encoder,
                                   decoder=decoder,
                                   optimizer=optimizer)
        ckpt_manager = tf.train.CheckpointManager(ckpt, checkpoint_path, max_to_keep=10)

        self.logger.info(f"OPTIMIZER SUMMARY:\n{optimizer.get_config()}")
        # start_epoch = 0
        # if ckpt_manager.latest_checkpoint:
        #     start_epoch = int(ckpt_manager.latest_checkpoint.split('-')[-1])
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
                batch_loss, t_loss = self.train_step(img_tensor, target, decoder, encoder, optimizer, loss_object)
                total_loss += t_loss

                if batch % 100 == 0:
                    average_batch_loss = batch_loss.numpy() / int(target.shape[1])
                    self.logger.info(
                        f'Epoch {epoch + 1} Batch {batch} Loss = {batch_loss.numpy()} / {int(target.shape[1])} = {average_batch_loss:.4f}')
            # storing the epoch end loss value to plot later
            loss_plot.append(total_loss / num_steps)

            ckpt_manager.save()
            self.logger.info(f"Epoch {epoch} checkpoint saved!")

            self.logger.info(f'Epoch {epoch + 1} Loss {total_loss / num_steps:.6f}')
            self.logger.info(f'Time taken for 1 epoch {time.time() - start:.2f} sec\n')

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
            with open(meta_path, "r") as meta:
                reg = re.findall(r"\w+", meta.readline())
                self.vocab_size = int(reg[0]) + 1
                self.units = int(reg[5])
                self.embedding_dim = int(reg[4])
        except:
            raise IOError("Meta file reading failed")

    def evaluate(self, image, decoder, encoder):
        global checkpoint_path
        attention_plot = np.zeros((self.max_length, 110))

        hidden = decoder.reset_state(batch_size=1)

        temp_input = tf.expand_dims(Training.load_image(image)[0], 0)
        # img_tensor_val = image_features_extract_model(temp_input)
        # img_tensor_val = tf.reshape(img_tensor_val, (img_tensor_val.shape[0],
        #                                             -1,
        #                                             img_tensor_val.shape[3]))

        features = encoder(temp_input)

        dec_input = tf.expand_dims([self.tokenizer.word_index['<start>']], 0)
        result = []

        for i in range(self.max_length):
            predictions, hidden, attention_weights = decoder(dec_input,
                                                             features,
                                                             hidden)

            attention_plot[i] = tf.reshape(attention_weights, (-1,)).numpy()

            predicted_id = tf.random.categorical(predictions, 1)[0][0].numpy()
            result.append(self.tokenizer.index_word[predicted_id])

            if self.tokenizer.index_word[predicted_id] == '<end>':
                return result, attention_plot

            dec_input = tf.expand_dims([predicted_id], 0)

        attention_plot = attention_plot[:len(result), :]
        return result, attention_plot

    def predict(self, image_path: str):
        global tokenizer_path, tokenizer_params_path, checkpoint_path
        try:
            with open(tokenizer_path, "r") as f:
                data = json.load(f)
                self.tokenizer = tokenizer_from_json(data)
            print("Tokenizer has been loaded")
            with open(tokenizer_params_path, "r") as txt:
                for line in txt:
                    regex = re.findall(r'\w+', line)
                    self.max_length = int(regex[0])
        except:
            raise IOError("Something went wrong with initializing tokenizer")

        encoder = CNN_Encoder(self.embedding_dim, m=8)
        decoder = RNN_Decoder(self.embedding_dim, self.units, self.vocab_size)
        optimizer = tf.keras.optimizers.Adam()

        ckpt = tf.train.Checkpoint(encoder=encoder,
                                   decoder=decoder,
                                   optimizer=optimizer)
        ckpt_manager = tf.train.CheckpointManager(ckpt, checkpoint_path, max_to_keep=5)
        if ckpt_manager.latest_checkpoint:
            # restoring the latest checkpoint in checkpoint_path
            ckpt.restore(ckpt_manager.latest_checkpoint).expect_partial()
            print("Restored from {}".format(ckpt_manager.latest_checkpoint))
        else:
            print("Restored from scratch")

        result, attention_plot = self.evaluate(image_path, decoder, encoder)
        print('Prediction Caption:', ' '.join(result))
        # Training.plot_attention(image_path, result, attention_plot)
        print(f"len res = {len(result)}")
        # opening the image
        Image.open(open(image_path, 'rb'))


class VAN:
    loaded = False
    d_p = "formula_images_png_4_resized\\"
    c_p = "4_dataset.json"
    top_k = 300
    image_count = 44500
    BATCH_SIZE = 64
    BUFFER_SIZE = 100
    embedding_dim = 200
    units = 200
    EPOCHS = 21
    conv_var = 1

    def __init__(self, model_name: str, working_path: str = ""):
        self.model_path = os.path.abspath(".") + "\\trained_models\\" + working_path + model_name
        self.init()

    def init(self):
        global tokenizer_path, tokenizer_params_path, checkpoint_path, meta_path
        tokenizer_path = self.model_path + "\\tokenizer.json"
        tokenizer_params_path = self.model_path + "\\tokenizer_params.txt"
        checkpoint_path = self.model_path + "\\checkpoints\\"
        meta_path = self.model_path + "\\meta.txt"

        if os.path.exists(checkpoint_path) and len(os.listdir(checkpoint_path)) != 0 and os.path.exists(tokenizer_path) \
                and os.path.exists(tokenizer_params_path) and os.path.exists(meta_path):
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

    def remove_temp(self):
        if not self.loaded:
            raise ValueError("Model is not loaded!")
        files = os.listdir(DATASET_PATH)
        for file in files:
            if file.split(".")[-1] == "npy":
                os.remove(DATASET_PATH + file)

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
                self.d_p = self.input(input(f"dataset_path - Default = {self.d_p}: "), self.d_p)
                self.c_p = self.input(input(f"caption file - Default = {self.c_p}: "), self.c_p)
                self.top_k = int(
                    self.input(input(f"top_k - Default = {self.top_k} (number of top used words in caps): "),
                               self.top_k))
                self.image_count = int(
                    self.input(input(f"image_count - Default = {self.image_count} (number of photos to be used): "),
                               self.image_count))
                self.BATCH_SIZE = int(self.input(input(f"BATCH_SIZE - Default = {self.BATCH_SIZE}: "), self.BATCH_SIZE))
                self.BUFFER_SIZE = int(
                    self.input(input(f"BUFFER_SIZE - Default = {self.BUFFER_SIZE}: "), self.BUFFER_SIZE))
                self.embedding_dim = int(
                    self.input(input(f"embedding_dim - Default = {self.embedding_dim}: "), self.embedding_dim))
                self.units = int(self.input(input(f"units - Default = {self.units}: "), self.units))
                self.EPOCHS = int(self.input(input(f"EPOCHS - Default = {self.EPOCHS}: "), self.EPOCHS))
                self.conv_var = int(self.input(input(f"conv_var - Default = {self.conv_var}: "), self.conv_var))
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

    def predict(self, image_path: str):
        if not self.loaded:
            raise ValueError("Model is not loaded!")
        pred = Prediction()
        pred.predict(image_path)


# physical_devices = tf.config.list_physical_devices('GPU')
# tf.config.experimental.set_memory_growth(physical_devices[0], True)
# tf.data.experimental.enable_debug_mode()


enable_gpu(True, gb=6)
van = VAN("model_latex_x9")

van.train()


#van.predict("C:/Users/shace/Desktop/1d1fa28f44.png")
