import tensorflow as tf
from tensorflow.python.keras.models import Sequential
import numpy as np
from tensorflow import keras
from tensorflow.keras.optimizers import RMSprop
from tensorflow.keras.preprocessing.image import ImageDataGenerator
import os
import zipfile
import scipy
import random
from matplotlib import pyplot as plt
from tensorflow.keras.preprocessing import image
from tensorflow.keras import layers, models

from tensorflow.keras.applications.vgg16 import VGG16
from tensorflow.keras.models import Model
from tensorflow.keras.layers import Dense
from tensorflow.keras.layers import Flatten
from tensorflow.keras.layers import Input
from tensorflow.keras.layers import Conv2D
from tensorflow.keras.layers import MaxPooling2D
from tensorflow.keras.layers import Dropout

from tqdm import tqdm 
from PIL import Image
from PIL import ImageOps

import albumentations as alb
import cv2

  
from tensorflow.keras.layers import Conv2D, MaxPooling2D, Activation, Dropout, Flatten, Dense, BatchNormalization, Input, Concatenate, AvgPool2D, GlobalAveragePooling2D
from tensorflow.keras.models import Sequential, Model


model_save_way = "C:/Users/shace/Documents/GitHub/MachineLearning/Models"
learning_data_way = "C:/Users/shace/Documents/GitHub/MachineLearning/LearningData"
testing_data_way = "C:/Users/shace/Documents/GitHub/MachineLearning/TestingData"


def gpu(turn, gb: int):  # enable of disable GPU backend
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
gpu(False, 4)
# https://colab.research.google.com/github/lmoroney/mlday-tokyo/blob/master/Lab1-Hello-ML-World.ipynb
# model = tf.keras.Sequential([keras.layers.Dense(units=1, input_shape=[1])])
# model.compile(optimizer='sgd', loss='mean_squared_error')

# xs = np.array([-1.0, 0.0, 1.0, 2.0, 3.0, 4.0, 5.0, 6.0], dtype=float)
# ys = np.array([-2.0, 1.0, 4.0,7.0, 10.0, p13.0, 16.0, 19.0], dtype=float)

# model.fit(xs, ys, epochs=1000)
# print(model.predict([15.0, 2.0]))

# https://colab.research.google.com/github/lmoroney/mlday-tokyo/blob/master/Lab2-Computer-Vision.ipynb
# mnist = tf.keras.datasets.fashion_mnist
# (training_images, training_labels), (test_images, test_labels) = mnist.load_data()
# training_images = training_images / 255.0
# test_images = test_images / 255.0
# model = tf.keras.models.Sequential([tf.keras.layers.Flatten(),
#                                     tf.keras.layers.Dense(512, activation=tf.nn.relu),
#                                     tf.keras.layers.Dense(128, activation=tf.nn.relu),
#                                     tf.keras.layers.Dense(10, activation=tf.nn.softmax)])
#
# model.compile(optimizer=tf.keras.optimizers.Adam(),
#               loss='sparse_categorical_crossentropy',
#               metrics=['accuracy'])
#
# model.fit(training_images, training_labels, epochs=50)
#
# model.evaluate(test_images, test_labels)
#
# classifications = model.predict(test_images)
#
# print(classifications[0])
#
# model.save(model_save_way + "/fashion_mnist")


# converter = tf.lite.TFLiteConverter.from_saved_model("C:/Users/shace/Desktop/ddd")  # path to the SavedModel directory
# tflite_model = converter.convert()
#
# with open('mnist.tflite', 'wb') as f:
#     f.write(tflite_model)

# https://colab.research.google.com/github/lmoroney/mlday-tokyo/blob/master/Lab4-Using-Convolutions.ipynb
# mnist = tf.keras.datasets.fashion_mnist
# (training_images, training_labels), (test_images, test_labels) = mnist.load_data()
# training_images=training_images.reshape(60000, 28, 28, 1)
# training_images=training_images / 255.0
# test_images = test_images.reshape(10000, 28, 28, 1)
# test_images=test_images/255.0
# model = tf.keras.models.Sequential([
#  tf.keras.layers.Conv2D(64, (3,3), activation='relu', input_shape=(28, 28, 1)),
#  tf.keras.layers.MaxPooling2D(2, 2),
#  tf.keras.layers.Conv2D(64, (3,3), activation='relu'),
#  tf.keras.layers.MaxPooling2D(2, 2),
#  tf.keras.layers.Flatten(),
#  tf.keras.layers.Dense(128, activation='relu'),
#  tf.keras.layers.Dense(10, activation='softmax')
# ])
# model.compile(optimizer='adam', loss='sparse_categorical_crossentropy', metrics=['accuracy'])
# model.summary()
# model.fit(training_images, training_labels, epochs=10)
# test_loss = model.evaluate(test_images, test_labels)

# https://colab.research.google.com/github/lmoroney/mlday-tokyo/blob/master/Lab5-Using-Convolutions-With-Complex-Images.ipynb

img_width = 200
img_height = 200
#
# os.environ['TF_FORCE_GPU_ALLOW_GROWTH'] = 'true'
#
# gpus = tf.config.experimental.list_physical_devices('GPU')
# if gpus:
#     # Restrict TensorFlow to only allocate 1*X GB of memory on the first GPU
#     try:
#         tf.config.experimental.set_virtual_device_configuration(
#             gpus[0],
#             [tf.config.experimental.VirtualDeviceConfiguration(memory_limit=(1024 * 6))])
#         logical_gpus = tf.config.experimental.list_logical_devices('GPU')
#         print(len(gpus), "Physical GPUs,", len(logical_gpus), "Logical GPUs")
#     except RuntimeError as e:
#         # Virtual devices must be set before GPUs have been initialized
#         print(e)
#
# # os.environ["CUDA_VISIBLE_DEVICES"] = "-1"
# # try:
# #     # Disable all GPUS
# #     tf.config.set_visible_devices([], 'GPU')
# #     visible_devices = tf.config.get_visible_devices()
# #     for device in visible_devices:
# #         assert device.device_type != 'GPU'
# # except:
# #     # Invalid device or cannot modify virtual devices once initialized.
# #     pass
#
# # local_zip = 'C:/Users/shace/Downloads/horse-or-human.zip'
# # zip_ref = zipfile.ZipFile(local_zip, 'r')
# # zip_ref.extractall('/tmp/horse-or-human')
# # local_zip = 'C:/Users/shace/Downloads/validation-horse-or-human.zip'
# # zip_ref = zipfile.ZipFile(local_zip, 'r')
# # zip_ref.extractall('/tmp/validation-horse-or-human')
# # zip_ref.close()
#
# # Directory with our training horse pictures
# # train_horse_dir = os.path.join('/tmp/horse-or-human/horses')
# #
# # # Directory with our training human pictures
# # train_human_dir = os.path.join('/tmp/horse-or-human/humans')
# #
# # # Directory with our training horse pictures
# # validation_horse_dir = os.path.join('/tmp/validation-horse-or-human/horses')
# #
# # # Directory with our training human pictures
# # validation_human_dir = os.path.join('/tmp/validation-horse-or-human/humans')
# #
# # train_horse_names = os.listdir(train_horse_dir)
# # print(train_horse_names[:10])
# #
# # train_human_names = os.listdir(train_human_dir)
# # print(train_human_names[:10])
#
# # print('total training horse images:', len(os.listdir(train_horse_dir)))
# # print('total training human images:', len(os.listdir(train_human_dir)))
#
# data_augmentation = keras.Sequential(
#     [
#         layers.experimental.preprocessing.RandomFlip("horizontal",
#                                                      input_shape=(img_height,
#                                                                   img_width,
#                                                                   3)),
#         layers.experimental.preprocessing.RandomRotation(0.1),
#         layers.experimental.preprocessing.RandomZoom(0.1),
#     ]
# )
#
# # model = tf.keras.models.Sequential([ //основная модель, работает неплохо
# #     data_augmentation,
# #     tf.keras.layers.Conv2D(16, (3, 3), activation='relu',
# #                            input_shape=(img_width, img_height, 3)),
# #     tf.keras.layers.MaxPooling2D(2, 2),
# #     tf.keras.layers.Conv2D(32, (3, 3), activation='relu'),
# #     tf.keras.layers.MaxPooling2D(2, 2),
# #     tf.keras.layers.Conv2D(64, (3, 3), activation='relu'),
# #     tf.keras.layers.MaxPooling2D(2, 2),
# #     tf.keras.layers.Conv2D(128, (3, 3), activation='relu'),
# #     tf.keras.layers.MaxPooling2D(2, 2),
# #     layers.Dropout(0.2),
# #     tf.keras.layers.Flatten(),
# #     tf.keras.layers.Dense(512, activation='relu'),
# #     tf.keras.layers.Dense(1, activation='sigmoid')
# # ])
# # model.compile(loss='binary_crossentropy',
# #               optimizer=RMSprop(lr=0.001),
# #               metrics=['accuracy'])
#
# # model = tf.keras.models.Sequential([
# #     data_augmentation,
# #     tf.keras.layers.Conv2D(16, (3, 3), activation='relu',
# #                            input_shape=(img_width, img_height, 3)),
# #     tf.keras.layers.MaxPooling2D(2, 2),
# #     tf.keras.layers.Conv2D(32, (3, 3), activation='relu'),
# #     tf.keras.layers.MaxPooling2D(2, 2),
# #     tf.keras.layers.Conv2D(64, (3, 3), activation='relu'),
# #     tf.keras.layers.MaxPooling2D(2, 2),
# #     tf.keras.layers.Conv2D(128, (3, 3), activation='relu'),
# #     tf.keras.layers.MaxPooling2D(2, 2),
# #     layers.Dropout(0.2),
# #     tf.keras.layers.Flatten(),
# #     tf.keras.layers.Dense(512, activation='relu'),
# #     tf.keras.layers.Dense(2)
# # ])
#
# model = tf.keras.models.Sequential(
#     [
#         layers.experimental.preprocessing.Rescaling(1./255, input_shape=(img_height, img_width, 3)),
#         tf.keras.layers.Conv2D(16, (3, 3), activation='relu'),
#         tf.keras.layers.MaxPooling2D(2, 2),

#         tf.keras.layers.Conv2D(32, (3, 3), activation='relu'),
#         tf.keras.layers.MaxPooling2D(2, 2),

#         tf.keras.layers.Conv2D(64, (3, 3), activation='relu'),
#         tf.keras.layers.MaxPooling2D(2, 2),

#         tf.keras.layers.Conv2D(128, (3, 3), activation='relu'),
#         tf.keras.layers.MaxPooling2D(2, 2),

#         tf.keras.layers.Conv2D(256, (3, 3), activation='relu'),
#         tf.keras.layers.MaxPooling2D(2, 2),

#         tf.keras.layers.Flatten(),
#         tf.keras.layers.Dense(512, activation='relu'),
#         layers.Dropout(0.2),
#         tf.keras.layers.Dense(400, activation='relu'),
#         tf.keras.layers.Dense(332)
#     ]
# )

class ImageClassificationModel:
    @staticmethod
    def build(width, height, number_of_classes, final_activation='softmax'):
        input_size = (height, width, 3) 
        
        input_tensor=Input(input_size)
        initial = Conv2D(64, (7,7), strides=2, activation='relu')(input_tensor)
        max_pooling_initial = MaxPooling2D(pool_size=(2,2))(initial)

        batch_1_batchNorm = BatchNormalization()(max_pooling_initial)
        batch_1_activ = Activation('relu')(batch_1_batchNorm)
        batch_1_conv2d_1 = Conv2D(128, (1,1), activation='relu', padding='same')(batch_1_activ)
        batch_1_drop = Dropout(0.3)(batch_1_conv2d_1)
        batch_1_conv2d_2 = Conv2D(32, (3,3), activation='relu', padding='same')(batch_1_drop)

        batch_2 = Concatenate()([max_pooling_initial, batch_1_conv2d_2])

        batch_2_batchNorm = BatchNormalization()(batch_2)
        batch_2_activ = Activation('relu')(batch_2_batchNorm)
        batch_2_conv2d_1 = Conv2D(128, (1,1), activation='relu', padding='same')(batch_2_activ)
        batch_2_drop = Dropout(0.4)(batch_2_conv2d_1)
        batch_2_conv2d_2 = Conv2D(32, (3,3), activation='relu', padding='same')(batch_2_drop)

        batch_3 = Concatenate()([batch_2, batch_2_conv2d_2])

        batch_3_batchNorm = BatchNormalization()(batch_3)
        batch_3_activ = Activation('relu')(batch_3_batchNorm)
        batch_3_conv2d_1 = Conv2D(128, (1,1), activation='relu', padding='same')(batch_3_activ)
        batch_3_drop = Dropout(0.4)(batch_3_conv2d_1)
        batch_3_conv2d_2 = Conv2D(32, (3,3), activation='relu', padding='same')(batch_3_drop)

        batch_4 = Concatenate()([batch_3, batch_3_conv2d_2])

        batch_4_batchNorm = BatchNormalization()(batch_4)
        batch_4_activ = Activation('relu')(batch_4_batchNorm)
        batch_4_conv2d_1 = Conv2D(128, (1,1), activation='relu', padding='same')(batch_4_activ)
        batch_4_drop = Dropout(0.4)(batch_4_conv2d_1)
        batch_4_conv2d_2 = Conv2D(32, (3,3), activation='relu', padding='same')(batch_4_drop)

        final_batch = Concatenate()([batch_4_conv2d_2, batch_4])

        downsampling_batchNorm = BatchNormalization()(final_batch)
        downsampling_activ = Activation('relu')(downsampling_batchNorm)
        downsampling_conv2d_1 = Conv2D(32, (1,1), activation='relu')(downsampling_activ)
        downsampling_avg = AvgPool2D(pool_size=(2,2), strides=2)(downsampling_conv2d_1)

        flatten = Flatten()(downsampling_avg)
        top_layer_dense_1 = Dense(1024, activation='relu')(flatten)
        top_layer_dropout = Dropout(0.4)(top_layer_dense_1)
        top_layer_dense_2 = Dense(number_of_classes, activation=final_activation)(top_layer_dropout)
        model = Model(inputs=input_tensor, outputs=top_layer_dense_2)
        
        model.summary()
        return model


def functional_cnn_model():
    def model_init():
        input_shape = (300, 300, 3)
        kernel_size = 3
        filters = 64
        dropout = 0.3

        inputs = Input(shape=input_shape)
        y = tf.keras.layers.experimental.preprocessing.Rescaling(1./127.5, offset= -1)
        y = Conv2D(filters=16,kernel_size=kernel_size,activation='relu')(inputs)
        y = MaxPooling2D()(y)
        y = Conv2D(filters=32,kernel_size=kernel_size,activation='relu')(y)
        y = MaxPooling2D()(y)
        y = Conv2D(filters=64,kernel_size=kernel_size,activation='relu')(y)
        y = MaxPooling2D()(y)
        y = Conv2D(filters=128,kernel_size=kernel_size,activation='relu')(y)
        y = MaxPooling2D()(y)
        y = Conv2D(filters=256,kernel_size=kernel_size,activation='relu')(y)
        y = MaxPooling2D()(y)
        # convert image to vector 
        y = Flatten()(y)
        y = Dense(512, activation='relu')(y)
        # dropout regularization
        y = Dropout(dropout)(y)
        y = Dense(400, activation='relu')(y)
        outputs = Dense(332)(y)
        # model building by supplying inputs/outputs
        return Model(inputs=inputs, outputs=outputs)
    
    def dataset_init():
        train_ds = tf.keras.preprocessing.image_dataset_from_directory(
        "C:\\Users\\shace\\Desktop\\conv_latex_exp",
        seed=123,
        image_size=(300, 300),
        batch_size=32)

        return train_ds

    model = model_init()
    ds = dataset_init()

    class_names = np.array(ds.class_names)
    # normalization_layer = tf.keras.layers.experimental.preprocessing.Rescaling(1./127.5, offset= -1)
    # ds = ds.map(lambda x, y: (normalization_layer(x), y))
    AUTOTUNE = tf.data.AUTOTUNE
    ds = ds.cache().prefetch(buffer_size=AUTOTUNE)

    model.compile(optimizer=tf.keras.optimizers.Adam(), 
    loss=tf.keras.losses.SparseCategoricalCrossentropy(from_logits=True), 
    metrics=['acc'])

    history = model.fit(ds, epochs=5)
    #model = tf.keras.models.load_model("cnn.h5")
    model.summary()
    model.save("cnn.h5")
    while True:
        inp = input()
        try:
            img = keras.preprocessing.image.load_img(inp, target_size=(300, 300))
            img_array = keras.preprocessing.image.img_to_array(img)
            img_array = tf.expand_dims(img_array, 0)  # Create batch axis
        
            predictions = model.predict(img_array)

            score = tf.nn.softmax(predictions[0])

            print(score)

            print(
            "This image most likely belongs to {} with a {:.2f} percent confidence."
            .format(class_names[np.argmax(score)], 100 * np.max(score))
            )
        except:
            print("smth wrong")

def InceptionV3_convolutional_model():
    image_model = tf.keras.applications.VGG16(include_top=False, weights='imagenet')
    # image_model.trainable = False
    new_input = image_model.input
    hidden_layer = image_model.layers[-1].output
    tf.keras.Model(new_input, hidden_layer).summary()
    return tf.keras.Model(new_input, hidden_layer)



print(tf.nn.softmax(np.array([50.23874, 49.774884, 48.1268736]) / 0.5).numpy())

#InceptionV3_convolutional_model()

#functional_cnn_model()





labels_path = tf.keras.utils.get_file('ImageNetLabels.txt','https://storage.googleapis.com/download.tensorflow.org/data/ImageNetLabels.txt')
imagenet_labels = np.array(open(labels_path).read().splitlines())
IMAGE_SHAPE = (229, 229)

# classifier_model ="https://tfhub.dev/google/imagenet/inception_v3/classification/5"

# classifier = tf.keras.Sequential([
#     hub.KerasLayer(classifier_model, input_shape=IMAGE_SHAPE+(3,))
# ])

batch_size = 32
img_height = 120
img_width = 700

train_ds = tf.keras.preprocessing.image_dataset_from_directory(
  "C:\\Users\\shace\\Desktop\\conv_latex_exp",
  seed=123,
  image_size=(img_height, img_width),
  batch_size=batch_size)

class_names = np.array(train_ds.class_names)
print(class_names)

normalization_layer = tf.keras.layers.experimental.preprocessing.Rescaling(1./127.5, offset= -1)

train_ds = train_ds.map(lambda x, y: (normalization_layer(x), y))
AUTOTUNE = tf.data.AUTOTUNE
train_ds = train_ds.cache().prefetch(buffer_size=AUTOTUNE)

for image_batch, labels_batch in train_ds:
  print(image_batch.shape)
  print(labels_batch.shape)
  break

feature_extractor_layer = hub.KerasLayer(
    "https://tfhub.dev/google/tf2-preview/inception_v3/feature_vector/4", input_shape=(299, 299, 3), trainable=False)

# feature_batch = feature_extractor_layer(image_batch)
# print(feature_batch.shape)

model_s = tf.keras.models.load_model('C:\\Users\\shace\\Documents\\GitHub\\MachineLearning\\PythonML\\inceptionv3_latex_3.h5')
model_s.summary()
print(f"len of stock = {len(model_s.layers)}")
# layer = model_s.get_layer("inception_v3")
# for l in layer.layers:
#   print(l)

# model = VGG16(include_top=False, input_shape=(300, 300, 3))
# # add new classifier layers
# flat1 = Flatten()(model.layers[-1].output)
# class1 = Dense(1024, activation='relu')(flat1)
# output = Dense(10, activation='softmax')(class1)
# # define new model
# model = Model(inputs=model.inputs, outputs=output)
# # summarize
# model.summary()


num_classes = len(class_names)

# model = tf.keras.Sequential([
#   feature_extractor_layer #, tf.keras.layers.Dense(num_classes)
# ])


# predictions = model(image_batch)
# print("predic shape = " + str(predictions.shape))


image_model = tf.keras.applications.InceptionV3(include_top = False,
                                                    weights = 'imagenet', input_shape=(120,700,3))
image_model.trainable = False

image_model.summary()
print(f"len of stock = {len(image_model.layers)}")
predictions = image_model(image_batch)
print("predic shape = " + str(predictions.shape))

preprocess_input = tf.keras.applications.inception_v3.preprocess_input
global_average_layer = tf.keras.layers.GlobalAveragePooling2D()
prediction_layer = tf.keras.layers.Dense(num_classes)

x = global_average_layer(image_model.layers[-1].output)
outputs = prediction_layer(x)
model = tf.keras.Model(image_model.inputs, outputs)

# inputs = tf.keras.Input(shape=(299, 299, 3))
# x = preprocess_input(inputs)
# x = image_model(x, training=False)
# x = global_average_layer(x)
# outputs = prediction_layer(x)
# model = tf.keras.Model(inputs, outputs)


model.summary()
print(f"len of NONstock = {len(model.layers)}")
# mdl.summary()
model.compile(
  optimizer=tf.keras.optimizers.RMSprop(),
  loss=tf.keras.losses.SparseCategoricalCrossentropy(from_logits=True),
  metrics=['acc'])


history = model.fit(train_ds, epochs=5)

predicted_batch = model.predict(image_batch)
predicted_id = np.argmax(predicted_batch, axis=-1)
predicted_label_batch = class_names[predicted_id]
# print()
print(predicted_label_batch)


tf.keras.Model(model.input, model.layers[-3].output).save("inceptionv3_latex_4.h5")


while True:
    inp = input()
    try:
        img = keras.preprocessing.image.load_img(inp, target_size=(img_width, img_height))
        img_array = keras.preprocessing.image.img_to_array(img)
        img_array = tf.expand_dims(img_array, 0)  # Create batch axis

        predictions = model.predict(img_array)

        score = tf.nn.softmax(predictions[0])

        print(score)

        print(
        "This image most likely belongs to {} with a {:.2f} percent confidence."
        .format(class_names[np.argmax(score)], 100 * np.max(score))
        )
    except:
        print("smth wrong")

# model.compile(loss=tf.keras.losses.SparseCategoricalCrossentropy(from_logits=True),
#               optimizer='adam',
#               metrics=['accuracy'])

# # Flow training images in batches of 128 using train_datagen generator
# train_generator = tf.keras.preprocessing.image_dataset_from_directory(
#     "C:\\Users\\shace\\Desktop\\conv_latex_exp",  # This is the source directory for training images
#     image_size=(img_width, img_height),  # All images will be resized to 150x150
#     batch_size=32,
#     # Since we use binary_crossentropy loss, we need binary labels
# )
# class_names = train_generator.class_names
# print(class_names)
#
# validation_datagen = ImageDataGenerator(rescale=1 / 255)
#
# # Flow training images in batches of 128 using train_datagen generator
# validation_generator = validation_datagen.flow_from_directory(
#     learning_data_way,  # This is the source directory for training images
#     target_size=(img_width, img_height),  # All images will be resized to 150x150
#     batch_size=32,
#     # Since we use binary_crossentropy loss, we need binary labels
#     #class_mode='binary'
# )
#
# history = model.fit(
#     train_generator,
#     # validation_data=validation_generator,
#     epochs=25,
#     verbose=1)

# while True:
#     inp = input()
#     try:
#         img = keras.preprocessing.image.load_img(inp, target_size=(img_width, img_height))
#         img_array = keras.preprocessing.image.img_to_array(img)
#         img_array = tf.expand_dims(img_array, 0)  # Create batch axis

#         predictions = model.predict(img_array)

#         score = tf.nn.softmax(predictions[0])

#         print(score)

#         print(
#         "This image most likely belongs to {} with a {:.2f} percent confidence."
#         .format(class_names[np.argmax(score)], 100 * np.max(score))
#         )
#     except:
#         print("smth wrong")
    
#
# # model.save(model_save_way + "/man_woman")
# #
# model = tf.keras.models.load_model(model_save_way + "/man_woman")
#
# img = keras.preprocessing.image.load_img(
#     testing_data_way + "/Andrew.jpg", target_size=(img_width, img_height)
# )
# img_array = keras.preprocessing.image.img_to_array(img)
# img_array = tf.expand_dims(img_array, 0)  # Create batch axis
# predictions = model.predict(img_array)
# print(predictions)
# # score = predictions[0]
# # print(score[0])
# # print(
# #     "This image is %.2f percent man and %.2f percent woman."
# #     % (100 * (1 - score), 100 * score)
# # )
# #
# converter = tf.lite.TFLiteConverter.from_saved_model("C:/Users/shace/Desktop/man_woman (best2)")  # path to the SavedModel directory
# tflite_model = converter.convert()
#
# with open('ManWoman2.tflite', 'wb') as f:
#     f.write(tflite_model)

# https://colab.research.google.com/github/lmoroney/mlday-tokyo/blob/master/Lab6-Cats-v-Dogs.ipynb

