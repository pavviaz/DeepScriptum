import collections
import math
import os
import re
import shutil
import random
import time
from black import json
from matplotlib import pyplot as plt
import inspect
import numpy as np
import tensorflow as tf
from tensorflow import keras
from keras import layers
from keras.applications import resnet
from keras.layers import Dense
from keras.layers import Conv2D
from keras.layers import MaxPooling2D
from keras.layers import TextVectorization
import logging
from munch import munchify
from utils.images_preprocessing import make_fix_size
from keras_preprocessing.text import tokenizer_from_json
from tqdm import tqdm


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

def get_cnn_model(IMAGE_SIZE):
    base_model = resnet.ResNet50(
        input_shape=(*IMAGE_SIZE, 3), include_top=False, weights=None,
    )
    # We freeze our feature extractor
    base_model.trainable = True
    base_model_out = base_model.output
    # print(base_model_out.shape)
    base_model_out = layers.Reshape((-1, base_model_out.shape[-1]))(base_model_out)
    cnn_model = keras.models.Model(base_model.input, base_model_out)
    return cnn_model

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


class CNN_Encoder(tf.keras.Model):
    def __init__(self, emb_dim):
        super(CNN_Encoder, self).__init__()
        
        self.rescale = tf.keras.layers.experimental.preprocessing.Rescaling(scale=1. / 127.5, offset=-1)
        # self.rescale = tf.keras.layers.experimental.preprocessing.Rescaling(scale=1. / 255.0)
        
        self.conv1_block1 = Conv2D(filters=64, kernel_size=(3, 3), padding='same', activation='relu')
        self.pool_block1 = MaxPooling2D(pool_size=(2, 2))

        self.conv1_block2 = Conv2D(filters=128, kernel_size=(3, 3), padding='same', activation='relu')
        self.pool_block2 = MaxPooling2D(pool_size=(2, 2))

        self.conv1_block3 = Conv2D(filters=256, kernel_size=(3, 3), padding='same', activation='relu')
        self.conv2_block3 = Conv2D(filters=256, kernel_size=(3, 3), padding='same', activation='relu')
        self.pool_block3 = MaxPooling2D(pool_size=(2, 2))

        self.conv1_block4 = Conv2D(filters=512, kernel_size=(3, 3), padding='same', activation='relu')
        self.pool_block4 = MaxPooling2D(pool_size=(2, 2))
        self.conv2_block4 = Conv2D(filters=512, kernel_size=(3, 3), padding='same', activation='relu')
        
        self.fc = Dense(units=emb_dim, activation='relu')
    

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
        
        x = self.add_timing_signal_nd(x, min_timescale=10.0)
        
        x = tf.reshape(x, (x.shape[0], -1, x.shape[3]))
        
        x = self.fc(x)
        
        # x = self.add_timing_signal_nd(x, min_timescale=10.0)
        
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
        inv_timescales = min_timescale * tf.exp(
                tf.cast(tf.range(num_timescales), tf.float32) * -log_timescale_increment)
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


class MultiHeadAttention(tf.keras.Model):
  def __init__(self,*, d_model, num_heads):
    super(MultiHeadAttention, self).__init__()
    self.num_heads = num_heads
    self.d_model = d_model

    assert d_model % self.num_heads == 0

    self.depth = d_model // self.num_heads

    self.wq = tf.keras.layers.Dense(d_model)
    self.wk = tf.keras.layers.Dense(d_model)
    self.wv = tf.keras.layers.Dense(d_model)

    self.dense = tf.keras.layers.Dense(d_model)

  def split_heads(self, x, batch_size):
    """Split the last dimension into (num_heads, depth).
    Transpose the result such that the shape is (batch_size, num_heads, seq_len, depth)
    """
    x = tf.reshape(x, (batch_size, -1, self.num_heads, self.depth))
    return tf.transpose(x, perm=[0, 2, 1, 3])

  def scaled_dot_product_attention(self, q, k, v, mask):
    """Calculate the attention weights.
    q, k, v must have matching leading dimensions.
    k, v must have matching penultimate dimension, i.e.: seq_len_k = seq_len_v.
    The mask has different shapes depending on its type(padding or look ahead)
    but it must be broadcastable for addition.

    Args:
        q: query shape == (..., seq_len_q, depth)
        k: key shape == (..., seq_len_k, depth)
        v: value shape == (..., seq_len_v, depth_v)
        mask: Float tensor with shape broadcastable
            to (..., seq_len_q, seq_len_k). Defaults to None.

    Returns:
        output, attention_weights
    """

    matmul_qk = tf.matmul(q, k, transpose_b=True)  # (..., seq_len_q, seq_len_k)

    # scale matmul_qk
    dk = tf.cast(tf.shape(k)[-1], tf.float32)
    scaled_attention_logits = matmul_qk / tf.math.sqrt(dk)

    # add the mask to the scaled tensor.
    if mask is not None:
        scaled_attention_logits += (mask * -1e9)

    # softmax is normalized on the last axis (seq_len_k) so that the scores
    # add up to 1.
    attention_weights = tf.nn.softmax(scaled_attention_logits, axis=-1)  # (..., seq_len_q, seq_len_k)

    output = tf.matmul(attention_weights, v)  # (..., seq_len_q, depth_v)

    return output, attention_weights

  def call(self, v, k, q, mask):
    batch_size = tf.shape(q)[0]

    q = self.wq(q)  # (batch_size, seq_len, d_model)
    k = self.wk(k)  # (batch_size, seq_len, d_model)
    v = self.wv(v)  # (batch_size, seq_len, d_model)

    q = self.split_heads(q, batch_size)  # (batch_size, num_heads, seq_len_q, depth)
    k = self.split_heads(k, batch_size)  # (batch_size, num_heads, seq_len_k, depth)
    v = self.split_heads(v, batch_size)  # (batch_size, num_heads, seq_len_v, depth)

    # scaled_attention.shape == (batch_size, num_heads, seq_len_q, depth)
    # attention_weights.shape == (batch_size, num_heads, seq_len_q, seq_len_k)
    scaled_attention, attention_weights = self.scaled_dot_product_attention(q, k, v, mask)

    scaled_attention = tf.transpose(scaled_attention, perm=[0, 2, 1, 3])  # (batch_size, seq_len_q, num_heads, depth)

    concat_attention = tf.reshape(scaled_attention,
                                  (batch_size, -1, self.d_model))  # (batch_size, seq_len_q, d_model)

    output = self.dense(concat_attention)  # (batch_size, seq_len_q, d_model)

    return output, attention_weights


class TransformerEncoderBlock(keras.Model):
    def __init__(self, embed_dim, dense_dim, num_heads, **kwargs):
        super().__init__(**kwargs)
        self.embed_dim = embed_dim
        self.dense_dim = dense_dim
        self.num_heads = num_heads
        # self.attention_1 = layers.MultiHeadAttention(
        #     num_heads=num_heads, key_dim=embed_dim, dropout=0.0
        # )
        self.attention_1 = MultiHeadAttention(
            num_heads=num_heads, d_model=embed_dim
        )
        self.layernorm_1 = layers.LayerNormalization()
        self.layernorm_2 = layers.LayerNormalization()
        self.dense_1 = layers.Dense(embed_dim, activation="relu")
        # self.dense_2 = layers.Dense(embed_dim, activation="relu")
        
        # self.attention_1_2 = layers.MultiHeadAttention(
        #     num_heads=num_heads, key_dim=embed_dim, dropout=0.0
        # )
        # self.layernorm_1_2 = layers.LayerNormalization()
        # self.layernorm_2_2 = layers.LayerNormalization()
        # self.dense_1_2 = layers.Dense(embed_dim, activation="relu")
        # self.dense_2_2 = layers.Dense(embed_dim, activation="relu")
        
    # def call(self, inputs, training, mask):
    #     # inputs = self.layernorm_1(inputs)
    #     # inputs = self.dense_1(inputs)
    #     padding_mask = None
    #     if mask is not None:
    #         padding_mask = tf.cast(mask[:, tf.newaxis, tf.newaxis, :], dtype="int32")
    #     attention_output_1 = self.attention_1(
    #         query=inputs,
    #         value=inputs,
    #         key=inputs,
    #         attention_mask=padding_mask,
    #         training=training,
    #     )
    #     out_1 = self.layernorm_1(inputs + attention_output_1)
    #     proj_output = self.dense_1(out_1)
    #     return self.layernorm_2(out_1 + proj_output)

    def call(self, inputs, training, mask):
        inputs = self.layernorm_1(inputs)
        inputs = self.dense_1(inputs)

        # attention_output_1 = self.attention_1(
        #     query=inputs,
        #     value=inputs,
        #     key=inputs,
        #     attention_mask=None,
        #     training=training,
        # )
        attention_output_1, _ = self.attention_1(
            q=inputs,
            v=inputs,
            k=inputs,
            mask=None,
        )
        out_1 = self.layernorm_2(inputs + attention_output_1)
        return out_1
    
    # def call(self, inputs, training, mask):
    #     inputs = self.layernorm_1(inputs)
    #     inputs = self.dense_1(inputs)

    #     attention_output_1 = self.attention_1(
    #         query=inputs,
    #         value=inputs,
    #         key=inputs,
    #         attention_mask=None,
    #         training=training,
    #     )
    #     out_1 = self.layernorm_2(inputs + attention_output_1)
    #     out_1 = self.dense_2(out_1)
        
    #     inp_2 = self.layernorm_1_2(out_1)
    #     inp_2 = self.dense_1_2(inp_2)
        
    #     attention_output_1_2 = self.attention_1_2(
    #         query=inp_2,
    #         value=inp_2,
    #         key=inp_2,
    #         attention_mask=None,
    #         training=training,
    #     )
    #     out_2 = self.layernorm_2_2(inp_2 + attention_output_1_2)
    #     return self.dense_2_2(out_2)


class PositionalEmbedding(keras.Model):
    def __init__(self, sequence_length, vocab_size, embed_dim, **kwargs):
        super().__init__(**kwargs)
        self.token_embeddings = layers.Embedding(
            input_dim=vocab_size, output_dim=embed_dim
        )
        self.position_embeddings = layers.Embedding(
            input_dim=sequence_length, output_dim=embed_dim
        )
        self.embed_scale = tf.math.sqrt(tf.cast(embed_dim, tf.float32))
        
    # def call(self, inputs):
    #     length = tf.shape(inputs)[-1]
    #     positions = tf.range(start=0, limit=length, delta=1)
    #     embedded_tokens = self.token_embeddings(inputs)
    #     embedded_positions = self.position_embeddings(positions)
    #     return embedded_tokens + embedded_positions

    def call(self, inputs):
        length = tf.shape(inputs)[-1]
        positions = tf.range(start=0, limit=length, delta=1)
        embedded_tokens = self.token_embeddings(inputs)
        embedded_tokens = embedded_tokens * self.embed_scale
        embedded_positions = self.position_embeddings(positions)
        return embedded_tokens + embedded_positions

    # def compute_mask(self, inputs, mask=None):
    #     return tf.math.not_equal(inputs, 0)


class TransformerDecoderBlock(keras.Model):
    def __init__(self, embed_dim, ff_dim, num_heads, vocab_size, max_len, **kwargs):
        super().__init__(**kwargs)
        self.embed_dim = embed_dim
        self.ff_dim = ff_dim
        self.num_heads = num_heads
        # self.attention_1 = layers.MultiHeadAttention(
        #     num_heads=num_heads, key_dim=embed_dim, dropout=0.1
        # )
        # self.attention_2 = layers.MultiHeadAttention(
        #     num_heads=num_heads, key_dim=embed_dim, dropout=0.1
        # )
        self.attention_1 = MultiHeadAttention(
            num_heads=num_heads, d_model=embed_dim
        )
        self.attention_2 = MultiHeadAttention(
            num_heads=num_heads, d_model=embed_dim
        )
        self.ffn_layer_1 = layers.Dense(ff_dim, activation="relu")
        self.ffn_layer_2 = layers.Dense(embed_dim)

        self.layernorm_1 = layers.LayerNormalization()
        self.layernorm_2 = layers.LayerNormalization()
        self.layernorm_3 = layers.LayerNormalization()
        
        
        # self.attention_1_2 = layers.MultiHeadAttention(
        #     num_heads=num_heads, key_dim=embed_dim, dropout=0.1
        # )
        # self.attention_2_2 = layers.MultiHeadAttention(
        #     num_heads=num_heads, key_dim=embed_dim, dropout=0.1
        # )
        # self.ffn_layer_1_2 = layers.Dense(ff_dim, activation="relu")
        # self.ffn_layer_2_2 = layers.Dense(embed_dim)

        # self.layernorm_1_2 = layers.LayerNormalization()
        # self.layernorm_2_2 = layers.LayerNormalization()
        # self.layernorm_3_2 = layers.LayerNormalization()


        self.embedding = PositionalEmbedding(
            embed_dim=embed_dim, sequence_length=max_len, vocab_size=vocab_size
        )
        self.out = layers.Dense(vocab_size, activation="softmax")

        self.dropout_1 = layers.Dropout(0.3)
        self.dropout_2 = layers.Dropout(0.5)

    def call(self, inputs, encoder_outputs, training, mask, b_s_t):
        
        padding_mask, causal_mask = self.create_padding_mask(inputs), self.create_look_ahead_mask(inputs.shape[1])
        
        inputs = self.embedding(inputs)
        
        
        # causal_mask, causal_mask_2 = self.get_causal_attention_mask(inputs), self.create_look_ahead_mask(inputs.shape[1])

        # if mask is not None:
        #     padding_mask = tf.cast(mask[:, :, tf.newaxis], dtype=tf.int32)
        #     combined_mask = tf.cast(mask[:, tf.newaxis, :], dtype=tf.int32)
        #     combined_mask = tf.minimum(combined_mask, causal_mask)

        # attention_output_1 = self.attention_1(
        #     query=inputs,
        #     value=inputs,
        #     key=inputs,
        #     attention_mask=combined_mask,
        #     training=training,
        # )
        attention_output_1, _ = self.attention_1(
            q=inputs,
            v=inputs,
            k=inputs,
            mask=causal_mask,
        )
        out_1 = self.layernorm_1(inputs + attention_output_1)

        # attention_output_2 = self.attention_2(
        #     query=out_1,
        #     value=encoder_outputs,
        #     key=encoder_outputs,
        #     attention_mask=padding_mask,
        #     training=training,
        # )
        attention_output_2, _ = self.attention_2(
            q=out_1,
            v=encoder_outputs,
            k=encoder_outputs,
            mask=padding_mask,
        )
        out_2 = self.layernorm_2(out_1 + attention_output_2)

        ffn_out = self.ffn_layer_1(out_2)
        ffn_out = self.dropout_1(ffn_out, training=training)
        ffn_out = self.ffn_layer_2(ffn_out)

        ffn_out = self.layernorm_3(ffn_out + out_2, training=training)
        ffn_out = self.dropout_2(ffn_out, training=training)
        preds = self.out(ffn_out)
        return preds

    def get_causal_attention_mask(self, inputs):
        input_shape = tf.shape(inputs)
        batch_size, sequence_length = input_shape[0], input_shape[1]
        i = tf.range(sequence_length)[:, tf.newaxis]
        j = tf.range(sequence_length)
        mask = tf.cast(i >= j, dtype="int32")
        mask = tf.reshape(mask, (1, input_shape[1], input_shape[1]))
        mult = tf.concat(
            [tf.expand_dims(batch_size, -1), tf.constant([1, 1], dtype=tf.int32)],
            axis=0,
        )
        return tf.tile(mask, mult)
    
    def create_masks(self, inp, tar):
        # Encoder padding mask (Used in the 2nd attention block in the decoder too.)
        padding_mask = self.create_padding_mask(inp)

        # Used in the 1st attention block in the decoder.
        # It is used to pad and mask future tokens in the input received by
        # the decoder.
        look_ahead_mask = self.create_look_ahead_mask(tf.shape(tar)[1])
        dec_target_padding_mask = self.create_padding_mask(tar)
        look_ahead_mask = tf.maximum(dec_target_padding_mask, look_ahead_mask)

        return padding_mask, look_ahead_mask
    
    def create_look_ahead_mask(self, size):
        mask = 1 - tf.linalg.band_part(tf.ones((size, size)), -1, 0)
        return mask  # (seq_len, seq_len)
    
    def create_padding_mask(self, seq):
        seq = tf.cast(tf.math.equal(seq, 0), tf.float32)

        # add extra dimensions to add the padding
        # to the attention logits.
        return seq[:, tf.newaxis, :, tf.newaxis]  # (batch_size, 1, 1, seq_len)


class Training:
    def __init__(self, **kwargs):
        self.params = munchify(kwargs)
        
        os.makedirs(os.path.dirname(self.params.meta_path), exist_ok=True)
        with open(self.params.meta_path, "w+") as meta:
            json.dump({"params": [kwargs]}, meta, indent=4)

        self.logger = log_init(self.params.model_path, "model_log")
        self.logger.info(
            f"Model '{self.params.model_path}' has been created with these params:")

        self.logger.info(
            f"---------------MODEL AND UTILS SUMMARY---------------")
        self.logger.info(
            f"CNN:\n{inspect.getsource(CNN_Encoder)}")
        self.logger.info(
            f"TRANSFORMER_ENCODER:\n{inspect.getsource(TransformerEncoderBlock)}")
        self.logger.info(
            f"TRANSFORMER_DECODER:\n{inspect.getsource(TransformerDecoderBlock)}")
        # self.logger.info(f"RESIZE_HEIGHT: {RESIZED_IMG_H}")
        # self.logger.info(f"RESIZE_WIDTH: {RESIZED_IMG_W}")
        # self.logger.info(f"DATA_SPLIT_INDEX: {DATA_SPLIT}")
        for k, v in kwargs.items():
            self.logger.info(f"{k}: {v}")
        self.logger.info(
            f"-----------------------------------------------------")
    
    def auto_train(self):
        self.data_preprocess()
        self.train()
    
    def calculate_loss(self, y_true, y_pred, mask, loss_fn):
        loss = loss_fn(y_true, y_pred)
        mask = tf.cast(mask, dtype=loss.dtype)
        loss *= mask
        a = tf.reduce_sum(mask)
        return tf.reduce_sum(loss) / tf.reduce_sum(mask)

    def calculate_accuracy(self, y_true, y_pred, mask):
        accuracy = tf.equal(y_true, tf.argmax(y_pred, axis=2))
        accuracy = tf.math.logical_and(mask, accuracy)
        accuracy = tf.cast(accuracy, dtype=tf.float32)
        mask = tf.cast(mask, dtype=tf.float32)
        return tf.reduce_sum(accuracy) / tf.reduce_sum(mask)

    def _compute_caption_loss_and_acc(self, encoder, decoder, img_embed, batch_seq, loss_fn, training=True):
        encoder_out = encoder(img_embed, training=training, mask=None)
        batch_seq_inp = batch_seq[:, :-1]
        batch_seq_true = batch_seq[:, 1:]
        mask = tf.math.not_equal(batch_seq_true, 0)
        batch_seq_pred = decoder(
            batch_seq_inp, encoder_out, training=training, mask=mask, b_s_t=batch_seq_true
        )
        loss = self.calculate_loss(batch_seq_true, batch_seq_pred, mask, loss_fn)
        # acc = self.calculate_accuracy(batch_seq_true, batch_seq_pred, mask)
        return loss, None
    
    # @tf.function
    def train_step(self, batch_data, cnn_model, encoder, decoder, optimizer, loss_fn):
        batch_img, batch_seq = batch_data
        batch_loss = 0
        batch_acc = 0
        
        # tf.keras.preprocessing.image.array_to_img(batch_img[0]).show()
        # print(batch_seq[0])

        # if image_aug:
        #     batch_img = image_aug(batch_img)
        
        # 1. Get image embeddings
        
        # 2. Pass each of the five captions one by one to the decoder
        # along with the encoder outputs and compute the loss as well as accuracy
        # for each caption.
        # batch_img = resnet.preprocess_input(batch_img)
        with tf.GradientTape() as tape:
            img_embed = cnn_model(batch_img)
            loss, _ = self._compute_caption_loss_and_acc(
                    encoder, decoder, img_embed, batch_seq[:, :], loss_fn, training=True
                )

            # 3. Update loss and accuracy
            batch_loss += loss
            # batch_acc += acc

            # 4. Get the list of all the trainable weights
        train_vars = (
                cnn_model.trainable_variables + encoder.trainable_variables + decoder.trainable_variables
            )

            # 5. Get the gradients
        grads = tape.gradient(loss, train_vars)

            # 6. Update the trainable weights
        optimizer.apply_gradients(zip(grads, train_vars))

        # 7. Update the trackers
        # batch_acc /= float(num_captions_per_image)
        # loss_tracker.update_state(batch_loss)
        # acc_tracker.update_state(batch_acc)

        # 8. Return the loss and accuracy values
        # return {"loss": loss_tracker.result(), "acc": acc_tracker.result()}
        return batch_loss, None
        
    # def test_step(self, batch_data):
    #     batch_img, batch_seq = batch_data
    #     batch_loss = 0
    #     batch_acc = 0

    #     # 1. Get image embeddings
    #     img_embed = self.cnn_model(batch_img)

    #     # 2. Pass each of the five captions one by one to the decoder
    #     # along with the encoder outputs and compute the loss as well as accuracy
    #     # for each caption.
    #     for i in range(self.num_captions_per_image):
    #         loss, acc = self._compute_caption_loss_and_acc(
    #             img_embed, batch_seq[:, i, :], training=False
    #         )

    #         # 3. Update batch loss and batch accuracy
    #         batch_loss += loss
    #         batch_acc += acc

    #     batch_acc /= float(self.num_captions_per_image)

    #     # 4. Update the trackers
    #     self.loss_tracker.update_state(batch_loss)
    #     self.acc_tracker.update_state(batch_acc)

    #     # 5. Return the loss and accuracy values
    #     return {"loss": self.loss_tracker.result(), "acc": self.acc_tracker.result()}

    def map_func_alb(self, img_name, cap):
        img = tf.io.read_file(img_name)
        img = tf.image.decode_png(img, channels=3)
        # img = tf.image.resize(img, (int(img.shape[0]*0.8), int(img.shape[1]*0.8)), ResizeMethod.GAUSSIAN)
        # enhancer = ImageEnhance.Sharpness(Image.fromarray(np.uint8(img.numpy())).convert('RGB'))
        # img = enhancer.enhance(3)
        # data = {"image":img.numpy()}

        # aug_data = transforms(**data)
        # aug_img = aug_data["image"]
        img = make_fix_size(img.numpy(), self.params.image_size[1], self.params.image_size[0], False)

        img = tf.image.resize(img, (self.params.image_size[0], self.params.image_size[1]))

        # to_img = tf.keras.preprocessing.image.array_to_img(img)
        # to_img.show()
        # print(tf.image.convert_image_dtype(img, dtype=tf.float32))
        # img = tf.image.convert_image_dtype(img, dtype=tf.float32)
        # img_tensor = np.load(img_name.decode('utf-8') + '.npy')
        # return img.astype(np.float32), cap
        return img.numpy(), cap

    def calc_max_length(self, tensor):
        return max(len(t) for t in tensor)

    def get_image_to_caption(self, annotations_file):
        # словарь, в который автоматически будет добавляться лист при попытке доступа к несущестувующему ключу
        image_path_to_caption = collections.defaultdict(list)
        for val in annotations_file['annotations']:
            caption = f"<start> {val['caption']} <end>"
            image_path = self.params.dataset_path + val["image_id"] + ".png"
            image_path_to_caption[image_path].append(
                caption)  # словарь типа 'путь_фото': ['описание1', 'описание2', ...]

        return image_path_to_caption
    
    def data_preprocess(self):
        with open(self.params.caption_path, 'r') as f:
            annotations = json.load(f)

        # with open("Flickr8k.token.txt") as caption_file:
        #     caption_data = caption_file.readlines()
        #     caption_mapping = {}
        #     text_data = []
        #     images_to_skip = set()

        #     for line in caption_data:
        #         line = line.rstrip("\n")
        #         # Image name and captions are separated using a tab
        #         img_name, caption = line.split("\t")

        #         # Each image is repeated five times for the five different captions.
        #         # Each image name has a suffix `#(caption_number)`
        #         img_name = img_name.split("#")[0]
        #         img_name = os.path.join("Flicker8k_Dataset", img_name.strip())

        #         # We will remove caption that are either too short to too long
        #         tokens = caption.strip().split()

        #         if len(tokens) < 5 or len(tokens) > 25:
        #             images_to_skip.add(img_name)
        #             continue

        #         if img_name.endswith("jpg") and img_name not in images_to_skip:
        #             # We will add a start and an end token to each caption
        #             caption = "<start> " + caption.strip() + " <end>"
        #             text_data.append(caption)

        #             if (img_name in caption_mapping) and len(caption_mapping[img_name]) < 1:
        #                 caption_mapping[img_name].append(caption)
        #             else:
        #                 caption_mapping[img_name] = [caption]

        #     for img_name in images_to_skip:
        #         if img_name in caption_mapping:
        #             del caption_mapping[img_name]
        
        # image_path_to_caption = caption_mapping
        image_path_to_caption = self.get_image_to_caption(annotations)
        image_paths = list(image_path_to_caption.keys())
        random.shuffle(image_paths)

        train_image_paths = image_paths[:self.params.image_count]
        self.logger.info(
            f"Annotations and image paths have been successfully extracted from {self.params.caption_path}\ntotal images count - {len(train_image_paths)}")

        train_captions = []  # описания для тренировки
        self.img_name_vector = []  # пути к фото

        for image_path in train_image_paths:
            caption_list = image_path_to_caption[image_path]
            train_captions.extend(caption_list)
            self.img_name_vector.extend([image_path] * len(caption_list))

        os.makedirs(os.path.dirname(self.params.tokenizer_path), exist_ok=True)
        with open(self.params.tokenizer_path, 'w+', encoding='utf-8') as f:
            # Choose the top top_k words from the vocabulary
            self.tokenizer = keras.preprocessing.text.Tokenizer(num_words=self.params.vocab_size, split=' ', oov_token="<unk>",
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
        cap_vector = keras.preprocessing.sequence.pad_sequences(
            train_seqs, padding='post')  # приведение всех
        # последовательностей к одинаковой длине
        # print(cap_vector[:100])

        # Calculates the max_length, which is used to store the attention weights

        # os.makedirs(os.path.dirname(tokenizer_params_path), exist_ok=True)
        self.max_length = self.calc_max_length(train_seqs)
        with open(self.params.meta_path, "r+") as meta:
            params = json.load(meta)
            params["params"][0].update({"max_len": self.max_length})
            meta.seek(0)
            json.dump(params, meta, indent=4)

        self.img_to_cap_vector = collections.defaultdict(list)
        for img, cap in zip(self.img_name_vector, cap_vector):
            self.img_to_cap_vector[img].append(cap)

        self.logger.info(
            f"Img_to_cap vector is compiled. Max cap length - {self.max_length}")
    
    def decode_and_resize(self, img_path, cap):
        img = tf.io.read_file(img_path)
        img = tf.image.decode_jpeg(img, channels=3)
        img = tf.image.resize(img, (299, 299))
        img = tf.image.convert_image_dtype(img, tf.float32)
        return img, cap
    
    def train_flickr(self):
        IMAGES_PATH = "Flicker8k_Dataset"


        # Vocabulary size
        VOCAB_SIZE = 10000

        # Fixed length allowed for any sequence
        SEQ_LENGTH = 25

        # Dimension for the image embeddings and token embeddings
        EMBED_DIM = 512

        # Per-layer units in the feed-forward network
        FF_DIM = 512

        # Other training parameters
        BATCH_SIZE = 64
        with open("Flickr8k.token.txt") as caption_file:
            caption_data = caption_file.readlines()
            caption_mapping = {}
            text_data = []
            images_to_skip = set()

            for line in caption_data:
                line = line.rstrip("\n")
                # Image name and captions are separated using a tab
                img_name, caption = line.split("\t")

                # Each image is repeated five times for the five different captions.
                # Each image name has a suffix `#(caption_number)`
                img_name = img_name.split("#")[0]
                img_name = os.path.join(IMAGES_PATH, img_name.strip())

                # We will remove caption that are either too short to too long
                tokens = caption.strip().split()

                if len(tokens) < 5 or len(tokens) > SEQ_LENGTH:
                    images_to_skip.add(img_name)
                    continue

                if img_name.endswith("jpg") and img_name not in images_to_skip:
                    # We will add a start and an end token to each caption
                    caption = "<start> " + caption.strip() + " <end>"
                    text_data.append(caption)

                    if img_name in caption_mapping:
                        caption_mapping[img_name].append(caption)
                    else:
                        caption_mapping[img_name] = [caption]

            for img_name in images_to_skip:
                if img_name in caption_mapping:
                    del caption_mapping[img_name]

        tmp = {}
        [tmp.update({k: v[0]}) for k, v in caption_mapping.items()]
        caption_mapping = tmp
        all_images = list(caption_mapping.keys())

        # 2. Shuffle if necessary

        np.random.shuffle(all_images)

        # 3. Split into training and validation sets
        train_size = int(len(caption_data) * 1)

        training_data = {
            img_name: caption_mapping[img_name] for img_name in all_images[:train_size]
        }
        validation_data = {
            img_name: caption_mapping[img_name] for img_name in all_images[train_size:]
        }
        
        strip_chars = "!\"#$%&'()*+,-./:;<=>?@[\]^_`{|}~"
        strip_chars = strip_chars.replace("<", "")
        strip_chars = strip_chars.replace(">", "")

        vectorization = TextVectorization(
            max_tokens=VOCAB_SIZE,
            output_mode="int",
            output_sequence_length=SEQ_LENGTH,
            standardize=lambda input_string: tf.strings.regex_replace(tf.strings.lower(input_string), "[%s]" % re.escape(strip_chars), ""),
        )
        vectorization.adapt(text_data)
        
        dataset = tf.data.Dataset.from_tensor_slices((list(training_data.keys()), list(training_data.values())))
        dataset = dataset.shuffle(len(list(training_data.keys())))
        dataset = dataset.map((lambda img_path, captions: (self.decode_and_resize(img_path), vectorization(captions))), num_parallel_calls=tf.data.AUTOTUNE)
        dataset = dataset.batch(24).prefetch(tf.data.AUTOTUNE)
        for _, cap in dataset:
            print(cap)
    
    def train(self):
        img_keys = list(self.img_to_cap_vector.keys())
        random.shuffle(img_keys)
        # print(img_keys[:100])

        slice_index = int(len(img_keys) * 1)
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

        num_steps = len(img_name_train) // self.params.batch_size
        dataset = tf.data.Dataset.from_tensor_slices(
            (img_name_train, cap_train))
        # Use map to load the numpy files in parallel
        dataset = dataset.map(lambda item1, item2: tf.numpy_function(
            self.map_func_alb, [item1, item2], [tf.float32, tf.int32]),
            num_parallel_calls=tf.data.AUTOTUNE)
        # dataset = dataset.map(self.map_func_alb, num_parallel_calls=tf.data.AUTOTUNE)

        # Shuffle and batch
        dataset = dataset.shuffle(self.params.buffer_size).batch(self.params.batch_size)
        dataset = dataset.prefetch(buffer_size=tf.data.AUTOTUNE)
        
        # for im, cap in dataset.take(1):
        #     print(im[0].shape)
        #     print(im[0])
        #     print(tf.keras.preprocessing.image.array_to_img(im[0]).show())
        #     print(cap[0])
            
        cnn_model = CNN_Encoder(self.params.embedding_dim)
        # cnn_model = get_cnn_model((110, 940))
        encoder = TransformerEncoderBlock(embed_dim=self.params.embedding_dim, dense_dim=self.params.ff_dim, num_heads=8)
        decoder = TransformerDecoderBlock(embed_dim=self.params.embedding_dim, ff_dim=self.params.ff_dim, num_heads=8, vocab_size=self.params.vocab_size, max_len=self.max_length)
        
        optimizer = keras.optimizers.Adam()
        loss_object = keras.losses.SparseCategoricalCrossentropy(
            from_logits=False, reduction='none')

        ckpt = tf.train.Checkpoint(encoder=encoder,
                                   decoder=decoder,
                                   cnn=cnn_model)
        ckpt_manager = tf.train.CheckpointManager(
            ckpt, self.params.checkpoint_path, max_to_keep=10)
        
        self.logger.info(f"OPTIMIZER SUMMARY:\n{optimizer.get_config()}")
        
        loss_plot = []

        for epoch in range(self.params.epochs):
            start = time.time()
            total_loss = 0

            for (batch, (img_tensor, target)) in tqdm(enumerate(dataset)):
                # print(f"img_tensor shape = {img_tensor.shape}")
                # print(f"target shape = {target.shape}")

                # print(img_tensor.numpy())
                # print(target.numpy())
                batch_loss, _ = self.train_step(
                    (img_tensor, target), cnn_model, encoder, decoder, optimizer, loss_object)
                # total_loss += t_loss
                # print(f"b_loss = {batch_loss}")
                if batch % 100 == 0:
                    # average_batch_loss = batch_loss.numpy() / \
                    #     int(target.shape[1])
                    self.logger.info(
                        f'Epoch {epoch + 1} Batch {batch} Loss = {batch_loss.numpy()}')
            # storing the epoch end loss value to plot later
            # loss_plot.append(total_loss / num_steps)

            ckpt_manager.save()
            self.logger.info(f"Epoch {epoch} checkpoint saved!")

            # self.logger.info(
            #     f'Epoch {epoch + 1} Loss {total_loss / num_steps:.6f}')
            self.logger.info(
                f'Time taken for 1 epoch {time.time() - start:.2f} sec\n')

        self.logger.info(f"Training is done!")



class Image2Latex_load:
    loaded = False
    d_p = "images_150\\"
    c_p = "5_dataset_large.json"
    VOCAB_SIZE = 300
    IMAGE_COUNT = 100
    IMG_H = 110
    IMG_W = 940
    BATCH_SIZE = 24
    BUFFER_SIZE = 100
    EMBEDDING_DIM = 200
    FF_DIM = 200
    EPOCHS = 100

    def __init__(self, model_name: str, working_path: str = ""):
        self.model_path = os.path.abspath(
            ".") + "\\trained_models_transformers\\" + working_path + model_name
        
        self.tokenizer_path = self.model_path + "\\tokenizer.json"
        self.checkpoint_path = self.model_path + "\\checkpoints\\"
        self.meta_path = self.model_path + "\\meta.json"

        if os.path.exists(self.checkpoint_path) and len(os.listdir(self.checkpoint_path)) != 0 and os.path.exists(self.tokenizer_path) and os.path.exists(self.meta_path):
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

        if self.loaded:
            try:
                with open(self.meta_path, "r") as meta:
                    self.params = munchify(json.load(meta)["params"])[0]
            except:
                raise IOError("config file reading error!")
            
            try:
                with open(self.params.tokenizer_path, "r") as f:
                    data = json.load(f)
                    self.tokenizer = tokenizer_from_json(data)
            except:
                raise IOError("Something went wrong with initializing tokenizer")
                    
            # self.cnn_model = get_cnn_model((299, 299))
            self.cnn_model = CNN_Encoder(self.params.embedding_dim)
            self.encoder = TransformerEncoderBlock(embed_dim=self.params.embedding_dim, dense_dim=self.params.ff_dim, num_heads=1)
            self.decoder = TransformerDecoderBlock(embed_dim=self.params.embedding_dim, ff_dim=self.params.ff_dim, num_heads=2, vocab_size=self.params.vocab_size, max_len=self.params.max_len)

            ckpt = tf.train.Checkpoint(encoder=self.encoder,
                                    decoder=self.decoder,
                                    cnn=self.cnn_model)
            ckpt_manager = tf.train.CheckpointManager(
                ckpt, self.params.checkpoint_path, max_to_keep=5)
            if ckpt_manager.latest_checkpoint:
                # restoring the latest checkpoint in checkpoint_path
                ckpt.restore(ckpt_manager.latest_checkpoint).expect_partial()
        
    def get_model_path(self):
        return self.model_path

    def input(self, inp, val):
        return val if inp == "" else inp

    def train(self):
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
                self.VOCAB_SIZE = int(
                    self.input(input(f"vocab size - Default = {self.VOCAB_SIZE} (number of top used words in caps): "),
                               self.VOCAB_SIZE))
                self.IMAGE_COUNT = int(
                    self.input(input(f"image_count - Default = {self.IMAGE_COUNT} (number of photos to be used): "),
                               self.IMAGE_COUNT))
                self.IMG_H = int(
                    self.input(input(f"image_height - Default = {self.IMG_H} (image final height after preprocessing): "),
                               self.IMG_H))
                self.IMG_W = int(
                    self.input(input(f"image_width - Default = {self.IMG_W} (image final width after preprocessing): "),
                               self.IMG_W))
                self.BATCH_SIZE = int(self.input(
                    input(f"BATCH_SIZE - Default = {self.BATCH_SIZE}: "), self.BATCH_SIZE))
                self.BUFFER_SIZE = int(
                    self.input(input(f"BUFFER_SIZE - Default = {self.BUFFER_SIZE}: "), self.BUFFER_SIZE))
                self.EMBEDDING_DIM = int(
                    self.input(input(f"embedding_dim - Default = {self.EMBEDDING_DIM}: "), self.EMBEDDING_DIM))
                self.FF_DIM = int(self.input(
                    input(f"ff_dim - Default = {self.FF_DIM} (feed-forward unit dimension): "), self.FF_DIM))
                self.EPOCHS = int(self.input(
                    input(f"EPOCHS - Default = {self.EPOCHS}: "), self.EPOCHS))
            except:
                raise TypeError("Model params initialization failed")

            self.loaded = True
            self.dataset_path = os.path.abspath(".") + "\\datasets\\" + self.d_p
            self.caption_path = os.path.abspath(".") + "\\" + self.c_p

            train = Training(model_path=self.model_path,
                             dataset_path=self.dataset_path,
                             caption_path=self.caption_path,
                             tokenizer_path=self.tokenizer_path,
                             checkpoint_path=self.checkpoint_path,
                             meta_path=self.meta_path,
                             vocab_size=self.VOCAB_SIZE,
                             image_count=self.IMAGE_COUNT,
                             image_size=(self.IMG_H, self.IMG_W),
                             batch_size=self.BATCH_SIZE,
                             buffer_size=self.BUFFER_SIZE,
                             embedding_dim=self.EMBEDDING_DIM,
                             ff_dim=self.FF_DIM,
                             epochs=self.EPOCHS)

            train.auto_train()
            
    def predict(self, image_path: str):
        if not self.loaded:
            raise ValueError("Model is not loaded!")
        return self.greedy_decoder(image_path=image_path)
        # return self.beam_decoder(image_path, beam_width=4, temperature=0.3)
        
    def index(self, array, item):
        for idx, val in np.ndenumerate(array):
            if val == item:
                return idx[1]

    def find_n_best(self, array, n):
        probs = [np.partition(array[0], i)[i] for i in range(-1, -n - 1, -1)]
        ids = [self.index(array, p) for p in probs]
        return [[prob, id] for prob, id in zip(probs, ids)]

    def beam_decoder(self, image, beam_width=10, temperature=0.3): 
        sample_img = self.load_image(image)

        # Pass the image to the CNN
        img = tf.expand_dims(sample_img, 0)
        img = self.cnn_model(img)

        # Pass the image features to the Transformer encoder
        encoded_img = self.encoder(img, training=False, mask=None)

        # Generate the caption using the Transformer decoder
        decoded_caption = "<start> "

        tokenized_caption = tf.keras.preprocessing.sequence.pad_sequences(self.tokenizer.texts_to_sequences([decoded_caption]), padding="post", maxlen=self.params.max_len)[:, :-1]
        mask = tf.math.not_equal(tokenized_caption, 0)
        predictions = self.decoder(
                tokenized_caption, encoded_img, training=False, mask=mask, b_s_t=None
            )
        predictions = tf.expand_dims((predictions[0, 0, :] / temperature), 0).numpy()
        init = self.find_n_best(predictions, beam_width)
        results = [[obj[0], obj[1], decoded_caption + self.tokenizer.index_word[int(obj[1])] + " "] for obj in
                   init]  # 0 - prob ; 1 - id ; 2 - hidden

        for i in range(1, self.params.max_len):
            tmp_res = []

            for r in results:
                tmp_tokenized_caption = tf.keras.preprocessing.sequence.pad_sequences(self.tokenizer.texts_to_sequences([r[2]]), padding="post", maxlen=self.params.max_len)[:, :-1]
                tmp_mask = tf.math.not_equal(tmp_tokenized_caption, 0)
                tmp_preds = self.decoder(tmp_tokenized_caption, encoded_img, training=False, mask=tmp_mask, b_s_t=None)

                for obj in self.find_n_best(tf.expand_dims((tmp_preds[0, i, :] / temperature), 0).numpy(), beam_width):
                    tmp_res.append(
                        [obj[0] + r[0], obj[1], r[2] + self.tokenizer.index_word[int(obj[1])] + " "])  # multiplied scores, curr id, hidden, prev id

            results.clear()
            tmp_res.sort(reverse=True, key=lambda x: x[0])
            # attention_plot[i] = tf.reshape(attention_weights, (-1,)).numpy()
            for el in range(beam_width):
                results.append(tmp_res[el])

            # for res, att in zip(results, attention_plots):
            #     att[i] = tf.reshape(res[5], (-1,)).numpy()

            # if any(self.tokenizer.index_word[int(results[i][1])] == '<end>' for i in range(len(results))):
            #     break

            if all(['<end>' in r[2] for r in results]):
                break
            
            # print()

        # for el in results:
        #     tf.print(el[3] + "\n")
        # tf.print(results[0][3])
        # return [results[0][3]], None
        return [el[2] for el in results]
        
    def greedy_decoder(self, image_path):
    # Select a random image from the validation dataset
        # Read the image from the disk
        sample_img = self.load_image(image_path)
        # img = sample_img.clip(0, 255).astype(np.uint8)
        # plt.imshow(img)
        # plt.show()

        # Pass the image to the CNN
        img = tf.expand_dims(sample_img, 0)
        img = self.cnn_model(img)

        # Pass the image features to the Transformer encoder
        encoded_img = self.encoder(img, training=False, mask=None)

        # Generate the caption using the Transformer decoder
        decoded_caption = "<start> "
        for i in range(self.params.max_len):
            # tokenized_caption = vectorization([decoded_caption])[:, :-1]
            tokenized_caption = tf.keras.preprocessing.sequence.pad_sequences(self.tokenizer.texts_to_sequences([decoded_caption]), padding="post", maxlen=self.params.max_len)[:, :-1]
            mask = tf.math.not_equal(tokenized_caption, 0)
            predictions = self.decoder(
                tokenized_caption, encoded_img, training=False, mask=mask, b_s_t=None
            )
            sampled_token_index = np.argmax(predictions[0, i, :])
            # sampled_token = index_lookup[sampled_token_index]
            sampled_token = self.tokenizer.index_word[sampled_token_index]
            if sampled_token == "<end>":
                break
            decoded_caption += " " + sampled_token

        decoded_caption = decoded_caption.replace("<start> ", "")
        decoded_caption = decoded_caption.replace(" <end>", "").strip()
        print("Predicted Caption: ", decoded_caption)

    def load_image(self, img_name):
        img = tf.io.read_file(img_name)
        img = tf.image.decode_png(img, channels=3)

        img = make_fix_size(img.numpy(), self.params.image_size[1], self.params.image_size[0], False)

        img = tf.image.resize(img, (self.params.image_size[0], self.params.image_size[1]))
        # img = tf.io.read_file(img_name)
        # img = tf.image.decode_jpeg(img, channels=3)
        # img = tf.image.resize(img, (299, 299))
        # img = tf.image.convert_image_dtype(img, tf.float32)

        # to_img = tf.keras.preprocessing.image.array_to_img(img)
        # to_img.show()
        return img.numpy()
    # def predict(self, decoder, image_path: str):
    #     if not self.loaded:
    #         raise ValueError("Model is not loaded!")
    #     pred = Prediction()
    #     res = pred.predict(decoder, image_path)
    #     if decoder == "categorical" or decoder == "beam2" or decoder == "beam3":
    #         print(' '.join(res))
    #     if decoder == "beam":
    #         [print(r[:r.index("<end>")] + "\n") for r in res]

    def random_predict(self, number=9):
        if not self.loaded:
            raise ValueError("Model is not loaded!")
        with open(self.params.caption_path, 'r+') as file:
            capt = json.load(file)["annotations"]

        parameters = {'axes.labelsize': 10,
                      'axes.titlesize': 10,
                      'figure.subplot.hspace': 0.999}
        plt.rcParams.update(parameters)
        # print(plt.rcParams.keys())
        
        images = random.choices(capt, k=number)
        plotting = []
        for im in images:
            plotting.append([im["caption"], self.predict(self.params.dataset_path + im["image_id"] + ".png"),
                             plt.imread(self.params.dataset_path + im["image_id"] + ".png"), im["image_id"]])  # real ; pred ; plt image

        _, axes = plt.subplots(nrows=5, ncols=1)

        for ax, plot in zip(axes.flat, plotting):
            ax.imshow(plot[2])
            # edit_dist = levenshteinDistance(
            #     plot[0], plot[1][0].replace('<end>', '')[:-2])
            # bleu = nltk.translate.bleu_score.sentence_bleu([list(filter(lambda a: a != " ", plot[0].split(" ")))], list(
            #     filter(lambda a: a != " ", plot[1][0][:plot[1][0].index("<end>")].split(" "))))
            ax.set(
                title=f"img_name = {plot[-1]}\nreal = {plot[0]}\npred = {plot[1][0][:plot[1][0].index('<end>')]}\nbleu = {0}")
            ax.axis('off')
        plt.show()
        
enable_gpu(False, 10)

model = Image2Latex_load("T_model_X_3")
model.train()
# model.random_predict(5)
# print(*model.predict("C:\\Users\\shace\\Desktop\\CodeCogsEqn (52).png"), sep="\n")