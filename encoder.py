import math
import tensorflow as tf
from tensorflow.keras.layers import Conv2D
from tensorflow.keras.layers import MaxPooling2D
from tensorflow.keras.layers import Dense
from tensorflow.keras.layers.experimental.preprocessing import Rescaling

class Encoder(tf.keras.Model):
    def __init__(self, embedding_dim):
        super(Encoder, self).__init__()

        self.rescale = Rescaling(scale=1. / 127.5, offset=-1)

        self.conv1_block1 = Conv2D(filters=64, kernel_size=(3, 3), padding='same', activation='relu')
        self.pool_block1 = MaxPooling2D(pool_size=(2, 2))

        self.conv1_block2 = Conv2D(filters=128, kernel_size=(3, 3), padding='same', activation='relu')
        self.pool_block2 = MaxPooling2D(pool_size=(4, 4))

        self.conv1_block3 = Conv2D(filters=256, kernel_size=(3, 3), padding='same', activation='relu')
        self.conv2_block3 = Conv2D(filters=256, kernel_size=(3, 3), padding='same', activation='relu')
        self.pool_block3 = MaxPooling2D(pool_size=(2, 2))

        self.conv1_block4 = Conv2D(filters=512, kernel_size=(3, 3), padding='same', activation='relu')
        self.pool_block4 = MaxPooling2D(pool_size=(2, 2))
        self.conv2_block4 = Conv2D(filters=512, kernel_size=(3, 3), padding='same', activation='relu')

        self.fc = Dense(embedding_dim)

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
        
        # Positional embeddings
        x = self.add_timing_signal_nd(x, min_timescale=10.0)

        # Reshaping to [batch_size, H*W, 512] shape
        x = tf.reshape(x, (x.shape[0], -1, x.shape[3]))

        # Final vector of shape [batch_size, H*W, embedding_dim]
        x = self.fc(x)
        x = tf.nn.relu(x)

        return x

    # taken from https://github.com/tensorflow/tensor2tensor/blob/37465a1759e278e8f073cd04cd9b4fe377d3c740/tensor2tensor/layers/common_attention.py
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