import tensorflow as tf
from keras.layers import Embedding
from keras.layers import LSTMCell
from keras.layers import Dense
from attention import BahdanauAttention

class Decoder(tf.keras.Model):
    def __init__(self, embedding_dim, units, vocab_size):
        super(Decoder, self).__init__()
        self.units = units

        self.embedding = Embedding(vocab_size, embedding_dim)

        self.lstm = LSTMCell(self.units, recurrent_initializer='glorot_uniform')

        self.fc1 = Dense(self.units)
        self.fc2 = Dense(vocab_size)

        self.attention = BahdanauAttention(self.units)

        # self.dropout = tf.keras.layers.Dropout(0.5)
        # self.b_n = tf.keras.layers.BatchNormalization()

        self.get_initial_state = self.lstm.get_initial_state

    def call(self, x, features, state_output, hidden):
        context_vector, attention_weights = self.attention(features, state_output)

        x = self.embedding(x)

        x = tf.concat([context_vector, tf.squeeze(x, axis=1)], axis=-1)

        # passing the concatenated vector to the LSTM
        state_output, state = self.lstm(x, hidden)

        x = self.fc1(state_output)

        # x = self.dropout(x)
        # x = self.b_n(x)

        x = self.fc2(x)

        return x, state_output, state, attention_weights

    def reset_state(self, batch_size):
        return tf.zeros((batch_size, self.units))