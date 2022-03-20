from tensorflow.keras import Model
from tensorflow.keras.layers import Dense, Flatten, Conv2D
from tensorflow_addons.layers import Maxout
import tensorflow as tf


class maxout_nn(Model):
    def __init__(self, width=1000, depth=1, q=2, n_out=10):
        super(maxout_nn, self).__init__()
        self.width = width
        self.q = q
        self.depth = depth
        self.n_out = n_out

        self.flatten = tf.keras.layers.Flatten()
        self.dense_1 = Dense(self.width * self.q)
        self.dense_2 = Dense(self.width * self.q)
        self.dense_3 = Dense(self.width * self.q)
        self.maxout = Maxout(self.width)
        self.output_layer = Dense(self.n_out)

    def call(self, x):
        x = self.flatten(x)
        if self.depth == 1:
            x = self.dense_1(x)
            x = self.maxout(x)
            x = self.output_layer(x)
            return x
        elif self.depth == 2:
            x = self.dense_1(x)
            x = self.maxout(x)
            x = self.dense_2(x)
            x = self.maxout(x)
            x = self.output_layer(x)
            return x
        elif self.depth == 3:
            x = self.dense_1(x)
            x = self.maxout(x)
            x = self.dense_2(x)
            x = self.maxout(x)
            x = self.dense_3(x)
            x = self.maxout(x)
            x = self.output_layer(x)
            return x






