
import tensorflow as tf
import tensorflow_probability as tfp
from scipy.stats import multivariate_normal
import multiprocessing as mp
import functools
import numpy as np
from itertools import product
import tensorflow_datasets as tfds
from utils import ReflectionPadding2D
from data_processing import get_data_loader

def mvn_cdf(z, rv):
    return rv.cdf(z)

def transform_data(x, mean, sd, padding, padding_type, random_crop = None):
    x = tf.cast(x, tf.float64)
    x = x / 255.0
    for d in range(x.shape[3]):
        x = (x - mean[d])/sd[d]

    if padding_type == 'zero':
        pad_layer = tf.keras.layers.ZeroPadding2D(padding = padding)
    elif padding_type == 'reflection':
        pad_layer = ReflectionPadding2D(padding = (padding, padding))
    x = pad_layer(x)

    if random_crop != None:
        random_crop_layer = tf.keras.layers.RandomCrop(height=random_crop[0], width=random_crop[1])
        x = random_crop_layer(x)
    return x

if __name__ == '__main__':
    train_ds = get_data_loader('train', 'cifar10', batch_size = 128, shuffle = True, number_sample = 1000)
    for images, labels in train_ds:
        print(images)


