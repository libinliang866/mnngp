
import numpy as np
import tensorflow as tf
from configuration import *

def _interpolate(cors, grid_data, cor_grid, n_corr):
    '''
    grid_data = grid_data.astype(np.float64)
    cors = tf.convert_to_tensor(cors, dtype=tf.float64)

    cor_grid = tf.cast(tf.linspace(-1, 1, n_corr), tf.float64)
    spacing = tf.cast(cor_grid[1] - cor_grid[0], tf.float64)
    cors_float = (cors - cor_grid[0])/spacing
    '''

    cor_gap = cor_grid[1] - cor_grid[0]

    if use_float64:
        cors = tf.convert_to_tensor(cors, dtype=tf.float64)
    else:
        cors = tf.convert_to_tensor(cors, dtype=tf.float32)
    cors_float = (cors - cor_grid[0]) / cor_gap


    ind_floor = tf.math.floor(cors_float)
    ind_ceiling = tf.math.ceil(cors_float)

    weight_floor = 1.0 - (cors_float - ind_floor)
    weight_ceiling = 1.0 - (ind_ceiling - cors_float)

    ind_floor = tf.cast(ind_floor, dtype = tf.int32)
    ind_ceiling = tf.cast(ind_ceiling, dtype = tf.int32)

    if use_float64:
        cor_floor = tf.cast(tf.gather(grid_data, ind_floor), tf.float64)
        cor_ceiling = tf.cast(tf.gather(grid_data, ind_ceiling), tf.float64)
    else:
        cor_floor = tf.cast(tf.gather(grid_data, ind_floor), tf.float32)
        cor_ceiling = tf.cast(tf.gather(grid_data, ind_ceiling), tf.float32)

    cor_out = weight_floor * cor_floor + weight_ceiling * cor_ceiling
    return cor_out




