import tensorflow as tf
from cluster.gen_grid import _compute_cov_grid, _compute_var
import configuration as config
import numpy as np
from interpolate import _interpolate
import tensorflow as tf
from data_processing import get_data

import os
os.environ["CUDA_VISIBLE_DEVICES"] = "-1"
#os.environ['TF_GPU_ALLOCATOR']='cuda_malloc_async'

class MNNGPKernel(object):
    def __init__(self, depth, weight_var = 1.0, bias_var = 1.0, q = 2, n_grid = 100, grid_max = 10, n_corr = 100):
        self.depth = depth
        self.q = q
        self.weight_var = weight_var
        self.bias_var = bias_var
        self.n_corr = n_corr
        self.n_grid = n_grid
        self.grid_max = grid_max

    def get_grid(self, n_grid = 100, grid_max = 10, n_corr = 100, q = 4):
        if config.read_cur_grid:
            self.grid_data = np.load(config.save_grid_path)
            self.cor_grid = np.load(config.save_cor_grid_path)
        else:
            self.grid_data, self.cor_grid = _compute_cov_grid(n_grid, grid_max, n_corr, q)
            np.save(config.save_grid_path, self.grid_data)
            np.save(config.save_cor_grid_path, self.cor_grid)

    def internpolate(self, cors):
        cors = _interpolate(cors, self.grid_data, self.cor_grid, self.n_corr)
        return cors

    def _input_layer_normalization(self, x):
        eps = 1e-15
        mean, var = tf.nn.moments(x, axes = [1], keepdims = True)
        x_normalized = (x - mean) / tf.sqrt(var + eps)
        return x_normalized

    def k_full_split(self, save_file, name,  n_blocks, input1, input2 = None):
        eps = 1e-4

        n_input = input1.shape[0]
        n_blocks_size = int(n_input/n_blocks)
        input1 = self._input_layer_normalization(input1)

        if input2 is None:
            for i in range(n_blocks):
                for j in range(n_blocks):
                    input_temp1 = input1[(i * n_blocks_size):((i + 1) * n_blocks_size)]
                    input_temp2 = input1[(j * n_blocks_size):((j + 1) * n_blocks_size)]

                    cov_init = tf.matmul(input_temp1, input_temp2, transpose_b = True)/input_temp1.shape[1]
                    del input_temp1, input_temp2

                    current_y_var = self.weight_var + self.bias_var
                    current_y_cov = self.weight_var * cov_init + self.bias_var

                    current_y_cor = current_y_cov / (current_y_var + eps)

                    gamma_q = _compute_var(self.n_grid, self.grid_max, self.q)
                    for l in range(self.depth):
                        current_z_cov = current_y_var * _interpolate(current_y_cor, grid_data=self.grid_data,
                                                                     cor_grid=self.cor_grid, n_corr=self.n_corr)
                        current_z_var = current_y_var * gamma_q
                        current_y_var = self.weight_var * current_z_var + self.bias_var
                        current_y_cov = self.weight_var * current_z_cov + self.bias_var
                        current_y_cor = current_y_cov / (current_y_var + eps)
                    np.save(save_file + name + '_' + str(i) + '_' + str(j) + '.npy', current_y_cov.numpy())
        else:
            input2 = self._input_layer_normalization(input2)
            for i in range(n_blocks):
                input_temp1 = input1[(i * n_blocks_size):((i + 1) * n_blocks_size)]

                cov_init = tf.matmul(input_temp1, input2, transpose_b=True) / input_temp1.shape[1]
                del input_temp1
                current_y_var = self.weight_var + self.bias_var
                current_y_cov = self.weight_var * cov_init + self.bias_var

                current_y_cor = current_y_cov / (current_y_var + eps)

                gamma_q = _compute_var(self.n_grid, self.grid_max, self.q)
                for l in range(self.depth):
                    current_z_cov = current_y_var * _interpolate(current_y_cor, grid_data=self.grid_data,
                                                                 cor_grid=self.cor_grid, n_corr=self.n_corr)
                    current_z_var = current_y_var * gamma_q
                    current_y_var = self.weight_var * current_z_var + self.bias_var
                    current_y_cov = self.weight_var * current_z_cov + self.bias_var
                    current_y_cor = current_y_cov / (current_y_var + eps)
                np.save(save_file + name + '_' + str(i) + '.npy', current_y_cov.numpy())

    def k_full(self, input1, input2 = None):
        eps = 1e-4

        input1 = self._input_layer_normalization(input1)
        if input2 is None:
            input2 = input1
        else:
            input2 = self._input_layer_normalization(input2)

        cov_init = tf.matmul(
            input1, input2, transpose_b=True) / input1.shape[1]

        current_y_var = self.weight_var + self.bias_var
        current_y_cov = self.weight_var * cov_init + self.bias_var
        current_y_cor = current_y_cov/(current_y_var + eps)

        gamma_q = _compute_var(self.n_grid, self.grid_max, self.q)
        for l in range(self.depth):
            current_z_cov = current_y_var * _interpolate(current_y_cor, grid_data = self.grid_data, cor_grid = self.cor_grid, n_corr = self.n_corr)
            current_z_var = current_y_var * gamma_q

            current_y_var = self.weight_var * current_z_var + self.bias_var
            current_y_cov = self.weight_var * current_z_cov + self.bias_var
            current_y_cor = current_y_cov/(current_y_var + eps)

        return current_y_cov


























