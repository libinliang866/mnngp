
import tensorflow as tf
from mnngp import MNNGPKernel
from data_processing import get_data
from configuration import *
import numpy as np

class gpr():
    def __init__(self, mnngp, x_train, x_test, y_train, stability_eps):
        self.mnngp = mnngp
        self.mnngp.get_grid()

        self.stability_eps = stability_eps
        self.x_train = x_train
        self.x_test = x_test
        self.y_train = y_train

    def _build_data_data(self, x_train):
        self.k_data_data = self.mnngp.k_full(x_train)
        #self.k_data_data = tf.cast(self.k_data_data, tf.float64)
        #self.k_data_data.numpy()
        #np.save('k_data_data_64.npy', self.k_data_data)

    def _build_data_test(self, x_train, x_test):
        self.k_data_test = self.mnngp.k_full(x_train, x_test)
        #self.k_data_test = tf.cast(self.k_data_test, tf.float64)

    def _predict(self):
        self._build_data_data(self.x_train)
        self._build_data_test(self.x_train, self.x_test)

        if use_float64:
            self.k_data_data_reg = self.k_data_data + tf.eye(
            self.x_train.shape[0], dtype=tf.float64) * self.stability_eps
        else:
            self.k_data_data_reg = self.k_data_data + tf.eye(
                self.x_train.shape[0], dtype=tf.float32) * self.stability_eps
        #self.k_data_data_reg = tf.cast(self.k_data_data_reg, tf.float64)
        #self.k_data_test = tf.cast(self.k_data_test, tf.float64)
        #self.y_train = tf.cast(self.y_train, tf.float64)

        #for i in range(10):
        #    print('-------')
        #    print(i)
        #    print(self.k_data_data_reg[i,i])

        self.l = tf.linalg.cholesky(self.k_data_data_reg)
        self.v = tf.linalg.triangular_solve(self.l, self.y_train)

        a = tf.linalg.triangular_solve(self.l, self.k_data_test)

        fmean = tf.matmul(a, self.v, transpose_a=True)

        return fmean



class gpr_block():
    def __init__(self, mnngp, x_train, x_test, y_train, size_b_1, stability_eps):
        self.mnngp = mnngp
        self.mnngp.get_grid()
        self.size_b_1 = size_b_1
        self.stability_eps = stability_eps
        self.x_train = x_train
        self.x_test = x_test
        self.y_train = y_train

    def _predict(self):
        self.k_data_data_b_11 = self.mnngp.k_full(self.x_train[:self.size_b_1, : ])
        self.k_data_data_b_22 = self.mnngp.k_full(self.x_train[self.size_b_1:, : ])
        self.k_data_data_b_12 = self.mnngp.k_full(self.x_train[:self.size_b_1, : ], self.x_train[self.size_b_1:, : ])
        self.k_data_data_b_21 = self.mnngp.k_full(self.x_train[self.size_b_1:, : ], self.x_train[:self.size_b_1, : ])

        self.k_data_test_1 = self.mnngp.k_full(self.x_test, self.x_train[:self.size_b_1, : ])
        self.k_data_test_2 = self.mnngp.k_full(self.x_test, self.x_train[self.size_b_1:, : ])

        if use_float64:
            self.k_data_data_b_11 = self.k_data_data_b_11 + tf.eye(
            self.size_b_1, dtype=tf.float64) * self.stability_eps
            
            self.k_data_data_b_22 = self.k_data_data_b_22 + tf.eye(
            self.x_train.shape[0] - self.size_b_1, dtype=tf.float64) * self.stability_eps
        else:
            self.k_data_data_b_11 = self.k_data_data_b_11 + tf.eye(
            self.size_b_1, dtype=tf.float32) * self.stability_eps
            
            self.k_data_data_b_22 = self.k_data_data_b_22 + tf.eye(
            self.x_train.shape[0] - self.size_b_1, dtype=tf.float32) * self.stability_eps
        
        self.k_data_data_b_11 = tf.linalg.inv(self.k_data_data_b_11)
        self.k_data_data_b_22 = tf.linalg.inv(self.k_data_data_b_22 - tf.matmul(tf.matmul(self.k_data_data_b_21, self.k_data_data_b_11), self.k_data_data_b_12))
        self.k_data_data_b_12 = -tf.matmul(tf.matmul(self.k_data_data_b_11, self.k_data_data_b_12), self.k_data_data_b_22) 
        self.k_data_data_b_11 = self.k_data_data_b_11 - tf.matmul(tf.matmul(self.k_data_data_b_12, self.k_data_data_b_21), self.k_data_data_b_11)

        fmean = tf.matmul(self.k_data_test_1 @ self.k_data_data_b_11 + tf.matmul(self.k_data_test_2, self.k_data_data_b_12, transpose_b=True), self.y_train[:self.size_b_1, :]) + tf.matmul(self.k_data_test_1 @ self.k_data_data_b_12 + self.k_data_test_2 @ self.k_data_data_b_22, self.y_train[self.size_b_1:,])

        return fmean



if __name__ == '__main__':

    dat_train, y_train = get_data(split = 'train', extract_number = 10000, one_hot = True, data_set = 'cifar10', shuffle = False)
    dat_test, y_test = get_data(split = 'test', extract_number = 10000, one_hot = True, data_set = 'cifar10')

    if use_float64:
        dat_train = tf.cast(dat_train, tf.float64)
        dat_test = tf.cast(dat_test, tf.float64)
        y_train = tf.cast(y_train, tf.float64)
        y_test = tf.cast(y_test, tf.float64)
    else:
        dat_train = tf.cast(dat_train, tf.float32)
        dat_test = tf.cast(dat_test, tf.float32)
        y_train = tf.cast(y_train, tf.float32)
        y_test = tf.cast(y_test, tf.float32)

    mnn_kernel = MNNGPKernel(depth = 3, q = 2, weight_var = 1, bias_var = 0.00, n_corr = 501)
    mnn_kernel.get_grid()
    #mnn_gpr = gpr(mnngp = mnn_kernel, x_train = dat_train, x_test = dat_test, y_train = y_train, stability_eps = 0.0)
    #size_b_1, size of first block
    mnn_gpr = gpr_block(mnngp = mnn_kernel, x_train = dat_train, x_test = dat_test, y_train = y_train, size_b_1 = 5000, stability_eps = 0.0)

    fmean = mnn_gpr._predict()
    pred_y = tf.math.argmax(fmean, axis = 1)
    true_lab = tf.math.argmax(y_test, axis = 1)

    print(sum(pred_y.numpy() == true_lab.numpy())/true_lab.numpy().shape[0])


