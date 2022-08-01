
import tensorflow as tf
from configuration import *

class gpr():
    def __init__(self, mnngp, x_train, x_test, y_train, stability_eps, size_b_1 = None):
        self.mnngp = mnngp
        self.mnngp.get_grid()
        self.size_b_1 = size_b_1
        self.stability_eps = stability_eps
        self.x_train = x_train
        self.x_test = x_test
        self.y_train = y_train

    def _build_data_v1(self, x_train, x_test):
        self.k_data_data = self.mnngp.k_full(x_train)
        self.k_data_test = self.mnngp.k_full(x_train, x_test)

    def _build_data_v2(self, x_train, x_test):
        self.k_data_data_b_11 = self.mnngp.k_full(self.x_train[:self.size_b_1, :])
        self.k_data_data_b_22 = self.mnngp.k_full(self.x_train[self.size_b_1:, :])
        self.k_data_data_b_12 = self.mnngp.k_full(self.x_train[:self.size_b_1, :], self.x_train[self.size_b_1:, :])
        self.k_data_data_b_21 = self.mnngp.k_full(self.x_train[self.size_b_1:, :], self.x_train[:self.size_b_1, :])

        self.k_data_test_1 = self.mnngp.k_full(self.x_test, self.x_train[:self.size_b_1, :])
        self.k_data_test_2 = self.mnngp.k_full(self.x_test, self.x_train[self.size_b_1:, :])

    def _predict_v1(self, variance = False):
        self._build_data_v1(self.x_train, self.x_test)

        if use_float64:
            self.k_data_data_reg = self.k_data_data + tf.eye(
            self.x_train.shape[0], dtype=tf.float64) * self.stability_eps
        else:
            self.k_data_data_reg = self.k_data_data + tf.eye(
                self.x_train.shape[0], dtype=tf.float32) * self.stability_eps

        self.l = tf.linalg.cholesky(self.k_data_data_reg)
        self.v = tf.linalg.triangular_solve(self.l, self.y_train)

        a = tf.linalg.triangular_solve(self.l, self.k_data_test)

        fmean = tf.matmul(a, self.v, transpose_a=True)

        if variance == True:
            k_test_test_diag = tf.linalg.diag_part(self.mnngp.k_full(self.x_test))
            a = tf.linalg.triangular_solve(self.l, self.k_data_test)

            fvar = k_test_test_diag - tf.reduce_sum(tf.square(a), 0)
            fvar = tf.tile(tf.reshape(fvar, (-1, 1)), [1, self.y_train.shape[1]])

            return fmean, fvar
        else:
            return fmean

    def _predict_v2(self, variance):
        self._build_data_v2(self.x_train, self.x_test)

        self.k_data_data_b_11 = self.k_data_data_b_11 + tf.eye(
            self.size_b_1, dtype=tf.float64) * self.stability_eps

        self.k_data_data_b_22 = self.k_data_data_b_22 + tf.eye(
            self.x_train.shape[0] - self.size_b_1, dtype=tf.float64) * self.stability_eps

        self.k_data_data_b_11 = tf.linalg.inv(self.k_data_data_b_11)
        self.k_data_data_b_22 = tf.linalg.inv(
            self.k_data_data_b_22 - tf.matmul(tf.matmul(self.k_data_data_b_21, self.k_data_data_b_11),
                                              self.k_data_data_b_12))
        self.k_data_data_b_12 = -tf.matmul(tf.matmul(self.k_data_data_b_11, self.k_data_data_b_12),
                                           self.k_data_data_b_22)
        self.k_data_data_b_11 = self.k_data_data_b_11 - tf.matmul(
            tf.matmul(self.k_data_data_b_12, self.k_data_data_b_21), self.k_data_data_b_11)

        fmean = tf.matmul(
            self.k_data_test_1 @ self.k_data_data_b_11 + tf.matmul(self.k_data_test_2, self.k_data_data_b_12,
                                                                   transpose_b=True),
            self.y_train[:self.size_b_1, :]) + tf.matmul(
            self.k_data_test_1 @ self.k_data_data_b_12 + self.k_data_test_2 @ self.k_data_data_b_22,
            self.y_train[self.size_b_1:, ])

        if variance == True:
            k_test_test_diag = tf.linalg.diag_part(self.mnngp.k_full(self.x_test))
            k_test_test_diag_part2 = tf.linalg.diag_part(
                tf.matmul(tf.matmul(self.k_data_test_1, self.k_data_data_b_11), self.k_data_test_1, transpose_b=True))
            k_test_test_diag_part2 += tf.linalg.diag_part(
                tf.matmul(tf.matmul(self.k_data_test_2, self.k_data_data_b_12, transpose_b=True), self.k_data_test_1,
                          transpose_b=True))
            k_test_test_diag_part2 += tf.linalg.diag_part(
                tf.matmul(tf.matmul(self.k_data_test_1, self.k_data_data_b_12), self.k_data_test_2, transpose_b=True))
            k_test_test_diag_part2 += tf.linalg.diag_part(
                tf.matmul(tf.matmul(self.k_data_test_2, self.k_data_data_b_22), self.k_data_test_2, transpose_b=True))
            fvar = k_test_test_diag - k_test_test_diag_part2

            fvar = tf.tile(tf.reshape(fvar, (-1, 1)), [1, self.y_train.shape[1]])
            return fmean, fvar
        else:
            return fmean

