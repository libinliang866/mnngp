
import tensorflow as tf
from mnngp import MNNGPKernel
from data_processing import get_data
from configuration import *
import numpy as np

import os
os.environ["CUDA_VISIBLE_DEVICES"]="0"
os.environ['TF_GPU_ALLOCATOR']='cuda_malloc_async'


class gpr():
    def __init__(self, mnngp, y_train, stability_eps):
        self.mnngp = mnngp
        self.mnngp.get_grid()

        self.stability_eps = stability_eps
        self.y_train = y_train

    def block_wise_inverse(self, file):
        ### inverse dat 11
        dat_0_0 = np.load(file + 'training_set_0_0.npy')
        dat_0_1 = np.load(file + 'training_set_0_1.npy')
        dat_1_0 = np.load(file + 'training_set_1_0.npy')
        dat_1_1 = np.load(file + 'training_set_1_1.npy')

        block_size = dat_0_0.shape[0]
        dat_11 = np.zeros((block_size*2, block_size*2))
        dat_11[:block_size,:block_size] = dat_0_0
        del dat_0_0
        dat_11[:block_size,block_size:] = dat_0_1
        del dat_0_1
        dat_11[block_size:,:block_size] = dat_1_0
        del dat_1_0
        dat_11[block_size:,block_size:] = dat_1_1
        del dat_1_1

        #dat_11 = tf.convert_to_tensor(dat_11)
        #dat_11 = tf.linalg.inv(dat_11)

        #dat_temp = tf.linalg.cholesky(dat_11)
        #dat_temp = tf.linalg.triangular_solve(dat_temp, tf.eye(block_size*2, tf.float64))
        #dat_11 = tf.matmul(dat_temp, dat_temp, transpose_a=True)

        cholMp = tf.linalg.cholesky(dat_11)
        icholMp = tf.linalg.triangular_solve(cholMp, tf.eye(block_size*2, dtype = tf.float64))
        dat_11 = tf.matmul(icholMp, icholMp, transpose_a = True)
        print('finishing inverse 11')
        del icholMp

        np.save(file + 'inverse_training_set' + '_' + str(0) + '_' + str(0) + '.npy', dat_11[:block_size,:block_size].numpy())
        np.save(file + 'inverse_training_set' + '_' + str(0) + '_' + str(1) + '.npy', dat_11[:block_size, block_size:].numpy())
        np.save(file + 'inverse_training_set' + '_' + str(1) + '_' + str(0) + '.npy', dat_11[block_size:, :block_size].numpy())
        np.save(file + 'inverse_training_set' + '_' + str(1) + '_' + str(1) + '.npy', dat_11[block_size:, block_size:].numpy())

        ### calculate dat_21_11
        dat_2_0 = np.load(file + 'training_set_2_0.npy')
        dat_2_1 = np.load(file + 'training_set_2_1.npy')
        dat_3_0 = np.load(file + 'training_set_3_0.npy')
        dat_3_1 = np.load(file + 'training_set_3_1.npy')

        block_size = dat_2_0.shape[0]
        dat_21 = np.zeros((block_size * 2, block_size * 2))
        dat_21[:block_size, :block_size] = dat_2_0
        del dat_2_0
        dat_21[:block_size, block_size:] = dat_2_1
        del dat_2_1
        dat_21[block_size:, :block_size] = dat_3_0
        del dat_3_0
        dat_21[block_size:, block_size:] = dat_3_1
        del dat_3_1

        dat_21 = tf.convert_to_tensor(dat_21)
        dat_21_11 = tf.matmul(dat_21, dat_11)
        del dat_11, dat_21

        ### calculate dat_21_11_12
        dat_0_2 = np.load(file + 'training_set_0_2.npy')
        dat_0_3 = np.load(file + 'training_set_0_3.npy')
        dat_1_2 = np.load(file + 'training_set_1_2.npy')
        dat_1_3 = np.load(file + 'training_set_1_3.npy')

        block_size = dat_0_2.shape[0]
        dat_12 = np.zeros((block_size * 2, block_size * 2))
        dat_12[:block_size, :block_size] = dat_0_2
        del dat_0_2
        dat_12[:block_size, block_size:] = dat_0_3
        del dat_0_3
        dat_12[block_size:, :block_size] = dat_1_2
        del dat_1_2
        dat_12[block_size:, block_size:] = dat_1_3
        del dat_1_3

        dat_12 = tf.convert_to_tensor(dat_12)
        dat_21_11_12 = tf.matmul(dat_21_11, dat_12)
        del dat_21_11

        ### calculate inverse dat 22
        dat_2_2 = np.load(file + 'training_set_2_2.npy')
        dat_2_3 = np.load(file + 'training_set_2_3.npy')
        dat_3_2 = np.load(file + 'training_set_3_2.npy')
        dat_3_3 = np.load(file + 'training_set_3_3.npy')


        block_size = dat_2_2.shape[0]
        dat_22 = np.zeros((block_size * 2, block_size * 2))
        dat_22[:block_size, :block_size] = dat_2_2
        del dat_2_2
        dat_22[:block_size, block_size:] = dat_2_3
        del dat_2_3
        dat_22[block_size:, :block_size] = dat_3_2

        del dat_3_2
        dat_22[block_size:, block_size:] = dat_3_3
        del dat_3_3

        dat_22_minus_21_11_12 = dat_22 - dat_21_11_12
        del dat_22, dat_21_11_12
        #dat_22_minus_21_11_12 = tf.linalg.inv(dat_22_minus_21_11_12)

        cholMp = tf.linalg.cholesky(dat_22_minus_21_11_12)
        icholMp = tf.linalg.triangular_solve(cholMp, tf.eye(block_size * 2, dtype=tf.float64))
        dat_22_minus_21_11_12 = tf.matmul(icholMp, icholMp, transpose_a=True)
        print('finishing inverse 22_minus_21_11_12')
        del icholMp


        np.save(file + 'inverse_training_set' + '_' + str(2) + '_' + str(2) + '.npy',
                dat_22_minus_21_11_12[:block_size, :block_size].numpy())
        np.save(file + 'inverse_training_set' + '_' + str(2) + '_' + str(3) + '.npy',
                dat_22_minus_21_11_12[:block_size, block_size:].numpy())
        np.save(file + 'inverse_training_set' + '_' + str(3) + '_' + str(2) + '.npy',
                dat_22_minus_21_11_12[block_size:, :block_size].numpy())
        np.save(file + 'inverse_training_set' + '_' + str(3) + '_' + str(3) + '.npy',
                dat_22_minus_21_11_12[block_size:, block_size:].numpy())
        del dat_22_minus_21_11_12



        ### calculate dat_11_12
        # load inverse dat 11
        dat_0_0 = np.load(file + 'inverse_training_set_0_0.npy')
        dat_0_1 = np.load(file + 'inverse_training_set_0_1.npy')
        dat_1_0 = np.load(file + 'inverse_training_set_1_0.npy')
        dat_1_1 = np.load(file + 'inverse_training_set_1_1.npy')

        block_size = dat_0_0.shape[0]
        dat_inv_11 = np.zeros((block_size * 2, block_size * 2))
        dat_inv_11[:block_size, :block_size] = dat_0_0
        del dat_0_0
        dat_inv_11[:block_size, block_size:] = dat_0_1
        del dat_0_1
        dat_inv_11[block_size:, :block_size] = dat_1_0
        del dat_1_0
        dat_inv_11[block_size:, block_size:] = dat_1_1
        del dat_1_1
        dat_inv_11 = tf.convert_to_tensor(dat_inv_11)

        # dat_12
        dat_0_2 = np.load(file + 'training_set_0_2.npy')
        dat_0_3 = np.load(file + 'training_set_0_3.npy')
        dat_1_2 = np.load(file + 'training_set_1_2.npy')
        dat_1_3 = np.load(file + 'training_set_1_3.npy')

        block_size = dat_0_2.shape[0]
        dat_12 = np.zeros((block_size * 2, block_size * 2))
        dat_12[:block_size, :block_size] = dat_0_2
        del dat_0_2
        dat_12[:block_size, block_size:] = dat_0_3
        del dat_0_3
        dat_12[block_size:, :block_size] = dat_1_2
        del dat_1_2
        dat_12[block_size:, block_size:] = dat_1_3
        del dat_1_3

        dat_12 = tf.convert_to_tensor(dat_12)

        dat_inv_11_12 = tf.matmul(dat_inv_11, dat_12)
        del dat_inv_11, dat_12


        ### calculate dat_inv_11_12_22
        ### calculate inverse dat 22
        dat_2_2 = np.load(file + 'inverse_training_set_2_2.npy')
        dat_2_3 = np.load(file + 'inverse_training_set_2_3.npy')
        dat_3_2 = np.load(file + 'inverse_training_set_3_2.npy')
        dat_3_3 = np.load(file + 'inverse_training_set_3_3.npy')

        block_size = dat_2_2.shape[0]
        dat_22 = np.zeros((block_size * 2, block_size * 2))
        dat_22[:block_size, :block_size] = dat_2_2
        del dat_2_2
        dat_22[:block_size, block_size:] = dat_2_3
        del dat_2_3
        dat_22[block_size:, :block_size] = dat_3_2
        del dat_3_2
        dat_22[block_size:, block_size:] = dat_3_3
        del dat_3_3

        dat_inv_11_12_22 = -tf.matmul(dat_inv_11_12, dat_22)


        np.save(file + 'inverse_training_set' + '_' + str(0) + '_' + str(2) + '.npy',
                dat_inv_11_12_22[:block_size, :block_size].numpy())
        np.save(file + 'inverse_training_set' + '_' + str(0) + '_' + str(3) + '.npy',
                dat_inv_11_12_22[:block_size, block_size:].numpy())
        np.save(file + 'inverse_training_set' + '_' + str(1) + '_' + str(2) + '.npy',
                dat_inv_11_12_22[block_size:, :block_size].numpy())
        np.save(file + 'inverse_training_set' + '_' + str(1) + '_' + str(3) + '.npy',
                dat_inv_11_12_22[block_size:, block_size:].numpy())


        ### calculate inv 11
        #load dat 21
        dat_2_0 = np.load(file + 'training_set_2_0.npy')
        dat_2_1 = np.load(file + 'training_set_2_1.npy')
        dat_3_0 = np.load(file + 'training_set_3_0.npy')
        dat_3_1 = np.load(file + 'training_set_3_1.npy')

        block_size = dat_2_0.shape[0]
        dat_21 = np.zeros((block_size * 2, block_size * 2))
        dat_21[:block_size, :block_size] = dat_2_0
        del dat_2_0
        dat_21[:block_size, block_size:] = dat_2_1
        del dat_2_1
        dat_21[block_size:, :block_size] = dat_3_0
        del dat_3_0
        dat_21[block_size:, block_size:] = dat_3_1
        del dat_3_1
        dat_21 = tf.convert_to_tensor(dat_21)

        dat_inv_12_21 = tf.matmul(dat_inv_11_12_22, dat_21)
        del dat_inv_11_12_22, dat_21

        ### load inv dat 11
        dat_0_0 = np.load(file + 'inverse_training_set_0_0.npy')
        dat_0_1 = np.load(file + 'inverse_training_set_0_1.npy')
        dat_1_0 = np.load(file + 'inverse_training_set_1_0.npy')
        dat_1_1 = np.load(file + 'inverse_training_set_1_1.npy')

        block_size = dat_0_0.shape[0]
        dat_inv_11 = np.zeros((block_size * 2, block_size * 2))
        dat_inv_11[:block_size, :block_size] = dat_0_0
        del dat_0_0
        dat_inv_11[:block_size, block_size:] = dat_0_1
        del dat_0_1
        dat_inv_11[block_size:, :block_size] = dat_1_0
        del dat_1_0
        dat_inv_11[block_size:, block_size:] = dat_1_1
        del dat_1_1
        #dat_inv_11 = tf.convert_to_tensor(dat_inv_11)

        dat_inv_11_out = dat_inv_11 - np.matmul(dat_inv_12_21.numpy(), dat_inv_11)

        np.save(file + 'inverse_training_set' + '_' + str(0) + '_' + str(0) + '.npy',
                dat_inv_11_out[:block_size, :block_size])
        np.save(file + 'inverse_training_set' + '_' + str(0) + '_' + str(1) + '.npy',
                dat_inv_11_out[:block_size, block_size:])
        np.save(file + 'inverse_training_set' + '_' + str(1) + '_' + str(0) + '.npy',
                dat_inv_11_out[block_size:, :block_size])
        np.save(file + 'inverse_training_set' + '_' + str(1) + '_' + str(1) + '.npy',
                dat_inv_11_out[block_size:, block_size:])

    def _predict(self, file):
        ###
        dat_0_0 = np.load(file + 'inverse_training_set_0_0.npy')
        dat_0_1 = np.load(file + 'inverse_training_set_0_1.npy')
        dat_1_0 = np.load(file + 'inverse_training_set_1_0.npy')
        dat_1_1 = np.load(file + 'inverse_training_set_1_1.npy')

        block_size = dat_0_0.shape[0]
        dat_inv_11 = np.zeros((block_size * 2, block_size * 2))
        dat_inv_11[:block_size, :block_size] = dat_0_0
        del dat_0_0
        dat_inv_11[:block_size, block_size:] = dat_0_1
        del dat_0_1
        dat_inv_11[block_size:, :block_size] = dat_1_0
        del dat_1_0
        dat_inv_11[block_size:, block_size:] = dat_1_1
        del dat_1_1
        dat_inv_11 = tf.convert_to_tensor(dat_inv_11)

        #print(dat_inv_11)
        Kxx_inv_y_part1 = tf.matmul(dat_inv_11, self.y_train[:block_size*2,:])
        del dat_inv_11

        dat_0_2 = np.load(file + 'inverse_training_set_0_2.npy')
        dat_0_3 = np.load(file + 'inverse_training_set_0_3.npy')
        dat_1_2 = np.load(file + 'inverse_training_set_1_2.npy')
        dat_1_3 = np.load(file + 'inverse_training_set_1_3.npy')

        block_size = dat_0_2.shape[0]
        dat_inv_12 = np.zeros((block_size * 2, block_size * 2))
        dat_inv_12[:block_size, :block_size] = dat_0_2
        del dat_0_2
        dat_inv_12[:block_size, block_size:] = dat_0_3
        del dat_0_3
        dat_inv_12[block_size:, :block_size] = dat_1_2
        del dat_1_2
        dat_inv_12[block_size:, block_size:] = dat_1_3
        del dat_1_3
        dat_inv_12 = tf.convert_to_tensor(dat_inv_12)

        Kxx_inv_y_part1 = Kxx_inv_y_part1 + tf.matmul(dat_inv_12, self.y_train[block_size*2:,:])

        ###
        dat_2_2 = np.load(file + 'inverse_training_set_2_2.npy')
        dat_2_3 = np.load(file + 'inverse_training_set_2_3.npy')
        dat_3_2 = np.load(file + 'inverse_training_set_3_2.npy')
        dat_3_3 = np.load(file + 'inverse_training_set_3_3.npy')

        block_size = dat_2_2.shape[0]
        dat_inv_22 = np.zeros((block_size * 2, block_size * 2))
        dat_inv_22[:block_size, :block_size] = dat_2_2
        del dat_2_2
        dat_inv_22[:block_size, block_size:] = dat_2_3
        del dat_2_3
        dat_inv_22[block_size:, :block_size] = dat_3_2
        del dat_3_2
        dat_inv_22[block_size:, block_size:] = dat_3_3
        del dat_3_3
        dat_inv_22 = tf.convert_to_tensor(dat_inv_22)

        Kxx_inv_y_part2 = tf.matmul(dat_inv_12, self.y_train[:block_size*2,:], transpose_a = True)
        del dat_inv_12


        Kxx_inv_y_part2 = Kxx_inv_y_part2 + tf.matmul(dat_inv_22, self.y_train[block_size*2:, :])
        del dat_inv_22

        ###
        dat_train_test_0 = np.load(file + 'training_testing_set_0.npy')
        dat_train_test_1 = np.load(file + 'training_testing_set_1.npy')

        ###
        n_block = dat_train_test_0.shape[0]
        n_test = dat_train_test_0.shape[1]
        dat_train_test_part1 = np.zeros((n_block * 2, n_test))
        dat_train_test_part1[:n_block,:] = dat_train_test_0
        del dat_train_test_0
        dat_train_test_part1[n_block:,:] = dat_train_test_1
        del dat_train_test_1

        f_mean_part1 = tf.matmul(dat_train_test_part1, Kxx_inv_y_part1, transpose_a = True)
        del dat_train_test_part1, Kxx_inv_y_part1


        ###
        dat_train_test_2 = np.load(file + 'training_testing_set_2.npy')
        dat_train_test_3 = np.load(file + 'training_testing_set_3.npy')

        ###
        n_block = dat_train_test_2.shape[0]
        n_test = dat_train_test_2.shape[1]
        dat_train_test_part2 = np.zeros((n_block * 2, n_test))
        dat_train_test_part2[:n_block, :] = dat_train_test_2
        del dat_train_test_2
        dat_train_test_part2[n_block:, :] = dat_train_test_3
        del dat_train_test_3

        f_mean_part2 = tf.matmul(dat_train_test_part2, Kxx_inv_y_part2, transpose_a=True)
        del dat_train_test_part2, Kxx_inv_y_part2

        f_mean = f_mean_part1 + f_mean_part2

        #print(f_mean_part1.shape)
        return f_mean


if __name__ == '__main__':

    dat_train, y_train = get_data(split = 'train', extract_number = 25000, one_hot = True, data_set = 'cifar10', shuffle = False)
    dat_test, y_test = get_data(split = 'test', extract_number = 10000, one_hot = True, data_set = 'cifar10')

    del dat_train, dat_test

    if use_float64:
        y_train = tf.cast(y_train, tf.float64)
        y_test = tf.cast(y_test, tf.float64)
    else:
        y_train = tf.cast(y_train, tf.float32)
        y_test = tf.cast(y_test, tf.float32)

    mnn_kernel = MNNGPKernel(depth = 3, q = 2, weight_var = 1, bias_var = 0.0, n_corr = 501)
    mnn_kernel.get_grid()
    mnn_gpr = gpr(mnngp = mnn_kernel, y_train = y_train, stability_eps = 0.5)
    mnn_gpr.block_wise_inverse('matrixs/')

    fmean = mnn_gpr._predict('matrixs/')
    pred_y = tf.math.argmax(fmean, axis = 1)
    true_lab = tf.math.argmax(y_test, axis = 1)
    print(sum(pred_y.numpy() == true_lab.numpy())/true_lab.numpy().shape[0])