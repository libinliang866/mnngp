
import tensorflow as tf
import numpy as np

def blocks_inverse_1(save_file, name, rows, cols):
    C = np.load(save_file + name + '_' + str(rows[1]) + '_' + str(cols[1]) + '.npy')
    A = np.load(save_file + name + '_' + str(rows[0]) + '_' + str(cols[0]) + '.npy')
    B = np.load(save_file + name + '_' + str(rows[0]) + '_' + str(cols[1]) + '.npy')

    C = tf.convert_to_tensor(C)
    A = tf.convert_to_tensor(A)
    B = tf.convert_to_tensor(B)

    C_inv = tf.linalg.inv(C)
    del C

    B_C_inv = tf.matmul(B, C_inv)
    B_C_inv_B_T = tf.matmul(B_C_inv, B, transpose_b = True)
    A_tilde = A - B_C_inv_B_T
    del B_C_inv_B_T

    A_tilde_inv = tf.linalg.inv(A_tilde)
    del A_tilde

    A_tilde_inv_B = tf.matmul(A_tilde_inv, B)
    minus_A_tilde_inv_B_C_inv = -tf.matmul(A_tilde_inv_B, C_inv)
    del A_tilde_inv_B

    C_inv_B_T = tf.matmul(C_inv, B, transpose_b = True)
    C_inv_B_T_A_tilde_inv = tf.matmul(C_inv_B_T, A_tilde_inv)
    del C_inv_B_T
    C_inv_B_T_A_tilde_inv_B = tf.matmul(C_inv_B_T_A_tilde_inv, B)
    del C_inv_B_T_A_tilde_inv
    C_inv_B_T_A_tilde_inv_B_C_inv = tf.matmul(C_inv_B_T_A_tilde_inv_B, C_inv)
    del C_inv_B_T_A_tilde_inv_B


    np.save(save_file + name + '_inv_' +'_' + str(rows[0]) + '_' + str(cols[0]) + '.npy', A_tilde_inv.numpy())
    np.save(save_file + name + '_inv_' + '_' + str(rows[0]) + '_' + str(cols[1]) + '.npy', minus_A_tilde_inv_B_C_inv.numpy())
    np.save(save_file + name + '_inv_' + '_' + str(rows[1]) + '_' + str(cols[0]) + '.npy',
            tf.transpose(minus_A_tilde_inv_B_C_inv).numpy())
    np.save(save_file + name + '_inv_' + '_' + str(rows[1]) + '_' + str(cols[1]) + '.npy',
            (C_inv + C_inv_B_T_A_tilde_inv_B_C_inv).numpy())

    #print(A_tilde)

blocks_inverse_1('matrixs/',  'training_set', [0,1], [0,1])