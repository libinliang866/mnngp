import argparse
from data_processing import get_data
from mnngp import MNNGPKernel
from gpr_integrate import gpr
import configuration as config
import tensorflow as tf

if __name__ == '__main__':
    parser = argparse.ArgumentParser(description='Optional app description')

    # add argument
    parser.add_argument('--dataset', type=str, default = 'mnist')
    parser.add_argument('--num_of_training', type=int, default = 1000)
    parser.add_argument('--num_of_testing', type=int, default = 1000)
    parser.add_argument('--maxout_rank', type=int, default = 2)
    parser.add_argument('--depth', type=int, default = 10)
    parser.add_argument('--sigma_w_sq', type=float, default = 3)
    parser.add_argument('--sigma_b_sq', type=float, default = 0.1)

    ###
    args = parser.parse_args()

    if args.dataset not in ['mnist', 'cifar10']:
        args.dataset = 'mnist'
    if args.num_of_training <= 0:
        args.num_of_training = 1000
    if args.num_of_testing <= 0:
        args.num_of_testing = 1000
    if args.maxout_rank not in [2, 3, 4]:
        args.maxout_rank = 2
    if args.depth <= 0:
        args.depth = 10
    if args.sigma_w_sq <= 0:
        args.sigma_w_sq = 3
    if args.sigma_b_sq <= 0:
        args.sigma_b_sq = 0.1

    cov_grid_paths = ['cov_grid_2_501_1001_10.npy', 'cov_grid_3_501_1001_10.npy', 'cov_grid_4_501_1001_10.npy']
    cor_grid_paths = ['cor_grid_2_501_1001_10.npy', 'cor_grid_3_501_1001_10.npy', 'cor_grid_4_501_1001_10.npy']

    save_grid_base = 'tables/'
    save_cor_grid_base = 'tables/'
    config.save_grid_path = save_grid_base + cov_grid_paths[args.maxout_rank - 2]
    config.save_cor_grid_path = save_cor_grid_base + cor_grid_paths[args.maxout_rank - 2]

    dat_train, y_train = get_data(split='train', extract_number=args.num_of_training, one_hot=True, data_set = args.dataset, shuffle=False)
    dat_test, y_test = get_data(split='test', extract_number=args.num_of_testing, one_hot=True, data_set = args.dataset)

    dat_train = tf.cast(dat_train, tf.float64)
    dat_test = tf.cast(dat_test, tf.float64)
    y_train = tf.cast(y_train, tf.float64)
    y_test = tf.cast(y_test, tf.float64)

    mnn_kernel = MNNGPKernel(depth=args.depth, q=args.maxout_rank, weight_var=args.sigma_w_sq, bias_var=args.sigma_b_sq, n_corr=501)
    mnn_kernel.get_grid()
    mnn_gpr = gpr(mnngp=mnn_kernel, x_train=dat_train, x_test=dat_test, y_train=y_train, stability_eps=1e-10)

    fmean = mnn_gpr._predict_v1()

    pred_y = tf.math.argmax(fmean, axis=1)
    true_lab = tf.math.argmax(y_test, axis=1)

    acc = (sum(pred_y.numpy() == true_lab.numpy()) / true_lab.numpy().shape[0])

    print('The accuracy in testing data is:')
    print(acc)