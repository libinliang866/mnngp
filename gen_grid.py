
import numpy as np
from scipy.stats import multivariate_normal, norm
from tqdm import tqdm
import multiprocessing as mp
import functools
from itertools import product


def norm_cdf(x, rv):
    return rv.cdf(x)

def norm_pdf(x, rv):
    return rv.pdf(x)

def square(x):
    return x**2

def mvn_cdf(z, rv):
    return rv.cdf(z)

def mvn_con_cdf(z, rv, rho):
    z_con = (z[0]-rho*z[1], z[1]-rho*z[0])
    return(rv.cdf(z_con))

def mvn_pdf(z, rv):
    return rv.pdf(z)

def multiply(z):
    return z[0]*z[1]

def min_max_pdf(z, rv, q):
    if z[1] > z[0]:
        return q*(q-1)*((rv.cdf(z[1])-rv.cdf(z[0]))**(q-2))*rv.pdf(z[0])*rv.pdf(z[1])
    else:
        return 0

def _compute_cov_grid(n_grid, grid_max, n_corr, q):
    lin_grid = np.linspace(-grid_max, grid_max, n_grid)
    int_grid = [x for x in product(lin_grid, lin_grid)]

    cor_grid = np.linspace(-1, 1, n_corr)

    cov_grid = np.zeros((n_corr))

    pool = mp.Pool()
    for idx, rho in tqdm(enumerate(cor_grid)):
        if rho != 1 and rho != -1:
            rv_rho = multivariate_normal([0, 0], [[1, rho], [rho, 1]])
            rv_con = multivariate_normal([0, 0], [[1-rho**2, 0], [0, 1-rho**2]])
            rv_ind = multivariate_normal([0, 0], [[1, 0], [0, 1]])

            weight_same = np.array(pool.map(functools.partial(mvn_pdf, rv=rv_rho), int_grid))
            weight_dif = np.array(pool.map(functools.partial(mvn_pdf, rv=rv_ind), int_grid))

            cdf_grid = np.array(pool.map(functools.partial(mvn_cdf, rv = rv_rho), int_grid))
            cdf_con_grid = np.array(pool.map(functools.partial(mvn_con_cdf, rv=rv_con, rho = rho), int_grid))

            product_grid = np.array(pool.map(multiply, int_grid))

            s1 = np.sum(product_grid*(cdf_grid**(q-1))*weight_same)
            s2 = np.sum(product_grid*(cdf_grid**(q-2))*cdf_con_grid*weight_dif)

            sw1 = np.sum(weight_same)
            sw2 = np.sum(weight_dif)

            cov_grid[idx] = q*s1/sw1 + q*(q-1)*s2/sw2
        elif rho == 1:
            cov_grid[idx] = _compute_var(int(n_grid*6), grid_max, q)
        elif rho == -1:
            cov_grid[idx] = -_compute_min_max(int(n_grid*6), grid_max, q)

    pool.close()
    return cov_grid, cor_grid

def _compute_var(n_grid, grid_max, q):
    lin_grid = np.linspace(-grid_max, grid_max, n_grid)
    pool = mp.Pool()

    rv = norm()
    cdf_grid = np.array(pool.map(functools.partial(norm_cdf, rv = rv), lin_grid))
    weight = np.array(pool.map(functools.partial(norm_pdf, rv = rv), lin_grid))
    square_x = np.array(pool.map(square, lin_grid))

    return np.sum(q*square_x*(cdf_grid**(q-1))*weight)/np.sum(q*(cdf_grid**(q-1))*weight)


def _compute_min_max(n_grid, grid_max, q):
    lin_grid = np.linspace(-grid_max, grid_max, n_grid)
    int_grid = [x for x in product(lin_grid, lin_grid)]

    pool = mp.Pool()
    rv = norm()
    weight = np.array(pool.map(functools.partial(min_max_pdf, rv=rv, q =q), int_grid))
    product_z = np.array(pool.map(multiply, int_grid))

    return np.sum(product_z*weight)/np.sum(weight)



if __name__ == '__main__':
    cov_grid, cor_grid = _compute_cov_grid(n_grid = 200, grid_max = 10, n_corr = 501, q = 2)
    np.save('cov_grid.npy', cov_grid)
    np.save('cor_grid.npy', cor_grid)

    #cov_grid = np.load('cov_grid.npy')
    #cor_grid = np.load('cor_grid.npy')
    #print(cov_grid)
    #print(cor_grid)
    #print(_compute_var(200, 10, 4))
