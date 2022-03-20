
from multiprocessing import Pool
from scipy.stats import multivariate_normal

rv = multivariate_normal([0, 0], [[1, 0.3], [0.3, 1]])
print(rv())