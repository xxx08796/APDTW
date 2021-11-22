from numpy import array, zeros, full, argmin, inf, ndim
from scipy.spatial.distance import cdist
from math import isinf
import numpy as np
import pandas as pd


def get_apdtw(x, y, c_x, c_y, weight):
    """
    Computes Dynamic Time Warping (DTW) of two sequences.
    :param array x: N1*M array
    :param array y: N2*M array
    :param func dist: distance used as cost measure
    Returns the minimum distance, the cost matrix, the accumulated cost matrix, and the wrap path.
    """
    r, c = len(x), len(y)
    dp = zeros((r + 1, c + 1))
    dp[0, 1:] = inf
    dp[1:, 0] = inf
    for i in range(1, r + 1):
        for j in range(1, c + 1):
            dp[i, j] = dist_fun(x[i - 1], y[j - 1], c_x[i - 1], c_y[j - 1], weight)
            min_list = [dp[i - 1, j - 1]]
            min_list += [dp[i - 1, j], dp[i, j - 1]]
            dp[i, j] += min(min_list)
    if len(x) == 1:
        path = zeros(len(y)), range(len(y))
    elif len(y) == 1:
        path = range(len(x)), zeros(len(x))
    else:
        path = _traceback(dp)
    return dp[-1, -1], path


def _traceback(D):
    i, j = array(D.shape) - 2
    p, q = [i], [j]
    while (i > 0) or (j > 0):
        tb = argmin((D[i, j], D[i, j + 1], D[i + 1, j]))
        if tb == 0:
            i -= 1
            j -= 1
        elif tb == 1:
            i -= 1
        else:  # (tb == 2):
            j -= 1
        p.insert(0, i)
        q.insert(0, j)
    return array(p), array(q)


def dist_fun(x, y, c_x, c_y, weight):
    dist = np.power(x - y, 2)
    weight_dist = np.multiply(weight[c_x, c_y], dist)
    weight_sum_dist = np.sum(weight_dist)
    return weight_sum_dist


if __name__ == '__main__':
    x = np.random.randn(5, 3, 4)
    y = np.random.randn(5, 3, 4)
    weight = np.random.randn(3, 4) + np.ones((3, 4))
    dist1, path1 = dtw1(x, y, weight=weight)
    print(dist1)
    print(path1)
