"""
--------------------------------------- description--------------------------------------------------
@author:Mengcheng Fang
@time:2021.4.20
@Use: Just run the funaction Permutation_Entropy(delay,m,data) directly.
@Parameter Description:
    delay:Time delay parameter, indicating the separation distance between data nodes.
    m:The dimension of permutation entropy, the recommended value range is 3 to 7.
    data:The input data, the requirement is that the data format is 1xn.
"""
import numpy as np
import random
from collections import Counter
import math
from matplotlib.pylab import plt
import pandas as pd


def Permutation_Entropy(delay, m, data):  # Permutation entropy function
    def key(a):
        return a[0]

    X = [[] for i in range(len(data) - (m - 1) * delay)]
    for i in range(len(X)):  # Map data to m-dimensional space
        for j in range(m):
            X[i].append([data[i + j * delay], j])
        X[i].sort(key=key)
    ordinal_patterns = []
    for i in range(len(X)):
        s = ''
        for j in range(m):
            s += str(X[i][j][1])
        ordinal_patterns.append(s)

    ordinal_patterns = Counter(ordinal_patterns)
    P = []
    for key in ordinal_patterns.keys():
        P.append(ordinal_patterns[key] / len(X))  # Calculate the probability distribution of the sequence model
    H = 0
    for i in range(len(P)):
        H += P[i] * math.log(P[i])
    H *= -1.0
    # return H / math.log(math.factorial(m))
    return H


if __name__ == '__main__':
    delay = 1
    m = 5
    # data = [4, 7, 9, 10, 19, 38, 3, 2, 1, 100, 6, 11, 3]
    data = pd.read_excel("Fig5.xlsx")
    print('The entropy of the current data arrangement is:', Permutation_Entropy(1, 3, data))
