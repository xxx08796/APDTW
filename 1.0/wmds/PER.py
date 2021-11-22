import ordpy
import pandas as pd
import numpy as np
from collections import Counter
import math


def get_per(data, delay, m, scale, seg_num, sval_num=None):
    """
    :param data: 时间序列，维度：1，ts_len
    :param delay: 延迟默认为1，可暂时忽略
    :param m: 嵌入维度范围，[3, m]
    :param: scale: 时间尺度范围，[1, scale]
    :param seg_num: 分段数量
    :return PER，即PES矩阵序列，维度：seg_len，m-3+1，scale
    """
    data = np.reshape(data, -1)  # 调整为1维数组格式
    ts_len = data.shape[0]  # 时间序列长度
    seg_len = int(ts_len / seg_num)  # 子段长度
    # 初始化数组，用于存储PER
    per = np.empty((seg_num, m - 3 + 1, scale))
    # 对于每一子段， 令嵌入维度和时间尺度分别从[3, m]和[1, scale]变化，计算对应的IMPE
    for k in range(seg_num):
        pes_matrix = []
        seg = data[k * seg_len:(k + 1) * seg_len]  # 取第k个子段
        for i in range(3, m + 1):
            temp = []
            for j in range(1, scale + 1):
                temp.append(get_impe(delay, m=i, scale=j, data=seg))  # 计算子段的IMPE
            pes_matrix.append(temp)
        per[k] = np.array(pes_matrix)

    # per_reduced = np.empty((seg_num, sval_num, sval_num))  # reduced PER of the time series
    # for k in range(seg_num):
    #     per_reduced[k] = pes_mat_reduce(per[k], sval_num)
    return per


def get_impe(delay, m, scale, data):
    """
    :param delay: 延迟
    :param m: 嵌入维度
    :param scale: 时间尺度
    :param data: 时间序列
    :return: 时间序列在m和scale下对应的IMPE
    """
    pe_list = []
    for i in range(scale):
        tmp = disjoint_data(data[i:], scale)  # 对以第i个元素为起点的子序列进行处理，得到由均值组成的序列
        pe_list.append(get_pe(delay, m, data=tmp))  # 计算均值序列的PE
        # pe_list.append(ordpy.permutation_entropy(tmp, dx=m, base='e', normalized=False))
    return sum(pe_list) / len(pe_list)  # 求所有均值序列PE的平均值


def disjoint_data(data, scale):
    """
    对输入的序列，以每scale个相邻元素求均值，并将这些均值组成一条新的序列
    :param data: 序列，一维数组
    :param scale: 时间尺度
    :return: 原始序列对应的均值序列
    """
    l = len(data)
    num = int(l / scale)
    d_data = []
    for i in range(num):
        tmp = data[i * scale:(i + 1) * scale]
        d_data.append(sum(tmp) / len(tmp))
    return d_data


def get_pe(delay, m, data):
    """
    计算均值序列的PE
    :param delay: 延迟
    :param m: 嵌入维度
    :param data: 均值序列
    :return: 序列的PE
    """
    def key(a):
        return a[0]
    # 将data分成长度为m的子段
    X = [[] for i in range(len(data) - (m - 1) * delay)]
    for i in range(len(X)):
        for j in range(m):
            X[i].append([data[i + j * delay], j])
        X[i].sort(key=key)
    # 存储每个子段的顺序模式
    ordinal_patterns = []
    for i in range(len(X)):
        s = ''
        for j in range(m):
            s += str(X[i][j][1])
        ordinal_patterns.append(s)
    # 计算每一种顺序模式出现的次数
    ordinal_patterns = Counter(ordinal_patterns)
    # 计算每一种顺序模式的概率
    P = []
    for key in ordinal_patterns.keys():
        P.append(ordinal_patterns[key] / len(X))
    # 计算PE
    H = 0
    for i in range(len(P)):
        H += P[i] * math.log(P[i])
    H *= -1.0
    # return H / math.log(math.factorial(m))
    return H


def pes_mat_reduce(pes_matrix, singular_val_num):
    """
    PES矩阵降维，暂不使用
    :param pes_matrix: 原始PES矩阵
    :param singular_val_num: 保留最大奇异值的个数
    :return: 降维后矩阵
    """
    u, s, vh = np.linalg.svd(pes_matrix, full_matrices=True)
    v = vh.T
    u1 = u[:, 0:singular_val_num]
    s1 = np.diag(s[0:singular_val_num])
    v1 = v[:, 0:singular_val_num]
    reduced_pes = s1 @ u1.T @ pes_matrix @ v1 @ s1
    # red = s1.dot(u1.T).dot(pes_matrix).dot(v1).dot(s1)
    return reduced_pes

# if __name__ == '__main__':
#     delay = 1
#     m = 7  # m>3
#     scale = 2  # 10>scale>2
#     data = pd.read_excel('Fig5.xlsx', sheet_name=1, header=None)
#     data = np.array(data)
#     PER = get_per(data, delay, m, scale, seg_num=10, singular_val_num=2)

# 仓库名：PER-APDTW Learning
