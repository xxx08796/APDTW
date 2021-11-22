import numpy as np
import random
import math


def get_ewkm(data, cluster_num, lam):
    """
    对所有子段进行EWKM聚类
    :param data: 所有样本的PES矩阵，维度：obj_num，seg_num，pes_l，pes_w
    :param cluster_num: 簇中心数量
    :param lam: 系数
    :return: 所有子段的簇标签，维度：obj_num，seg_num；簇中心，维度：cluster_num，pes_l，pes_w；权重，维度：cluster_num，pes_l，pes_w
    """
    obj_num, seg_num, pes_l, pes_w = data.shape  # 样本数量，子段数量，PES矩阵的行数和列数
    # 将数据的维度从obj_num，seg_num，pes_l，pes_w转换成obj_num * seg_num，pes_l，pes_w
    all_obj = np.empty((obj_num * seg_num, pes_l, pes_w))
    for i in range(obj_num):
        for j in range(seg_num):
            all_obj[i * seg_num + j] = data[i][j]
    # EWKM
    best_labels, best_centers, best_weight = ewkmeans(data=all_obj, cluster_number=cluster_num, lam=lam,
                                                      iterations=20)
    return np.reshape(best_labels, (obj_num, seg_num)), best_centers, best_weight


def ewkmeans(data, cluster_number, lam, iterations):
    """
    实现EWKM
    :param data: 所有样本，维度：obj_num * seg_num，pes_l，pes_w
    :param cluster_number: 簇中心数量
    :param lam: 参数
    :param iterations: 迭代次数
    :return: 所有子段的簇标签，维度：obj_num * seg_num；其他同get_ewkm
    """
    n, pes_l, pes_w = data.shape  # 样本数量，PES矩阵的行数和列数
    cost_func = np.zeros(iterations)  # 存储每次迭代的损失函数值
    weight = np.zeros((cluster_number, pes_l, pes_w), dtype=float) + np.divide(1, float(pes_l * pes_w))  # 初始化权重
    cluster_centers = init_cluster_center(data, cluster_number)  # 初始化簇中心
    cluster_index = None
    for i in range(iterations):
        cluster_index = find_closest_cluster_center(data, weight, cluster_centers)  # 更新簇标签
        cluster_centers = compute_cluster_center(data, cluster_index, cluster_number)  # 更新簇中心
        weight = compute_weight(data, cluster_centers, cluster_index, lam)  # 更新权重
        cost_func[i] = cost_function(data, cluster_centers, cluster_index, weight, lam)  # 计算损失函数
    print(cost_func)
    # 得到最终的结果
    best_labels = cluster_index
    best_centers = cluster_centers
    best_weight = weight
    if is_convergence(cost_func):
        return best_labels, best_centers, best_weight
    else:
        return None, None, None


def init_cluster_center(data, cluster_number):
    """
    初始化簇中心
    :param data: 所有样本
    :param cluster_number: 簇中心数量
    :return: 簇中心，维度：cluster_num，pes_l，pes_w
    """
    n = np.size(data, 0)
    rand_index = np.array(random.sample(range(n), cluster_number))  # 随机选择cluster_number个样本序号
    cluster_centers = data[rand_index]  # 根据序号得到对应簇中心
    return cluster_centers


def find_closest_cluster_center(data, weight, cluster_centers):
    """
    更新簇标签
    :param data: 所有样本
    :param weight: 权重，维度：cluster_num，pes_l，pes_w
    :param cluster_centers: 簇中心，维度：cluster_num，pes_l，pes_w
    :return: 所有样本的簇标签
    """
    cluster_number = np.size(cluster_centers, axis=0)  # 簇中心数量
    n = data.shape[0]  # 样本数量
    # 初始化簇标签
    cluster_index = np.empty(n, dtype=int)
    for i in range(n):
        distance = np.power((cluster_centers - data[i]), 2)  # 所有簇中心减第i个样本，并把每个元素平方
        weight_distance = np.multiply(distance, weight)  # 与权重矩阵的对应位置相乘
        # 计算样本i到每个簇中心的距离
        weight_distance_sum = np.sum(weight_distance, axis=2)
        weight_distance_sum = np.sum(weight_distance_sum, axis=1)
        # if math.isinf(weight_distance_sum.sum()) or math.isnan(weight_distance_sum.sum()):
        #     weight_distance_sum = np.zeros(cluster_number)
        # 选择距离最小的簇中心序号作为簇标签
        cluster_index[i] = np.where(weight_distance_sum == weight_distance_sum.min())[0][0]
    return cluster_index


def compute_cluster_center(data, cluster_index, cluster_number):
    """
    更新簇中心
    :param data: 所有样本
    :param cluster_index: 簇标签
    :param cluster_number: 簇数量
    :return:
    """
    n, pes_l, pes_w = data.shape
    cluster_centers = np.zeros((cluster_number, pes_l, pes_w), dtype=float)
    for k in range(cluster_number):
        index = np.where(cluster_index == k)[0]   # 取所有属于第k个簇的样本序号
        temp = data[index]  # 取所有属于第k个簇的样本
        # 将这些样本相加并除以样本数量，得到簇中心
        all_dimension_sum = np.sum(temp, axis=0)
        cluster_centers[k] = all_dimension_sum / np.size(index)
    return cluster_centers


def compute_weight(data, cluster_centers, cluster_index, lam):
    """
    更新权重
    :param data: 所有样本
    :param cluster_centers: 簇数量
    :param cluster_index: 簇标签
    :param lam: 系数
    :return: 权重
    """
    cluster_number = np.size(cluster_centers, 0)
    n, pes_l, pes_w = data.shape
    weight = np.zeros((cluster_number, pes_l, pes_w))
    # 计算D， 维度：cluster_number, pes_l, pes_w
    distance_sum = np.zeros((cluster_number, pes_l, pes_w), dtype=float)
    for k in range(cluster_number):
        index = np.where(cluster_index == k)[0]  # 取所有属于第k个簇的样本序号
        temp = data[index]  # 取所有属于第k个簇的样本
        distance = np.power((temp - cluster_centers[k]), 2)  # 将这些样本分别减去第k个簇，对应元素平方
        # distance = np.sum(distance, axis=2)
        distance_sum[k] = np.sum(distance, axis=0)  # 将结果按第0个维度加总，得到D_k
    for k in range(cluster_number):
        numerator = np.exp(np.divide(-distance_sum[k], lam))  # 将D_k取负号并除以lambda
        denominator = np.sum(numerator)  # 将每一个元素加总，得到分母
        weight[k] = np.divide(numerator, denominator)  # 将每一个元素分别除以分母，结果即为第k个簇更新后的权重矩阵
    return weight


def cost_function(data, cluster_centers, cluster_index, weight, lam):
    """
    计算损失函数
    :param data: 所有样本
    :param cluster_centers: 簇中心
    :param cluster_index: 簇标签
    :param weight: 权重
    :param lam: 系数
    :return: 损失函数值
    """
    cost = 0
    cluster_number = np.size(cluster_centers, 0)
    for k in range(cluster_number):
        index = np.where(cluster_index == k)[0]  # 取所有属于第k个簇的样本序号
        temp = data[index]  # 取所有属于第k个簇的样本
        distance = np.power((temp - cluster_centers[k]), 2)  # 将这些样本分别减去第k个簇，对应元素平方
        # distance = np.sum(distance, axis=2)
        weight_distance = np.multiply(distance, weight[k])  # 与第k个簇的权重矩阵相乘
        temp = lam * np.sum(np.multiply(weight[k], np.log(weight[k])))  # 计算权重的熵
        cost = cost + np.sum(weight_distance) + temp  # np.sum将weight_distance所有元素进行加总
    return cost


def is_convergence(cost_func):
    """
    判断损失函数值是否收敛
    :param cost_func: 损失函数值列表
    :return: 是否收敛
    """
    result = True
    # 若损失函数值不存在或不是递减的，则返回false，否则返回True
    if math.isnan(np.sum(cost_func)):
        result = False
    iteration = np.size(cost_func)
    for i in range(iteration - 1):
        if cost_func[i] < cost_func[i + 1]:
            result = False
        i += 1
    return result

# class EWKmeans:
#     n_cluster = 0
#     max_iter = 0
#     gamma = 0
#     best_labels, best_centers, best_weight = None, None, None
#     is_converge = False
#
#     def __init__(self, n_cluster=3, max_iter=20, gamma=10.0):
#         self.n_cluster = n_cluster
#         self.max_iter = max_iter
#         self.gamma = gamma
#
#     def fit(self, data):
#         self.is_converge, self.best_labels, self.best_centers, self.best_weight = ewkmeans(
#             data=data, cluster_number=self.n_cluster, gamma=self.gamma, iterations=self.max_iter)
#         return self.best_labels, self.best_centers, self.best_weight


# if __name__ == '__main__':
#     DATA = np.random.randn(200, 50, 1, 1)
#     label = get_ewkm(data=DATA, cluster_num=4, lam=2)
#     print(label)

# 仓库名：PER-APDTW Learning
