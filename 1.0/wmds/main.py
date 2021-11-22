import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import PER
import EWKM
import KMLMNN

# def MaxMinNormalization(x):
#     x = (x - np.min(x)) / (np.max(x) - np.min(x))
#     return x
seg_num = 4  # 分段数量
delay = 1  # 排列熵延迟
scale = 4  # 时间尺度的最大值
m = 4  # 嵌入维度的最大值
# sval_num = 3  # 保留的奇异值数目，暂时不使用SVD降维
cluster_num = 4  # 簇中心数量
lam = 1  # 目标函数中熵的系数
iteration = 5  # 迭代次数
nea_num = 5  # K近邻数量
path = '/Users/yuanhanyang/OneDrive/experiment/数据集说明/UCRArchive_2018/DodgerLoopDay/DodgerLoopDay_TRAIN.tsv'
df = pd.read_csv(path, header=None, sep='\t')
df = df.dropna()  # 去除缺失行
df = df.values  # 转为数组
data = df[:, 1:]  # 取第一列到最后一列为数据， 第0列为标签
obj_label = df[:, 0]
pes_l = m - 3 + 1  # pes矩阵列行，范围：[3,m]
pes_w = scale  # pes矩阵列数，范围：[1:scale]
obj_num, ts_len = data.shape  # 样本数量，时间序列长度
seg_len = int(ts_len / seg_num)  # 每个子段长度
# 计算所有样本的PER
all_obj_per = np.empty((obj_num, seg_num, pes_l, pes_w))  # 初始化维度是obj_num, seg_num, pes_l, pes_w的空数组
for i in range(obj_num):  # 对每一个样本进行分段，并计算PER
    all_obj_per[i] = PER.get_per(data[i, :], delay, m, scale, seg_num)
# all_obj_per = MaxMinNormalization(all_obj_per)  # 暂不做归一化处理
# 初始化权重（包括w_k及w_pq），聚类中心c_k的初始化在EKWM中完成
# 权重数组维度：cluster_num, cluster_num, pes_l, pes_w
weight = np.zeros((cluster_num, cluster_num, pes_l, pes_w)) + np.divide(1, float(pes_l * pes_w))
# 对所有样本的pes矩阵进行EWKM聚类， 得到所有PES矩阵的簇标签，簇中心，以及w_k，
cluster_label, cluster_centers, weight_k = EWKM.get_ewkm(all_obj_per, cluster_num, lam)
for i in range(cluster_num):  # 将更新后的w_k存入权重矩阵中
    weight[i, i] = weight_k[i]
# 初始化全部完成，进入迭代过程
opt = KMLMNN.JointLearning(all_obj_per, weight, cluster_label, obj_label, cluster_centers, nea_num, lam)
cost = []  # 存储损失函数大小
for i in range(iteration):
    print(i)
    # 更新每一对样本之间的DTW距离、路径，更新每一个样本的target set，imposter set
    opt.dtw_dist_mat, opt.path = opt.get_ap_dtw()
    opt.target_set, opt.imposter_set = opt.update_target_imposter()
    # 更新w_k和w_pq
    opt.weight = opt.update_weight()
    opt.target_set, opt.imposter_set = opt.update_target_imposter()
    # 更新所有PES矩阵的簇标签
    opt.update_cluster_label()
    opt.target_set, opt.imposter_set = opt.update_target_imposter()
    # 更新簇中心
    opt.cluster_center = opt.update_cluster_center()
    # 计算损失函数
    cost.append(opt.cost_function())

# plt.plot(cost)
# plt.show()
print(cost)
