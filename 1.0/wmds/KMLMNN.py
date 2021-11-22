import numpy as np
import copy


class JointLearning:
    def __init__(self, data, weight, cluster_label, obj_label, cluster_centers, nea_num, lam):
        # 以下属性不需要更新
        self.cluster_num = weight.shape[0]  # 簇数量
        self.obj_label = obj_label  # 样本标签, 维度：obj_num
        self.nea_num = nea_num  # k，取前k个最近的同类样本作为target neighbor
        self.obj_num, self.seg_num, self.pes_l, self.pes_w = data.shape  # 样本数量，分段数量，PES矩阵的行数，列数
        self.data = data  # 所有样本的PES矩阵, 维度：obj_num，seg_num，pes_l，pes_w
        self.lam = lam  # 系数
        # 以下属性需要在每次迭代中更新
        self.dtw_dist_mat = None  # 样本之间的DTW距离矩阵，维度：obj_num，obj_num
        self.path = None  # 样本之间的DTW路径，维度：obj_num，obj_num，其中每一元素是一组DTW路径
        self.target_set = None  # 每一样本的target set，维度：obj_num，nea_num
        self.imposter_set = None  # 每一样本的imposter set，每一样本只取最近的一个imposter即可，因此维度为：obj_num
        self.weight = weight  # 权重, 维度：cluster_num，cluster_num，pes_l，pes_w
        self.cluster_label = cluster_label  # 簇标签，维度：obj_num，seg_num
        self.cluster_center = cluster_centers  # 簇中心, 维度：cluster_num，pes_l，pes_w


    def get_ap_dtw(self):
        """
        更新每一对样本之间的DTW距离和路径，更新target set和imposter set
        :return: 更新后的DTW距离，路径，target set，imposter set
        """
        # 确定每一对样本的DTW距离和路径
        dtw_dist_mat = np.zeros((self.obj_num, self.obj_num))  # 存储所有样本间的DTW距离
        path_list = []  # 存储所有样本间的DTW路径
        for i in range(self.obj_num):
            tmp_path_list = []
            for j in range(self.obj_num):
                # if j < i:
                #     dist = dtw_dist_mat[j, i]
                #     path = (path_list[j][i][1], path_list[j][i][0])
                # else:
                dist, path = self.apdtw(i, j)  # 计算第i个样本和第j个样本之间DTW距离和路径
                dtw_dist_mat[i, j] = dist
                tmp_path_list.append(path)
            path_list.append(tmp_path_list)

        return dtw_dist_mat, path_list

    def update_target_imposter(self):
        """
        保持DTW路径不变，更新DTW距离矩阵，更新target set，imposter set
        :return: 更新后的target set，imposter set
        """
        # 更新样本间的DTW距离（但保持DTW路径不变）
        dtw_dist_mat = np.zeros((self.obj_num, self.obj_num))  # 存储所有样本间的DTW距离
        for i in range(self.obj_num):
            for j in range(self.obj_num):
                path_1 = self.path[i][j]  # 取样本i与样本j的DTW路径
                # 计算i与j的DTW距离
                sub = 0
                for k in range(path_1[0].shape[0]):  # 遍历路径，用更新后的簇标签或更新后权重计算DTW距离
                    re_index = (path_1[0][k], path_1[1][k])
                    dist = np.power(self.data[i][re_index[0]] - self.data[j][re_index[1]], 2)
                    # 计算DTW路径中第k个位置上两个PES矩阵所属的簇
                    c_1 = self.cluster_label[i][re_index[0]]
                    c_2 = self.cluster_label[j][re_index[1]]
                    weight_dist = np.multiply(dist, self.weight[c_1][c_2])  # 计算加权距离
                    sub += np.sum(weight_dist)
                dtw_dist_mat[i, j] = sub
        self.dtw_dist_mat = dtw_dist_mat

        sorted_dist_mat = np.argsort(self.dtw_dist_mat)  # 将DTW距离矩阵按行排序，排序后,得到由样本序号️组成的矩阵
        target_set = np.empty((self.obj_num, self.nea_num), dtype=int)  # 所有样本的target set
        imposter_set = np.empty(self.obj_num, dtype=int)  # 所有样本的imposter set
        for i in range(self.obj_num):  # 对于样本i
            cnt = 0  # 选出前k个最近的同类样本
            for j in range(sorted_dist_mat[i].shape[0]):
                if sorted_dist_mat[i][j] == i:  # 跳过样本i本身
                    continue
                if self.obj_label[sorted_dist_mat[i][j]] == self.obj_label[i]:  # 如果第j个最近的样本与i同类
                    target_set[i][cnt] = sorted_dist_mat[i][j]
                    cnt += 1
                if cnt > self.nea_num - 1:  # 找到前k个，停止
                    break
            for j in range(sorted_dist_mat[i].shape[0]):  # 找到最近的一个不同类样本，停止
                if self.obj_label[sorted_dist_mat[i][j]] != self.obj_label[i]:
                    imposter_set[i] = sorted_dist_mat[i][j]
                    break
        return target_set, imposter_set

    def apdtw(self, x_index, y_index):
        """
        计算两个样本间的DTW距离和路径
        :param x_index: 样本序号
        :param y_index: 样本序号
        :return: DTW距离，DTW路径
        """
        x = self.data[x_index]  # 取序号为x_index的样本
        y = self.data[y_index]  # 取序号为y_index的样本
        r, c = len(x), len(y)  # x，y的长度
        # 初始化dp矩阵
        # dp矩阵中以序号1，2，3，...表示第1，2，3，...个样本
        dp = np.zeros((r + 1, c + 1))
        dp[0, 1:] = np.inf
        dp[1:, 0] = np.inf
        # 计算dp矩阵
        for i in range(1, r + 1):
            for j in range(1, c + 1):
                clu_1 = self.cluster_label[x_index][i - 1]  # 样本x的第i个PES矩阵的簇标签
                clu_2 = self.cluster_label[y_index][j - 1]  # 样本y的第j个PES矩阵的簇标签
                weight = self.weight[clu_1][clu_2]  # 对应于簇clu_1和簇clu_2的权重矩阵
                dp[i, j] = dist_fun(x[i - 1], y[j - 1], weight)  # 令dp[i,j]等于i和j的点对距离
                min_list = [dp[i - 1, j - 1]]
                min_list += [dp[i - 1, j], dp[i, j - 1]]
                dp[i, j] += min(min_list)  # 选择（i-1，j-1），（i，j-1）和（i-1，j）中最小的一者相加
        # if len(x) == 1:
        #     path = np.zeros(len(y)), range(len(y))
        # elif len(y) == 1:
        #     path = range(len(x)), np.zeros(len(x))
        # else:
        path = _traceback(dp)  # 计算x，y的DTW路径
        return dp[-1, -1], path

    def update_weight(self):
        """
        更新权重
        :return: 更新后的权重
        """
        # 初始化权重，theta，D，维度如下
        weight = np.zeros((self.cluster_num, self.cluster_num, self.pes_l, self.pes_w))
        theta = np.zeros((self.cluster_num, self.cluster_num, self.pes_l, self.pes_w))
        d = np.zeros((self.cluster_num, self.pes_l, self.pes_w))
        # 计算theta_pq
        for i in range(self.cluster_num):
            for j in range(self.cluster_num):
                theta[i][j] = self.get_theta_pq(i, j)
        # 计算D_k
        for i in range(self.cluster_num):
            d[i] = self.get_d_k(i)
        # 计算权重
        for i in range(self.cluster_num):
            for j in range(self.cluster_num):
                # 计算w_K：
                if i == j:
                    numerator = np.exp(np.divide(-theta[i][j] - d[i], self.lam))
                    denominator = np.sum(numerator)
                    weight[i][j] = np.divide(numerator, denominator)
                # 计算w_pq
                else:
                    numerator = np.exp(np.divide(-theta[i][j], self.lam))
                    denominator = np.sum(numerator)
                    weight[i][j] = np.divide(numerator, denominator)
        return weight

    def get_theta_pq(self, p, q):
        """
        计算theta_pq
        :param p: 簇中心序号
        :param q: 簇中心序号
        :return: 簇p与簇q对应的theta
        """
        # 初始化theta_pq
        theta_pq = np.zeros((self.pes_l, self.pes_w))
        # 对于第i个样本
        for i in range(self.obj_num):
            # 找到样本i的imposter
            lm = self.imposter_set[i]
            tmp1 = np.zeros((self.pes_l, self.pes_w))
            path_2 = self.path[i][lm]  # 取样本i与样本lm的DTW路径
            sub_2 = np.zeros((self.pes_l, self.pes_w))  # 计算公式（20）内中括号的后项
            for k in range(path_2[0].shape[0]):
                re_index = (path_2[0][k], path_2[1][k])
                if self.cluster_label[i][re_index[0]] != p or self.cluster_label[lm][re_index[1]] != q:
                    continue
                # 若分别属于簇p和簇q
                sub_2 += np.power(self.data[i][re_index[0]] - self.data[lm][re_index[1]], 2)
            # 对于样本i的每一个target neighbor
            for j in range(self.target_set[i].shape[0]):
                z_index = self.target_set[i][j]  # 取第j个target neighbor的序号
                if 1 - self.dtw_dist_mat[i][lm] + self.dtw_dist_mat[i][z_index] <= 0:  # 检查公式（20）的第二个等式是否小于0
                    continue
                path_1 = self.path[i][z_index]  # 取样本i与第j个target neighbor的DTW路径
                sub_1 = np.zeros((self.pes_l, self.pes_w))  # 计算公式（20）内中括号的前项
                for k in range(path_1[1].shape[0]):
                    re_index = (path_1[0][k], path_1[1][k])  # path_1[0][k]， path_1[1][k]表示路径中第k个元组
                    if self.cluster_label[i][re_index[0]] != p or self.cluster_label[z_index][re_index[1]] != q:
                        continue
                    # 若分别属于簇p和簇q
                    sub_1 += np.power(self.data[i][re_index[0]] - self.data[z_index][re_index[1]], 2)

                # 计算公式（20）的中括号
                tmp1 += (sub_1 - sub_2)
            theta_pq += tmp1
        return theta_pq

    def get_d_k(self, k):
        """
        计算D_k
        :param k: 簇序号
        :return: D_k
        """
        d_k = np.zeros((self.pes_l, self.pes_w))
        # 对所有PES矩阵遍历
        for i in range(self.obj_num):
            for j in range(self.seg_num):
                if self.cluster_label[i][j] == k:  # 若x_ij属于簇k
                    d_k += np.power(self.cluster_center[k] - self.data[i][j], 2)
        return d_k

    def update_cluster_label(self):
        """
        更新簇标签
        :return: 更新后的簇标签
        """
        cluster_label = np.zeros((self.obj_num, self.seg_num), dtype=int)
        weight_k = np.zeros((self.cluster_num, self.pes_l, self.pes_w))  # 存储所有的w_k
        # 从self.weight中得到所有的w_k
        for i in range(self.cluster_num):
            weight_k[i] = self.weight[i][i]
        # 以下与EKWM.find_closest_cluster_center()类似
        for i in range(self.obj_num):
            for j in range(self.seg_num):
                distance = np.power((self.cluster_center - self.data[i][j]), 2)
                weight_distance = np.multiply(distance, weight_k)
                weight_distance_sum = np.sum(weight_distance, axis=2)
                weight_distance_sum = np.sum(weight_distance_sum, axis=1)
                # if math.isinf(weight_distance_sum.sum()) or math.isnan(weight_distance_sum.sum()):
                #     weight_distance_sum = np.zeros(self.cluster_number)
                cluster_label[i][j] = np.where(weight_distance_sum == weight_distance_sum.min())[0][0]
        return cluster_label

    # def update_cluster_label(self):
    #     """
    #     更新簇标签
    #     :return: 更新后的簇标签
    #     """
    #     cluster_label = np.zeros((self.obj_num, self.seg_num), dtype=int)
    #     weight_k = np.zeros((self.cluster_num, self.pes_l, self.pes_w))  # 存储所有的w_k
    #     # 从self.weight中得到所有的w_k
    #     for i in range(self.cluster_num):
    #         weight_k[i] = self.weight[i][i]
    #     for i in range(self.obj_num):
    #         for j in range(self.seg_num):
    #             # 对于x_ij
    #             dist_list = np.empty(self.cluster_num)
    #             old_label = self.cluster_label[i][j]
    #             for k in range(self.cluster_num):
    #                 # 计算x_ij与簇k的加权距离
    #                 distance = np.power((self.cluster_center[k] - self.data[i][j]), 2)
    #                 weight_distance = np.multiply(distance, weight_k[k])
    #                 weight_distance_sum = np.sum(weight_distance)
    #                 # 计算在x_ij属于簇k的情况下，公式（15）LMNN项的值
    #                 self.cluster_label[i][j] = k
    #                 self.update_target_imposter()
    #                 lmnn = self.get_LMNN()
    #                 dist_list[k] = weight_distance_sum + lmnn  # 将x_ij属于簇k时对损失函数的贡献添加到dist_list中
    #             # self.cluster_label[i][j] = old_label  # 为保证清晰，此处不对类属性cluster_label进行修改，而在外部统一更新
    #             self.cluster_label[i][j] = np.argmin(dist_list)  # 选出dist_list中最小项的序号，作为x_ij的簇标签
    #             self.update_target_imposter()

    def update_cluster_center(self):
        """
        更新簇中心
        :return: 更新后的簇中心
        """
        cluster_centers = np.zeros((self.cluster_num, self.pes_l, self.pes_w), dtype=float)
        for i in range(self.cluster_num):  # 对于每一个簇
            cnt = 0  # 计算属于簇i的样本数量
            # 遍历所有点PES矩阵
            for j in range(self.obj_num):
                for k in range(self.seg_num):
                    if self.cluster_label[j][k] == i:  # 若该PES矩阵属于簇i
                        cluster_centers[i] += self.data[j][k]
                        cnt += 1
            # 除以属于簇i的样本数量
            cluster_centers[i] = np.divide(cluster_centers[i], cnt)
        return cluster_centers

    def cost_function(self):
        """
        计算损失函数
        :return: 损失函数值
        """
        cost = 0
        # 计算簇内距离和
        for i in range(self.cluster_num):  # 对于簇i
            # 遍历属于PES矩阵
            for j in range(self.obj_num):
                for k in range(self.seg_num):
                    if self.cluster_label[j][k] == i:
                        dist = np.power(self.cluster_center[i] - self.data[j][k], 2)
                        weight_dist = np.multiply(self.weight[i][i], dist)  # weight[i][i]即为w_i，计算加权距离
                        cost += np.sum(weight_dist)  # np.sum对所有元素求和
        # 计算所有权重矩阵的熵，并乘以系数lambda
        for i in range(self.cluster_num):
            for j in range(self.cluster_num):
                # 先对权重矩阵weight[i][j]的每一元素取对数，将对应位置元素相乘，再进行加总，最后乘以lambda
                cost += self.lam * np.sum(np.multiply(self.weight[i][j], np.log(self.weight[i][j])))
        # 加PFLMNN项的值
        cost = cost + self.get_LMNN()
        return cost

    def get_LMNN(self):
        """
        计算PFLMNN项，与get_theta_pq()类似
        :return: PFLMNN项的值
        """
        result = 0
        for i in range(self.obj_num):  # 对于每一样本
            lm = self.imposter_set[i]
            sub_2 = 0  # 计算公式（15）内，中括号第二项
            path_2 = self.path[i][lm]  # 取样本i与样本lm的DTW路径
            for k in range(path_2[0].shape[0]):  # 遍历路径，用更新后的簇标签和权重计算DTW距离
                re_index = (path_2[0][k], path_2[1][k])
                dist = np.power(self.data[i][re_index[0]] - self.data[lm][re_index[1]], 2)
                # 计算DTW路径中第k个位置上两个PES矩阵所属的簇
                c_1 = self.cluster_label[i][re_index[0]]
                c_2 = self.cluster_label[lm][re_index[1]]
                weight_dist = np.multiply(dist, self.weight[c_1][c_2])  # 计算加权距离
                sub_2 += np.sum(weight_dist)
            for j in range(self.target_set[i].shape[0]):  # 计算公式（15）内，中括号第三项
                # 以下与上一个for循环类似
                z_index = self.target_set[i][j]
                path_1 = self.path[i][z_index]
                sub_1 = 0
                for k in range(path_1[1].shape[0]):
                    re_index = (path_1[0][k], path_1[1][k])
                    dist = np.power(self.data[i][re_index[0]] - self.data[z_index][re_index[1]], 2)
                    c_1 = self.cluster_label[i][re_index[0]]
                    c_2 = self.cluster_label[z_index][re_index[1]]
                    weight_dist = np.multiply(dist, self.weight[c_1][c_2])
                    sub_1 += np.sum(weight_dist)
                # 判断公式（15）的中括号是否大于0
                if 1 - sub_2 + sub_1 > 0:
                    result += 1 - sub_2 + sub_1
        return result


def dist_fun(x, y, weight):
    """
    计算点对距离
    :param x: PES矩阵
    :param y: PES矩阵
    :param weight: 权重矩阵
    :return:
    """
    dist = np.power(x - y, 2)  # 矩阵x减矩阵y，每个元素平方
    weight_dist = np.multiply(weight, dist)  # 与权重矩阵对应位置相乘
    weight_sum_dist = np.sum(weight_dist)  # 将相乘后的所有元素加总
    return weight_sum_dist


def _traceback(D):
    """
    计算DTW路径
    :param D: dp矩阵，以序号1，2，3，...表示第1，2，3，...个样本
    :return: DTW路径
    """
    i, j = np.array(D.shape) - 2
    p, q = [i], [j]
    while (i > 0) or (j > 0):
        tb = np.argmin((D[i, j], D[i, j + 1], D[i + 1, j]))
        if tb == 0:
            i -= 1
            j -= 1
        elif tb == 1:
            i -= 1
        else:  # (tb == 2):
            j -= 1
        p.insert(0, i)
        q.insert(0, j)
    return np.array(p), np.array(q)
