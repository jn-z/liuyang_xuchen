"-----------这个文件能够在给定列之间的相关系数的条件下分析Cxx或者Cyy的特征值的大小 采用瑞利商的方法进行估计"
import numpy as np
# from scipy.spatial.distance import cosine
# def compute_column_cosine_similarity(matrix):
#     # 获取矩阵的列数
#     num_columns = matrix.shape[1]
#
#     # 初始化一个用于存储相似度的矩阵
#     similarity_matrix = np.zeros((num_columns, num_columns))
#
#     # 遍历所有列对
#     for i in range(num_columns):
#         for j in range(i , num_columns):  # 只需要计算上半部分，因为是对称的
#             # 提取第i列和第j列作为向量
#             vec_i = matrix[:, i]
#             vec_j = matrix[:, j]
#             a = vec_i.reshape(-1, 1)
#             # 计算余弦相似度
#             similarity = 1- cosine(vec_i, vec_j)
#
#             # 存储相似度值
#             similarity_matrix[i, j] = similarity
#             similarity_matrix[j, i] = similarity  # 因为是对称的，所以也设置下半部分的值
#
#     return similarity_matrix

import scipy.io as sio
# filepath1 = "D:\\原华为云盘-微信\\WeChat Files\\wxid_6iy8wv10garl22\\FileStorage\\File\\2024-03\\tezhengjuzhengenghuan.mat"
# matrix = sio.loadmat(filepath1)
# matrix = matrix['tezhengjuzhen']
#
# similarity_matrix = compute_column_cosine_similarity(matrix)
# print(similarity_matrix)

import numpy as np

# 假设矩阵的大小为n x n
n = 500  # 你可以更改这个值以匹配你的矩阵大小

# 创建一个实对称矩阵，对角线上元素为n，其他元素为0
matrix = n * np.eye(n)  # 显式地指定dtype为float64

# 添加其他元素，这些元素是从-n/10到n/10的随机数
matrix += np.random.uniform(n / 5 - 0.05*n/3 , n / 5 + 0.05*n/3  , size=(n, n))  # 因为matrix已经是float64类型，这里不会有问题

# 确保矩阵是对称的（如果之前的操作没有破坏对称性）
matrix = (matrix + matrix.T) / 2
for i in range(0, n):
    for j in range(0, n):
        if (i == j) :
            matrix[i][j] = n
# 接下使用rayleigh quotient或者其他方法估计特征值范围


# 选择一个向量x来计算瑞利熵，这里我们使用全1向量作为示例
x = np.ones(n)

# 计算瑞利熵R(A, x)
rayleigh_quotient = (x.dot(matrix.dot(x))) / (x.dot(x))

# 选择试探向量的数量
num_vectors = 1000 # 可以根据需要增加或减少

# 初始化一个列表来保存所有瑞利熵的值
rayleigh_quotients = []

# 生成随机试探向量并计算瑞利熵
for _ in range(num_vectors):
    # 生成一个随机向量，其元素服从标准正态分布
    x = np.random.randn(n)

    # 计算瑞利熵
    rayleigh_quotient = (x.dot(matrix.dot(x))) / (x.dot(x))

    # 将瑞利熵添加到列表中
    rayleigh_quotients.append(rayleigh_quotient)

# 找到并打印瑞利熵的最小值，作为最小特征值的粗略估计
min_rayleigh_quotient = min(rayleigh_quotients)
print(f"使用瑞利熵估计的最小特征值约为: {min_rayleigh_quotient}")


max_rayleigh_quotient = max(rayleigh_quotients)
print(f"使用瑞利熵估计的最大特征值约为: {max_rayleigh_quotient}")
# 注意：这只是一个粗略的估计，不代表真实的最小特征值。
# 要获得更准确的特征值，应该使用数值方法，如NumPy的eigvals函数。

# 使用NumPy的eigvals函数计算真实的特征值
eigenvalues = np.linalg.eigvals(matrix)
min_eigenvalue = np.min(eigenvalues)
max_eigenvalue = np.max(eigenvalues)
print(f"真实的最小特征值为: {min_eigenvalue}")
print(f"真实的最大特征值为: {max_eigenvalue}")
#
# # 使用NumPy的eig函数计算真实的特征值




