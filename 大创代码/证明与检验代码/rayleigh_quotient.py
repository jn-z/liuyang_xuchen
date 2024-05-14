"-----------这个文件能够在给定列之间的相关系数的条件下分析Cxx或者Cyy的特征值的大小 采用瑞利商的方法进行估计"
import numpy as np

# 假设矩阵的大小为n x n
n = 120  # 你可以更改这个值以匹配你的矩阵大小

# 创建一个实对称矩阵，对角线上元素为n，其他元素为0
matrix = np.diag(np.full(n, n, dtype=np.float64))   # 显式地指定dtype为float64

# 添加其他元素，这些元素是从-n/10到n/10的随机数
matrix += np.random.uniform(0, n / 20, size=(n, n))  # 因为matrix已经是float64类型，这里不会有问题

# 确保矩阵是对称的（如果之前的操作没有破坏对称性）
matrix = (matrix + matrix.T) / 2

# 接下来你可以使用rayleigh quotient或者其他方法估计特征值范围
# ...

# # 选择一个向量x来计算瑞利熵，这里我们使用全1向量作为示例
# x = np.ones(n)
#
# # 计算瑞利熵R(A, x)
# rayleigh_quotient = (x.dot(matrix.dot(x))) / (x.dot(x))

# 选择试探向量的数量
num_vectors = 1000  # 可以根据需要增加或减少

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

# 注意：这只是一个粗略的估计，不代表真实的最小特征值。
# 要获得更准确的特征值，应该使用数值方法，如NumPy的eigvals函数。

# 使用NumPy的eigvals函数计算真实的特征值
eigenvalues = np.linalg.eigvals(matrix)
min_eigenvalue = np.min(eigenvalues)
print(f"真实的最小特征值为: {min_eigenvalue}")
