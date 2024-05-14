"----------------------------------基于相关系数选取特征的证明分析----------------三个变量的情况"
# 定义目标函数，计算|x-y| + |x-z| + |y-z|
def objective(xyz):
    x, y, z = xyz
    return (np.abs(x - y)**2 + np.abs(x - z)**2 + np.abs(y - z)**2)


# 定义约束条件，确保x^2 + y^2 + z^2 = 1
def constraint(xyz):
    x, y, z = xyz
    return x ** 2 + y ** 2 + z ** 2 - 1


# 初始化r的值
r_values = np.linspace(-1,1,100) * np.sqrt(3)
max_value = -np.inf
max_r = None

# 遍历r的每一个值，找到最大值
for r in r_values:
    # 定义约束条件，确保x + y + z = r
    cons = ({'type': 'eq', 'fun': lambda xyz: xyz[0] + xyz[1] + xyz[2] - r},
            {'type': 'eq', 'fun': constraint})

    # 初始猜测值
    x0 = [r / 3, r / 3, r / 3]

    # 使用SciPy的约束优化函数求解
    res = minimize(objective, x0, method='SLSQP', constraints=cons)

    # 如果优化成功，则更新最大值和对应的r值
    if res.success:
        current_value = objective(res.x)
        if current_value > max_value:
            max_value = current_value
            max_r = r / np.sqrt(3)
            max_xyz = res.x
print(f"The  value of |x-y| + |x-z| + |y-z| is: {max_value}")
print(f"The corresponding value of r is: {max_r}")
print(f"The corresponding value of xyz is: {max_xyz}")

def find_max_min_distance(r_range):
    max_value = 0
    max_xyz = None

    # 假设我们使用随机搜索，但你可以替换为网格搜索或其他方法
    for _ in range(10):  # 你可以增加迭代次数以获得更好的近似
        xyz = np.random.uniform(-1, 1, 3)  # 在单位立方体内随机选择点
        xyz /= np.linalg.norm(xyz)  # 归一化到单位球面上

        # 确保x+y+z接近给定的r值
        while abs(xyz[0] + xyz[1] + xyz[2] - r_range) > 1e-3:
            xyz += np.random.normal(0, 0.01, 3)  # 在当前点附近进行小的随机扰动
            xyz /= np.linalg.norm(xyz)  # 重新归一化

        # 计算min{|x-y|, |y-z|, |z-x|}
        distances = np.array([abs(a - b) for a, b in zip(xyz, np.roll(xyz, 1))])
        distances = np.append(distances, abs(xyz[0] - xyz[2]))  # 添加|z-x|
        min_distance = np.min(distances)

        # 更新最大值和对应的xyz（如果需要）
        if min_distance > max_value:
            max_value = min_distance
            max_xyz = xyz

    return max_value, max_xyz


# 给定的r的范围
r_range = np.linspace(-np.sqrt(3), np.sqrt(3), 100)
max_values = []

# 对每个r值进行搜索
for r in r_range:
    max_value, max_xyz = find_max_min_distance(r)
    max_values.append(max_value)

# 找到全局最大值及其对应的r值
global_max_value = np.max(max_values)
global_max_r = r_range[np.argmax(max_values)]
print(f"The approximate maximum value of min{'| x-y |, | y-z |, | z-x |'} is: {global_max_value}")
print(f"The corresponding value of r is: {global_max_r}")


"---------------------六个变量的情况-------------------"
import numpy as np
from scipy.optimize import minimize
#
#
# # 目标函数，计算min{|x-y|, |y-z|, |z-x|}的负值
# def objective(xyz, sign=1):
#     x, y, z = xyz
#     distances = np.abs(np.array([x - y, y - z, z - x]))
#     return sign * np.min(distances)
#
#
# # 约束条件，确保x1^2 + x2^2 + x3^2 = 1
# def constraint(xyz):
#     x, y, z = xyz
#     return x ** 2 + y ** 2 + z ** 2 - 1
#
#
# # 约束条件字典
# con = {'type': 'eq', 'fun': constraint}
#
# # 初始化随机种子以便结果可复现
# np.random.seed(0)
#
# # 初始化搜索的最大迭代次数
# max_iter = 10000
#
# # 初始化最大min{|x-y|, |y-z|, |z-x|}的值和对应的xyz
# max_value = -np.inf
# best_xyz = None
#
# # 进行随机搜索
# for _ in range(max_iter):
#     # 在单位球面上随机采样一个点
#     xyz = np.random.randn(3)
#     xyz /= np.linalg.norm(xyz)
#
#     # 使用SciPy的minimize函数来找到局部最小值（实际上是目标函数的负值的最大值）
#     # 注意：我们传递-1作为sign参数，因为我们正在寻找目标函数的负值的最大值
#     res = minimize(objective, xyz, method='SLSQP', constraints=con, args=(-1,))
#
#     if res.success:
#         current_value = -res.fun  # 将负值转回正值
#         if current_value > max_value:
#             max_value = current_value
#             best_xyz = res.x
#
# print(f"The approximate maximum value of min{'| x-y |, | y-z |, | z-x |'} is: {max_value}")
# print(f"The corresponding point (x, y, z) is: {best_xyz}")
import numpy as np

# 初始化最大m值和对应的点
max_m = 0
max_m_point = None

# 设定搜索次数
num_searches = 1000000
count = 0
# 进行随机搜索

while count < num_searches:
    # 在单位球面上随机生成一个点
    point = np.random.randn(5)

    six_th = -sum(point)
    point_added = np.insert(point, point.size, six_th)
    point_added /= np.linalg.norm(point_added)  # 归一化，确保在单位球面上
    if 1:
        x1, x2, x3, x4, x5, x6 = point_added
        if x1 > x2 > x3 > x4 > x5 > x6:
            count += 1
            # 计算三个绝对值
            # abs_diffs = ([x1 - x2, x1 - x3, x2 - x3])
            abs_diffs = (
            [x1 - x2, x1 - x3, x1 - x4, x1 - x5, x1 - x6, x2 - x3, x2 - x4, x2 - x5, x2 - x6, x3 - x4, x3 - x5, x3 - x6,
             x4 - x5, x4 - x6, x5 - x6])
            # 找出其中的最小值m
            m = np.min(abs_diffs)

            # 更新最大m值和对应的点
            if m > max_m:
                max_m = m
                max_m_point = point_added

    # 输出结果
print(f"The approximate maximum value of m (min{'| x1-x2 |, | x2-x3 |, | x3-x1 |'}) is: {max_m}")
print(f"The  point (x1, x2, x3, x4, x5, x6) is: {(max_m_point)}")
print(f"The sum of maximum point (x1, x2, x3) is: {sum(max_m_point)}")
print(sum(max_m_point)/np.sqrt(6))
