"----------这个函数调用类Fractal()计算一个3D-Hilbert谱图的差分盒维数盒多重分形维数-------------"
import os
from scipy.sparse import load_npz  # 假设矩阵以npz格式存储
import concurrent.futures
import numpy as np
from scipy.sparse import csc_matrix
from sparse_processing import process_sparse_matrix
from DBC import Fractal
import os
import re
obj = Fractal()

import os
import scipy.io as sio
from sparse_processing import process_sparse_matrix

# 定义文件夹路径
import os

# 定义文件夹路径
folder_path = ""#放每个信号3D-Hilbert谱的稀疏矩阵

# 获取.mat文件列表
file_paths = [os.path.join(folder_path, f) for f in os.listdir(folder_path) if f.endswith('.mat')]


# 定义一个函数来提取文件名中最后一个_后面的数字并转换为整数
def extract_sort_number(file_path):
    filename = os.path.basename(file_path)  # 获取文件名（不含路径）
    match = re.search(r'_(\d+)\.txt.mat$', filename)  # 查找最后一个_和.mat之间的数字
    return int(match.group(1)) if match else float('inf')  # 如果没有找到，则返回无穷大以便放在列表末尾


# 使用sorted函数和自定义的key函数对文件路径进行排序
sorted_file_paths = sorted(file_paths, key=extract_sort_number)

# 打印排序后的文件路径
for file_path in sorted_file_paths:
    print(file_path)


# 按照文件名中最后一个_后面的数字排序
sorted_file_paths = sorted(file_paths, key=extract_sort_number)


# 初始化结果列表
fd_list = []
md_list = []
R_1_list = []
R_2_list = []
w1_list = []
w2_list = []
# 使用for循环遍历文件
for file_path in sorted_file_paths:
    # 加载.mat文件
    try:
        data = sio.loadmat(file_path)
        # 假设sparse_matrix是.mat文件中的变量名，根据实际情况调整
        sparse_matrix = data['hs']
    except Exception as e:
        print(f"Error loading {file_path}: {e}")
        continue  # 跳过当前文件并继续下一个

    # 处理稀疏矩阵
    try:
        # max_value = process_sparse_matrix(sparse_matrix)
        # 假设obj.execute是处理processed_matrix的函数，返回fd和md
        fd, md, r1, r2, w1, w2 = obj.execute(sparse_matrix)
        fd_list.append(fd[0])
        md_list.append(md[0])
        R_1_list.append(r1[0])
        R_2_list.append(r2[0])
        w1_list.append(w1[0])
        w2_list.append(w2[0])
    except Exception as e:
        print(f"Error processing {file_path}: {e}")
        continue  # 跳过当前文件并继续下一个

import csv
# # 1. 创建文件对象（指定文件名，模式，编码方式）a模式 为 下次写入在这次的下一行
with open("D:\\pycharmmm\\six_lei_w1w2\\S_9001_0005.csv", "w", encoding="utf-8", newline="") as f:
    # 2. 基于文件对象构建 csv写入对象
    csv_writer = csv.writer(f)
    # 3. 构建列表头
    name = ['分形盒维数', '差分盒维数']  # w1是差分盒维数 w2是多重分形维数
    csv_writer.writerow(name)
    # 4. 写入csv文件内容
    w1_list = np.array(w1_list).reshape(20, 1)
    w2_list = np.array(w2_list).reshape(20, 1)
    w_list_together_s = np.concatenate((w1_list, w2_list), axis=1)
    csv_writer.writerows(w_list_together_s)  # writerowswriterows
    print("写入数据成功")
    # 5. 关闭文件
    f.close()

# 现在fd_list和md_list包含了所有处理后的结果
