
"-------以下代码实现CCA的特征融合算法------------"
import csv
import pandas as pd
import numpy as np
import cv2
import matplotlib
from matplotlib import pyplot as plt

matplotlib.use('Qt5Agg')
from scipy import io as sio
"-----------读取数据文件----------时域和时频域已经提取号的特征矩阵"
filepath1 = "D:\\原华为云盘-微信\\WeChat Files\\wxid_6iy8wv10garl22\\FileStorage\\File\\2024-03\\tezhengjuzhengenghuan.mat"
filepath2 =  "D:\\原华为云盘-微信\\WeChat Files\\wxid_6iy8wv10garl22\\FileStorage\\File\\2024-03\\biaozhunhuahou.mat"
baoluo_feature_raw = sio.loadmat(filepath1)
baoluo_feature_raw = baoluo_feature_raw['tezhengjuzhen']


# 假设你的CSV文件名为 'data.csv' 并且它位于当前工作目录中
filename_iq1 = "D:\\pycharmmm\six_lei_w1w2\\iq1_alone.csv"
filename_iq2 = "D:\\pycharmmm\six_lei_w1w2\\iq2_alone.csv"
filename_iq3 = "D:\\pycharmmm\six_lei_w1w2\\iq3_alone.csv"
filename_90001_0002 = "D:\\pycharmmm\\six_lei_w1w2\\S_9001_0002.csv"
filename_90001_0005 = "D:\\pycharmmm\\six_lei_w1w2\\S_9001_0002.csv"
filename_tfsk = "D:\\pycharmmm\\six_lei_w1w2\\TFSK.csv"
# 使用列表推导式一次性读取所有行到data_array中
# mean_arr = np.mean(baoluo_feature_raw, axis=0)
# std_arr = np.std(baoluo_feature_raw, axis=0)
# baoluo_feature = (baoluo_feature_raw - mean_arr) / std_arr
baoluo_feature = baoluo_feature_raw[:,]

w_iq1 = np.array(pd.read_csv(filename_iq1))
w_iq2 = np.array(pd.read_csv(filename_iq2))
w_iq3 = np.array(pd.read_csv(filename_iq3))
w_90001_0002 = np.array(pd.read_csv(filename_90001_0002))
w_90001_0005 = np.array(pd.read_csv(filename_90001_0005))
w_tfsk = np.array(pd.read_csv(filename_tfsk))
w_together_ordered = np.concatenate((w_iq1, w_iq2, w_iq3, w_90001_0002, w_90001_0005, w_tfsk), axis=0)
import csv
# # # 1. 创建文件对象（指定文件名，模式，编码方式）a模式 为 下次写入在这次的下一行
with open("D:\\pycharmmm\\six_lei_w1w2\\single_shipinyu_feature.csv", "w", encoding="utf-8", newline="") as f:
    # 2. 基于文件对象构建 csv写入对象
    csv_writer = csv.writer(f)
    # 3. 构建列表头
    # name = ['', '']  # w1是差分盒维数 w2是多重分形维数
    # csv_writer.writerow(name)
    # 4. 写入csv文件内容

    csv_writer.writerows(w_together_ordered)  # writerowswriterows
    print("写入数据成功")
w_together_ordered = sio.loadmat("D:\\原华为云盘-微信\\WeChat Files\\wxid_6iy8wv10garl22\\FileStorage\\File\\2024-03\\ronghejieguo2.mat")
w_together_ordered = w_together_ordered['ronghejieguo2']
import csv
mean_arr = np.mean(w_together_ordered, axis=0)
std_arr = np.std(w_together_ordered, axis=0)
w_together_ordered = (w_together_ordered - mean_arr) / std_arr





"---------按照CCA公式计算Cxx,Cyy,和典型相关变量Z_x,Z_y"
Cxx = np.dot(baoluo_feature.T, baoluo_feature)
Cyy = np.dot(w_together_ordered.T, w_together_ordered)
Cxy = np.dot(baoluo_feature.T, w_together_ordered)

from scipy.linalg import eigvals
#
v_x, Q_x = np.linalg.eig(Cxx)
u,s,v = np.linalg.svd(Cxx)
V_x = np.diag(v_x ** (-0.5))
Cxx_half_inv = Q_x @ V_x @ np.linalg.inv(Q_x)

v_y, Q_y = np.linalg.eig(Cyy)

V_y = np.diag(v_y ** (-0.5))
Cyy_half_inv = Q_y @ V_y @ np.linalg.inv(Q_y)
#
H = Cxx_half_inv @  Cxy @ Cyy_half_inv
U, S, Vt = np.linalg.svd(H)
wx = np.zeros([2, 4])
wy = np.zeros([2, 2])
col_num = H.shape[0]
V = Vt.T

wx = Cxx_half_inv @ U
wy = Cyy_half_inv @ V
zx = np.zeros((120, 4))
zy = np.zeros((120, 2))

zx = np.dot(baoluo_feature, wx)
zy = np.dot(w_together_ordered, wy)
# Z = np.concatenate((zx, zy), axis=0)
import matplotlib.pyplot as plt
plt.figure()

Z = zx[:,[0,1]] + zy


print(zx.T @ zx)
plt.scatter(Z[0:20, 0], Z[0:20, 1], color='red', marker='s', label='iq1')
plt.scatter(Z[20:40, 0], Z[20:40, 1], color='green', marker='D', label='iq2')
plt.scatter(Z[40:60, 0], Z[40:60, 1], color='blue', marker='*', label='iq3')
plt.scatter(Z[60:80, 0], Z[60:80, 1], color='cyan', marker='o', label='002')
plt.scatter(Z[80:100, 0], Z[80:100, 1], color='magenta', marker='p', label='005')
plt.scatter(Z[100:120, 0], Z[100:120, 1], color='black', marker='h', label='tfsk')
# plt.title('time24column')
plt.legend()
plt.show()
#

with open("D:\\pycharmmm\\six_lei_w1w2\\34融合.csv", "w", encoding="utf-8", newline="") as f:
    # 2. 基于文件对象构建 csv写入对象
    csv_writer = csv.writer(f)
    # 3. 构建列表头z
    # name = ['分形盒维数', '差分盒维数']  # w1是差分盒维数 w2是多重分形维数
    # csv_writer.writerow(name)
    # 4. 写入csv文件内容

    csv_writer.writerows(Z)  # writerowswriterows
    print("写入数据成功")
    # 5. 关闭文件
    f.close()





for i in range(0,6):
    time_freq_std_iq2 = 10*np.std(w_together_ordered[i*20:(i+1)*20][:], axis=0)

    time_std_iq2 =10* np.std(baoluo_feature[i*20:(i+1)*20][:], axis=0)
    zx_std = 10*np.std(zx[i*20:(i+1)*20][:], axis=0)
    zy_std = 10*np.std(zy[i*20:(i+1)*20][:], axis=0)

    # cov0 =  np.cov(w_together_ordered[i*20:(i+1)*20][0], baoluo_feature[i*20:(i+1)*20][0])
    # cov1 = np.cov(w_together_ordered[i * 20:(i + 1) * 20][1], baoluo_feature[i * 20:(i + 1) * 20][1])
    Z_std_iq2 = 10*np.std(Z[i*20:(i+1)*20][:], axis=0)
    print("时域方差：", time_std_iq2)
    print("时频域方差:",time_freq_std_iq2)
    # print('\n')

    # print("\n")
    print("zx方差",zx_std)
    # print('\n')
    #
    print("zy方差",zy_std)
    # print('\n')

    print('融合后方差：',Z_std_iq2)
    print('\n')





"-----------------------CCA典型特征变量典型相关性的检验---------------------------------------"
import sympy as sp

Lambda_1 =  (1 - S[1]*S[1])
Q_1 = -(118-0.5*7)*np.log(Lambda_1)
import scipy.stats as stats

# 设置显著水平为0.05，置信水平为1 - 0.05 = 0.95
confidence_level = 0.95

# 设置自由度
df = 3

# 计算卡方分布的临界值
critical_value = stats.chi2.ppf(confidence_level, df)

