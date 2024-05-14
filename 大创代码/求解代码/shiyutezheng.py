import numpy as np
import math
import signal
from numpy.fft import fft, ifft
from numpy import array, sign, zeros
from scipy.interpolate import interp1d

# 输入fs为信号采样频率，up_fre为宽带上的起始频率，down_fre为宽带上的截止频率，N为采样点数
fs = 200000
up_fre = -3000
down_fre = 3000
N = 200000


# 内部函数，求频谱斜截率
def general_equation(first_x, first_y, second_x, second_y):
	# 斜截式 y = kx + b
	A = second_y - first_y
	B = first_x - second_x
	C = second_x * first_y - first_x * second_y
	k = -1 * A / B
	b = -1 * C / B
	return k, b


# 输入中频序列，提取包络谱特征，可单独作为特征也可以复用到其他特征提取方法里面，包络特征优于直接计算幅值或者希尔伯特变换
# 返回数值为上包络与下包络
def envelope_extraction(signal):
	s = signal.astype(float)
	q_u = np.zeros(s.shape)
	q_l = np.zeros(s.shape)
	# 在插值值前加上第一个值。这将强制模型对上包络和下包络模型使用相同的起点。
	u_x = [0, ]  # 上包络的x序列
	u_y = [s[0], ]  # 上包络的y序列
	l_x = [0, ]  # 下包络的x序列
	l_y = [s[0], ]  # 下包络的y序列

	# 检测波峰和波谷，并分别标记它们在u_x,u_y,l_x,l_中的位置。
	# Detect peaks and troughs and mark their location in u_x,u_y,l_x,l_y respectively.
	for k in range(1, len(s) - 1):
		if (sign(s[k] - s[k - 1]) == 1) and (sign(s[k] - s[k + 1]) == 1):
			u_x.append(k)
			u_y.append(s[k])
		if (sign(s[k] - s[k - 1]) == -1) and ((sign(s[k] - s[k + 1])) == -1):
			l_x.append(k)
			l_y.append(s[k])
	u_x.append(len(s) - 1)  # 上包络与原始数据切点x
	u_y.append(s[-1])  # 对应的值
	l_x.append(len(s) - 1)  # 下包络与原始数据切点x
	l_y.append(s[-1])  # 对应的值

	# u_x,l_y是不连续的，以下代码把包络转为和输入数据相同大小的数组[便于后续处理，如滤波]
	upper_envelope_y = np.zeros(len(signal))
	lower_envelope_y = np.zeros(len(signal))
	upper_envelope_y[0] = u_y[0]  # 边界值处理
	upper_envelope_y[-1] = u_y[-1]
	lower_envelope_y[0] = l_y[0]  # 边界值处理
	lower_envelope_y[-1] = l_y[-1]

	# 上包络
	last_idx, next_idx = 0, 0
	k, b = general_equation(u_x[0], u_y[0], u_x[1], u_y[1])  # 初始的k,b
	for e in range(1, len(upper_envelope_y) - 1):
		if e not in u_x:
			v = k * e + b
			upper_envelope_y[e] = v
		else:
			idx = u_x.index(e)
			upper_envelope_y[e] = u_y[idx]
			last_idx = u_x.index(e)
			next_idx = u_x.index(e) + 1
			# 求连续两个点之间的直线方程
			k, b = general_equation(u_x[last_idx], u_y[last_idx], u_x[next_idx], u_y[next_idx])
	# 下包络
	last_idx, next_idx = 0, 0
	k, b = general_equation(l_x[0], l_y[0], l_x[1], l_y[1])  # 初始的k,b
	for e in range(1, len(lower_envelope_y) - 1):
		if e not in l_x:
			v = k * e + b
			lower_envelope_y[e] = v
		else:
			idx = l_x.index(e)
			lower_envelope_y[e] = l_y[idx]
			last_idx = l_x.index(e)
			next_idx = l_x.index(e) + 1
			# 求连续两个切点之间的直线方程
			k, b = general_equation(l_x[last_idx], l_y[last_idx], l_x[next_idx], l_y[next_idx])

	# 也可以使用三次样条进行拟合，针对不同数据进行尝试
	# u_p = interp1d(u_x,u_y, kind = 'cubic',bounds_error = False, fill_value=0.0)
	# l_p = interp1d(l_x,l_y, kind = 'cubic',bounds_error = False, fill_value=0.0)
	# for k in range(0,len(s)):
	#   q_u[k] = u_p(k)
	#   q_l[k] = l_p(k)
	return upper_envelope_y, lower_envelope_y


#
def complex_data(data):  # 将复杂数据从实部和虚部分开，并将其合并为一个复数形式的数据。
	data_I = data[::2]
	data_Q = data[1::2]
	com_data = data_I + 1j * data_Q
	return com_data


def getArr(start, step, end, type='arr'):  # 生成数字序列，返回数组或列表
	result = []
	length = math.ceil((end - start) / step)
	for i in range(length + 1):
		num = start + i * step
		if (num > end):
			break
		result.append(num)
	# pdb.set_trace()
	if type == 'arr':
		result = np.array(result)
		return result
	elif type == 'list':
		return result
	else:
		return []


def shiftFreq(x, Freq, fs):  # 频谱偏移
	ts = (getArr(0, 1, len(x) - 1)) / fs
	Y = x * np.exp(1j * 2 * math.pi * Freq * ts)
	return Y


def bandpass_complex(x, fs, freqmin, freqmax):  # 带通滤波
	freqmin1 = -fs / 2 + fs / 2
	freqmax1 = freqmin + fs / 2 - 1
	freqmin2 = freqmax + fs / 2
	freqmax2 = fs / 2 + fs / 2 - 1
	x = shiftFreq(x, fs / 2, fs)
	n = len(x)
	ni1 = math.floor(2 * freqmin1 * (n / 2) / fs)
	na1 = math.floor(2 * freqmax1 * (n / 2) / fs)
	ni2 = math.floor(2 * freqmin2 * (n / 2) / fs)
	na2 = math.floor(2 * freqmax1 * (n / 2) / fs)
	y = fft(x, n)
	y[ni1:na1] = 0
	y[ni2:na2] = 0
	y = ifft(y, n)
	y = shiftFreq(y, -fs / 2, fs)
	return y


# 频谱图特征，在模拟CW数据集上效果较好，一线数据上效果差
def spectrogram(data):  # 对输入信号进行谱图分析，并返回对应的频谱图。
	com_data = complex_data(data)
	y = bandpass_complex(com_data, fs, up_fre, down_fre)
	# fft_window = signal.windows.kaiser(segment_for_fft,beta = 10)
	# f,t,spet = signal.spectrogram(com_data, 20000, window = fft_window, nperseg = segment_for_fft, nfft = segment_for_fft*2, return_onesided=False)
	f, t, spet = signal.spectrogram(y, fs)
	return spet


# 参差功率谱，在功率谱上加上对数参差值，一线通信数据上效果非常好
def pow_spectrum(data):  # 计算输入信号的功率谱，并返回以对数形式表示的功率谱。
	com_data = complex_data(data)
	y = bandpass_complex(com_data, fs, up_fre, down_fre)
	power = abs(y)
	log_power = 20 * np.log10(power)
	return log_power


# 二阶矩
def two_order(data):  # 计算输入信号的二阶中心矩
	com_data = complex_data(data)
	y = bandpass_complex(com_data, fs, up_fre, down_fre)
	abs_x = abs(y)
	N = len(abs_x)
	x1 = np.sum(abs_x) / N
	S2 = np.sum((abs_x - x1) ** 2) / N
	return S2


# 三阶矩
def there_order(data):  # 计算输入信号的三阶中心矩
	com_data = complex_data(data)
	y = bandpass_complex(com_data, fs, up_fre, down_fre)
	abs_x = abs(y)
	N = len(abs_x)
	x1 = np.sum(abs_x) / N
	P2 = np.std(abs_x, ddof=1)
	S3 = np.sum(((abs_x - x1) / P2) ** 3) / N
	return S3


# 四阶矩
def four_order(data):  # 计算输入信号的四阶中心矩
	com_data = complex_data(data)
	y = bandpass_complex(com_data, fs, up_fre, down_fre)
	abs_x = abs(y)
	N = len(abs_x)
	x1 = np.sum(abs_x) / N
	S4 = (np.sum((abs_x - x1) ** 4) / N) / ((np.sum((abs_x - x1) ** 2) / N) ** 2)
	return S4


# 计算信号的lzc特征
# 计算信号的lzc特征
def lempel_ziv_complexity(sequence):
    dictionary = {}
    w = ""
    complexity = 0

    for c in sequence:
        wc = w + c
        if wc in dictionary:
            w = wc
        else:
            dictionary[wc] = len(dictionary) + 1
            w = c
            complexity += 1

    return complexity
def LZC(data, type):#计算复杂度
    '''
    :param data: np.array的数据
    :param type:'upper'计算上包络，'lower'计算下包络，可选项内容
    :return:返回数组功率谱的LZC值
    '''
    data_freq = fft(data)
    data_freq = (data_freq * np.conj(data_freq)).real/max(data_freq.shape)
    upp_envelope, lower_envelope = envelope_extraction(data_freq) # 获取上包络、下包络
    if(type == 'upper'):
        x = upp_envelope
    elif(type=='lower'):
        x = lower_envelope
    else:
        print('type选择错误')
        return -10000
    upp_mean = np.mean(x)
    sa = x - upp_mean
    sc = []
    for i in range(len(sa)-1):
        sc.append(sa[i+1] - sa[i])
    sc_mean = np.mean(sc)
    sq = []
    for i in range(len(sc)):
        if sc[i] < sc_mean:
            sq.append('0')
        else:
            sq.append('1')
    data = ''.join(sq)
    result = lempel_ziv_complexity(data)
    return result


# 计算信号的信息维度特征
def informationDimension(data):
	'''
	:param data:
	:return: 返回信息维数
	'''
	data_freq = fft(data)
	data = (data_freq * np.conj(data_freq)).real
	length_data = max(data.shape)
	so = []
	for i in range(length_data - 1):
		so.append(data[i + 1] + data[i])
	S = np.sum(so)
	P = so / S
	D = P * np.log10(P)
	result = np.sum(D)
	return -result


# 计算信号的模糊熵特征
def Fuzzy_Entropy(x, m, r=0.25, n=2):
	"""
	模糊熵
	m 滑动时窗的长度
	r 阈值系数 取值范围一般为：0.1~0.25
	n 计算模糊隶属度时的维度，参照默认值
	"""
	# 将x转化为数组
	x = np.array(x)
	# 检查x是否为一维数据
	if x.ndim != 1:
		raise ValueError("x的维度不是一维")
	# 计算x的行数是否小于m+1
	if len(x) < m + 1:
		raise ValueError("len(x)小于m+1")
	# 将x以m为窗口进行划分
	entropy = 0  # 近似熵
	for temp in range(2):
		X = []
		for i in range(len(x) - m + 1 - temp):
			X.append(x[i:i + m + temp])
		X = np.array(X)
		# 计算X任意一行数据与其他行数据对应索引数据的差值绝对值的最大值
		D_value = []  # 存储差值
		for index1, i in enumerate(X):
			sub = []
			for index2, j in enumerate(X):
				if index1 != index2:
					sub.append(max(np.abs(i - j)))
			D_value.append(sub)
		# 计算模糊隶属度
		D = np.exp(-np.power(D_value, n) / r)
		# 计算所有隶属度的平均值
		Lm = np.average(D.ravel())
		entropy = abs(entropy) - Lm
	return entropy


# 计算信号的分形维度特征
def fractal_dim(data, cellmax):
	'''
	:param data: 输入信号
	:param cellmax: 分形份数，设置数值必须大于等于信号长度
	:return:
	'''
	L = max(data.shape)
	if cellmax < L:
		raise ValueError('cellmax必须大于等于信号长度')
	y_min = min(data)
	y_shift = data - y_min
	x_ord = np.array(getArr(0, 1, L - 1)) / (L - 1)
	xx_ord = np.array(getArr(0, 1, cellmax)) / cellmax
	yy_ord = interp1d(x_ord, y_shift, kind='linear')  # 三次样条插值
	# yy_ord = interp1d(x_ord, y_shift, kind='cubic')  # 三次样条插值
	y_interp = yy_ord(xx_ord)

	ys_max = max(y_interp)
	factory = cellmax / ys_max
	yy = abs(y_interp * factory)

	t = math.floor(math.log(cellmax) / math.log(2) + 1)
	NumSeg = []
	N = []
	for e in range(t):
		Ne = 0
		cellsize = 2 ** (e - 1 + 1)
		NumSeg.append(math.floor(cellmax / cellsize))
		for j in range(NumSeg[e]):
			begin = cellsize * (j - 1) + 1
			tail = cellsize * j + 1
			seg = getArr(begin, 1, tail)
			yy_max = max(yy[seg])
			yy_min = min(yy[seg])
			up = math.ceil(yy_max / cellsize)
			down = math.floor(yy_min / cellsize)
			Ns = up - down
			Ne = Ne + Ns
		N.append(Ne)
	Nr = []
	for i in range(len(N)):
		Nr.append(math.log(N[i]) / math.log(2))
	r = -np.diff(np.array(Nr))
	Ne = []
	E = []
	for i in range(len(r)):
		if r[i] <= 2 and r[i] >= 1:
			Ne.append(N[i])
			E.append(NumSeg[i])
	for i in range(len(Ne)):
		Ne[i] = math.log(Ne[i]) / math.log(2)
	for i in range(len(E)):
		E[i] = math.log(E[i]) / math.log(2)
	P = np.polyfit(np.array(E), np.array(Ne), 1)
	D = P[0]
	return D


# 计算信号的盒维度特征
def box_dimension(x):
	def getN(data, r):
		'''
		:param data: 一维数据
		:param r: 半径
		:return: 填满数据的所有最小闭球个数
		'''
		data = np.array(data)
		return math.floor(max(data.shape) / (2 * r + 1))

	data_freq = fft(x)
	data_freq = (data_freq * np.conj(data_freq)).real  # /max(data_freq.shape)
	upp_envelope, lower_envelope = envelope_extraction(data_freq)  # 获取上包络、下包络
	da_N = max(upp_envelope.shape)
	N = 16
	d = 1 / N
	sum_max = 0;
	sum_min = 0
	for i in range(da_N - 1):
		sum_max = sum_max + max(upp_envelope[i], upp_envelope[i + 1])
		sum_min = sum_min + min(upp_envelope[i], upp_envelope[i + 1])
	Nd = N + (sum_max * d - sum_min * d) / (d ** 2)
	a = math.log(d) / math.log(math.e)
	Db = (math.log(Nd) / math.log(math.e)) / (math.log(d) / math.log(math.e))
	return Db















#此处之前都是函数（用于直接计算时域特征或者辅助计算时域特征）的定义 之后开始计算各种时域特征，相应的修改所使用下面所使用的函数即可（此处以计算模糊熵为例）
import scipy.io
from tqdm import tqdm
import time
# 从MAT文件中加载数据
mat_data = scipy.io.loadmat('D:/新数据3.2/buweiling.mat') # 此处使用的是将包络信号补零后使其长度相同之后的mat文件

# 获取数组
data_array = mat_data['buweiling']
# 替换 'array' 为你MAT文件中数组的变量名
cellmax=2500
m=3
mohushang=np.zeros((120,))
# 遍历每个数组进行操作
for i in tqdm(range(120)):

	new_signal = data_array[i, np.nonzero(data_array[i])]
	# new_signal = new_signal.T
	new_signal = new_signal.flatten()
	mohushang[i]=Fuzzy_Entropy(new_signal, m, r=0.25, n=2)
	time.sleep(0.1)  # 模拟任务执行时间



