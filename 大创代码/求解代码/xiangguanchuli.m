load('C_640KHz_250KHz_INT16_iq_2_201_1.mat')  
signal1 = wav_data_r; % 第一个信号数据
load('C_640KHz_250KHz_INT16_iq_2_201_2.mat') 
signal2 =wav_data_r; % 第二个信号数据  
  
% 两个信号长度如果不同，可以通过截断或填充零来使其相同  
len = min(length(signal1), length(signal2));  
signal1 = signal1(1:len);  
signal2 = signal2(1:len);  
  
% 计算互相关函数  
[corr, lags] = xcorr(signal1, signal2, 'coeff');  
  
% 找到互相关函数的峰值位置  
[~, peak_lag_idx] = max(abs(corr));  
  
% 计算时延（以采样点为单位）  
delay_samples = lags(peak_lag_idx);  
