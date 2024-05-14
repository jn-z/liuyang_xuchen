load('shiyutezheng.mat')
features=shiyutezheng;  
% 计算各时域特征之间的相关系数矩阵，时频域对应修改  
R = corr(features);  


