load('shiyutezheng.mat')
D=[shiyutezheng(:,1),shiyutezheng(:,2)];%这里的数据集D是特征矩阵中的其中两维，这里以时域特征矩阵的1、2列为例
% 步骤1：计算距离分布矩阵
dist_matrix = pdist2(D, D); % 计算数据集 D 中每个对象之间的距禷

% 步骤2：对距离矩阵的每一行元素进行升序排序
sorted_dist_matrix = sort(dist_matrix, 2);

% 步骤3：计算 K-平均最近邻距离向量
max_k = size(sorted_dist_matrix, 2);
eps_candidates = zeros(1, max_k);

for k = 1:max_k
    eps_candidates(k) = mean(sorted_dist_matrix(:, k));
end

% 假设已知数据集D和数据对象数量n以及Eps参数列表 DEps
% 假设已经计算得到的Eps参数列表 DEps，请确保DEps已经被赋值

% 初始化变量
n=max_k;
MinPtsList = zeros(1, n); % 初始化MinPts参数列表向量

for i = 1:n
    p = D(i, :); % 当前对象p的坐标
    neighborhoodCount = zeros(n, 1); % 初始化存储每个对象的Eps邻域对象数量的向量
    
    for j = 1:n
        q = D(j, :); % 对象q的坐标
        dist_pq = norm(p - q); % 计算对象p与对象q之间的欧氏距离
        
        if dist_pq <= eps_candidates(i)
            neighborhoodCount(j) = neighborhoodCount(j) + 1; % 如果距离小于等于Eps，则对象q在对象p的Eps邻域内
        end
    end
    
    MinPtsList(i) =   sum(neighborhoodCount); % 计算MinPts参数列表
end



f_score=zeros(1,max_k);%计算聚类综合正确率
for i = 1:max_k
    cluster=dbscan(D,eps_candidates(i),MinPtsList(i));
    clust(i)=max(cluster);
    true_positive = sum(cluster(1:20)==1)+sum(cluster(21:40)==2)+sum(cluster(41:60)==3)+sum(cluster(61:80)==4)+sum(cluster(81:100)==5)+sum(cluster(101:120)==6); % 判断正确的数据个数
    false_negative = sum(cluster == -1); % 未能正确识别的数据个数(噪声)
    false_positive =120-true_positive-false_negative; % 判断错误的数据个数
end



