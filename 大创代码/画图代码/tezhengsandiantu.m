load('LZC.mat')
%横坐标为序列，纵坐标为时域特征
x= 1:120;
y =LZC;%以画LZC复杂度为例，画其它特征对应修改即可

%设置颜色和形状
colors = {'r', 'g', 'b','y','m','c'};
markers = {'o', 'o', 'o','o','o','o'};

%创建画布
figure;
set(gcf,'color','white');
%分别绘制不同区间的点
scatter(x(1:20), y(1:20), [], colors{1}, markers{1},'filled');
hold on;
scatter(x(21:40), y(21:40), [], colors{2}, markers{2},'filled');
scatter(x(41:60), y(41:60), [], colors{3}, markers{3},'filled');
scatter(x(61:80), y(61:80), [], [0.8 0.6 0], markers{4},'filled');
scatter(x(81:100), y(81:100), [], colors{5}, markers{5},'filled');
scatter(x(101:120), y(101:120), [], colors{6}, markers{6},'filled');
%设置图例
legend('iq 1', 'iq 2', 'iq 3','iq 4','iq 5','iq 6');
xlabel('雷达序列')
ylabel('LZC复杂度')
title('所有雷达的LZC复杂度')
grid on
%显示图形
hold off;