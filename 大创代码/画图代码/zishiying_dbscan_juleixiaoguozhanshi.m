
%先画出两维特征数据集的散点图作为对照
x= shiyutezheng(:,1);
y =shiyutezheng(:,2);

colors = {'r', 'g', 'b','y','m','c'};
markers = {'o', 'o', 'o','o','o','o'};

%创建画布
figure(1);
set(gcf,'color','white');
%分别绘制不同区间的点
scatter(x(1:20), y(1:20), [], colors{1}, markers{1},'filled');
hold on;
scatter(x(21:40), y(21:40), [], colors{2}, markers{2},'filled');
scatter(x(41:60), y(41:60), [], colors{3}, markers{3},'filled');
scatter(x(61:80), y(61:80), [], [0.8 0.6 0], markers{4},'filled');
% scatter(x(61:80), y(61:80), [], colors{4}, markers{4},'filled');
hold on
scatter(x(81:100), y(81:100), [], colors{5}, markers{5},'filled');
scatter(x(101:120), y(101:120), [], colors{6}, markers{6},'filled');
%设置图例
legend('iq 1', 'iq 2', 'iq 3','iq 4','iq 5','iq 6');
xlabel('分形维数')
ylabel('LZC复杂度')
title('特征散点图')
grid on
%显示图形
hold off;


%在确定最佳聚类标签结果cluster之后，画出它的聚类效果图
colors = {'r', 'g', 'b','y','m','c','k'};
figure(1)
hold on
for i = 1:120
    if cluster(i)==1
        h1=scatter(D(i,1),D(i,2),[],colors{1},'o',"filled");
    end
    if cluster(i)==2
        h2=scatter(D(i,1),D(i,2),[],colors{2},'o',"filled");
    end
    if cluster(i)==3
        h3=scatter(D(i,1),D(i,2),[],colors{3},'o',"filled");
    end
    if cluster(i)==4
        h4=scatter(D(i,1),D(i,2),[],[0.8 0.6 0],'o',"filled");
    end
    if cluster(i)==5
        h5=scatter(D(i,1),D(i,2),[],colors{5},'o',"filled");
    end
    if cluster(i)==6
        h6=scatter(D(i,1),D(i,2),[],colors{6},'o',"filled");
    end
%     if cluster(i)==7
%         h7=scatter(D(i,1),D(i,2),[],[0.5,0,0],'o',"filled");
%     end
%     if cluster(i)==8
%         h8=scatter(D(i,1),D(i,2),[],[0.5 0.5 0.5],'o','filled');
%     end
    if cluster(i)==-1
        h9=scatter(D(i,1),D(i,2),[],colors{7},'o',"filled");
    end
end
hold off


% 绘制散点图并保存句柄  

  
% 添加图例  
legend([h1, h2, h3,h4,h5,h6,h9], {'cluster1 #1', 'cluster2 #2','cluster3 #3','cluster4 #4','cluster5 #5','cluster6 #6','noise'}); 
xlabel('分形维数')
ylabel('LZC复杂度')
grid on
title('聚类可视化')
