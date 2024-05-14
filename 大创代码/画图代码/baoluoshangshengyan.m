
figure(1); % 创建新的图形窗口
% 将图形窗口背景色设置为白色
set(gcf,'color','white');
hold on;
% 定义 legend 标记和对应颜色
plot(baoluozhengli{1},'r--')%每个雷达画出一个包络上升沿信号
plot(baoluozhengli{21},'g--')
plot(baoluozhengli{41},'b--')
plot(baoluozhengli{61},'y--')
plot(baoluozhengli{81},'m--')
plot(baoluozhengli{101},'c--')
grid on
legend('iq 1', 'iq 2', 'iq 3','iq 4','iq 5','iq 6');
xlabel('时间采样点');
ylabel('幅度');
title('各个雷达个体的一组包络前沿波形')
figure(2)
% 将图形窗口背景色设置为白色
set(gcf,'color','white');
hold on

    h1=plot(baoluozhengli{1}, 'r--');%每个雷达画出五个包络上升沿
    plot(baoluozhengli{2}, 'r--');
    plot(baoluozhengli{3}, 'r--');
    plot(baoluozhengli{4}, 'r--');
    plot(baoluozhengli{5}, 'r--');
   
    h2=plot(baoluozhengli{21}, 'g--');
    plot(baoluozhengli{22}, 'g--');
    plot(baoluozhengli{23}, 'g--');
    plot(baoluozhengli{24}, 'g--');
    plot(baoluozhengli{25}, 'g--');

    h3=plot(baoluozhengli{41}, 'b--');
    plot(baoluozhengli{42}, 'b--');
    plot(baoluozhengli{43}, 'b--');
    plot(baoluozhengli{44}, 'b--');
    plot(baoluozhengli{45}, 'b--');

    h4=plot(baoluozhengli{61}, 'y--');
    plot(baoluozhengli{62}, 'y--');
    plot(baoluozhengli{63}, 'y--');
    plot(baoluozhengli{64}, 'y--');
    plot(baoluozhengli{65}, 'y--');

    h5=plot(baoluozhengli{81}, 'm--');
    plot(baoluozhengli{82}, 'm--');
    plot(baoluozhengli{83}, 'm--');
    plot(baoluozhengli{84}, 'm--');
    plot(baoluozhengli{85}, 'm--');

    h6=plot(baoluozhengli{101}, 'c--');
    plot(baoluozhengli{102}, 'c--');
    plot(baoluozhengli{103}, 'c--');
    plot(baoluozhengli{104}, 'c--');
    plot(baoluozhengli{105}, 'c--');
legend([h1, h2, h3,h4,h5,h6], {'iq1', 'iq2','iq3','iq4','iq5','iq6'}); 
hold off;
xlabel('时间采样点');
ylabel('幅度');
title('各个雷达个体的多组包络前沿波形')
grid on