load('C_640KHz_250KHz_INT16_iq_2_201_1.mat')%导入一个mat文件
signal=wav_data_r;%原始信号
figure(1)
set(gcf,'color','white');
plot(1:length(signal),signal)
xlabel('时间采样点')
ylabel('幅度')
title('原始信号波形')


lvbo_signal=wdenoise(signal,5,'DenoisingMethod','BlockJS');%小波降噪
figure(2)
set(gcf,'color','white');
plot(1:length(lvbo_signal),lvbo_signal)
xlabel('时间采样点')
ylabel('幅度')
title('滤波后的信号波形')


hilbertTransform_signal = hilbert(signal);%原始信号希尔伯特求包络
envelope_signal = abs(hilbertTransform_signal);
figure(3)
set(gcf,'color','white');
plot(1:length(envelope_signal),envelope_signal)
xlabel('时间采样点')
ylabel('幅度')
title('原始包络波形')


hilbertTransform_lvbo_signal = hilbert(lvbo_signal);%滤波后希尔伯特求包络
envelope_lvbo_signal = abs(hilbertTransform_lvbo_signal);
figure(4)
set(gcf,'color','white');
plot(1:length(envelope_lvbo_signal),envelope_lvbo_signal)
xlabel('时间采样点')
ylabel('幅度')
title('滤波后的包络波形')


figure(5)
set(gcf,'color','white');
imagesc(lvbo_signal);
colorbar;  % 添加颜色条
xlabel('时间采样点');
ylabel('脉冲数量');
title('原始伪彩图');

figure(6)
set(gcf,'color','white');
imagesc(lvbo_signal(1000:20000));%这里的1000是根据相关处理所求出的时延来选择
colorbar;  % 添加颜色条
xlabel('时间采样点');
ylabel('脉冲数量');
title('对齐后的伪彩图');

figure(7)
set(gcf,'color','white');
imagesc(lvbo_signal(1:2000));%这里的2000是为了放大包络上升沿起点前后部分的伪彩图
colorbar;  % 添加颜色条
xlabel('时间采样点');
ylabel('脉冲数量');
title('放大前端部分 ');









