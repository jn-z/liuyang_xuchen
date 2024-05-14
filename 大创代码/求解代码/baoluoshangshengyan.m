
folder_path = 'D:\新数据3.2\其它\4';% 信号文件夹路径，文件夹中为mat格式文件
file_list = dir(fullfile(folder_path, 'T_FSK_blind_lf_pulse_*.mat'));
xinshuju=zeros(20,2560);%用于存储前20个信号
xinshujulvbohou=zeros(20,2560);%用于存储前20个滤波后的信号
for i = 1:20
    file_name = fullfile(folder_path, file_list(i).name);
    st = load(file_name);
    data=st.data;
    data=transpose(data);
    xinshuju(i,:)=data;
    xinshujulvbohou(i,:)=wdenoise(xinshuju(i,:),5,'DenoisingMethod','BlockJS');%使用小波降噪法滤波
end
baoluo=cell(1,20);%用于存储滤波后的包络上升沿，由于包络上升沿长度不等，故使用cell来存储
for i = 1:20

    
    signal_data = xinshujulvbohou(i,:);
    max_val = max(signal_data);
    min_val = min(signal_data);

    % 对信号数据进行归一化
    normalized_signal = (signal_data - min_val) / (max_val - min_val);
    hilbertTransform = hilbert(signal_data);
    envelope = abs(hilbertTransform);
    max_val = max(envelope);
    min_val = min(envelope);

    % 对包络数据进行归一化
    normalized_envelope = (envelope - min_val) / (max_val - min_val);
    % 找到包络信号的所有极大值点
    [~, locs] = findpeaks(normalized_envelope); 

    % 从所有极大值点中找到第一个大于0.5的点
    first_max_below_threshold = find(normalized_envelope(locs) > 0.5, 1);
       first_max_idx = locs(first_max_below_threshold);
        first_max_val = normalized_envelope(first_max_idx);
            % 从第一个大于0.5的极大值点位置之前寻找所有的小于0.1的极小值点
            sub=normalized_envelope(1:first_max_idx);
    [~, locs_min] = findpeaks(-sub,'MinPeakHeight', -0.1);

    % 找到第一个极大值点前第一个小于0.1的极小值点
    first_min_idx = locs_min(end);
    baoluo{i}=normalized_envelope(first_min_idx:first_max_idx);
end

    
