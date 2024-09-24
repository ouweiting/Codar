% 见过的人，分人
clc;
clear;
close all;
addpath(genpath('F:\穿墙WIFI\crossfloor'));
addpath('F:/穿墙WIFI/代码/myPreprocess/linux-80211n-csitool');
addpath('F:\穿墙WIFI\crossfloor')
addpath(genpath('F:\异常检测\代码\PTA-HAD-master\tecoda-main'));
addpath('..\preprocess_tool')

%%
empty_dataName_list = {'54_gt.dat' '56_gt.dat'};
p1_dataName_list = {'45_dyp_1.dat'	'45_dyp_2.dat'	'45_dyp_3.dat'	'45_dyp_4.dat'   '54_dyp_1.dat'	'54_dyp_2.dat'	'54_dyp_3.dat' '54_dyp_4.dat'   '56_dyp_1.dat'	'56_dyp_2.dat'	'56_dyp_3.dat'   '56_dyp_4.dat'     '65_dyp_1.dat'	'65_dyp_2.dat'	'65_dyp_3.dat'	'65_dyp_4.dat' };
p2_dataName_list = {'45_lyj_1.dat'	'45_lyj_2.dat'	'45_lyj_3.dat' '45_lyj_4.dat' '54_lyj_1.dat'	'54_lyj_2.dat'	'54_lyj_3.dat' '54_lyj_4.dat' '56_lyj_1.dat'	'56_lyj_2.dat'	'56_lyj_3.dat' '56_lyj_4.dat' '65_lyj_1.dat'	'65_lyj_2.dat'	'65_lyj_3.dat'  '65_lyj_4.dat'};
p3_dataName_list = { '45_szj_1.dat'	'45_szj_2.dat'	'45_szj_3.dat' '45_szj_4.dat' '54_szj_1.dat'	'54_szj_2.dat'	'54_szj_3.dat' '54_szj_4.dat' '56_szj_1.dat'	'56_szj_2.dat'	'56_szj_3.dat' '56_szj_4.dat' '65_szj_1.dat'	'65_szj_2.dat'	'65_szj_3.dat' '65_szj_4.dat'};

%%
n_timestamps = 4;
step_size = 1;
test_ratio = 0.2;
train_amp_empty = [];
train_pha_empty = [];
train_CSI_empty = [];
train_amp_presence1 = [];
train_pha_presence1 = [];
train_CSI_presence1 = [];
train_amp_presence2 = [];
train_pha_presence2 = [];
train_CSI_presence2 = [];
train_amp_presence3 = [];
train_pha_presence3 = [];
train_CSI_presence3 = [];

test_amp_empty = [];
test_pha_empty = [];
test_CSI_empty = [];
test_amp_presence1 = [];
test_pha_presence1 = [];
test_CSI_presence1 = [];
test_amp_presence2 = [];
test_pha_presence2 = [];
test_CSI_presence2 = [];
test_amp_presence3 = [];
test_pha_presence3 = [];
test_CSI_presence3 = [];

%% empty tx*rx*sc*pa
for dataName = empty_dataName_list

    %% CSI
    csi = read_bf_file(dataName{:});
    csi = copy_empty_csi(csi);% 补全缺失天线，tx 1缺1
    csi = interpolate_amp0(csi); % 补全amp=0,用其它子载波的平均值
    csi = scale_csi_5300(csi,"5300");% 以绝对单位计算CSI
    csi = interpolate_timestamp(csi,'linear');%时间插值对齐

    %% AMP
    amp = get_amplitude(csi);
    amp = amp_hampel(amp);
    amp = amp_DWT(amp);
    amp = normalize_data(amp);

    n = length(csi);
    startIdx = randi(n-ceil(test_ratio*n));
    tmp1 = get_window(amp(1:startIdx-1,:,:,:),n_timestamps,step_size);
    tmp2 = get_window(amp(startIdx:startIdx+floor(test_ratio*n),:,:,:),n_timestamps,step_size);
    tmp3 = get_window(amp(startIdx+floor(test_ratio*n)+1:end,:,:,:),n_timestamps,step_size);
    train_amp_empty = cat(1,train_amp_empty,tmp1,tmp3);
    test_amp_empty = cat(1,test_amp_empty,tmp2);

    %% Phase
    phase = get_phase(csi);
    % phase = calibrate_phase(phase);%相位校正
    phase0 = phase;
    phase0 = calibrate_phase(phase0);
    phase0 = normalize_data(phase0);
    cCSI = amp.* exp(1i * phase0);
    tmp1 = get_window(cCSI(1:startIdx-1,:,:,:),n_timestamps,step_size);
    tmp2 = get_window(cCSI(startIdx:startIdx+floor(test_ratio*n),:,:,:),n_timestamps,step_size);
    tmp3 = get_window(cCSI(startIdx+floor(test_ratio*n)+1:end,:,:,:),n_timestamps,step_size);
    train_CSI_empty = cat(1,train_CSI_empty,tmp1,tmp3);
    test_CSI_empty = cat(1,test_CSI_empty,tmp2);


    phase = difference_phase(phase);%作差
    % phase = normalize_data(phase);%归一化
    phase = phase2AOA(phase,1,1);%AOA,波长，距离d约5cm，2.4G波长12.5

    tmp1 = get_window(phase(1:startIdx-1,:,:,:),n_timestamps,step_size);
    tmp2 = get_window(phase(startIdx:startIdx+floor(test_ratio*n),:,:,:),n_timestamps,step_size);
    tmp3 = get_window(phase(startIdx+floor(test_ratio*n)+1:end,:,:,:),n_timestamps,step_size);
    train_pha_empty = cat(1,train_pha_empty,tmp1,tmp3);
    test_pha_empty = cat(1,test_pha_empty,tmp2);
end

%% presence tx*rx*sc*pa
for dataName = p1_dataName_list

    %% CSI
    csi = read_bf_file(dataName{:});
    csi = copy_empty_csi(csi);% 补全缺失天线，tx 1缺1
    csi = interpolate_amp0(csi); % 补全amp=0,用其它子载波的平均值
    csi = scale_csi_5300(csi,"5300");% 以绝对单位计算CSI
    csi = interpolate_timestamp(csi,'linear');%时间插值对齐

    %% AMP
    amp = get_amplitude(csi);
    amp = amp_hampel(amp);
    amp = amp_DWT(amp);
	amp = normalize_data(amp);

    n = length(csi);
    startIdx = randi(n-ceil(test_ratio*n));
    tmp1 = get_window(amp(1:startIdx-1,:,:,:),n_timestamps,step_size);
    tmp2 = get_window(amp(startIdx:startIdx+floor(test_ratio*n),:,:,:),n_timestamps,step_size);
    tmp3 = get_window(amp(startIdx+floor(test_ratio*n)+1:end,:,:,:),n_timestamps,step_size);
    train_amp_presence1 = cat(1,train_amp_presence1,tmp1,tmp3);
    test_amp_presence1 = cat(1,test_amp_presence1,tmp2);

    %% Phase
    phase = get_phase(csi);
    % phase = calibrate_phase(phase);%相位校正
    phase0 = phase;
    phase0 = calibrate_phase(phase0);
    phase0 = normalize_data(phase0);
    cCSI = amp.* exp(1i * phase0);
    tmp1 = get_window(cCSI,n_timestamps,step_size);
    train_CSI_presence1 = cat(1,train_CSI_presence1,tmp1);

    phase = difference_phase(phase);%作差
    % phase = normalize_data(phase);%归一化
    phase = phase2AOA(phase,1,1);%AOA,波长，距离d约5cm，2.4G波长12.5
    tmp1 = get_window(phase(1:startIdx-1,:,:,:),n_timestamps,step_size);
    tmp2 = get_window(phase(startIdx:startIdx+floor(test_ratio*n),:,:,:),n_timestamps,step_size);
    tmp3 = get_window(phase(startIdx+floor(test_ratio*n)+1:end,:,:,:),n_timestamps,step_size);
    train_pha_presence1 = cat(1,train_pha_presence1,tmp1,tmp3);
    test_pha_presence1 = cat(1,test_pha_presence1,tmp2);
end

for dataName = p2_dataName_list

    %% CSI
    csi = read_bf_file(dataName{:});
    csi = copy_empty_csi(csi);% 补全缺失天线，tx 1缺1
    csi = interpolate_amp0(csi); % 补全amp=0,用其它子载波的平均值
    csi = scale_csi_5300(csi,"5300");% 以绝对单位计算CSI
    csi = interpolate_timestamp(csi,'linear');%时间插值对齐

    %% AMP
    amp = get_amplitude(csi);
    amp = amp_hampel(amp);
    amp = amp_DWT(amp);
    amp = normalize_data(amp);
	
    n = length(csi);
    startIdx = randi(n-ceil(test_ratio*n));
    tmp1 = get_window(amp(1:startIdx-1,:,:,:),n_timestamps,step_size);
    tmp2 = get_window(amp(startIdx:startIdx+floor(test_ratio*n),:,:,:),n_timestamps,step_size);
    tmp3 = get_window(amp(startIdx+floor(test_ratio*n)+1:end,:,:,:),n_timestamps,step_size);
    train_amp_presence2 = cat(1,train_amp_presence2,tmp1,tmp3);
    test_amp_presence2 = cat(1,test_amp_presence2,tmp2);

    %% Phase
    phase = get_phase(csi);
    % phase = calibrate_phase(phase);%相位校正
    phase0 = phase;
    phase0 = calibrate_phase(phase0);
    phase0 = normalize_data(phase0);
    cCSI = amp.* exp(1i * phase0);
    tmp1 = get_window(cCSI,n_timestamps,step_size);
    train_CSI_presence2 = cat(1,train_CSI_presence2,tmp1);

    phase = difference_phase(phase);%作差
    % phase = normalize_data(phase);%归一化
    phase = phase2AOA(phase,1,1);%AOA,波长，距离d约5cm，2.4G波长12.5
    tmp1 = get_window(phase(1:startIdx-1,:,:,:),n_timestamps,step_size);
    tmp2 = get_window(phase(startIdx:startIdx+floor(test_ratio*n),:,:,:),n_timestamps,step_size);
    tmp3 = get_window(phase(startIdx+floor(test_ratio*n)+1:end,:,:,:),n_timestamps,step_size);
    train_pha_presence2 = cat(1,train_pha_presence2,tmp1,tmp3);
    test_pha_presence2 = cat(1,test_pha_presence2,tmp2);
end

for dataName = p3_dataName_list

    %% CSI
    csi = read_bf_file(dataName{:});
    csi = copy_empty_csi(csi);% 补全缺失天线，tx 1缺1
    csi = interpolate_amp0(csi); % 补全amp=0,用其它子载波的平均值
    csi = scale_csi_5300(csi,"5300");% 以绝对单位计算CSI
    csi = interpolate_timestamp(csi,'linear');%时间插值对齐

    %% AMP
    amp = get_amplitude(csi);
    amp = amp_hampel(amp);
    amp = amp_DWT(amp);
    amp = normalize_data(amp);
	
    n = length(csi);
    startIdx = randi(n-ceil(test_ratio*n));
    tmp1 = get_window(amp(1:startIdx-1,:,:,:),n_timestamps,step_size);
    tmp2 = get_window(amp(startIdx:startIdx+floor(test_ratio*n),:,:,:),n_timestamps,step_size);
    tmp3 = get_window(amp(startIdx+floor(test_ratio*n)+1:end,:,:,:),n_timestamps,step_size);
    train_amp_presence3 = cat(1,train_amp_presence3,tmp1,tmp3);
    test_amp_presence3 = cat(1,test_amp_presence3,tmp2);

    %% Phase
    phase = get_phase(csi);
    % phase = calibrate_phase(phase);%相位校正
    phase0 = phase;
    phase0 = calibrate_phase(phase0);
    phase0 = normalize_data(phase0);
    cCSI = amp.* exp(1i * phase0);
    tmp1 = get_window(cCSI,n_timestamps,step_size);
    test_CSI_presence3 = cat(1,test_CSI_presence3,tmp1);

    phase = difference_phase(phase);%作差
    % phase = normalize_data(phase);%归一化
    phase = phase2AOA(phase,1,1);%AOA,波长，距离d约5cm，2.4G波长12.5
    tmp1 = get_window(phase(1:startIdx-1,:,:,:),n_timestamps,step_size);
    tmp2 = get_window(phase(startIdx:startIdx+floor(test_ratio*n),:,:,:),n_timestamps,step_size);
    tmp3 = get_window(phase(startIdx+floor(test_ratio*n)+1:end,:,:,:),n_timestamps,step_size);
    train_pha_presence3 = cat(1,train_pha_presence3,tmp1,tmp3);
    test_pha_presence3 = cat(1,test_pha_presence3,tmp2);

end

%% EEGNet

%% CSI
X_train_CSI_empty = train_CSI_empty;
X_test_CSI_empty = test_CSI_empty;
X_train_CSI_presence1 = train_CSI_presence1;
X_test_CSI_presence1 = test_CSI_presence1;
X_train_CSI_presence2 = train_CSI_presence2;
X_test_CSI_presence2 = test_CSI_presence2;
X_train_CSI_presence3 = train_CSI_presence3;
X_test_CSI_presence3 = test_CSI_presence3;

rand_indices = randperm(size(X_test_CSI_presence3,1), size(X_test_CSI_empty,1));
X_test_CSI_presence3 = X_test_CSI_presence3(rand_indices,:,:,:,:);

% 2. n*r移到package上train_amp_empty



%% N*f

Y_train = cat(1,ones(size(X_train_CSI_empty,1),1),ones(size(X_train_CSI_empty,1),1),ones(size(X_train_CSI_empty,1),1),2*ones(size(X_train_CSI_presence1,1),1),2*ones(size(X_train_CSI_presence2,1),1));
CSI_train = cat(1,X_train_CSI_empty,X_train_CSI_empty,X_train_CSI_empty,X_train_CSI_presence1,X_train_CSI_presence2);
% Y_train = categorical(Y_train);
CSI_test = cat(1,X_test_CSI_empty,X_test_CSI_presence3);
Y_test = cat(1,ones(size(X_test_CSI_empty,1),1),2*ones(size(X_test_CSI_presence3,1),1));
% Y_test = categorical(Y_test);

save('CSI_train.mat', 'CSI_train');
save('Y_train.mat', 'Y_train');
save('CSI_test.mat', 'CSI_test');
save('Y_test.mat', 'Y_test');

