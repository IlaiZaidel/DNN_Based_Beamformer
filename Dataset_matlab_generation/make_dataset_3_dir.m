%%
clear all;
clc;
%% BASIC Parameters 
fs = 16000;
T = 4;
mic_ref = 4;
%path = 'C:\Users\adid5\Dropbox (BIU)\All DNN Beamformer\dataset\clean_sounds_test';
%path = '/dsi/gannot-lab1/datasets/LibriSpeech/LibriSpeech/Train';
path = '/dsi/gannot-lab1/datasets/LibriSpeech/LibriSpeech/Test';
foldersNames = {dir(path).name};
foldersNames = foldersNames(2:end);
NumOfFolders = numel(foldersNames);
channels = 8;
flagwhiteNoise  = 1;
noise_time = 0.5;
PlotRoom = 0;
%% INPUT

% missing_numbers = [522, 705, 1371, 1375, 2000, 2299, 2541, 3020, 3051, 3090, ...
%                    3990, 5038, 6042, 8191, 8195, 8820, 9119, 9361, 9840, 9871, ...
%                    9910, 10810, 11858, 12862, 13653, 13749, 13762, 13798, 15271, ...
%                    15275, 15900, 16199, 16441, 16920, 16951, 16990, 17890, 18938, 19942];


for example =  0
%example = 18500;
%while example <= 19999
    x =[];
    folder_num = randi(NumOfFolders);
    folderName = fullfile(path,foldersNames(folder_num));
    subfolders = dir(folderName{:});
    subfoldersNames = {subfolders([subfolders.isdir]).name};
    subfoldersNames = subfoldersNames(3:end);
    NumOfSubFolders = numel(subfoldersNames);
    % Check if there are any subfolders
    if NumOfSubFolders == 0
        disp(['No subfolders found in folder: ', folderName]);
        continue;  % Skip to the next iteration if no subfolders
    end
    subfolder_num = randi(NumOfSubFolders);
    subfolderName = fullfile(folderName,subfoldersNames(subfolder_num));
    files = {dir(fullfile(subfolderName{:}, '*.wav')).name}; %files = {dir([subfolderName{:}, '\*.wav']).name};
    %disp(['Subfolder path: ', subfolderName{:}]);
    %disp(['Full search path: ', fullfile(subfolderName{:}, '*.wav')]);
    %disp(['Checking folder: ', subfolderName{:}]);
       % Check if the directory contains .wav files
    if isempty(files)
        disp(['No .wav files found in folder: ', subfolderName]);
        continue;  % Skip to the next iteration if no files are found 236400
    end
    
    %example = example + 1;
     
    counter = 0;
    temp = T*fs;
    while (length(x) < T*fs) %4[s]
        idx = randi(numel(files));
        counter=counter+1;
        file = files{idx};   
        name = fullfile(subfolderName,file);
        [signal,fs] = audioread(name{:});
        %p = length(signal)
        G_x = max(abs(signal));
        signal = 2*signal/G_x;
            % Ensure the length does not exceed the signal length
        if temp > length(signal)
            temp = length(signal); % Adjust `temp` to fit within `signal`
        end
        x = [x ; signal(1:temp)];
        %k = length(x)
        %disp(['X length: ', num2str(length(x))])
    end
    %% room - h 
    flag = 0;
    mycount = 0;
    while flag == 0 
        try
            [h,a,a_2,information] = create_room_3_dir(PlotRoom);
            disp(['arriveed here']);
            flag = 1;
        catch
            mycount= mycount +1;
            flag = 0;
        end
    end
    
    h = h.';
    a = a.';
    a_2 = a_2.';
    %% CONV 
    d = cell(1,channels);
    for c = 1:channels
        d(c) = {filter(h(:,c),1,x)};
    end
    d = cell2mat(d);
    d = [zeros(fs*noise_time,channels);d(1:fs*(T-noise_time),:)]; % 0.5 seconds of noisy signal
    s = [zeros(fs*noise_time,1);x(1:fs*(T-noise_time),:)]; 
    %% NOISE(Directional noise)
    w = randn(T*fs,1);
    AR = [1 -0.7];
    noise = filter(1,AR,w);
    
    n = cell(1,channels);
    for c = 1:channels
        n(c) = {filter(a(:,c),1,noise)};
    end
    n_1 = cell2mat(n);
    
    %% SNR x & n 
    % SNR = 6; %12db
    % dVSn = sqrt((sum(d(:,mic_ref).^2)*10^(-SNR/10))./(sum(n(:,mic_ref).^2)));
    % n = dVSn.*n;
    % SNR_dn = snr(d,n);

    %% SECOND NOISE( Second Directional noise)
    w_2 = randn(T*fs,1);
    AR = [1 -0.7];
    noise_2 = filter(1,AR,w_2);
    
    n_2 = cell(1,channels);
    for c = 1:channels
        n_2(c) = {filter(a_2(:,c),1,noise_2)};
    end
    n_2 = cell2mat(n_2);
    
    %% SNR x & n 
    % SNR = 6; %12db
    % dVSn = sqrt((sum(d(:,mic_ref).^2)*10^(-SNR/10))./(sum(n_2(:,mic_ref).^2)));
    % n_2 = dVSn.*n_2;
    % SNR_dn_2 = snr(d,n_2);

    %% SNR of both noise sources
    SNR = 3; %12db
    dVSn = sqrt((sum(d(:,mic_ref).^2)*10^(-SNR/10))./(sum((n_1(:,mic_ref)   +   n_2(:,mic_ref)).^2)));
    n_1 = dVSn.*n_1;
    n_2 = dVSn.*n_2;
    SNR_dn = snr(d,n_1+n_2)
    %% white Noise 
    SNR = 30; %30; %30db
    v = randn(size(d)); %noise
    if flagwhiteNoise == 1
        dVSv = sqrt((sum(d(:,mic_ref).^2)*10^(-SNR/10))./(sum(v(:,mic_ref).^2)));
    else
        dVSv = 0;
    end
    v = dVSv.*v;
    SNR_dv = snr(d,v)
    %% signal +noise
    y =  d + v + n_1 + n_2 ;
    %% Sounds
    %soundsc(y(1:T*fs,mic_ref),fs) 
%     %% Save h as one of the feature_vector labels # 14.12.2024 - I
%     %information(:).impulse_response = h;  % Add h to the structure
%     max_y = max(abs(y(:,mic_ref)));
%     [information(:).feature] = y/max_y; 
%     [information(:).fulloriginal] = d/max_y;
%     [information(:).fullnoise_first] = n_1/max_y;
%     [information(:).fullnoise_second] = n_2/max_y;
%     max_s = max(abs(s));
%     [information(:).target_s] = s/max_s;
%     %path_folder = '/home/dsi/ilaiz/DNN_Based_Beamformer/Code/processed_data';
%     %path_folder = '/dsi/gannot-lab1/datasets/Ilai_data/Test_set'; 
%     %path_folder = '/home/dsi/ilaiz/DNN_Based_Beamformer/Code/beampattern_gen';
%    % path_folder = '/dsi/gannot-lab1/datasets/Ilai_data/WhiteNoiseOnly_Train';
%     path_folder = '/dsi/gannot-lab1/datasets/Ilai_data/Two_Directional_Noises_Train';
%     %path_folder = '/dsi/gannot-lab1/datasets/Ilai_data/Two_Directional_Noises_Test';
%     name_file = append('my_feature_vector_',num2str(example),'.mat');    
%     %save(fullfile(pwd,'..',path_folder,name_file),'-struct','information') 
%     save(fullfile(path_folder,name_file),'-struct','information') 
%     disp(['arriveed here']);


    %% Save h as one of the feature_vector labels # 14.12.2024 - I
    %information(:).impulse_response = h;  % Add h to the structure
    max_y = max(abs(y(:,mic_ref)));
    [information(:).feature] = y/max_y; 
    [information(:).fulloriginal] = d/max_y;
    [information(:).fullnoise_first] = n_1/max_y;
    [information(:).fullnoise_second] = n_2/max_y;
    max_s = max(abs(s));
    [information(:).target_s] = s/max_s;

    % Specify the output folder and file name
  %  path_folder = '/dsi/gannot-lab1/datasets/Ilai_data/Two_Directional_Noises_Train';
    %path_folder = '/dsi/gannot-lab1/datasets/Ilai_data/Two_Directional_Noises_Test';
    path_folder ='/home/dsi/ilaiz/DNN_Based_Beamformer/Code/beampattern_gen';
    name_file = append('my_feature_vector_', num2str(example), '.mat');

    % Attempt to save; if an error occurs, retry until successful
    saved = false;
    while ~saved
        try
            save(fullfile(path_folder, name_file), '-struct', 'information');
            saved = true;
        catch err
            disp(['Error saving file ' fullfile(path_folder, name_file) '. Retrying...']);
            pause(1);  % Wait for 1 second before retrying
        end
    end
    disp('arriveed here');

end

