%%
clear all;
clc;
%% BASIC Parameters 
fs = 16000;
T = 4;
mic_ref = 4;
%path = 'C:\Users\adid5\Dropbox (BIU)\All DNN Beamformer\dataset\clean_sounds_test';
path = '/dsi/gannot-lab1/datasets/LibriSpeech/LibriSpeech/Train';

foldersNames = {dir(path).name};
foldersNames = foldersNames(2:end);
NumOfFolders = numel(foldersNames);
channels = 8;
flagwhiteNoise  = 1;
noise_time = 0.5;
PlotRoom = 0;
%% INPUT



for example = 136 %0:1:180
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
    disp("im here")
    counter = 0:1:179;
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
    %disp("hello")
    flag = 0;
    mycount = 0;
    while flag == 0 
        try
            [h,a,information] = create_surround(PlotRoom, example);
            flag = 1;
        catch
            mycount= mycount +1;
            flag = 0;
        end
    end
    h = h.';
    a = a.';
    %disp("am i here?")
    %% CONV 
    d = cell(1,channels);
    for c = 1:channels
        d(c) = {filter(h(:,c),1,x)};
    end
    d = cell2mat(d);
    d = [zeros(fs*noise_time,channels);d(1:fs*(T-noise_time),:)]; % 0.5 seconds of noisy signal
    s = [zeros(fs*noise_time,1);x(1:fs*(T-noise_time),:)]; 
    %% NOISE(Directional noise)
    % w = randn(T*fs,1);
    % AR = [1 -0.7];

    %noise = filter(1,AR,w);
    noise = randn(T*fs,1);
    n = cell(1,channels);
    for c = 1:channels
        n(c) = {filter(h(:,c),1,noise)};
    end
    n = cell2mat(n);

    %% signal +noise    
    y = n ;
    %% Sounds
    %soundsc(y(1:T*fs,mic_ref),fs)  
    %% Save h as one of the feature_vector labels # 14.12.2024 - I  
    %information(:).impulse_response = h;  % Add h to the structure
    max_y = max(abs(y(:,mic_ref)));     
    [information(:).feature] = y;         
    [information(:).fulloriginal] = d/max_y;
    [information(:).fullnoise] = (n)/max_y;   
    max_s = max(abs(s));    
    [information(:).target_s] = s/max_s;
    disp("did i came here?")
    %path_folder = '/home/dsi/ilaiz/DNN_Based_Beamformer/Code/processed_data';
    %path_folder = '/dsi/gannot-lab1/datasets/Ilai_data/Noisy_Beampattern_Surround';
    path_folder = '/dsi/gannot-lab1/datasets/Ilai_data/Correct_White_Beampattern_Surround' ;
    %path_folder = '/home/dsi/ilaiz/DNN_Based_Beamformer/Code/beampattern_gen';
    name_file = append('my_surround_feature_vector_angle_',num2str(example),'.mat');    
    %save(fullfile(pwd,'..',path_folder,name_file),'-struct','information') 
    save(fullfile(path_folder,name_file),'-struct','information') 
    
end

