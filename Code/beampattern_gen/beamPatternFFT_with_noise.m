function beamPatternFFT_with_noise(mvdr_results_W,h_16K,information,example,SaveFolder,flag)
    %% Parameters
    win_len = 512;
    overLap = 0.75;
    R = win_len-win_len*overLap;
    M = 8;
    K = win_len/2+1;
    win = hamming(win_len, 'periodic');
    if flag % flag=0 for me
        Vangle = (-90:1:90);    
    else
        Vangle = (0:1:180);    
    end
    results_Beampattern = zeros(length(Vangle),2);
    results_Beampattern(:,1) = deg2rad(Vangle);
    FreqVSDoa = cell(length(Vangle),1);
    fs = 16000;
    channels = 8;
    T = 4;
    %% Load W
    W = squeeze(mvdr_results_W(example,:,:,:));
    W(:,1) = real(W(:,1));
    W(:,end) = real(W(:,end));
    
    %% room - h 48k
    noise = randn(T*fs,1);
    for j = 1:length(h_16K)
        %%%%%%%%%%%%%%%%%
        h = h_16K{j}.';
        % conv
        n = cell(1,channels);
        for c = 1:channels
            n(c) = {filter(h(:,c),1,noise)}; % filter
        end
        n = cell2mat(n);
        
        %STFT synthesis
        L = floor((length(n)-win_len)/R+1);
        N = zeros(M,K,L);

        for i = 1:M
            N(i,:,:) = my_stft(squeeze(n(:,i)), win, R, win_len);
        end
    
        %Beamformer
        z_out = zeros(K,1,L);
        for k = 1:K
            z_out(k,:,:) = squeeze(W(:,k))'*squeeze(N(:,k,:));
        end
        
        %IFFT synthesis        
        [first_channel, t1] = my_istft(squeeze(z_out(:,1,:)), win, win, R, win_len, fs);

        norm_wh = norm((first_channel))^2;
        results_Beampattern(j,2) = norm_wh;
        FreqVSDoa(j) = {z_out};
        %%%%%%%%%%%%%%%%%% 
    end
    %% norm
    x_position = information.angle_x;
    idx = Vangle == x_position;
    gain = results_Beampattern(idx,2);
    results_Beampattern(:,2) = results_Beampattern(:,2)/gain;
    %% Plot Beampattern
    f1 = figure('Visible','off');
    polarplot(results_Beampattern(:,1),results_Beampattern(:,2),'-')

    disp(x_position)

    hold on
    max_dis = (max(results_Beampattern(:,2)));
    polarscatter(deg2rad(x_position),max_dis,'filled')
    n_position = information.n_position;
    n_position = deg2rad(n_position);
    polarscatter(n_position,max_dis+zeros(size(n_position)),'filled')
    legend('','source','noise')
    filePath = fullfile(SaveFolder,append('beampattern_example',num2str(example-1),'.png'));
    saveas(f1,filePath)
    %%
    FreqVSDoaInput = FreqVSDoa;
    FreqVSDoaPlot = cat(2,FreqVSDoaInput{:});
    FreqVSDoaPlot_time = sum(FreqVSDoaPlot.^2,3);%mean(FreqVSDoaPlot,3);
    lenF = size(FreqVSDoaPlot_time,1);
    F = (0:lenF-1)*(fs/2)/(lenF-1);
    if flag
        t = (-90:1:90);
    else
        t = (0:1:180);
    end
    f = figure('Visible','off');
    imagesc(t,F,20*log10(abs(FreqVSDoaPlot_time)));
    axis xy
    xlabel('DOA[angles]','fontsize',14);
    ylabel('Frequency[Hz]','fontsize',14);
    set(gca,'fontsize',14);
    colorbar
    filePath = fullfile(SaveFolder,append('FreqVSDoa_beampattern_example',num2str(example-1),'.png'));
    saveas(f,filePath)
end