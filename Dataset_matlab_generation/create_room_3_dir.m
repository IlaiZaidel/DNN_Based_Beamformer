function [h_n,a_n,a_n_scnd,information] = create_room_3_dir(PlotRoom)

   %load handel.mat
   mFlag = 0;
   disp(['be here'])
   % Parameters
   c_k = 340;                      % Sound velocity (m/s)
   fs = 16000;                     % Sample frequency (samples/s)
   %Vbeta = (0.3:0.05:0.5);         % Possible values for beta
   %idx = randi(numel(Vbeta));      % Random index selection
   beta = 0; % Vbeta(idx);              % Reverberation time (s)
   n = fs*beta;                    % Number of samples
   mtype = 'omnidirectional';      % Type of microphone
   order = 0;                     % -1 equals maximum reflection order!
   dim = 3;                        % Room dimension
   orientation = 0;                % Microphone orientation (rad)
   hp_filter = 1;                  % Disable high-pass filter

   % room dimensions 
   x_lim = 6 + 3*rand(1);         % x position (m) - number in the range [6,9](m)
   y_lim = 6 + 3*rand(1);         % y position (m) - number in the range [6,9](m)
   L = [x_lim y_lim 3];           % Room dimensions [x_lim y_lim z=3] (m)

   % mic position
   mic_height = 1 + 0.5*rand(1);    % microphone array height  - in the range [1,1.5](m) 
   mic_x = 2.5 + (x_lim-5)*rand(1);   % [1.5, x_room - 1.5] 0.5m for the source and 1 for min radius
   mic_y = 0.5 + (y_lim-3)*rand(1);   % [0.5, y_room - 1.5] 0.5m for the source 
   mic_position = [[mic_x-0.17 ,mic_y, mic_height];...
                   [mic_x-0.12, mic_y, mic_height];...
                   [mic_x-0.07, mic_y, mic_height];...
                   [mic_x-0.04, mic_y, mic_height];...
                   [mic_x+0.04, mic_y, mic_height];...
                   [mic_x+0.07, mic_y, mic_height];...
                   [mic_x+0.12, mic_y, mic_height];...
                   [mic_x+0.17, mic_y, mic_height]]; % Receiver position [x y z] (m)
   %mic1 - 5cm - mic2 - 5cm - mic3 - 3cm - mic4 - 8cm - mic5 - 3cm - mic6 -5cm - mic7 - 5cm - mic8
   
   Vangle = (-45:1:45);                   % Possible values for orientation of microphone array
   idx_angle = randi(numel(Vangle));     % Random index selection
   angleOrientation = Vangle(idx_angle); 
   
   %mic_position_ori = mic_position;
   mic_position_final = mic_position;
   %mic_position_ori(:,1:2) = mic_position(:,1:2)*[cosd(angleOrientation),sind(angleOrientation);...
                                                    %        -sind(angleOrientation),cosd(angleOrientation)];
   %mic_x = 2.5 + (x_lim-5)*rand(1);   % [1.5, x_room - 1.5] 0.5m for the source and 1 for min radius
   %mic_y = 0.5 + (y_lim-3)*rand(1);   % [0.5, y_room - 1.5] 0.5m for the source    
   center = [mic_x,mic_y]; % center of the array
   %mic_position_final(:,1:2) = center + mic_position_ori(:,1:2);



    % Ilai Zaidel - 18/03
    % I have found an error. The previous code rotated the array without taking into account the center of the array, resulting in incorrect positioning of the mics.
    % I fixed that using the following lines:

   mic_rel = mic_position(:,1:2) - center;  % Step 1: Shift to origin
   mic_rotated = mic_rel * [cosd(angleOrientation), sind(angleOrientation);
                            -sind(angleOrientation), cosd(angleOrientation)];  % Step 2: Rotate
   mic_position_final(:,1:2) = mic_rotated + center;  % Step 3: Shift back




   % source position
   lim_radius = min([center(1),L(1:2)-center]) - 0.5; % The limit of the radius according to the dimensions of the room
   if lim_radius < 1 % lim_ radius can be less than one only if l(2)-center is smaller than one. 
                         % In this case we will take the second half of the room
       lim_radius = min([center(1:2),L(1)-center(1)]); 
       mFlag = 1;
   end
   if lim_radius > 2.2  % limit the distance to 5m
       lim_radius = 2.2;
   end

   %%%%% This is the speaker %%%%%

   radius = rand(1)*(lim_radius-1.8) + 1.8;  % [1, lim_radius](m) - The radius from the center of the array
   Vangle = (0:1:180);                    % Possible values for angle
   idx_angle = randi(numel(Vangle));       % Random index selection
   angle = Vangle(idx_angle);              % angle (degree)
   %angle = 100; % DELETE ONLY FOR REFERENCE EXAMPLE
   [source_x_temp,source_y_temp] = sph2cart(deg2rad(angle+angleOrientation),0,radius);  % Source position from the center of the array 

   source_x = source_x_temp + center(1);     

   if ~mFlag
       source_y = source_y_temp + center(2);
   else
       source_y = center(2) - source_y_temp;
   end
   source_height = 1.2;                              % sound source height - in the range [1.2,1.9](m) 
   

  % =====Ilai Z only test:====
   source_y = max(0.5, min(source_y, y_lim - 0.5));
   source_x = max(0.5, min(source_x, x_lim - 0.5));
   x_position = [source_x source_y source_height]; % Source position [x y z] (m)
% =====Ilai Z only test:====
   %disp([' Source_y equals = ' num2str(source_y) ' y_lim - 0.5 = ' num2str(y_lim - 0.5)]) % IlaiZ  
   
   if source_x > x_lim - 0.5 || source_x < 0.5
       disp(['Error in x; x_pos = ' num2str(source_x) ' ; Room Dim = ' num2str(x_lim)])
       return;
   end
   if source_y > y_lim - 0.5|| source_y < 0.5
       disp(['Error in y; y_pos = ' num2str(source_y) ' ; Room Dim = ' num2str(y_lim)])
       return;
   end
    %%%%%%%%

%%%%%%%%%%%%%%%%%%%%%%%%%%%%
% Noise Terms 

   [angle_n1, angle_n2] = randomize_noise_angles(angle);
%%%%%%%%%%%%%%%%%%%%%%%%%%%%
   radius_n = radius;                            % [1,lim_radius](m) - The radius from the center of the array
   [noise_x_temp,noise_y_temp] = sph2cart(deg2rad(angle_n1+angleOrientation),0,radius_n); % Noise position from the center of the array 

   noise_x = noise_x_temp + center(1);  % Random x position
   if ~mFlag
       noise_y = noise_y_temp + center(2);
   else
       noise_y = center(2) - noise_y_temp;
   end
   noise_height = 1;                    % noise source height - in the range [1.2,1.9](m) 
    % Noise position [x y z] (m)
     % =====Ilai Z only test:============================
   noise_y = max(0.5, min(noise_y, y_lim - 0.5));
   noise_x = max(0.5, min(noise_x, x_lim - 0.5));
   n_position = [noise_x noise_y noise_height]; 
     % =====Ilai Z only test:============================
   if noise_x > x_lim - 0.5 || noise_x < 0.5
       disp(['Error in noise x; x_pos = ' num2str(noise_x) ' ; Room Dim = ' num2str(x_lim)])
       return;
   end
   if noise_y > y_lim - 0.5 || noise_y < 0.5
       disp(['Error in noise y; y_pos = ' num2str(noise_y) ' ; Room Dim = ' num2str(y_lim)])
       return;
   end

%%% 2nd noise

    %%%%%%%%
   % noise position
   
   radius_n = radius;                            % [1,lim_radius](m) - The radius from the center of the array
   [noise_x_temp,noise_y_temp] = sph2cart(deg2rad(angle_n2+angleOrientation),0,radius_n); % Noise position from the center of the array 

   noise_scnd_x = noise_x_temp + center(1);  % Random x position
   if ~mFlag
       noise_scnd_y = noise_y_temp + center(2);
   else
       noise_scnd_y = center(2) - noise_y_temp;
   end
   noise_height = 1.2; 
   noise_scnd_y = max(0.5, min(noise_scnd_y, y_lim - 0.5));                   % noise source height - in the range [1.2,1.9](m) 
   noise_scnd_x = max(0.5, min(noise_scnd_x, x_lim - 0.5));  
   n_position_scnd = [noise_scnd_x noise_scnd_y noise_height];  % Noise position [x y z] (m)
     % =====Ilai Z only test:============================
                                                                                    
     % =====Ilai Z only test:============================
   if noise_x > x_lim - 0.5 || noise_x < 0.5
       disp(['Error in noise x; x_pos = ' num2str(noise_x) ' ; Room Dim = ' num2str(x_lim)])
       return;
   end
   if noise_y > y_lim - 0.5 || noise_y < 0.5
       disp(['Error in noise y; y_pos = ' num2str(noise_y) ' ; Room Dim = ' num2str(y_lim)])
       return;
   end







% ---------- Using rir_generator---------- %
   
   h_n =      rir_generator(c_k, fs, mic_position_final, x_position, L, beta, n, mtype, order, dim, orientation, hp_filter);
   a_n =      rir_generator(c_k, fs, mic_position_final, n_position, L, beta, n, mtype, order, dim, orientation, hp_filter);
   a_n_scnd = rir_generator(c_k, fs, mic_position_final, n_position_scnd, L, beta, n, mtype, order, dim, orientation, hp_filter); % second noise
   
  % disp(['I passed the rir_generator'])
   information = struct('beta', beta, 'x_position', x_position,...
        'n_position', n_position,'n_position_scnd', n_position_scnd,'mic_position', mic_position_final,...
        'room_dim', L, 'angleOrientation', angleOrientation,...
        'angle_x', angle, 'angle_n1', angle_n1, 'angle_n2', angle_n2,'radius', radius);
   disp(['Hello my friend i am here'])



%    mic_position
%    disp('Mic Positions before:');
%    disp(mic_position);

%    disp('Mic Positions:');
%    disp(mic_position_final);

%    disp('Center');
%    disp(center);

   if PlotRoom
    f = figure('visible', 'on');
    hold on;
    axis([0 L(1) 0 L(2)]);
    grid on;
    xlabel('X Position (m)');
    ylabel('Y Position (m)');
    title('Room Layout with Speaker and Noise Sources');
    
    % Plot room boundaries
    plot([0 L(1) L(1) 0 0], [0 0 L(2) L(2) 0], 'k-', 'LineWidth', 1.5);
    
    % Plot microphone array
    plot(mic_position_final(:,1), mic_position_final(:,2), 'sr', 'MarkerSize', 8, 'DisplayName', 'Microphones');
    plot(center(1), center(2), '*g', 'MarkerSize', 10, 'DisplayName', 'Mic Center');
    
    % Plot speaker position
    plot(x_position(1), x_position(2), 'm*', 'MarkerSize', 10, 'DisplayName', 'Speaker');
    line([x_position(1), center(1)], [x_position(2), center(2)], 'Color', 'm', 'LineWidth', 1);
    
    % Plot noise positions
    plot(n_position(1), n_position(2), 'c^', 'MarkerSize', 10, 'DisplayName', 'Noise 1');
    line([n_position(1), center(1)], [n_position(2), center(2)], 'Color', 'c', 'LineWidth', 1);
    
    plot(n_position_scnd(1), n_position_scnd(2), 'b^', 'MarkerSize', 10, 'DisplayName', 'Noise 2');
    line([n_position_scnd(1), center(1)], [n_position_scnd(2), center(2)], 'Color', 'b', 'LineWidth', 1);
    
    legend;
    saveas(f,'room.png');
    end 
end
