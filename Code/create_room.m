function [h_n,a_n,information] = create_room(PlotRoom)

   %load handel.mat
   mFlag = 0;

   % Parameters
   c_k = 340;                      % Sound velocity (m/s)
   fs = 16000;                     % Sample frequency (samples/s)
   Vbeta = (0.3:0.05:0.5);         % Possible values for beta
   idx = randi(numel(Vbeta));      % Random index selection
   beta = Vbeta(idx);              % Reverberation time (s)
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
   mic_height = 1;% + 0.5*rand(1);    % microphone array height  - in the range [1,1.5](m) 
   mic_x = 0;%2.5 + (x_lim-5)*rand(1);   % [1.5, x_room - 1.5] 0.5m for the source and 1 for min radius
   mic_y = 0;%0.5 + (y_lim-3)*rand(1);   % [0.5, y_room - 1.5] 0.5m for the source 
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
   
   mic_position_ori = mic_position;
   mic_position_final = mic_position;
   mic_position_ori(:,1:2) = mic_position(:,1:2)*[cosd(angleOrientation),sind(angleOrientation);...
                                                            -sind(angleOrientation),cosd(angleOrientation)];
   mic_x = 2.5 + (x_lim-5)*rand(1);   % [1.5, x_room - 1.5] 0.5m for the source and 1 for min radius
   mic_y = 0.5 + (y_lim-3)*rand(1);   % [0.5, y_room - 1.5] 0.5m for the source    
   center = [mic_x,mic_y]; % center of the array
   mic_position_final(:,1:2) = center + mic_position_ori(:,1:2);

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
   radius = rand(1)*(lim_radius-1.8) + 1.8;  % [1, lim_radius](m) - The radius from the center of the array
   Vangle = (0:1:180);                    % Possible values for angle
   idx_angle = randi(numel(Vangle));       % Random index selection
   angle = Vangle(idx_angle);              % angle (degree)

   [source_x_temp,source_y_temp] = sph2cart(deg2rad(angle+angleOrientation),0,radius);  % Source position from the center of the array 

   source_x = source_x_temp + center(1);     

   if ~mFlag
       source_y = source_y_temp + center(2);
   else
       source_y = center(2) - source_y_temp;
   end
   source_height = 1;                              % sound source height - in the range [1.2,1.9](m) 
   x_position = [source_x source_y source_height]; % Source position [x y z] (m)

  % =====Ilai Z only test:====
   source_y = max(0.5, min(source_y, y_lim - 0.5));
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
 
   % noise position
   Vangle = [(angle+20:1:180) (0:1:angle-20)];  % Possible values for angle
   idx_angle = randi(numel(Vangle));             % Random index selection
   angle_n = Vangle(idx_angle);                  % angle (degree)
   radius_n = radius;                            % [1,lim_radius](m) - The radius from the center of the array
   [noise_x_temp,noise_y_temp] = sph2cart(deg2rad(angle_n+angleOrientation),0,radius_n); % Noise position from the center of the array 

   noise_x = noise_x_temp + center(1);  % Random x position
   if ~mFlag
       noise_y = noise_y_temp + center(2);
   else
       noise_y = center(2) - noise_y_temp;
   end
   noise_height = 1;                    % noise source height - in the range [1.2,1.9](m) 
   n_position = [noise_x noise_y noise_height];  % Noise position [x y z] (m)
     % =====Ilai Z only test:============================
   noise_y = max(0.5, min(noise_y, y_lim - 0.5));
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
 %disp(['Arrived here'])
   h_n = rir_generator(c_k, fs, mic_position_final, x_position, L, beta, n, mtype, order, dim, orientation, hp_filter);
   a_n = rir_generator(c_k, fs, mic_position_final, n_position, L, beta, n, mtype, order, dim, orientation, hp_filter);
  % disp(['I passed the rir_generator'])
   information = struct('beta', beta, 'x_position', x_position,...
        'n_position', n_position,'mic_position', mic_position_final,...
        'room_dim', L, 'angleOrientation', angleOrientation,...
        'angle_x', angle, 'angle_n', angle_n, 'radius', radius);
    
   if PlotRoom
       f = figure('visible', 'on');
       p = zeros(5);
       hold on;
       % Plot room dimensions 
       p(1) = line([0,L(1)],[L(2),L(2)]);
       line([L(1),L(1)],[0,L(2)]);  
       line([0,L(1)],[0,0]);
       line([0,0],[0,L(2)]);
       % Plot mic array position
       p(2) = plot(mic_position_final(:,1),mic_position_final(:,2),'xr');
       % Plot the center of the mic array 
       p(3) = plot(center(1),center(2),'*g');       
       % Plot x (source) position 
       p(4) = plot(x_position(1),x_position(2),'m*');
       line([x_position(1),center(1)],[x_position(2),center(2)]);
       text((x_position(1)+center(1))/2,(x_position(2)+center(2))/2,num2str(round(angle,2)))
       % Plot n (noise) position 
       p(5) = plot(n_position(1),n_position(2),'c*');
       line([n_position(1),center(1)],[n_position(2),center(2)]);
       text((n_position(1)+center(1))/2,(n_position(2)+center(2))/2,num2str(round(angle_n,2)))
       legend(p(1:5),'room dim','mic position','center','x position','n position','Location','northeastoutside')
       saveas(f,'room_original.png');
   end 
end
