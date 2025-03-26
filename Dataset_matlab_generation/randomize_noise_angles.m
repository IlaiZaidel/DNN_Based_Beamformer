function [angle_n1, angle_n2] = randomize_noise_angles(Vangle)
    % Define possible angles excluding those too close to Vangle
    possible_angles = setdiff(0:1:180, Vangle-20:Vangle+20);
    
    % Randomly select the first noise angle
    idx1 = randi(numel(possible_angles));
    angle_n1 = possible_angles(idx1);
    
    % Exclude angles too close to the first noise angle
    possible_angles = setdiff(possible_angles, angle_n1-20:angle_n1+20);
    
    % If there are no valid angles left, throw an error
    if isempty(possible_angles)
        error('No valid noise angles available with the given constraints.');
    end
    
    % Randomly select the second noise angle
    idx2 = randi(numel(possible_angles));
    angle_n2 = possible_angles(idx2);
end
