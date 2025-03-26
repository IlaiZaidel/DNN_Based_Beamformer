% Define the file path
filePath = '/dsi/gannot-lab1/datasets/Ilai_data/Train/feature_vector_20000.mat';

% Check if the file exists
if isfile(filePath)
    % Delete the file
    delete(filePath);
    disp('File deleted successfully.');
else
    disp('File does not exist.');
end