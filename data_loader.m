function [data_matrix, label_vector] = data_loader(file_directory, INSTANCES)
%DATA_LOADER: Processing and loading data
% Author: Aaron Sintayehu
% ---------------------- Purpose of this code --------------------------- 
% In this file we aim to preprocess our data by separating it into testing
% and training sets. This fortunately was completed by the author who
% created our data, and we explor it later on.
% 
% Our training matrix is a 34627 x 784 matrix, wherein each row is an 
% observation, and each column of any row represents the i-th pixel of 
% that observation.
%
% Our label vector, of size 1x34627 is indexed in parallel with our
% training matrix. This is, the true class of the i-th row of our training
% matrix corresponds with the i-th entry in our label vector. Our label 
% vector is encoded numerically, such that 1 maps to A, 2 maps to B, and so
% on, excluding J and Z, respectively.

% We flatten our 28x28 matrix into a 1x784 vector of pixels
NUM_FEATURES= 784;

% Observed classes
classes= {"A/", "B/", "C/", "D/", "E/", "F/", "G/", "H/", "I/", "K/", "L/", "M/", "N/", "O/", "P/", "Q/", "R/", "S/", "T/", "U/", "V/", "W/", "X/", "Y/"};


% Instantiating matrix which will hold our observations
data_matrix= zeros(INSTANCES, NUM_FEATURES);
label_vector= zeros(1, INSTANCES);


% Row counter
curr_row_idx= 0;

% Outer loop, over classes
for i=1:length(classes)

    % Initial file path building. Test/Y
    pre_file_path= fullfile(file_directory, classes{i});
    files= dir(fullfile(pre_file_path, "*.jpg"));

    % Inner loop, over images
    % Loop over dir(classes{i}) length
    for j=1:length(files)

        % Increment row counter
        curr_row_idx = curr_row_idx + 1;

        % Building file_path for image name
        % At this moment, we have Test/Y
        % After the below call, we have Test/Y/im_name.jpg
        file_path= strcat(pre_file_path, files(j).name);
        
        % Use global row counter to index train_matrix
        data_matrix(curr_row_idx,:)= reshape(imread(file_path), [1, NUM_FEATURES]);

        % Building label vector
        label_vector(curr_row_idx)= i;
    end
end

data_matrix= data_matrix/255;

end
