%% Function that wraps the preprocessing together

[train_matrix, train_labels] = data_loader(train_directory, NUM_TRAIN_INSTANCES);
[test_matrix, test_labels] = data_loader(test_directory, NUM_TEST_INSTANCES);

% To use the matrices in further tasks, uncomment the below code
% save('asl_data.mat', 'train_matrix', 'train_labels', 'test_matrix', 'test_labels');
% load('asl_data.mat');

% After doing this, you have the data you need to feed to classifiers.
