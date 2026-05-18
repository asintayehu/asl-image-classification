% Classifier : Soft-Margin SVM (linear kernel, BoxConstraint C=1)
% Strategy   : One-vs-All (OAA)

%% 0. Load Data
train_directory     = fullfile('archive', 'Train');
test_directory      = fullfile('archive', 'Test');
NUM_TRAIN_INSTANCES = 27455;
NUM_TEST_INSTANCES  = 7172;

[train_matrix, train_labels] = data_loader(train_directory, NUM_TRAIN_INSTANCES);
[test_matrix,  test_labels]  = data_loader(test_directory,  NUM_TEST_INSTANCES);

X_train = train_matrix;
X_test  = test_matrix;
Y_train = train_labels(:);
Y_test  = test_labels(:);

%% 1. Train OAA Soft-Margin SVMs
% One binary SVM is trained per class:
%   positive (+1) = samples belonging to class c
%   negative (-1) = all other samples

classes     = {'A','B','C','D','E','F','G','H','I','K','L','M', ...
               'N','O','P','Q','R','S','T','U','V','W','X','Y'};
num_classes = numel(classes);
Mdl         = cell(num_classes, 1);

for c = 1:num_classes
    Ybin          = double(Y_train == c);
    Ybin(Ybin==0) = -1;
    Mdl{c} = fitcsvm(X_train, Ybin, ...
                     'Standardize',    true, ...
                     'BoxConstraint',  1,    ...
                     'KernelFunction', 'linear');
end

%% 2. Predict on Test Set
% Each SVM returns a decision score. The class with the highest score wins.
scores = zeros(size(X_test,1), num_classes);
for c = 1:num_classes
    [~, s]      = predict(Mdl{c}, X_test);
    scores(:,c) = s(:,2);
end
[~, Y_pred] = max(scores, [], 2);

%% 3. Error Metrics

con_mat = confusionmat(Y_test, Y_pred);

% Metric 1: Accuracy
accuracy = sum(Y_pred == Y_test)/numel(Y_test);

% Metric 2: Precision
precision_per_class = diag(con_mat) ./ sum(con_mat,1)';
precision = mean(precision_per_class);

% Metric 3: Recall
recall_per_class = diag(con_mat) ./ sum(con_mat,2)';
recall = mean(recall_per_class);

fprintf("Accuracy: %.3f\n", accuracy);
fprintf("Precision: %.3f\n", precision);
fprintf("Recall: %.3f\n", recall);
