% Outline
% Take existing training and testing arrays and recombine into a single
% dataset, retaining labels on the images
% Take this dataset and split into k subsets, approximately evenly across
% labeled letters
% Apply our SVM model to the split for each of the k subsets, rotating
% through each with k-1 training sets and 1 testing set
% Compute accuracy of each by comparing sorted labels for data with true
% labels for data
% Take average of errors/accuracy for each runthrough and result is our KF
% CV result
% Compute for k= 1...6 and communicate the result with the lowest
% error/highest accuracy
% This result is the optimal training/testing split

% Combine datasets to put everything together
total_data = [train_matrix; test_matrix];
total_labels = [train_labels, test_labels];

% Put labels together with data to allow to sort and put classes together

total_labels = total_labels' ;
labeled_data = [total_labels, total_data];

% Now that we have our data and labels together, we can sort our array to
% ensure that all ones are together, etc

sorted_ldata = sortrows(labeled_data,[1]); 
% The one in brackets forces the function to NOT inpliment a tiebreaker 
% sort into columns 2 and on when the first column has the same values, in 
% this case 1 for a number of values

% Now that our data is all in one array, lets split the data into groups of
% arbitrary size, each including digits 

% Define number of groups for cross validation

K=4

% Using this function, we can generate n folds of data which mirrors the
% proportion of a larger dataset

all_true = [];
all_pred = [];

X = sorted_ldata(:,2:785);
Y = sorted_ldata(:,1);

c = cvpartition(Y, 'KFold',K);

acc = zeros(c.NumTestSets, 1);

for k = 1:c.NumTestSets
    tic

    trainIdx = training(c, k);
    testIdx  = test(c, k);

    X_Train = X(trainIdx, :);
    Y_Train = Y(trainIdx);

    X_Test = X(testIdx, :);
    Y_Test = Y(testIdx);

    Mdl = cell(num_classes, 1);

    for C = 1:num_classes
        Ybin = double(Y_Train == C);
        Ybin(Ybin == 0) = -1;

        Mdl{C} = fitcsvm(X_Train, Ybin, ...
            'Standardize', true, ...
            'BoxConstraint', 1, ...
            'KernelFunction', 'linear');
    end

    scores = zeros(size(X_Test,1), num_classes);
    for C = 1:num_classes
        [~, s] = predict(Mdl{C}, X_Test);
        scores(:,C) = s(:,2);
    end

    [~, Ypred] = max(scores, [], 2);
    

    fprintf('Fold %d complete in %.2f seconds\n', k, toc)
end

acctot = mean(acc)
