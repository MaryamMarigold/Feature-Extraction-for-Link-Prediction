function [c, bestDegree] = cross_kfold(trainX,trainY,n_fold)
degrees = [6, 8];
size_degree = numel(degrees);
cv_error = zeros(1, size_degree);
for i=1:size_degree
    cv_err_arr = zeros(1, n_fold);
    size_each_part = floor(size(trainX,1)/n_fold);
    rand_map = randperm(size(trainX,1));
    %     for fold = 1: n_fold
    fold = 1;
    cv_test_idx = rand_map((fold-1)*size_each_part+1:(fold)*size_each_part);
    cv_train_idx = true(size(trainX,1),1);
    cv_train_idx(cv_test_idx) = false;
    cv_trainX = trainX(cv_train_idx,:);
    cv_trainY = trainY(cv_train_idx,:);
    cv_testX = trainX(cv_test_idx,:);
    cv_testY = trainY(cv_test_idx,:);
    [samples, ~] = size(cv_trainX);
    cv_trainX = [ones(samples, 1), cv_trainX];
    [samplestest, ~] = size(cv_testX);
    cv_testX = [ones(samplestest, 1), cv_testX];
    str = sprintfc('-t 0 , -h 0 , -d %d', degrees(i));
    model1 = svmtrain(cv_trainY, cv_trainX, str(1, 1));
    disp(degrees(i));
    [newvalue, ~, ~] = svmpredict(cv_testY, cv_testX, model1);
    cv_err_arr(fold)=sqrt(mean((newvalue-cv_testY).^2));
    %     end
    cv_error(i) = mean(cv_err_arr);
    cv_error(i) = sqrt(mean((newvalue-cv_testY).^2));
    clear model1; clear cv_trainX; clear size_each_part; clear rand_map; clear cv_test_idx; clear cv_trainY; clear cv_testX; clear cv_testX; clear newvalue; clear cv_test_idx; clear cv_train_idx;
end
idxMin = 1;
for i=2:size_degree
    if(cv_error(i)<=cv_error(idxMin))
        idxMin =  i;
    end
end
 clear cv_err_arr;clear cv_error;
 cv_error = zeros(1, 4);
bestDegree=degrees(idxMin);
for i=1:-1:-2
    cv_err_arr = zeros(1, n_fold);
    size_each_part = floor(size(trainX,1)/n_fold);
    rand_map = randperm(size(trainX,1));
    %     for fold = 1: n_fold
    fold = 1;
    cv_test_idx = rand_map((fold-1)*size_each_part+1:(fold)*size_each_part);
    cv_train_idx = true(size(trainX,1),1);
    cv_train_idx(cv_test_idx) = false;
    cv_trainX = trainX(cv_train_idx,:);
    cv_trainY = trainY(cv_train_idx,:);
    cv_testX = trainX(cv_test_idx,:);
    cv_testY = trainY(cv_test_idx,:);
    [samples, ~] = size(cv_trainX);
    cv_trainX = [ones(samples, 1), cv_trainX];
    [samplestest, ~] = size(cv_testX);
    cv_testX = [ones(samplestest, 1), cv_testX];
    str = sprintfc('-t 0 , -h 0 , -d %d', bestDegree);
    str = strcat(str, ' ', sprintfc(' ,-c %d', 10^i));
    model1 = svmtrain(cv_trainY, cv_trainX, str(1, 1));
    [newvalue, ~, ~] = svmpredict(cv_testY, cv_testX, model1);
    cv_err_arr(fold, 1)=sqrt(mean((newvalue-cv_testY).^2));
    cv_error(i+3, 1) = mean(cv_err_arr);
    cv_error(i+3, 1) = sqrt(mean((newvalue-cv_testY).^2));
    clear model1; clear cv_trainX; clear size_each_part; clear rand_map; clear cv_test_idx; clear cv_trainY; clear cv_testX; clear cv_testX; clear newvalue; clear cv_test_idx; clear cv_train_idx;
end
idxMin = 1;
for i=1:4
    if(cv_error(i)<=cv_error(idxMin))
        idxMin =  i;
    end
end
c = 10^(idxMin-3);
clear cv_err_arr;clear cv_error;
end
 
