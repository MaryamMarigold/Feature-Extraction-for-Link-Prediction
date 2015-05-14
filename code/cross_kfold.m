function bestDegree = cross_kfold(trainX,trainY,n_fold)

degrees = [2, 4, 6, 8];
size_degree = numel(degrees);
cv_error = zeros(1, size_degree);

for i=1:size_degree
    cv_err_arr = zeros(1, n_fold);
    size_each_part = floor(size(trainX,1)/n_fold);
    rand_map = randperm(size(trainX,1));
    %     for fold = 1: n_fold
    fold = 3;
    cv_test_idx = rand_map((fold-1)*size_each_part+1:(fold)*size_each_part);
    cv_train_idx = true(size(trainX,1),1);
    cv_train_idx(cv_test_idx) = false;
    cv_trainX = trainX(cv_train_idx,:);
    cv_trainY = trainY(cv_train_idx,:);
    cv_testX = trainX(cv_test_idx,:);
    cv_testY = trainY(cv_test_idx,:);
    [samples, features] = size(cv_trainX);
    cv_trainX = [ones(samples, 1), cv_trainX];
    [samplestest, features] = size(cv_testX);
    cv_testX = [ones(samplestest, 1), cv_testX];
    options = optimset('maxiter',10000000);
    disp(degrees(i));
    svmModel = svmtrain(cv_trainX, cv_trainY, 'BoxConstraint',1e-1, 'Kernel_Function','polynomial','Polyorder',degrees(i),'options',options);
    newvalue= svmclassify(svmModel, cv_testX);
    %     cv_err_arr(fold)=sqrt(mean((newvalue-cv_testY).^2));
    %     end
    %     cv_error(i) = mean(cv_err_arr);
    cv_error(i) = sqrt(mean((newvalue-cv_testY).^2));
end
idxMin = 1;
for i=2:size_degree
    if(cv_error(i)<=cv_error(idxMin))
        idxMin =  i;
    end
end
bestDegree=degrees(idxMin);
end
