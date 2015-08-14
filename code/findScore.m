function [labels, decisionvalue] = findScore(data, clusterLists, row)
global model1;
if(row)
    for i = 1: size(data, 1)
        tempData = repmat(data(i, :)', 1, size(clusterLists, 1));
        newData((i-1)*(size(clusterLists, 1))+1:(i)*(size(clusterLists, 1)), :) = [clusterLists, tempData'];
    end
else
    for i = 1: size(data, 1)
        tempData = repmat(data(i, :)', 1, size(clusterLists, 1));
        newData((i-1)*(size(clusterLists, 1))+1:(i)*(size(clusterLists, 1)), :) = [tempData', clusterLists];
    end
end
[labels, ~, decisionvalue] = ...
    svmpredict(zeros(size(newData, 1), 1), newData, model1);
clear newData;clear tempData;
end
