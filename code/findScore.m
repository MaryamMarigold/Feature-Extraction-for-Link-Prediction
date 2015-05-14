function [score] = findScore(data, clusterLists)
global model1;
sv = model1.SupportVectors;
alphaHat = model1.Alpha;
bias = model1.Bias;
kfun = model1.KernelFunction;
kfunargs = model1.KernelFunctionArgs;
for i = 1: size(data, 1)
    tempData = repmat(data(i, :)', 1, size(clusterLists, 1));
    score(i, :) = kfun(sv, [clusterLists, tempData], kfunargs{:})'*alphaHat(:) + bias;
end
end
