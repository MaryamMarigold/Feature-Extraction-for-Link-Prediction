function [predictedInteractionMatrixTest] = testFeatureExtraction(data, otherDomainData, k, count1, count2)
diff = 1;
if(size(data, 1) <k)
    diff = k-size(data, 1);
    data = [zeros(diff, size(data, 2)); data];
end
[IDX, C] = kmeans(data, k, 'start', 'sample', 'emptyaction','drop');
% [C, IDX] = vl_kmeans(data, k, 'Initialization', 'plusplus');
 [row, ~] = find(isnan(C) == 1);
C(row, :) = [];
[labels, ~] = findScore(otherDomainData, C, 1);
distinctCrulters = unique(IDX);
k = size(distinctCrulters, 1);
predictedInteractionMatrixTest = zeros(count1, count2);
for i = 1: size(otherDomainData, 1)
    for j = 1:k
        indexes = find(find(IDX ==distinctCrulters(j, 1) ) >= diff);
        if(~isempty(indexes))
            indexes = rem(indexes, count1);
            indexes = indexes+1;
            predictedInteractionMatrixTest(i, indexes) = max(labels((i-1)*k+j,1), 0);
        end
    end
end
end
