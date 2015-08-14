function [predictedInteractionMatrixTest] = testFeatureExtraction(data, otherDomainData, k, count1, count2)
[IDX, C] = kmeans(data, k, 'start', 'sample', 'emptyaction','drop');
% [C, IDX] = vl_kmeans(data, k, 'Initialization', 'plusplus');
[row, ~] = find(isnan(C) == 1);
C(row, :) = [];
[labels, ~] = findScore(otherDomainData, C, 1);
distinctCrulters = unique(IDX);
k = size(distinctCrulters, 1);
predictedInteractionMatrixTest = zeros(count2, count1);
for i = 1: size(otherDomainData, 1)
    iperim = rem(i, count2);
        if(iperim == 0)
            iperim = count2;
        end
    for j = 1:k
        indexes = find(IDX == distinctCrulters(j, 1));
        indexes = rem(indexes, count1);
        indexes(find(indexes == 0)) = count1;
        predictedInteractionMatrixTest(iperim, indexes) = max(max(labels((i-1)*k+j,1), 0),...
            predictedInteractionMatrixTest(iperim, indexes));
    end
end
clear IDX; clear C; clear indexes; clear distinctCrulters;
end
