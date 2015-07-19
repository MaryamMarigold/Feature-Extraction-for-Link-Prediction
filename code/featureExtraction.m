function [scores] = featureExtraction(data, otherDomainData, k, row, counts)
[IDX, C] = kmeans(data, k,'start','sample', 'emptyaction','singleton');
% [C, IDX] = vl_kmeans(data, k, 'Initialization', 'plusplus');
[rows, ~] = find(isnan(C) == 1);
C(rows, :) = [];
[labels, decisionvalue] = findScore(otherDomainData, C, row);
global predictedInteractionMatrix;
if(row)
    for i = 1: size(otherDomainData, 1)
        for j = 1:k
            indexes = find(IDX == j);
            if(~isempty(indexes))
                indexes = rem(indexes, counts);
                indexes = indexes+1;
                predictedInteractionMatrix(i, indexes) = max(labels((i-1)*k+j,1), 0);
            end
        end
    end
end
for i = 1: size(otherDomainData, 1)
    scores(i, :) = decisionvalue((i-1)*k+1:i*k, 1);
end
end
