function [scores] = featureExtraction(data, otherDomainData, k, row, count1, count2)
[IDX, C] = kmeans(data, k,'start','sample', 'emptyaction','singleton');
% [C, IDX] = vl_kmeans(data, k, 'Initialization', 'plusplus');
[rows, ~] = find(isnan(C) == 1);
C(rows, :) = [];
[labels, decisionvalue] = findScore(otherDomainData, C, row);
global predictedInteractionMatrix;
distinctCrulters = unique(IDX);
kp = size(distinctCrulters, 1);
if(row)
    for i = 1: size(otherDomainData, 1)
        iperim = rem(i, count2);
        if(iperim == 0)
            iperim = count2;
        end
        for j = 1:kp
            indexes = find(IDX == distinctCrulters(j, 1));
            if(~isempty(indexes))
                indexes = rem(indexes, count1);
                indexes(find(indexes == 0)) = count1;
                predictedInteractionMatrix(iperim, indexes) = max(max(labels((i-1)*k+j,1), 0), predictedInteractionMatrix(iperim, indexes));
            end
        end
    end
else
    for i = 1: size(otherDomainData, 1)
        iperim = rem(i, count2);
        if(iperim == 0)
            iperim = count2;
        end
        for j = 1:kp
            indexes = find(IDX == distinctCrulters(j, 1));
            if(~isempty(indexes))
                indexes = rem(indexes, count1);
                indexes(find(indexes == 0)) = count1;
                predictedInteractionMatrix(indexes, iperim) =max(max(labels((i-1)*k+j,1), 0), predictedInteractionMatrix(indexes, iperim));
            end
        end
    end
end
scores = zeros(count2, k);
for i = 1: size(otherDomainData, 1)
        iperim = rem(i, count2);
        if(iperim == 0)
            iperim = count2;
        end
    scores(iperim, :) = decisionvalue((i-1)*k+1:i*k, 1)';
end
clear IDX; clear C; clear indexes; clear distinctCrulters; clear rows; clear labels; clear decisionvalueV; clear iperim;
end
