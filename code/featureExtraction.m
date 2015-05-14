function [score] = featureExtraction(data, otherDomainData, k)
[IDX,C] = kmeans(data, k,'start','sample', 'emptyaction','singleton');
C(isnan(C)) = 0;
score = findScore(otherDomainData, C);
end
