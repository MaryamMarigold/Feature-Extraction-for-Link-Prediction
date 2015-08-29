function [prec, recall] = calcPrecRecall(maskData, labels)
global predictedInteractionMatrix;
summ = labels+predictedInteractionMatrix;
diff = labels - predictedInteractionMatrix;
mask = false(250, 315);
mask((maskData(:, 2)-1)*(size(mask, 1))+ maskData(:, 1)) = true;
tp = size(find((mask(summ == 2) == 1)), 1);
prec = tp/size(find(mask(labels == 1) == 1), 1);
recall = tp/size(find(mask(diff == 0) == 1), 1);
clear summ;clear diff; clear tp;
end
