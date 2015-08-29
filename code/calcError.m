function [rmse, mae] = calcError(maskData, labels)
global predictedInteractionMatrix;
diff = labels - predictedInteractionMatrix;
mask = false(250, 315);
mask((maskData(:, 2)-1)*(size(mask, 1))+ maskData(:, 1)) = true;
rmse = sqrt(size(find(mask(diff ~= 0) == 1), 1)/size(diff, 1));
mae = size(find(mask(diff ~= 0) == 1), 1)/size(diff, 1);
clear summ;clear diff; clear tp;
end
