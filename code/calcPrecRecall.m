function [prec, recall] = calcPrecRecall(labels, score)
summ = labels+score;
diff = labels - score;
tp = size(find(summ == 2), 1);
prec = tp/size(find(labels == 1), 1);
recall = tp/size(find(diff == 0), 1);
clear summ;clear diff; clear tp;
end
