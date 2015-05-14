load('C:\Users\Administrator\Desktop\Link Prediction Problem\data\Perlman_Data');
%%%%%%%  drug cmdsScale, fining min dim for reduction
Y_DrugSim_ATCHierDrugsCommonSimilarityMat = cmdscale(DrugSim_ATCHierDrugsCommonSimilarityMat);
% Y_DrugSim_chemicalDrugsCommonSimilarityMat = cmdscale(DrugSim_chemicalDrugsCommonSimilarityMat);
% Y_DrugSim_ligandJaccardDrugsCommonSimilarityMat = cmdscale(DrugSim_ligandJaccardDrugsCommonSimilarityMat);
% Y_DrugSim_newCMapJaccardDrugsCommonSimilarityMat = cmdscale(DrugSim_newCMapJaccardDrugsCommonSimilarityMat);
% Y_DrugSim_pSideEffectDrugsCommonSimilarityMat = cmdscale(DrugSim_pSideEffectDrugsCommonSimilarityMat);
% drug_dimentionSizes = [size(Y_DrugSim_ATCHierDrugsCommonSimilarityMat, 2),...
%                      size(Y_DrugSim_ATCHierDrugsCommonSimilarityMat, 2),...
%                      size(Y_DrugSim_ATCHierDrugsCommonSimilarityMat, 2),...
%                      size(Y_DrugSim_ATCHierDrugsCommonSimilarityMat, 2),...
%                      size(Y_DrugSim_ATCHierDrugsCommonSimilarityMat, 2)];
% drug_finalDimention = max(drug_dimentionSizes);
drug_finalDimention = size(Y_DrugSim_ATCHierDrugsCommonSimilarityMat, 2);
drug_feature(1, :) = num2cell((mdscale(DrugSim_ATCHierDrugsCommonSimilarityMat, drug_finalDimention, 'criterion','metricstress'))', [1 size(DrugSim_ATCHierDrugsCommonSimilarityMat, 1)]);
% drug_feature(2, :) = num2cell((mdscale(DrugSim_chemicalDrugsCommonSimilarityMat, drug_finalDimention, 'criterion','metricstress'))', [1 size(DrugSim_ATCHierDrugsCommonSimilarityMat, 1)]);
% drug_feature(3, :) = num2cell((mdscale(DrugSim_ligandJaccardDrugsCommonSimilarityMat, drug_finalDimention, 'criterion','metricstress'))', [1 drug_finalDimention]);
% drug_feature(4,:) = num2cell((mdscale(DrugSim_newCMapJaccardDrugsCommonSimilarityMat, drug_finalDimention, 'criterion','metricstress'))', [1 size(DrugSim_ATCHierDrugsCommonSimilarityMat, 1)]);
% drug_feature(5, :) = num2cell((mdscale(DrugSim_pSideEffectDrugsCommonSimilarityMat, drug_finalDimention, 'criterion','metricstress'))', [1 size(DrugSim_ATCHierDrugsCommonSimilarityMat, 1)]);
%%%%%%%%%%%   target smdScale, fining min dim for reduction
Y_TargetSim_GOTargetsCommonSimilarityMat = cmdscale(TargetSim_GOTargetsCommonSimilarityMat);
% Y_TargetSim_distTargetsCommonSimilarityMat = cmdscale(TargetSim_distTargetsCommonSimilarityMat);
% Y_TargetSim_seqTargetsCommonSimilarityMat = cmdscale(TargetSim_seqTargetsCommonSimilarityMat);
% target_dimentionSizes = [size(Y_TargetSim_GOTargetsCommonSimilarityMat, 2),...
%                      size(Y_TargetSim_distTargetsCommonSimilarityMat, 2),...
%                      size(Y_TargetSim_seqTargetsCommonSimilarityMat, 2)];
% target_finalDimention = max(target_dimentionSizes);
target_finalDimention = size(Y_TargetSim_GOTargetsCommonSimilarityMat, 2);
%%%%%%%%%%%%   target mdsScale
target_feature(1, :) = num2cell((mdscale(TargetSim_GOTargetsCommonSimilarityMat, target_finalDimention, 'criterion','metricstress'))', [1 size(TargetSim_GOTargetsCommonSimilarityMat, 1)]);
% target_feature(2, :) = num2cell((mdscale(TargetSim_distTargetsCommonSimilarityMat, target_finalDimention, 'criterion','metricstress'))', [1 size(TargetSim_GOTargetsCommonSimilarityMat, 1)]);
% target_feature(3, :) = num2cell((mdscale(TargetSim_seqTargetsCommonSimilarityMat, target_finalDimention, 'criterion','metricstress'))', [1 size(TargetSim_GOTargetsCommonSimilarityMat, 1)]);
lastFeatureIndexDrug = 1;
lastFeatureIndexTarget = 1;
%%%%%%%%%%% create new features

global model1;
% for featureNum = 1 : 3
    xPostrain = cell2mat([target_feature(lastFeatureIndexTarget, Interactions_Drug_Target(:,2));...
                                        drug_feature(lastFeatureIndexDrug, Interactions_Drug_Target(:,1))]);
    [rowZerIdx, colZeoIdx] = find(Interactions_Matrix == 0);
    randomIndx = randperm(size(rowZerIdx, 1));
    ZeroIndexes = [rowZerIdx(randomIndx, 1), colZeoIdx(randomIndx, 1)];
    xNegtrain = cell2mat([target_feature(lastFeatureIndexTarget, ZeroIndexes(1:size(xPostrain, 2),2)); drug_feature(lastFeatureIndexDrug, ZeroIndexes(1:size(xPostrain, 2),1))]);
    xTrain = [xPostrain, xNegtrain]';
    yTrain = [ones(size(xPostrain,2), 1); repmat(-1,size(xPostrain,2), 1)];
    bestDegree = cross_kfold(xTrain, yTrain, 15);
    model1 = svmtrain(xTrain, yTrain, 'BoxConstraint',1e-1, 'Kernel_Function','polynomial','Polyorder', bestDegree);
    targetFeature = featureExtraction(cell2mat(drug_feature(lastFeatureIndexDrug, Interactions_Drug_Target(:,1)))',...
                                      cell2mat(target_feature(lastFeatureIndexTarget,:))',...
                                      target_finalDimention);
    lastFeatureIndexTarget = lastFeatureIndexTarget+1;
    target_feature(lastFeatureIndexTarget, :) = num2cell(targetFeature', [1 size(targetFeature, 2)]);
    drugFeature = featureExtraction(cell2mat(target_feature(lastFeatureIndexTarget, Interactions_Drug_Target(:,2)))',...
                                    cell2mat(drug_feature(lastFeatureIndexDrug,:))',...
                                    drug_finalDimention);
    lastFeatureIndexDrug = lastFeatureIndexDrug+1;
    drug_feature(lastFeatureIndexDrug, :) = num2cell(drugFeature', [1 size(drugFeature, 2)]);
% end
