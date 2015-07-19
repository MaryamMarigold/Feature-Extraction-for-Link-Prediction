load('..\data\Perlman_Data');
load('..\data\New_Interactions');
addpath('C:\Program Files\MATLAB\R2012b\toolbox\libsvm-3.20\matlab');
addpath('C:\Program Files\MATLAB\R2012b\toolbox\vlfeat-0.9.20-bin\vlfeat-0.9.20');
%%%%%%%%%%%  drug mdScale
drug_finalDimention = 10;
drug_feature(1, :) = num2cell((mdscale(DrugSim_ATCHierDrugsCommonSimilarityMat, drug_finalDimention, 'criterion','metricstress'))', [1 size(DrugSim_ATCHierDrugsCommonSimilarityMat, 1)]);
drug_feature(2, :) = num2cell((mdscale(DrugSim_chemicalDrugsCommonSimilarityMat, drug_finalDimention, 'criterion','metricstress'))', [1 size(DrugSim_ATCHierDrugsCommonSimilarityMat, 1)]);
drug_feature(3, :) = num2cell((mdscale(DrugSim_ligandJaccardDrugsCommonSimilarityMat, drug_finalDimention, 'criterion','metricstress'))', [1 drug_finalDimention]);
drug_feature(4,:) = num2cell((mdscale(DrugSim_newCMapJaccardDrugsCommonSimilarityMat, drug_finalDimention, 'criterion','metricstress'))', [1 size(DrugSim_ATCHierDrugsCommonSimilarityMat, 1)]);
drug_feature(5, :) = num2cell((mdscale(DrugSim_pSideEffectDrugsCommonSimilarityMat, drug_finalDimention, 'criterion','metricstress'))', [1 size(DrugSim_ATCHierDrugsCommonSimilarityMat, 1)]);
%%%%%%%%%%%%   target mdsScale
target_finalDimention = 10;
target_feature(1, :) = num2cell((mdscale(TargetSim_GOTargetsCommonSimilarityMat, target_finalDimention, 'criterion','metricstress'))', [1 size(TargetSim_GOTargetsCommonSimilarityMat, 1)]);
target_feature(2, :) = num2cell((mdscale(TargetSim_distTargetsCommonSimilarityMat, target_finalDimention, 'criterion','metricstress'))', [1 size(TargetSim_GOTargetsCommonSimilarityMat, 1)]);
target_feature(3, :) = num2cell((mdscale(TargetSim_seqTargetsCommonSimilarityMat, target_finalDimention, 'criterion','metricstress'))', [1 size(TargetSim_GOTargetsCommonSimilarityMat, 1)]);
lastFeatureIndexDrug = 5;
lastFeatureIndexTarget = 3;

% Setting the parameters
nDrugs = 315; % No of drugs
nTargets = 250; % No of targets
num_folds = 10; % No of folds for cross validation
indices =  PSLFolds(:,3); % Setting the folds to the same one used with PSL
global predictedInteractionMatrix;
global model1;
for currentFold = 1:num_folds,
    test = ((indices == currentFold) | (indices == -1));
    train = ~test;
    diffSize = size(Interactions_Drug_Target, 1)- size(train, 1);
    if(diffSize < 0)
        Interactions_Drug_Target_temp = [zeros(abs(diffSize), 2);Interactions_Drug_Target];
    else
        train = [train, zeros(diffSize, 1)];
        test = [test, zeros(diffSize, 1)];
    end
    train_Drug_Target_Interaction = intersect(Interactions_Drug_Target_temp, PSLFolds(find(train == 1), 1:2),'rows');
    test_Drug_Target_Interaction = intersect(Interactions_Drug_Target_temp, PSLFolds(find(test == 1), 1:2),'rows');
    newInteractions = NewInteractionFolds(find(NewInteractionFolds(:, 3) == currentFold), 1:2);
    diffSize2 = size(newInteractions, 1)- size(train, 1);
    if(diffSize2 < 0)
        newInteractions_temp = [zeros(abs(diffSize2), 2);newInteractions];
    else
        train = [train, zeros(diffSize2, 1)];
        test = [test, zeros(diffSize2, 1)];
    end
    train_newInteractions = intersect(newInteractions_temp, newInteractions,'rows');
    test_newInteractions = intersect(newInteractions_temp, newInteractions,'rows');
    xPosTrain = []; xPosTest = []; xNegTrain = []; xNegTest = [];
    [rowZeroIdx, colZeroIdx] = find(Interactions_Matrix == 0);
    diffSize = size(rowZeroIdx, 1)- size(train, 1);
    if(diffSize < 0)
        zeroIndexes = [zeros(abs(diffSize), 2); [rowZeroIdx, colZeroIdx]];
    else
        train = [train, zeros(diffSize, 1)];
        test = [test, zeros(diffSize, 1)];
    end
    train_zeroIndexes = intersect(zeroIndexes, PSLFolds(find(train == 1), 1:2),'rows');
    test_zeroIndexes = intersect(zeroIndexes, PSLFolds(find(test == 1), 1:2),'rows');
    randomIndx = randperm(size(train_zeroIndexes, 1));
    ZeroIndexes = train_zeroIndexes(randomIndx, :);
    drug = []; target = [];
    for i = 1:5
        drug =  [drug, drug_feature(i, :)];
        for j = 1:3
            xPosTrain = [xPosTrain, cell2mat([target_feature(j, train_Drug_Target_Interaction(:,2));...
                drug_feature(i, train_Drug_Target_Interaction(:,1))])];
            zeroIndexNum = min(size(ZeroIndexes, 1), size(xPosTrain, 2)) - size(xNegTrain, 2);
            xNegTrain = [xNegTrain, cell2mat([target_feature(j, ZeroIndexes(1:zeroIndexNum,2));...
                drug_feature(i, ZeroIndexes(1:zeroIndexNum,1))])];
            target =  [target, target_feature(j, :)];
        end
    end
    zeroIndexNumTest = min(size(test_zeroIndexes, 1), size(xPosTest, 2)) - size(xNegTest, 2);
    for featureNum = 80:-20:20
        predictedInteractionMatrix = zeros(nTargets, nDrugs);
        xTrain = [xPosTrain, xNegTrain]';
        yTrain = [ones(size(xPosTrain,2), 1); repmat(-1,size(xNegTrain,2), 1)];
        xTest = cell2mat(drug_feature(lastFeatureIndexDrug, unique(PSLFolds(find(test == 1), 1))));
        %     bestDegree = cross_kfold(xTrain, yTrain, 15);
        model1 = svmtrain(yTrain, xTrain, '-t 0, -h 0');
        targetFeature = featureExtraction(cell2mat(drug)',...
            cell2mat(target_feature(lastFeatureIndexTarget,:))',...
            featureNum, 1, nDrugs);
        lastFeatureIndexTarget = lastFeatureIndexTarget+1;
        target_feature(lastFeatureIndexTarget, :) = num2cell(targetFeature', [1 size(targetFeature, 2)]);
        target = target_feature(lastFeatureIndexTarget, :);
        drugFeature = featureExtraction(cell2mat(target)',...
            cell2mat(drug_feature(lastFeatureIndexDrug,:))',...
            featureNum, 0, nTargets);
        lastFeatureIndexDrug = lastFeatureIndexDrug+1;
        drug_feature(lastFeatureIndexDrug, :) = num2cell(drugFeature', [1 size(drugFeature, 2)]);
        [rows, cols] = find(predictedInteractionMatrix == 1);
        train_prediction = [cols, rows];
        %%%%testing and reporting test and train calculation
        predictedInteractionMatrixTest = testFeatureExtraction(xTest, cell2mat(target_feature(lastFeatureIndexTarget-1,:))',...
            featureNum, nTargets, nDrugs);
        [testrows, testcols] = find(predictedInteractionMatrixTest == 1);
        testPredixtion = [testcols, testrows];
        test_distract = intersect(testPredixtion, test_newInteractions, 'rows');
        train_distract = intersect(train_prediction, train_newInteractions, 'rows');

        %%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
        clear model1;clear train_Drug_Target_Interaction; clear xPosTrain; clear xNegTrain; clear drug; clear target;
        train_Drug_Target_Interaction = [cols, rows];

        drug =  drug_feature(lastFeatureIndexDrug, :);
        xPosTrain = cell2mat([target_feature(lastFeatureIndexTarget, train_Drug_Target_Interaction(:,2));...
            drug_feature(lastFeatureIndexDrug, train_Drug_Target_Interaction(:,1))]);
        [rowZeroIdx, colZeroIdx] = find(predictedInteractionMatrix == 0);
        zeroIndexNum = min(size(rowZeroIdx, 1), size(xPosTrain, 2));
        diffSize = zeroIndexNum- size(find(train == 1), 1);
        zeroIndexes = [rowZeroIdx, colZeroIdx];
        if(diffSize < 0)
            zeroIndexes = [[rowZeroIdx(1:zeroIndexNum, 1), colZeroIdx(1:zeroIndexNum, 1)]; zeros(abs(diffSize), 2)];
        else
            train = [train, zeros(diffSize, 1)];
            test = [test, zeros(diffSize, 1)];
        end
        train_zeroIndexes = intersect(zeroIndexes(1:zeroIndexNum, :), PSLFolds(find(train == 1), 1:2),'rows');
        test_zeroIndexes = intersect(zeroIndexes(1:zeroIndexNum, :), PSLFolds(find(test == 1), 1:2),'rows');
        randomIndx = randperm(size(train_zeroIndexes, 1));
        zeroIndexes = train_zeroIndexes(randomIndx, :);
        xNegTrain = cell2mat([target_feature(lastFeatureIndexTarget, zeroIndexes(:,2)); drug_feature(lastFeatureIndexDrug, zeroIndexes(:,1))]);
        target =  target_feature(lastFeatureIndexTarget, :);
        zeroIndexNum = min(size(zeroIndexes, 1), size(xPosTrain, 2));
    end
end

