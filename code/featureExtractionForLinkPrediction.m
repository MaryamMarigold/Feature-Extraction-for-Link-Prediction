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
train_recall = zeros(4, 1); train_precision = zeros(4, 1); test_recall = zeros(4, 1); test_precision = zeros(4, 1);
for currentFold = 1:num_folds,
    test = ((indices == currentFold) | (indices == -1)); %test nodes
    train = ~test;%train nodes
    test_indexess = find(test == 1);
    train_indexess = find(train == 1);
    diffSize = size(train, 1)-size(Interactions_Drug_Target, 1);
    Interactions_Drug_Target_temp = [Interactions_Drug_Target;zeros(diffSize, 2)]; %intersection with the same size
    train_Drug_Target_Interaction = intersect(Interactions_Drug_Target_temp, ...
        PSLFolds(train_indexess, 1:2),'rows');%extracting train interactions for current fold
    %new interaction matrix for current fold:
    newInteractions = NewInteractionFolds(find(NewInteractionFolds(:, 3) == currentFold), 1:2);%new interactions of current fold
    diffSize2 = size(train, 1)-size(newInteractions, 1);
    newInteractions_temp = [newInteractions;zeros(diffSize2, 2)]; %intersection with the same size
    train_newInteractions = intersect(newInteractions_temp, PSLFolds(train_indexess, 1:2),'rows');%train new interactions
    test_newInteractions = intersect(newInteractions_temp, PSLFolds(test_indexess, 1:2),'rows');%test new interactions
    trainNewInteractionMat = zeros(nTargets, nDrugs); %train new interactions matrix
    testNewInteractionMat = zeros(nTargets, nDrugs); %test new interactions matrix
    trainNewInteractionMat(sub2ind(size(trainNewInteractionMat),train_newInteractions(:, 2), train_newInteractions(:, 1))) = 1;
    testNewInteractionMat(sub2ind(size(testNewInteractionMat),test_newInteractions(:, 2), test_newInteractions(:, 1))) = 1;
    
    xPosTrain = []; xNegTrain = [];
    [rowZeroIdx, colZeroIdx] = find(Interactions_Matrix == 0);
    diffSize = size(train, 1)-size(rowZeroIdx, 1);
    zeroIndexes = [rowZeroIdx, colZeroIdx];
    zeroIndexes = [zeroIndexes;zeros(diffSize, 2)];
    train_zeroIndexes = intersect(zeroIndexes, PSLFolds(train_indexess, 1:2),'rows');%train nodes without interaction
    randomIndx = randperm(size(train_zeroIndexes, 1));
    ZeroIndexesTrain = train_zeroIndexes(randomIndx, :);
    drug=[]; target=[]; drugTest=[]; xPosTrain=[]; xNegTrain = [];
    clear Interactions_Drug_Target_temp; clear newInteractions_temp; clear newInteractions;clear zeroIndexes;
    %train and test X with interaction and without interaction:
    target =  [target_feature(1, :), target_feature(2, :), target_feature(3, :)];
    for i = 1:5
        drug =  [drug, drug_feature(i, :)];
        drugTest = [drugTest, drug_feature(i, unique(PSLFolds(test_indexess, 1)))];
        for j = 1:3
            xPosTrain = [xPosTrain, cell2mat([target_feature(j, train_Drug_Target_Interaction(:,2));...
                drug_feature(i, train_Drug_Target_Interaction(:,1))])];
            zeroIndexNum = min(size(ZeroIndexesTrain, 1), size(xPosTrain, 2)) - size(xNegTrain, 2);
            xNegTrain = [xNegTrain, cell2mat([target_feature(j, ZeroIndexesTrain(1:zeroIndexNum,2));...
                drug_feature(i, ZeroIndexesTrain(1:zeroIndexNum,1))])];
        end
    end
    clear ZeroIndexesTrain;clear randomIndx;
    for featureNum = 80:-20:20
        predictedInteractionMatrix = zeros(nTargets, nDrugs);%predicted interaction for train
        trainSize =  min(size(xPosTrain, 2),size(xNegTrain, 2));
        xTrain = [xPosTrain(:, 1:trainSize), xNegTrain(:, 1:trainSize)]';%train X
        yTrain = [ones(trainSize, 1); repmat(-1,trainSize, 1)]; %train labels
        [c, bestDegree] = cross_kfold(xTrain, yTrain, 5); %finding best degree and c parameter for polynomial kernel
        str = sprintfc('-t 0, -h 0, -d %d', bestDegree);
        str = strcat(str, ' ', sprintfc(', -c %d', c));
        model1 = svmtrain(yTrain, xTrain, str(1, 1)); 
        drugFeature = featureExtraction(cell2mat(target)', cell2mat(drug)', featureNum, 0, nTargets, nDrugs);%calc new feature for drug
        targetFeature = featureExtraction(cell2mat(drug)', cell2mat(target)',...
            featureNum, 1, nDrugs, nTargets);%calc new feature for target
        lastFeatureIndexTarget = lastFeatureIndexTarget+1;
        lastFeatureIndexDrug = lastFeatureIndexDrug+1;
        target_feature(lastFeatureIndexTarget, :) = num2cell(targetFeature', [1 size(targetFeature, 2)]);
        drug_feature(lastFeatureIndexDrug, :) = num2cell(drugFeature', [1 size(drugFeature, 2)]);
        [rows, cols] = find(predictedInteractionMatrix == 1);
        %%%testing
        predictedInteractionMatrixTest = testFeatureExtraction(cell2mat(drugTest)', cell2mat(target)',...
            featureNum, nDrugs, nTargets); %test predicted interactions
        [temp4, temp3] = calcPrecRecall(trainNewInteractionMat, predictedInteractionMatrix);%precision and recall for train
        [temp2, temp1] = calcPrecRecall(testNewInteractionMat, predictedInteractionMatrixTest);%precision and recall for test
        %sum recall and precision for all folds with the same feature dim:
        test_recall((featureNum-20)/20+1, 1) = temp1 + test_recall((featureNum-20)/20+1, 1);
        test_precision((featureNum-20)/20+1, 1) = temp2 + test_precision((featureNum-20)/20+1, 1);
        train_recall((featureNum-20)/20+1, 1) = train_recall((featureNum-20)/20+1, 1) + temp3;
        train_precision((featureNum-20)/20+1, 1) = train_precision((featureNum-20)/20+1, 1)+temp4;
        disp('current fold is : ');disp(currentFold);
        disp('feature dimention is: ');disp( featureNum);
        disp('test recall is: ');disp(test_recall((featureNum-20)/20+1, 1));
        disp('test precision is: ');disp(test_precision((featureNum-20)/20+1, 1));
        disp('train recall is: ');disp(train_recall((featureNum-20)/20+1, 1));
        disp('train precision is: ');disp(train_precision((featureNum-20)/20+1, 1));
        
        %clear all variables for better performance:
        clear train_Drug_Target_Interaction; clear xPosTrain; clear xNegTrain; clear drug; clear target; clear xTest; 
        clear targetFeature; clear predictedInteractionMatrixTest;
        %calc all variables for next loop:
        Drug_Target_Interaction = [cols, rows];
        clear rows; clear cols;
        diffSize = size(train, 1)-size(Drug_Target_Interaction, 1);
        Interactions_Drug_Target_temp = [Drug_Target_Interaction; zeros(diffSize, 2)]; %intersection with the same size
        train_Drug_Target_Interaction = intersect(Interactions_Drug_Target_temp, ...
            PSLFolds(train_indexess, 1:2),'rows');%extracting train interactions for current fold
        xPosTrain = []; xNegTrain = [];
        [rowZeroIdx, colZeroIdx] = find(predictedInteractionMatrix == 0);%predicted interaction matrix as interaction matrix for the next loop
        xPosTrain = cell2mat([target_feature(lastFeatureIndexTarget, train_Drug_Target_Interaction(:,2));...
            drug_feature(lastFeatureIndexDrug, train_Drug_Target_Interaction(:,1))]);
        zeroIndexNum = min(size(rowZeroIdx, 1), size(xPosTrain, 2));
        diffSize = size(train_indexess, 1) - zeroIndexNum;
        ZeroIndexesTrain = [[rowZeroIdx(1:zeroIndexNum, 1), colZeroIdx(1:zeroIndexNum, 1)]; zeros(abs(diffSize), 2)];
        train_zeroIndexes = intersect(ZeroIndexesTrain(1:zeroIndexNum, :), PSLFolds(train_indexess, 1:2),'rows');
        randomIndx = randperm(size(train_zeroIndexes, 1));
        ZeroIndexesTrain = train_zeroIndexes(randomIndx, :);
        zeroIndexNum = min(size(ZeroIndexesTrain, 1), size(xPosTrain, 2));
        target = target_feature(lastFeatureIndexTarget, :);
        drug =  drug_feature(lastFeatureIndexDrug, :);
        drugTest = {cell2mat(drug_feature(lastFeatureIndexDrug, unique(PSLFolds(test_indexess, 1))))};
        xNegTrain = cell2mat([target_feature(lastFeatureIndexTarget, ZeroIndexesTrain(1:zeroIndexNum,2));...
            drug_feature(lastFeatureIndexDrug, ZeroIndexesTrain(1:zeroIndexNum,1))]);
        if(size(xNegTrain, 1) == 0 || size(xPosTrain, 1) == 0)
            break;
        end
    end
end

test_recall = test_recall/num_folds;
test_precision = test_precision/num_folds;
test_recall = test_recall/num_folds;
test_precision = test_precision/num_folds;
