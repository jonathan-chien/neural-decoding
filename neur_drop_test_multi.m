function performance = neur_drop_test_multi(TxN,groundTruth,dropIdc,nvp)

arguments
    TxN
    groundTruth
    dropIdc
    nvp.nClasses {mustBeInteger} = 3
    nvp.nBootstraps {mustBeInteger}= 1000
    nvp.nullMethod {string} = 'datasample' % 'datasample' or 'kNN cluster'
    nvp.kFolds {mustBeInteger} = 5
    nvp.learner {string} = 'linear'
    nvp.loadings {mustBeNumeric} = []
    nvp.pval = 'two-sided' % 'two-sided', 'left', or 'right'
    nvp.plotHist = true
end

% Obtain parameters and calculate basic variables.
nClasses = nvp.nClasses;
nBootstraps = nvp.nBootstraps;
nFolds = nvp.kFolds;
learners = nvp.learner;
nullMethod = nvp.nullMethod;
nTrials = size(TxN, 1);
nNeurons = size(TxN, 2); % number of neurons in the full population
clustSize = length(dropIdc);

% Permute trials and groundTruth together so that trials and labels are
% still matched (crossvalind produces random indices but do this here just
% to be safe).
permutation = randperm(nTrials);
TxN = TxN(permutation,:);
groundTruth = groundTruth(permutation);

%% Evaluate decoder on full population and after targeted neuron drop

acc = NaN(1,2);
macroPrec = NaN(nClasses, 2);
macroReca = NaN(nClasses, 2);
macroFmea = NaN(nClasses, 2);
macroAUCPR = NaN(nClasses, 2);
binaryPrec = NaN(nClasses, 2);
binaryReca = NaN(nClasses, 2);
binaryFmea = NaN(nClasses, 2);
binaryAUCPR = NaN(nClasses, 2);

for iRun = 1:2
    
    % Set data matrix to be given to decoder.
    switch iRun 
        case 1
            % Use full population to train and test decoder.
            targetTxN = TxN;
        case 2
            % Calculate decoder performance based on dropping the neurons
            % specified in dropIdc.
            targetTxN = TxN(:, setdiff(1:nNeurons,dropIdc));
    end
    
    % Set up cross-validation indices and preallocate.
    cvIdc = crossvalind('KFold', nTrials, nFolds);
    confMat = struct('truePos', zeros(nClasses,1), ...
                     'falsePos', zeros(nClasses,1), ...
                     'falseNeg', zeros(nClasses,1)); % TN currently not used.
    allTestingLabels = [];
    scores = [];
    
    % Train/test decoder with k-fold cross-validation and concatenate
    % results to calculate performance metrics afterwards.
    for k = 1:nFolds
        
        % Designate train and test sets. Train and test decoder.
        trainingSet = targetTxN(cvIdc~=k,:);
        testingSet = targetTxN(cvIdc==k,:);
        trainingLabels = groundTruth(cvIdc ~= k);
        testingLabels = groundTruth(cvIdc == k);
        allTestingLabels = [allTestingLabels; testingLabels]; % used to calculate AUC after iterating across all k folds
        decoder = fitcecoc(trainingSet, trainingLabels, 'Coding', 'onevsall', ...
                           'Learners', learners);
        [label, ~, PBScore] = predict(decoder, testingSet); % ~ is negLoss
        
        % Accumulate TP,FP, and FN to calculate performance metrics after
        % iterating through all k folds.
        for iClass = 1:nClasses
            % True positives.
            confMat.truePos(iClass)  = confMat.truePos(iClass) + ...
                                       sum(label == iClass & testingLabels == iClass);
            % False positives.
            confMat.falsePos(iClass) = confMat.falsePos(iClass) + ...
                                       sum(label == iClass & testingLabels ~= iClass);
            % False negatives.
            confMat.falseNeg(iClass) = confMat.falseNeg(iClass) + ...
                                       sum(label ~= iClass & testingLabels == iClass);
        end
        
        % Append labels and scores for each k. Inefficient but small loop
        % so doesn't matter and also is more readable this way.
        scores = [scores; PBScore];    
    end
    
    % Calculate accuracy ( = micro-prec = micro-reca = micro-F-mea)
    acc(iRun) = sum(confMat.truePos) / nTrials;
    
    % Calculate metrics for each binary classifier.
    binaryPrec(:,iRun) = confMat.truePos./(confMat.truePos + confMat.falsePos);
    binaryReca(:,iRun) = confMat.truePos./(confMat.truePos + confMat.falseNeg);
    binaryFmea(:,iRun) = harmmean([binaryPrec(:,iRun) binaryReca(:,iRun)], 2); % nClasses x 1 vector
    for iClass = 1:nClasses
        [~,~,~,binaryAUCPR(iClass,iRun)] ...
            = perfcurve(allTestingLabels, ...
                        scores(:,iClass), ...
                        iClass, 'XCrit', 'reca', 'YCrit', 'prec');
    end
    
    % Calculate macro metrics across all binary classifiers. Note again
    % that class sizes are evenly balanced.
    macroPrec(iRun) = mean(binaryPrec(:,iRun));
    macroReca(iRun) = mean(binaryReca(:,iRun));
    macroFmea(iRun) = mean(binaryFmea(:,iRun));
    macroAUCPR(iRun) = mean(binaryAUCPR(:,iRun));
end


%% Evaluate decoder on population after *random* neuron drop to generate null

if strcmp(nullMethod, 'kNN cluster')
    % Find indices of kNN, where k = clustSize-1. Note that 1st NN for each
    % neuron is itself, and removing this "NN" amounts to subtracting 1
    % from clustSize. Note as well that 'SortIndices' nvp for knnsearch is
    % true by default.
    assert(~isempty(nvp.loadings), ...
           "Must supply loadings/eigenvectors if 'kNN cluster' null method desired.")
    kNNIdc = knnsearch(nvp.loadings, nvp.loadings, ...
                       'K', clustSize, 'Distance', 'cosine');
    kNNIdc = kNNIdc(:,2:end);
else 
    % Variable must be defined in workspace in or parfor below will fail.
    kNNIdc = [];
end

% Preallocate.
nullAcc = NaN(nBootstraps, 1);
% The following 4 vars don't have to be preallocated since they're computed
% from mean of binary metrics. Just here as a conceptual place holder.
% nullMacroPrec = NaN(nBootstraps, 1); 
% nullMacroReca = NaN(nBootstraps, 1);
% nullMacroFmea = NaN(nBootstraps, 1);
% nullMacroAUCPR = NaN(nBootstraps, 1);
nullBinaryPrec = NaN(nBootstraps, nClasses);
nullBinaryReca = NaN(nBootstraps, nClasses);
nullBinaryFmea = NaN(nBootstraps, nClasses);
nullBinaryAUCPR = NaN(nBootstraps, nClasses);

% Generate null distribution of performance metrics via permutation.
parfor iBoot = 1:nBootstraps
    
    % Randomly drop neurons using one of two methods.
    switch nullMethod
        case 'datasample'
            % Drop clustSize random neurons.
            bootTxN = TxN(:, datasample(1:nNeurons, nNeurons-clustSize, ...
                                        'Replace', false));
                                    
        case 'kNN cluster'
            % Select one neuron as a "cluster seed" and drop its
            % clustSize-1 nearest neighbors.
            iClusterSeed = datasample(1:nNeurons, 1);
            bootTxN = TxN(:, setdiff(1:nNeurons, ...
                                     [iClusterSeed kNNIdc(iClusterSeed,:)]));
    end 
    
    % Set up CV indices and initialize for iteration over k folds.
    cvIdc = crossvalind('Kfold', nTrials, nFolds);
    allTestingLabels = [];
    currBootBinScores = []
    confMat = struct('correctPred', 0, ...
                     'truePos', zeros(nClasses,1), ...
                     'falsePos', zeros(nClasses,1), ...
                     'falseNeg', zeros(nClasses,1));

    % Iterate over k folds.
    for k = 1:nFolds   

        % Designate train and test sets. Train and test decoder.
        trainingSet = bootTxN(cvIdc~=k,:);
        testingSet = bootTxN(cvIdc==k,:);
        trainingLabels = groundTruth(cvIdc ~= k);
        testingLabels = groundTruth(cvIdc == k);
        allTestingLabels = [allTestingLabels; testingLabels]; % used to calculate AUC after iterating across all k folds
        decoder = fitcecoc(trainingSet, trainingLabels, 'Coding', 'onevsall', ...
                           'Learners', learners);
        [label, ~, PBScore] = predict(decoder, testingSet); % ~ is negLoss

        % Accumulate correct TP/FP/FN to calculate
        % performance metrics after iterating through all k folds. Also accumulate
        % scores.
        for iClass = 1:nClasses
            % True positives in current fold.
            confMat.truePos(iClass) ...
                = confMat.truePos(iClass) + ...
                  sum(label == iClass & testingLabels == iClass);
            % False positives in current fold.
            confMat.falsePos(iClass) ...
                = confMat.falsePos(iClass) + ...
                  sum(label == iClass & testingLabels ~= iClass);
            % False negatives in current fold.
            confMat.falseNeg(iClass) ...
                = confMat.falseNeg(iClass) + ...
                  sum(label ~= iClass & testingLabels == iClass);
        end

        % Add scores from current fold.
        currBootBinScores = [currBootBinScores; PBScore];
    end

    % Calculate null accuracy = null micro precision = null micro
    % recall = null micro F-meausure.
    nullAcc(iBoot) = sum(confMat.truePos) ./ nTrials;

    % Calculate precision, recall, F-measure, and AUCPR for null binary
    % classifiers.
    nullBinaryPrec(iBoot,:) = confMat.truePos./(confMat.truePos ...
                                                + confMat.falsePos);
    nullBinaryReca(iBoot,:) = confMat.truePos./(confMat.truePos ...
                                                + confMat.falseNeg);
    nullBinaryFmea(iBoot,:) = harmmean([nullBinaryPrec(iBoot,:) ...
                                        nullBinaryReca(iBoot,:)], 2);
    for iClass = 1:nClasses
        [~,~,~,nullBinaryAUCPR(iBoot,iClass)] ...
            = perfcurve(allTestingLabels, ...
                        currBootBinScores(:,iClass), ...
                        iClass, 'XCrit', 'reca', 'YCrit', 'prec');
    end
end

% Calculate macro precision, macro recall, macro F-meausure, and
% macro AUCPR. Note again that class sizes are balanced by design. Doing
% this outside of the parfor loop to avoid indexing issues and because
% vectorized operation is straightforward.
nullMacroPrec = mean(nullBinaryPrec, 2);
nullMacroReca = mean(nullBinaryReca, 2);
nullMacroFmea = mean(nullBinaryFmea, 2);
nullMacroAUCPR = mean(nullBinaryAUCPR, 2);


%% Compute p value

% Subtract performance on full population from performance on targeted
% population to obtain observed effect size (will be negative if there is a
% reduction in performance after dropping neurons).
obs.accDiff = acc(2) - acc(1);
obs.binaryPrecDiff = binaryPrec(:,2) - binaryPrec(:,1);
obs.binaryRecaDiff = binaryReca(:,2) - binaryReca(:,1);
obs.binaryFmeaDiff = binaryFmea(:,2) - binaryFmea(:,1);
obs.binaryAUCPRDiff = binaryAUCPR(:,2) - binaryAUCPR(:,1);
obs.macroPrecDiff = macroPrec(2) - macroPrec(1);
obs.macroRecaDiff = macroReca(2) - macroReca(1);
obs.macroFmeaDiff = macroFmea(2) - macroFmea(1);
obs.macroAUCPRDiff = macroAUCPR(2) - macroAUCPR(1);

% Subtract performance of the decoder on the full population from the
% performance with randomly dropped neurons in order to create null
% distributions for the effect size (again will be negative if dropping
% neurons causes reduction in performance).
null.accDiff = nullAcc - acc(1);
null.binaryPrecDiff = nullBinaryPrec - binaryPrec(:,1)';
null.binaryRecaDiff = nullBinaryReca - binaryReca(:,1)';
null.binaryFmeaDiff = nullBinaryFmea - binaryFmea(:,1)';
null.binaryAUCPRDiff = nullBinaryAUCPR - binaryAUCPR(:,1)';
null.macroPrecDiff = nullMacroPrec - macroPrec(1);
null.macroRecaDiff = nullMacroReca - macroReca(1);
null.macroFmeaDiff = nullMacroFmea - macroFmea(1);
null.macroAUCPRDiff = nullMacroAUCPR - macroAUCPR(1);

% Calculate fraction of null that observed is less than (frac less = fl)
% and convert this fraction into a p value.
fracLess = cell(9,1);
fracLess{1} = sum(obs.accDiff < null.accDiff) / nBootstraps; % acc
fracLess{2} = sum(obs.binaryPrecDiff' < null.binaryPrecDiff) / nBootstraps; % binaryPrec
fracLess{3} = sum(obs.binaryRecaDiff' < null.binaryRecaDiff) / nBootstraps; % binaryReca
fracLess{4} = sum(obs.binaryFmeaDiff' < null.binaryFmeaDiff) / nBootstraps; % binaryFmea
fracLess{5} = sum(obs.binaryAUCPRDiff' < null.binaryAUCPRDiff) / nBootstraps; % binaryAUCPR
fracLess{6} = sum(obs.macroPrecDiff < null.macroPrecDiff) / nBootstraps; % macroPrec
fracLess{7} = sum(obs.macroRecaDiff < null.macroRecaDiff) / nBootstraps; % macroReca
fracLess{8} = sum(obs.macroFmeaDiff < null.macroFmeaDiff) / nBootstraps; % macroFmea
fracLess{9} = sum(obs.macroAUCPRDiff < null.macroAUCPRDiff) / nBootstraps; % macroAUCPR

pvalues = cell(9,1);
for iMetric = 1:9
    switch nvp.pval
        case 'two-sided'
            pvalues{iMetric} = convert_to_twosided_pval(fracLess{iMetric});
        case 'left'
            pvalues{iMetric} = convert_to_left_pval(fracLess{iMetric});
        case 'right'
            pvalues{iMetric} = convert_to_right_pval(fracLess{iMetric});  
    end
end

% Store effect sizes and p values in struct for export.
performance.obsAccDiff = obs.accDiff;
performance.obsBinaryPrecDiff = obs.binaryPrecDiff;
performance.obsBinaryRecaDiff = obs.binaryRecaDiff;
performance.obsBinaryFmeaDiff = obs.binaryFmeaDiff;
performance.obsBinaryAUCPRDiff = obs.binaryAUCPRDiff;
performance.obsMacroPrecDiff = obs.macroPrecDiff;
performance.obsMacroRecaDiff = obs.macroRecaDiff;
performance.obsMacroFmeaDiff = obs.macroFmeaDiff;
performance.obsMacroAUCPRDiff = obs.macroAUCPRDiff;
performance.pAcc = pvalues{1};
performance.pBinaryPrec = pvalues{2};
performance.pBinaryReca = pvalues{3};
performance.pBinaryFmea = pvalues{4};
performance.pBinaryAUCPR = pvalues{5};
performance.pMacroPrec = pvalues{6};
performance.pMacroReca = pvalues{7};
performance.pMacroFmea = pvalues{8};
performance.pMacroAUCPR = pvalues{9};

end

% Local functions for conversion from fraction of null less than observed
% to p value.
function pval = convert_to_twosided_pval(fractionLess)

% Determine number of values = length of fractionLess. Will be > 1 and =
% nClasses for binary metrics (based on one-vs-all classifiers) and = 1 for
% micro and macro metrics. 
nValues = length(fractionLess);
pval = NaN(nValues, 1);
for iValue = 1:nValues
    if fractionLess(iValue) <= 0.5
        pval(iValue) = fractionLess(iValue)*2;
    elseif fractionLess(iValue) >= 0.5
        pval(iValue) = (1-fractionLess(iValue)) * 2;
    end
end

end

function pval = convert_to_left_pval(fractionLess)

pval = 1 - fractionLess;

end

function pval = convert_to_right_pval(fractionLess)

pval = fractionLess;

end

