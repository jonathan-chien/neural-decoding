function performance = neur_drop_test_binary(TxN,groundTruth,dropIdc,nvp)
% Takes as input TxN, a nTrials x nNeurons matrix corresponding to one
% slice along the first axis of BxTxN, and uses groundTruth, a vector of
% trial labels with length = nTrials, to train a decoder on a binary
% variable, with performance measured via accuracy and AUC. Next, specific
% features (neurons) specified through the argument dropIdc are eliminated,
% and the decoder trained again on the reduced feature set (same trials of
% course), with performance once again measured via accuracy and AUC.
% Finally, random neurons are dropped nBootstraps times to generate a null
% distribution of reduced performance (calculated as difference between
% reduced and full features set performance) under random feature removal;
% there are two methods for defining this random drop (see the 'nullMethod'
% name value pair under PARAMETERS).
%
% PARAMETERS
% ----------
% TxN         -- nTrials x nNeurons array, where each column corresponds to
%                a neuron, and each row to the instantaneous firing rates
%                of those neurons on a given trial.
% groundTruth -- nTrials x 1 array of trial labels, with labels
%                corresponding to conditions (values of a binary task
%                variable).
% dropIdc     -- nBins x 1 cell array, where each cell is either empty or
%                contains a clustSize x 1 array of neuron indices denoting
%                which neurons belong to a particular cluster. Only cells
%                corresponding to bins with significant clustering are
%                nonempty.
% Name-Value Pair 
%   'nBootstraps' -- Number of times to randomly drop (see 'nullMethod' for
%                    two definitions of random drop) neurons/features to
%                    generate null distribution of performance difference.
%   'nullMethod'  -- May have string value 'datasample' or 'kNN cluster'.
%                    If 'datasample', n neurons, where n = clustSize will
%                    be randomly selected without replacement using
%                    MATLAB's datasample function and eliminated. If 'kNN
%                    cluster', for each bootstrap iteration, a single
%                    random neuron will be randomly selected and removed
%                    along with its n-1 nearest neighbors using a cosine
%                    distance metric. Note that if 'kNN cluster',
%                    'loadings' may not be empty.
%   'kFolds'      -- Number of folds to use during cross-validation for
%                    evaluation of decoder performance.
%   'learner'     -- Template to be used in linear svm. 
%   'loadings'    -- If 'nullMethod' is 'kNN cluster', supply the nNeurons
%                    nPCs loadings matrix in order to calculate the kNN for
%                    each neuron.
%   'pval'        -- May have string value 'two-sided', 'left', or 'right',
%                    corresponding to the method used to compute the p
%                    value.
%   'plotHist'    -- Logical true or false. If true, plot the null
%                    distributions of performance shifts (one each for
%                    accuracy and AUC), with the observed performance shift
%                    marked with a dashed line. 
%
% RETURNS
% -------
% performance -- 1x1 struct with the following fields:
%   .obsAccDiff -- Observed difference in accuracy between decoder trained
%                  with full neural population vs decoder trained on a
%                  reduced neural population (after dropping neurons in
%                  cluster). This metric is calculated as accReduced -
%                  accFull, such that a reduction in performance after
%                  dropping neurons results in a negative value.
%   .obsAUCDiff -- Observed difference in AUC between decoder trained
%                  with full neural population vs decoder trained on a
%                  reduced neural population (after dropping neurons in
%                  cluster). This metric is calculated as AUCReduced -
%                  AUCFull, such that a reduction in performance after
%                  dropping neurons results in a negative value.
%   .p_acc      -- P value for the observed accuracy difference vs the null
%                  distribution of accuracy differences upon dropping
%                  random neurons/features.
%   .p_AUC      -- P value for the observed AUC difference vs the null
%                  distribution of AUC differences upon dropping random
%                  neurons/features.
%
% Author: Jonathan Chien 6/6/21. Version 1.0. Last edit:6/8/21.

arguments
    TxN
    groundTruth {mustBeInteger}
    dropIdc
    nvp.nBootstraps {mustBeInteger}= 1000
    nvp.nullMethod {string} = 'datasample' % 'datasample' or 'kNN cluster'
    nvp.kFolds {mustBeInteger} = 5
    nvp.learner {string} = 'svm'
    nvp.loadings {mustBeNumeric} = []
    nvp.pval = 'two-sided' % 'two-sided', 'left', or 'right'
    nvp.plotHist = true
end

% Obtain parameters and calculate basic variables.
nBootstraps = nvp.nBootstraps;
nFolds = nvp.kFolds;
learner = nvp.learner;
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

% Preallocate.
acc = NaN(1,2);
AUC = NaN(1,2);

% On decoding run 1, use full neural population. On decoding run 2, use
% population after making a targeted drop.
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
    correctPred = 0;
    allTestingLabels = [];
    allScores = [];
    
    % Train/test decoder with k-fold cross-validation and concatenate
    % results to calculate accuracy and AUC afterwards.
    for k = 1:nFolds
        
        % Designate train and test sets. Train and test decoder.
        trainingSet = targetTxN(cvIdc~=k,:);
        testingSet = targetTxN(cvIdc==k,:);
        trainingLabels = groundTruth(cvIdc~=k);
        testingLabels = groundTruth(cvIdc==k);
        allTestingLabels = [allTestingLabels; testingLabels]; % used to calculate AUC after iterating across all k folds
        decoder = fitclinear(trainingSet, trainingLabels, 'Learner', learner);
        [label, scores] = predict(decoder, testingSet); 

        % Accumulate correct classifications to calculate accuracy/PR after
        % iterating through all k folds. Correct predictions = TP + TN.
        correctPred = correctPred + (sum(label == 1 & testingLabels == 1) + ...
                                     sum(label == 2 & testingLabels == 2));             

        % Append scores for each k in order to calculuate AUC after iterating
        % over all k folds. Inefficient but small loop so doesn't matter and
        % also is more readable this way.
        allScores = [allScores; scores];
    end

    % Calculate accuracy and AUC.
    acc(iRun) = correctPred / nTrials;
    [~,~,~,AUC(iRun)] = perfcurve(allTestingLabels, allScores(:,1), 1);
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
nullAUC = NaN(nBootstraps, 1);

% Generate null distribution of accuracy and AUC.
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
    
    % Set up cross-validation indices and preallocate.
    cvIdc = crossvalind('KFold', nTrials, nFolds);
    correctPred = 0;
    allTestingLabels = [];
    allScores = [];
    
    % Train decoder as before but on the randomly reduced population.
    for k = 1:nFolds  
        
        % Designate train and test sets. Train and test decoder.
        trainingSet = bootTxN(cvIdc~=k,:);
        testingSet = bootTxN(cvIdc==k,:);
        trainingLabels = groundTruth(cvIdc~=k);
        testingLabels = groundTruth(cvIdc==k);
        allTestingLabels = [allTestingLabels; testingLabels]; % used to calculate AUC after iterating across all k folds
        decoder = fitclinear(trainingSet, trainingLabels, 'Learner', learner);
        [label, scores] = predict(decoder, testingSet); 

        % Accumulate correct classifications to calculate accuracy/PR after
        % iterating through all k folds. Correct predictions = TP + TN.
        correctPred = correctPred + (sum(label == 1 & testingLabels == 1) + ...
                                     sum(label == 2 & testingLabels == 2));             

        % Append scores for each k in order to calculuate AUC after iterating
        % over all k folds. Inefficient but small loop so doesn't matter and
        % also is more readable this way.
        allScores = [allScores; scores];
    end
    
    % Calculate null accuracy and AUC.
    nullAcc(iBoot) = correctPred / nTrials;
    [~,~,~,nullAUC(iBoot)] = perfcurve(allTestingLabels, allScores(:,1), 1);
end

% Subtract performance on full population from performance on targeted
% population to obtain observed effect size (will be negative if there is a
% reduction in performance after dropping neurons).
obsAccDiff = acc(2) - acc(1);
obsAUCDiff = AUC(2) - AUC(1);

% Subtract performance of the decoder on the full population from the
% performance with randomly dropped neurons in order to create null
% distributions for the effect size (again will be negative if dropping
% neurons causes reduction in performance).
nullAccDiff = nullAcc - acc(1);
nullAUCDiff = nullAUC - AUC(1);

% Calculate p value for effect size.
fracLessAcc = sum(obsAccDiff < nullAccDiff) / nBootstraps; % accuracy
fracLessAUC = sum(obsAUCDiff < nullAUCDiff) / nBootstraps; % AUC
switch nvp.pval
    case 'two-sided'
        if fracLessAcc < 0.5
            p_acc = fracLessAcc * 2;
        else 
            p_acc = (1-fracLessAcc) * 2;
        end
        
        if fracLessAUC < 0.5
            p_AUC = fracLessAUC * 2;
        else
            p_AUC = (1- fracLessAUC) * 2;
        end
            
    case 'left'
        p_acc = 1 - fracLessAcc; 
        p_AUC = 1 - fracLessAUC; 
        
    case 'right'
        p_acc = fracLessAcc; 
        p_AUC = fracLessAUC; 
end

% Place results in struct for export.
performance.acc = acc;
performance.AUC = AUC;
performance.obsAccDiff = obsAccDiff;
performance.obsAUCDiff = obsAUCDiff;
performance.p_acc = p_acc;
performance.p_AUC = p_AUC;

% Option to plot sampling distributions with observed effects marked with
% dashed line.
if nvp.plotHist
    figure
    subplot(2,1,1)
    hold on
    histogram(nullAccDiff, 30)
    yl = ylim;
    plot([obsAccDiff obsAccDiff], [yl(1) yl(2)], '--', ...
         'DisplayName', 'Accuracy after targeted neuron drop')
    plot([mean(nullAccDiff) mean(nullAccDiff)], [yl(1) yl(2)], '--', ...
         'Color', 'k', 'DisplayName', 'Mean accuracy after random neuron drop')
    hold off
    title('Accuracy shift')
    legend

    subplot(2,1,2)
    hold on
    histogram(nullAUCDiff, 30)
    yl = ylim;
    plot([obsAUCDiff obsAUCDiff], [yl(1) yl(2)], '--', ...
         'DisplayName', 'AUC after targeted neuron drop')
    plot([mean(nullAUCDiff) mean(nullAUCDiff)], [yl(1) yl(2)], '--', ...
         'Color', 'k', 'DisplayName', 'Mean AUC after random neuron drop')
    hold off
    title('AUC shift')
    legend

    sgtitle('Distribution of performance shift under random feature (neuron) elimination')
end
        
end
