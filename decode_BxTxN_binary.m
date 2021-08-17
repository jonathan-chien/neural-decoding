function decoderPerf = decode_BxTxN_binary(BxTxN,groundTruth,nvp)
% Accepts as input a b x t x n array of neural firing rates, where b is the
% number of bins, t the number of trials, and n the number of neurons.
% However, only a subset of the n neurons are retained, specified through
% the keepIdc name-value pair. Then, for each b, i.e., for each t x n slice
% of the array, a decoder model will be trained and tested in a kFold
% cross-validation loop, and performance metrics (one value/set of values
% per bin), labels (t values per bin), and scores (t values per bin) will
% be registered. P values, if desired, will be computed via bootstrapping
% (classifiers trained on permuted trial labels).
%
% PARAMETERS
% ----------
%   BxTxN       -- b x t x n array of neural firing rates, where b = nBins,
%                  t = nTrials, and n = nNeurons. Out of the t trials, each
%                  condition is represented by an equal number of trials
%                  (each condition contributes t/nCond trials to the total
%                  t trials).
%   groundTruth -- t x 1 vector of trial labels (labels correspond to some
%                  condition in the task, e.g. positive vs negative
%                  valence).
%   Name-Value Pairs (nvp)
%       'dropIdc'      -- 1D cell array of length b, where b = nBins. Each
%                         cell contains a 1D array of length m, where m is
%                         the number of neurons to be dropped for decoding
%                         during that bin. Elements correspond to indices
%                         of the dropped neurons in the overall population.
%                         Default is empty.
%       'learner'      -- Specify template to be used for binary learner,
%                         e.g. 'svm' (default), 'linear', etc. See
%                         documentation for fitcecoc.m Name-Value Pairs
%                         for more info.
%       'kFolds'       -- Number of iterations within a cross-validation
%                         loop (one loop per bin).
%       'compute_pval' -- Specifies whether to compute p values for
%                         performance metrics in each bin via permutation
%                         test (the kind neuro people seem to do i.e.). If
%                         p values are desired, must specify 'two-sided',
%                         'left', or 'right'. Set to false to skip
%                         computation of p values (a costly process), e.g.
%                         if values for empirical data alone are desired.
%       'nBootstraps'  -- Number of permutations used to generate null
%                         distribution of no-skill performance metrics. 
%
% RETURNS
% -------
%   decoderPerf -- 1 x b struct with the following fields:
%       .labels     -- t x 1 array of predicted class labels (one label for
%                      each trial/observation).
%       .scores     -- Array of scores produced by the decoder within
%                      the given bin. 
%       .acc        -- Scalar value which is decoder accuracy within the
%                      given bin. 
%       .AUC        -- Scalar value which is decoder AUC within the given
%                      bin. 
%       .p_acc      -- p value for accuracy of the classifier, computed
%                      using permutation.
%       .p_AUC      -- p value for AUC of the classifier, computed using
%                      permutation.
%
% Author: Jonathan Chien Adapted from decode_BxTxN_svm on 6/2/21. 
% Version 1.0. Last edit: 6/2/21.

arguments
    BxTxN {mustBeNumeric}
    groundTruth {mustBeInteger}
    nvp.dropIdc = []
    nvp.learner string = 'svm'
    nvp.kFolds {mustBeInteger} = 5 
    nvp.compute_pval = false % false, 'two-sided', 'left', or 'right'
    nvp.nBootstraps {mustBeInteger} = 100
end

% Obtain and set parameters.
nBins = size(BxTxN, 1);
nTrials = size(BxTxN, 2);
nFolds = nvp.kFolds; 
nBootstraps = nvp.nBootstraps;
learner = nvp.learner;
if isempty(nvp.dropIdc)
    dropIdc = cell(nBins, 1);
else
    dropIdc = nvp.dropIdc;
end

% Permute trials and groundTruth together so that trials and labels are
% still matched (crossvalind produces random indices but do this here just
% to be safe).
permutation = randperm(nTrials);
BxTxN = BxTxN(:,permutation,:);
groundTruth = groundTruth(permutation);

%% Train and test on empirical data

% Preallocate struct 'decoderPerformance' and initialize waitbar.
decoderPerf = struct('labels', cell(1, nBins), ...
                     'scores', cell(1, nBins), ...
                     'acc', cell(1, nBins), ...
                     'AUC', cell(1, nBins), ...
                     'p_acc', cell(1, nBins), ...
                     'p_AUC', cell(1, nBins));
w = waitbar(0, '');

for iBin = 1:nBins
    
    % Update waitbar.
    waitbar(iBin./nBins, w,...
            sprintf('Training and testing over %d folds for bin %d of %d.', ...
                    nFolds, iBin, nBins));
    
    % Obtain slice corresponding to nTrials x nNeurons for current bin,
    % drop neurons if desired, and set up indices for cross-validation.
    currentBin = squeeze(BxTxN(iBin,:,:));
    currentBin(:,dropIdc{iBin}) = [];
    cvIndices = crossvalind('Kfold', nTrials, nFolds);
    
    % Train and test decoders.
    correctPred = 0;
    allTestingLabels = [];
    for k = 1:nFolds
        
        % Designate train and test sets. Train and test decoder.
        trainingSet = currentBin(cvIndices~=k,:);
        testingSet = currentBin(cvIndices==k,:);
        trainingLabels = groundTruth(cvIndices ~= k);
        testingLabels = groundTruth(cvIndices == k);
        allTestingLabels = [allTestingLabels; testingLabels]; % used to calculate AUC after iterating across all k folds
        decoder = fitclinear(trainingSet, trainingLabels, 'Learner', learner);
        [label, scores] = predict(decoder, testingSet); 
        
        % Accumulate correct classifications to calculate accuracy/PR after
        % iterating through all k folds. Correct predictions = TP + TN.
        correctPred = correctPred + (sum(label == 1 & testingLabels == 1) + ...
                                     sum(label == 2 & testingLabels == 2));             
        
        % Append labels and scores for each k. Inefficient but small loop
        % so doesn't matter and also is more readable this way.
        decoderPerf(iBin).labels = [decoderPerf(iBin).labels; label];
        decoderPerf(iBin).scores = [decoderPerf(iBin).scores; scores];    
    end
    
    % Calculate accuracy and AUC for current bin after iterating
    % over all k folds.
    decoderPerf(iBin).acc = correctPred./nTrials;
    [~,~,~,decoderPerf(iBin).AUC] ...
        = perfcurve(allTestingLabels, decoderPerf(iBin).scores(:,1), 1);  
end
close(w)


%% Generate null distribution

if nvp.compute_pval

% Prealllocate.
nullAcc = NaN(nBootstraps, nBins);
nullAUC = NaN(nBootstraps, nBins);

% Generate null distribution of performance metrics via permutation.
parfor iBoot = 1:nBootstraps
    
    for iBin = 1:nBins
        
        % Obtain current slice, shuffle labels, and set up CV indices.
        currentBin = squeeze(BxTxN(iBin,:,:));
        currentBin(:,dropIdc{iBin}) = [];
        currentGroundTruth = groundTruth(randperm(nTrials));
        cvIndices = crossvalind('Kfold', nTrials, nFolds);
        
        % Initialize for iteration over k folds.
        allTestingLabels = [];
        currBootBinScores = []
        correctPred = 0;
        
        % Iterate over k folds.
        for k = 1:nFolds   

            % Designate train and test sets. Train and test decoder.
            trainingSet = currentBin(cvIndices~=k,:);
            testingSet = currentBin(cvIndices==k,:);
            trainingLabels = currentGroundTruth(cvIndices ~= k);
            testingLabels = currentGroundTruth(cvIndices == k);
            allTestingLabels = [allTestingLabels; testingLabels]; % used to calculate AUC after iterating across all k folds
            decoder = fitclinear(trainingSet, trainingLabels, 'Learner', learner);
            [label, scores] = predict(decoder, testingSet); 

            % Accumulate correct classifications, adding those in current
            % fold.
            correctPred = correctPred + (sum(label == 1 & testingLabels == 1) + ...
                                         sum(label == 2 & testingLabels == 2));
                                  
            % Add scores from current fold.
            currBootBinScores = [currBootBinScores; scores];
        end

        % Calculate performance metrics for current bootstrap in current
        % bin after iterating over all k folds.
        nullAcc(iBoot,iBin) = correctPred./nTrials;
        [~,~,~,nullAUC(iBoot,iBin)] ...
            = perfcurve(allTestingLabels, currBootBinScores(:,1), 1);
    end
end


%% Calculate p values for each bin

% Preallocate.
fracGreaterAcc = NaN(nBins, 1);
fracGreaterAUC = NaN(nBins, 1);

% Calculate p values for each metric and store in respective field of
% decoderPerf.
for iBin = 1:nBins
    % Permutation P-values Should Never Be Zero: Calculating Exact P-values
    % When Permutations Are Randomly Drawn (Phipson & Smyth 2010)
    fracGreaterAcc(iBin) = (sum(decoderPerf(iBin).acc < ...
                            nullAcc(:,iBin)) + 1)./(nBootstraps + 1);
    fracGreaterAUC(iBin) = (sum(decoderPerf(iBin).AUC < ...
                            nullAUC(:,iBin)) + 1)./(nBootstraps + 1);
    
    switch nvp.compute_pval
        case 'two-sided'
            % Accuracy (binary classes)
            if fracGreaterAcc(iBin) < 0.5
                decoderPerf(iBin).p_acc = fracGreaterAcc(iBin)*2;  
            else 
                decoderPerf(iBin).p_acc = (1 - fracGreaterAcc(iBin))*2;  
            end
            % AUC (binary classes)
            if fracGreaterAUC(iBin) < 0.5
                decoderPerf(iBin).p_AUC = fracGreaterAUC(iBin)*2; 
            else 
                decoderPerf(iBin).p_AUC = (1 - fracGreaterAUC(iBin))*2;
            end
            
        case 'left'
            decoderPerf(iBin).p_acc = 1 - fracGreaterAcc(iBin); 
            decoderPerf(iBin).p_AUC = 1 - fracGreaterAUC(iBin);
            decoderPerf(iBin).p_prec = 1 - fracGreaterPrec(iBin,:);
            decoderPerf(iBin).p_reca = 1 - fracGreaterReca(iBin,:);
            decoderPerf(iBin).p_fmeasure = 1 - fracGreaterFmeasure(iBin,:);
            decoderPerf(iBin).p_AUCPR = 1 - fracGreaterAUCPR(iBin,:);
            
        case 'right'
            decoderPerf(iBin).p_acc = fracGreaterAcc(iBin); 
            decoderPerf(iBin).p_AUC = fracGreaterAUC(iBin);
            decoderPerf(iBin).p_prec = fracGreaterPrec(iBin,:);
            decoderPerf(iBin).p_reca = fracGreaterReca(iBin,:);
            decoderPerf(iBin).p_fmeasure = fracGreaterFmeasure(iBin,:);
            decoderPerf(iBin).p_AUCPR = fracGreaterAUCPR(iBin,:);
    end
end

end

end
