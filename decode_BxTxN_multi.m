function decoderPerf = decode_BxTxN_multi(BxTxN,groundTruth,nvp)
% Accepts as input a b x t x n array of neural firing rates, where b is the
% number of bins, t the number of trials, and n the number of neurons.
% However, only a subset of the n neurons are retained, specified through
% the keepIdc name-value pair. Then, for each b, i.e., for each t x n slice
% of the array, a decoder model will be trained and tested in a kFold
% cross-validation loop, and performance metrics (one value/set of values
% per bin), labels (t values per bin), and scores (t values per bin) will
% be registered. P values, if desired, will be computed via comparison to
% bootstrapped distribution generated from no-skill classifiers trained on
% permuted trial labels).
%
% PARAMETERS
% ----------
% BxTxN       -- b x t x n array of neural firing rates, where b = nBins,
%                t = nTrials, and n = nNeurons. Out of the t trials, each
%                condition is represented by an equal number of trials
%                (each condition contributes t/nCond trials to the total t
%                trials).
% groundTruth -- t x 1 vector of trial labels (labels correspond to some
%                condition in the task, e.g. positive vs negative valence).
% Name-Value Pairs (nvp)
%   'dropIdc'      -- 1D cell array of length b, where b = nBins. Each
%                     cell contains a 1D array of length m, where m is the
%                     number of neurons to be dropped for decoding during
%                     that bin. Elements correspond to indices of the
%                     dropped neurons in the overall population. Default is
%                     empty.
%   'nClasses'     -- Specify the number of values that each task
%                     variable may take on (corresponds to the number of
%                     classes that the classifier will be trained for).
%                     This field may not be left empty (else parfor will
%                     fail).
%   'learners'     -- Specify template to be used for binary learners,
%                     e.g. 'svm' (default), 'linear', etc. See
%                     documentation for fitcecoc.m Name-Value Pairs for
%                     more info.
%   'kFolds'       -- Number of iterations within a cross-validation loop
%                     (one loop per bin).
%   'compute_pval' -- Specifies whether to compute p values for
%                     performance metrics in each bin via permutation test
%                     (the kind neuro people seem to do i.e.). If p values
%                     are desired, must specify 'two-sided', 'left', or
%                     'right'. Set to false to skip computation of p values
%                     (a costly process), e.g. if values for empirical data
%                     alone are desired.
%   'nBootstraps'  -- Number of permutations used to generate null
%                     distribution of no-skill performance metrics.
%
% RETURNS
% -------
% decoderPerf -- 1 x b struct with the following fields, where b is the
%                number of bins:
%   .labels         -- t x 1 array of predicted class labels (one label
%                      for each trial/observation).
%   .scores         -- Array of scores produced by the decoder within
%                      the given bin. Shape is t x c where t is the number
%                      of trials/observations and c is the number of
%                      classes (each column corresponds to a one vs all
%                      classifier for that class/condition).
%   .acc            -- Equivalent to the micro-averaged precision (which
%                      is also equivalent to the micro-averaged recall and
%                      thus the micro-averaged F-measure).
%   .p_acc          -- p value for accuracy (equivalent here to
%                      micro-averaged precision, recall, and F-measure).
%   .macro_prec     -- Average precision of all binary classifiers. Note
%                      that class sizes are all equal by design.
%   .macro_reca     -- Average recall of all binary classifiers. Note that
%                      class sizes are all equal by design.
%   .macro_fmea     -- Average F-measure of all binary classifiers. Note
%                      that class sizes are all equal by design.
%   .macro_AUCPR    -- Average AUCPR of all binary classifiers. Note that
%                      class sizes are all equal by design.
%   .p_macro_prec   -- Scalar value that is the bootstrapped p value for
%                      macro precision.
%   .p_macro_reca   -- Scalar value that is the bootstrapped p value for
%                      macro recall.
%   .p_macro_fmea   -- Scalar value that is the bootstrapped p value for 
%                      macro F-measure.
%   .p_macro_AUCPR  -- Scalar value that is the bootstrapped p value for 
%                      macro AUCPR.
%   .binary_prec    -- c x 1 array of precision scores for each of the c
%                      binary classifiers, where c = nClasses.
%   .binary_reca    -- c x 1 array of recall scores for each of the c
%                      binary classifiers, where c = nClasses.
%   .binary_fmea    -- c x 1 array of F-measure scores for each of the c
%                      binary classifiers, where c = nClasses.
%   .binary_AUCPR   -- c x 1 array of AUCPR values for each of the c binary
%                      classifiers, where c = nClasses.
%   .p_binary_prec  -- 1 x c vector of bootstrapped p values for precision
%                      of each of the c binary classifiers, where c =
%                      nClasses.
%   .p_binary_reca  -- 1 x c vector of bootstrapped p values for recall of
%                      each of the c binary classifiers, where c =
%                      nClasses.
%   .p_binary_fmea  -- 1 x c vector of bootstrapped p values for F-measure
%                      of each of the c binary classifiers, where c =
%                      nClasses.
%   .p_binary_AUCPR -- 1 x c vector of bootstrapped p values for AUCPR of
%                      each of the c binary classifiers, where c =
%                      nClasses.
%
% Author: Jonathan Chien Adapted from decode_BxTxN_svm on 6/2/21.
% Version 1.0. Last edit: 6/2/21.

arguments
    BxTxN {mustBeNumeric}
    groundTruth {mustBeInteger}
    nvp.dropIdc = []
    nvp.nClasses {mustBeInteger} = 3
    nvp.learners string = 'linear'
    nvp.kFolds {mustBeInteger} = 5 
    nvp.compute_pval = false % false, 'two-sided', 'left', or 'right'
    nvp.nBootstraps {mustBeInteger} = 100
end

% Obtain and set parameters.
nBins = size(BxTxN, 1);
nTrials = size(BxTxN, 2);
nFolds = nvp.kFolds; 
nBootstraps = nvp.nBootstraps;
nClasses = nvp.nClasses;
learners = nvp.learners;
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
                     'p_acc', cell(1, nBins), ...
                     'macro_prec', cell(1, nBins), ...
                     'macro_reca', cell(1, nBins), ...
                     'macro_fmea', cell(1, nBins), ...
                     'macro_AUCPR', cell(1, nBins), ...
                     'p_macro_prec', cell(1, nBins), ...
                     'p_macro_reca', cell(1, nBins), ...
                     'p_macro_fmea', cell(1, nBins), ...
                     'p_macro_AUCPR', cell(1, nBins), ...
                     'binary_prec', cell(1, nBins), ...
                     'binary_reca', cell(1, nBins), ...
                     'binary_fmea', cell(1, nBins), ...
                     'binary_AUCPR', cell(1, nBins), ...
                     'p_binary_prec', cell(1, nBins), ...
                     'p_binary_reca', cell(1, nBins), ...
                     'p_binary_fmea', cell(1, nBins), ...
                     'p_binary_AUCPR', cell(1, nBins));
w = waitbar(0, '');

for iBin = 1:nBins
    
    % Update waitbar.
    waitbar(iBin./nBins, w,...
            sprintf('Training and testing over %d folds for bin %d of %d.', ...
                    nFolds, iBin, nBins));
    
    % Obtain slice corresponding to nTrials x nNeurons for current bin and
    % set up indices for cross-validation.
    currentBin = squeeze(BxTxN(iBin,:,:));
    currentBin(:,dropIdc{iBin}) = [];
    cvIndices = crossvalind('Kfold', nTrials, nFolds);
    
    % Train and test decoders.
    confMat = struct('truePos', zeros(nClasses,1), ...
                     'falsePos', zeros(nClasses,1), ...
                     'falseNeg', zeros(nClasses,1)); % TN currently not used.
    allTestingLabels = [];
    for k = 1:nFolds
        
        % Designate train and test sets. Train and test decoder.
        trainingSet = currentBin(cvIndices~=k,:);
        testingSet = currentBin(cvIndices==k,:);
        trainingLabels = groundTruth(cvIndices ~= k);
        testingLabels = groundTruth(cvIndices == k);
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
        decoderPerf(iBin).labels = [decoderPerf(iBin).labels; label];
        decoderPerf(iBin).scores = [decoderPerf(iBin).scores; PBScore];    
    end
    
    % Calculate accuracy = micro-precision = micro-recall = micro
    % F-meausure.
    decoderPerf(iBin).acc = sum(confMat.truePos) ./ nTrials;
    
    % Calculate binary classifier precision, recall, F-measure, and AUCPR
    % across all k folds (concatenated). Store all three in decoderPerf
    % struct.
    binaryPrec = confMat.truePos./(confMat.truePos + confMat.falsePos);
    binaryReca = confMat.truePos./(confMat.truePos + confMat.falseNeg);
    decoderPerf(iBin).binary_prec = binaryPrec; % nClasses x 1 vector
    decoderPerf(iBin).binary_reca = binaryReca; % nClasses x 1 vector
    decoderPerf(iBin).binary_fmea ...
        = harmmean([binaryPrec binaryReca], 2); % nClasses x 1 vector
    for iClass = 1:nClasses
        [~,~,~,decoderPerf(iBin).binary_AUCPR(iClass)] ...
            = perfcurve(allTestingLabels, ...
                        decoderPerf(iBin).scores(:,iClass), ...
                        iClass, 'XCrit', 'reca', 'YCrit', 'prec');
    end
    
    % Calculate macro precision, recall, F-meausure, and AUCPR.
    decoderPerf(iBin).macro_prec = mean(decoderPerf(iBin).binary_prec);
    decoderPerf(iBin).macro_reca = mean(decoderPerf(iBin).binary_reca);
    decoderPerf(iBin).macro_fmea = mean(decoderPerf(iBin).binary_fmea);
    decoderPerf(iBin).macro_AUCPR = mean(decoderPerf(iBin).binary_AUCPR);
end
close(w)


%% Bootstrap null distribution

if nvp.compute_pval

% Prealllocate.
nullAcc = NaN(nBootstraps, nBins);
% The following 4 vars don't have to be preallocated since they're computed
% from mean of binary metrics. Just here as a conceptual place holder.
% nullMacroPrec = NaN(nBootstraps, nBins); 
% nullMacroReca = NaN(nBootstraps, nBins);
% nullMacroFmea = NaN(nBootstraps, nBins);
% nullMacroAUCPR = NaN(nBootstraps, nBins);
nullBinaryPrec = NaN(nBootstraps, nBins, nClasses);
nullBinaryReca = NaN(nBootstraps, nBins, nClasses);
nullBinaryFmea = NaN(nBootstraps, nBins, nClasses);
nullBinaryAUCPR = NaN(nBootstraps, nBins, nClasses);

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
        confMat = struct('correctPred', 0, ...
                         'truePos', zeros(nClasses,1), ...
                         'falsePos', zeros(nClasses,1), ...
                         'falseNeg', zeros(nClasses,1));
        
        % Iterate over k folds.
        for k = 1:nFolds   

            % Designate train and test sets. Train and test decoder.
            trainingSet = currentBin(cvIndices~=k,:);
            testingSet = currentBin(cvIndices==k,:);
            trainingLabels = currentGroundTruth(cvIndices ~= k);
            testingLabels = currentGroundTruth(cvIndices == k);
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
        nullAcc(iBoot,iBin) = sum(confMat.truePos) ./ nTrials;

        % Calculate precision, recall, F-measure, and AUCPR for null binary
        % classifiers.
        binaryPrec = confMat.truePos./(confMat.truePos + confMat.falsePos);
        binaryReca = confMat.truePos./(confMat.truePos + confMat.falseNeg);
        nullBinaryPrec(iBoot,iBin,:) = binaryPrec;
        nullBinaryReca(iBoot,iBin,:) = binaryReca;
        nullBinaryFmea(iBoot,iBin,:) = harmmean([binaryPrec binaryReca], 2);
        for iClass = 1:nClasses
            [~,~,~,nullBinaryAUCPR(iBoot,iBin,iClass)] ...
                = perfcurve(allTestingLabels, ...
                            currBootBinScores(:,iClass), ...
                            iClass, 'XCrit', 'reca', 'YCrit', 'prec');
        end
    end
end

% Calculate macro precision, macro recall, macro F-meausure, and
% macro AUCPR. Note again that class sizes are balanced by design. Doing
% this outside of the parfor loop to avoid indexing issues and because
% vectorized operation is straightforward.
nullMacroPrec = mean(nullBinaryPrec, 3);
nullMacroReca = mean(nullBinaryReca, 3);
nullMacroFmea = mean(nullBinaryFmea, 3);
nullMacroAUCPR = mean(nullBinaryAUCPR, 3);


%% Calculate p values for each bin

% Preallocate. fl = fraction less (ehf el, not efh one)
flAcc = NaN(nBins, 1);
flBinaryPrec = NaN(nBins, nClasses);
flBinaryReca = NaN(nBins, nClasses);
flBinaryFmea = NaN(nBins, nClasses);
flBinaryAUCPR = NaN(nBins, nClasses);
flMacroPrec = NaN(nBins, 1);
flMacroReca = NaN(nBins, 1);
flMacroFmea = NaN(nBins, 1);
flMacroAUCPR = NaN(nBins, 1);

% Calculate p values for each metric and store in respective field of
% decoderPerf.
for iBin = 1:nBins
    % Permutation P-values Should Never Be Zero: Calculating Exact P-values
    % When Permutations Are Randomly Drawn (Phipson & Smyth 2010).
    
    % Fraction for accuracy.
    flAcc(iBin) = (sum(decoderPerf(iBin).acc < ...
                       nullAcc(:,iBin)) + 1)./(nBootstraps + 1);

    % Fractions for binary classifiers vectorized over all classes.
    flBinaryPrec(iBin,:) ...
        = (sum(decoderPerf(iBin).binary_prec' < ...
               squeeze(nullBinaryPrec(:,iBin,:))) + 1)./(nBootstraps + 1); 
    flBinaryReca(iBin,:) ...
        = (sum(decoderPerf(iBin).binary_reca' < ...
               squeeze(nullBinaryReca(:,iBin,:))) + 1)./(nBootstraps + 1);
    flBinaryFmea(iBin,:) ...
        = (sum(decoderPerf(iBin).binary_fmea' < ...
               squeeze(nullBinaryFmea(:,iBin,:))) + 1)./(nBootstraps + 1);
    flBinaryAUCPR(iBin,:) ...
        = (sum(decoderPerf(iBin).binary_AUCPR < ...
               squeeze(nullBinaryAUCPR(:,iBin,:))) + 1)./(nBootstraps + 1);
    
    % Fractions for macro metrics.
    flMacroPrec(iBin) = (sum(decoderPerf(iBin).macro_prec < ...
                             nullMacroPrec(:,iBin)) + 1) ./ (nBootstraps + 1);
    flMacroReca(iBin) = (sum(decoderPerf(iBin).macro_reca < ...
                             nullMacroReca(:,iBin)) + 1) ./ (nBootstraps + 1);
    flMacroFmea(iBin) = (sum(decoderPerf(iBin).macro_fmea < ...
                             nullMacroFmea(:,iBin)) + 1) ./ (nBootstraps + 1);
    flMacroAUCPR(iBin) = (sum(decoderPerf(iBin).macro_AUCPR < ...
                             nullMacroAUCPR(:,iBin)) + 1) ./ (nBootstraps + 1);
    
    % Turn fractions into p values.
    switch nvp.compute_pval
        case 'two-sided'
            % Accuracy 
            if flAcc(iBin) < 0.5
                decoderPerf(iBin).p_acc = flAcc(iBin)*2;  
            else 
                decoderPerf(iBin).p_acc = (1 - flAcc(iBin))*2;  
            end
            % Binary performance metrics
            for iClass = 1:nClasses
                % precision
                if flBinaryPrec(iBin,iClass) < 0.5
                    decoderPerf(iBin).p_binary_prec(iClass) = flBinaryPrec(iBin,iClass)*2;
                else 
                    decoderPerf(iBin).p_binary_prec(iClass) = (1 - flBinaryPrec(iBin,iClass))*2; 
                end
                % recall
                if flBinaryReca(iBin,iClass) < 0.5
                    decoderPerf(iBin).p_binary_reca(iClass) = flBinaryReca(iBin,iClass)*2;
                else 
                    decoderPerf(iBin).p_binary_reca(iClass) = (1 - flBinaryReca(iBin,iClass))*2; 
                end
                % F-measure
                if flBinaryFmea(iBin,iClass) < 0.5
                    decoderPerf(iBin).p_binary_fmea(iClass) = flBinaryFmea(iBin,iClass)*2;
                else 
                    decoderPerf(iBin).p_binary_fmea(iClass) = (1 - flBinaryFmea(iBin,iClass))*2; 
                end
                % AUCPR
                if flBinaryAUCPR(iBin,iClass) < 0.5
                    decoderPerf(iBin).p_binary_AUCPR(iClass) = flBinaryAUCPR(iBin,iClass)*2;
                else 
                    decoderPerf(iBin).p_binary_AUCPR(iClass) = (1 - flBinaryAUCPR(iBin,iClass))*2; 
                end
            end
            % Macro precision.
            if flMacroPrec(iBin) < 0.5
                decoderPerf(iBin).p_macro_prec = flMacroPrec(iBin)*2;  
            else 
                decoderPerf(iBin).p_macro_prec = (1 - flMacroPrec(iBin))*2;  
            end
            % Macro recall.
            if flMacroReca(iBin) < 0.5
                decoderPerf(iBin).p_macro_reca = flMacroReca(iBin)*2;  
            else 
                decoderPerf(iBin).p_macro_reca = (1 - flMacroReca(iBin))*2;  
            end
            % Macro F-measure.
            if flMacroFmea(iBin) < 0.5
                decoderPerf(iBin).p_macro_fmea = flMacroFmea(iBin)*2;  
            else 
                decoderPerf(iBin).p_macro_fmea = (1 - flMacroFmea(iBin))*2;  
            end
            % Macro AUCPR.
            if flMacroAUCPR(iBin) < 0.5
                decoderPerf(iBin).p_macro_AUCPR = flMacroAUCPR(iBin)*2;  
            else 
                decoderPerf(iBin).p_macro_AUCPR = (1 - flMacroAUCPR(iBin))*2;  
            end
            
        case 'left'
            decoderPerf(iBin).p_acc = 1 - flAcc(iBin); 
            decoderPerf(iBin).p_binary_prec = 1 - flBinaryPrec(iBin,:);
            decoderPerf(iBin).p_binary_reca = 1 - flBinaryReca(iBin,:);
            decoderPerf(iBin).p_binary_fmea = 1 - flBinaryFmea(iBin,:);
            decoderPerf(iBin).p_binary_AUCPR = 1 - flBinaryAUCPR(iBin,:);
            decoderPerf(iBin).p_macro_prec = 1 - flMacroPrec(iBin,:);
            decoderPerf(iBin).p_macro_reca = 1 - flMacroReca(iBin,:);
            decoderPerf(iBin).p_macro_fmea = 1 - flMacroFmea(iBin,:);
            decoderPerf(iBin).p_macro_AUCPR = 1 - flMacroAUCPR(iBin,:);
            
        case 'right'
            decoderPerf(iBin).p_acc = flAcc(iBin); 
            decoderPerf(iBin).p_binary_prec = flBinaryPrec(iBin,:);
            decoderPerf(iBin).p_binary_reca = flBinaryReca(iBin,:);
            decoderPerf(iBin).p_binary_fmea = flBinaryFmea(iBin,:);
            decoderPerf(iBin).p_binary_AUCPR = flBinaryAUCPR(iBin,:);
            decoderPerf(iBin).p_macro_prec = flMacroPrec(iBin,:);
            decoderPerf(iBin).p_macro_reca = flMacroReca(iBin,:);
            decoderPerf(iBin).p_macro_fmea = flMacroFmea(iBin,:);
            decoderPerf(iBin).p_macro_AUCPR = flMacroAUCPR(iBin,:);
    end
end

end

end
