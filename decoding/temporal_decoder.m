function performance = temporal_decoder(BxTxN,groundTruth,nvp)
% Wrapper function accepting as input a m x n x p array consisting of m
% datasets, each of which is an n x p slice featuring n observations across
% p features. For example, m could index bins, n single trials, and p
% neurons. For each n x p slice of the input array, a classifier model will
% be evaluated via k-fold cross-validation. If desired, significance
% measures such as p values will be computed via permutation.
%
% PARAMETERS
% ----------
% BxTxN       -- b x t x n array of neural firing rates, where b = nBins,
%                t = nTrials, and n = nNeurons. The t trials may draw from
%                some a set of c conditions, and each of the c conditions
%                may or may not be equally represented.
% groundTruth -- nObservations x 1 vector of integer class labels; each
%                label corresponds to some class recognized by the
%                classifier, not necessarily a task condition.
% Name-Value Pairs (nvp)
%   'dropIdc'         -- Vector of length q, where q is the number of
%                        neurons (out of the p total) to be dropped for
%                        decoding across all bins/datasets. Elements
%                        correspond to indices of the dropped neurons in
%                        the overall population. Default is empty.
%   'classifier'      -- Function handle specifying the classifier to be
%                        used. Default = @fitclinear for binary
%                        classification and @fitcecoc for nClasses > 2.
%   'cvFun'           -- (1 (default) | 2), specify whether to use
%                        crossval_model_1 or crossval_model_2 to evaluate
%                        model.
%   'kFold'           -- Integer value specifying the number of
%                        partitions/folds for cross-validation (default =
%                        10).
%   'nReps'           -- Integer number of repetitions of entire
%                        cross-validation process (performance metrics are
%                        calculated for each repetition and then averaged)
%                        for the unpermuted data only. Only one repetition
%                        is performed for each permuted dataset.
%   'useGpu'          -- (1 | 0 (default)), specify whether or not to
%                        convert predictor matrix to GPU array. Note that
%                        GPU arrays are not supported by all MATLAB
%                        classifier functions.
%   'oversampleTrain' -- (string | false (default)), specify whether or not
%                        to oversample train set. If so, specify either
%                        'byClass' to oversample within classes or 'byCond'
%                        to oversample within conditions (see 'condLabels'
%                        below). Set false to suppress oversampling.
%   'oversampleTest'  -- (string | false (default)), specify whether or not
%                        to oversample test set. If so, specify either
%                        'byClass' to oversample within classes or 'byCond'
%                        to oversample within conditions (see 'condLabels'
%                        below). Set false to suppres oversampling.
%   'condLabels'      -- nObservations x 1 vector in one-to-one
%                        correspondence to both the rows of X and the
%                        elements of groundTruth. The i_th element of this 
%   'nResamples'      -- Integer number of resamples to take (either within
%                        each class or within each condition, see
%                        'oversampleTrain' and 'oversampleTest' above).
%   'pval'            -- ('right-tailed' (default) | 'left-tailed' |
%                        'two-tailed' | false). Specify the sidedness of
%                        the test. If false, no permutations will be
%                        performed (resulting in no signfiicance measures
%                        of any kind); it may be useful to suppress these
%                        permutations if this function is called by another
%                        function, and significance measures are not
%                        desired at that time. 
%   'nullInt'         -- Scalar value that is the null distribution
%                        interval size to be returned.          
%   'nPerms'          -- Scalar value specifying the number of random
%                        permutations (each resulting in one sample from
%                        the null distribution). Default = 1000.
%   'permute'         -- ('features' | 'labels' (default)), string value
%                        specifying how to generate each dataset (one
%                        permutation). If 'features', each column of the
%                        predictor matrix (vector across observations for
%                        one feature) is permuted independently; this
%                        essentially shuffles the labels with respect to
%                        the observations, independently for each feature.
%                        If 'labels', the labels themselves are permuted
%                        once (leaving intact any correlations among
%                        features).
%   'saveLabels'      -- (1 | 0 (default)), specify whether or not to save
%                        the predicted labels (for the original unpermuted
%                        data only). Note that labels cannot be returned if
%                        'oversampleTest' evaluates to true.
%   'saveScores'      -- (1 | 0 (default)), specify whether or not to save
%                        the predicted scores (for the original unpermuted
%                        data only). Note that scores cannot be returned if
%                        'oversampleTest' evaluates to true.
%   'saveConfMat'     -- (1 | 0 (default)), specify whether or not to save
%                        the confusion matrices (for each of the reptitions
%                        on the original unpermuted data only).
%   'concatenate'     -- (1 (default) | 0), if true, results from each bin
%                        will be concatenated in a single numeric array
%                        under the corresponding field name. If false,
%                        performance will be returned as an nBins x 1 cell
%                        array, where each cell contains the output of
%                        classifier_signficance for the corresponding bin.
%
% RETURNS
% -------
% performance -- If 'concatenate' = false, an nBins x 1 cell array, where
%                the i_th cell contains the output of
%                classifier_significance.m (see RETURNS under that
%                function's documentation) for the i_th bin. If
%                'concatenate' = true, performance is a scalar struct with
%                the following fields:
%   .accuracy  -- nBins x 1 vector whose i_th element is the the
%                 micro-averaged accuracy across classes (for an
%                 aggregation across folds, this is the sum of true
%                 positive for each class, divided by the total number of
%                 observations; this is also equivalent to micro-averaged
%                 precision, recall, and f-measure), averaged across
%                 repetitions, for the i_th bin/dataset.
%   .precision -- nBins x nClasses matrix whose i_th j_th element is the
%                 precision (number of true positives divided by number of
%                 predicted positives) for the j_th class as positive,
%                 averaged across repetitions, in the i_th bin.
%   .recall    -- nBins x nClasses matrix whose i_th j_th element is the
%                 precision (number of true positives divided by number of
%                 actual positives) for the j_th class as positive,
%                 averaged across repetitions, in the i_th bin.
%   .fmeasure  -- nBins x nClasses matrix of f-measure scores whose i_th
%                 j_th element is the f-measure for the j_th class as
%                 positive, averaged across repetitions, in the i_th bin.
%   .aucroc    -- nBins x nClasses matrix of AUC ROC values whose i_th
%                 j_th element is the AUC ROC for the j_th class as
%                 positive, averaged across repetitions, in the i_th bin.
%   .aucpr     -- nBins x nClasses matrix of AUC PR (precision-recall)
%                 values whose i_th j_th element is the AUC PR for the j_th
%                 class as positive, averaged across repetitions, in the
%                 i_th bin.
%   .confMat   -- Optionally returned field (if 'saveConfMat' = true)
%                 consisitng of an nBins x nReps x nClasses x nClasses
%                 array where the (i,j,:,:) slice contains the nClasses x
%                 nClasses confusion matrix (rows: true class, columns:
%                 predicted class) for the j_th repetition in the i_th bin.
%   .labels    -- Optionally returned field (if 'saveLabels' = true and
%                 'oversampleTest' = false) consisting of an nBins x nReps
%                 x nObservations numeric array, where the i_th row
%                 contains the predicted labels (aggregated across folds)
%                 for the i_th repetition.
%   .scores    -- Optionally returned field (if 'saveScores' = true and
%                 'oversampleTest' = false) consisting of an nBins x nReps
%                 x nObservations x nClasses numeric array, where the
%                 (i,j,:,:) slice contains the nObservations x nClasses
%                 matrix for the j_th repetition in the i_th bin, and the
%                 k_th l_th element (of this slice) is the score for the
%                 k_th observation being classified into the l_th class.
%   .sig       -- Scalar struct with the following fields (all individual
%                 null performance values, from whose aggregate the
%                 following measures are calculated are generated from
%                 only one repetition of a given permutation).
%       .p        -- Scalar struct with the following fields (note: these p
%                    values are calculated using an observed performance
%                    values that are the average across repetitions).
%           .accuracy  -- nBinx x 1 vector of p values for micro-averaged
%                         accuracy.
%           .precision -- nBins x nClasses matrix of p values for precision.
%           .recall    -- nBins x nClasses matrix of p values for recall. 
%           .fmeasure  -- nBins x nClasses matrix of p values for f-measure.
%           .aucroc    -- nBins x nClasses matrix of p values for AUC ROC.
%           .aucpr     -- nBins x nClasses matrix of p values for AUC PR.
%       .nullInt  -- Scalar struct with the following fields (size of
%                    interval dictated by the 'nullInt' name-value pair.
%           .accuracy  -- nBins x 2 matrix whose i_th row has as its 1st
%                         and 2nd elements the lower and upper bounds of
%                         the interval on the accuracy null distribution
%                         for the i_th bin.
%           .precision -- nBins x nClasses x 2 array where the (i,:,:)
%                         slice is a matrix whose j_th row has as its 1st
%                         and 2nd elements the lower and upper interval
%                         bounds on the precision null distribution, where
%                         the j_th class is positive, in the i_th bin.
%           .recall    -- nBins x nClasses x 2 array where the (i,:,:)
%                         slice is a matrix whose j_th row has as its 1st
%                         and 2nd elements the lower and upper interval
%                         bounds on the recall null distribution, where
%                         the j_th class is positive, in the i_th bin.
%           .fmeasure  -- nBins x nClasses x 2 array where the (i,:,:)
%                         slice is a matrix whose j_th row has as its 1st
%                         and 2nd elements the lower and upper interval
%                         bounds on the f-measure null distribution, where
%                         the j_th class is positive, in the i_th bin.
%           .aucroc    -- nBins x nClasses x 2 array where the (i,:,:)
%                         slice is a matrix whose j_th row has as its 1st
%                         and 2nd elements the lower and upper interval
%                         bounds on the AUC ROC null distribution, where
%                         the j_th class is positive, in the i_th bin.
%           .aucpr     -- nBins x nClasses x 2 array where the (i,:,:)
%                         slice is a matrix whose j_th row has as its 1st
%                         and 2nd elements the lower and upper interval
%                         bounds on the AUC PR null distribution, where
%                         the j_th class is positive, in the i_th bin.
%       .obsStdev -- Scalar struct with the following fields:
%           .accuracy  -- nBins x 1 vector whose i_th element is the number
%                         of standard deviations from the mean of the null
%                         accuracy distribution that the observed accuracy
%                         (averaged across repetitions) lies, for the i_th
%                         bin.
%           .precision -- nBins x nClasses matrix whose i_th j_th element
%                         is the number of standard deviations from the
%                         mean of the null precision distribution (for the
%                         j_th class as positive) that the observed
%                         precision (averaged across repetitions) lies, for
%                         the i_th bin.
%           .recall    -- nBins x nClasses matrix whose i_th j_th element
%                         is the number of standard deviations from the
%                         mean of the null recall distribution (for the
%                         j_th class as positive) that the observed recall
%                         (averaged across repetitions) lies, for the i_th
%                         bin.
%           .fmeasure  -- nBins x nClasses matrix whose i_th j_th element
%                         is the number of standard deviations from the
%                         mean of the null f-measure distribution (for the
%                         j_th class as positive) that the observed
%                         f-measure (averaged across repetitions) lies, for
%                         the i_th bin.
%           .aucroc    -- nBins x nClasses matrix whose i_th j_th element
%                         is the number of standard deviations from the
%                         mean of the null AUC ROC distribution (for the
%                         j_th class as positive) that the observed AUC ROC
%                         (averaged across repetitions) lies, for the i_th
%                         bin.
%           .aucpr     -- nBins x nClasses matrix whose i_th j_th element
%                         is the number of standard deviations from the
%                         mean of the null AUC PR distribution (for the
%                         j_th class as positive) that the observed AUC PR
%                         (averaged across repetitions) lies, for the i_th
%                         bin.
% 
% Author: Jonathan Chien

arguments
    BxTxN {mustBeNumeric}
    groundTruth {mustBeInteger}
    nvp.dropIdc = []
    nvp.classifier = @fitclinear
    nvp.kFold {mustBeInteger} = 5 
    nvp.nReps = 25
    nvp.cvFun = 1
    nvp.useGpu = false
    nvp.oversampleTrain = false
    nvp.oversampleTest = false
    nvp.condLabels = []
    nvp.nResamples = 100  
    nvp.pval = false 
    nvp.nullInt = 95
    nvp.nPerms {mustBeInteger} = 1000
    nvp.permute = 'labels'
    nvp.saveLabels = false;
    nvp.saveScores = false;
    nvp.saveConfMat = false;
    nvp.concatenate = true
end

% Remove any neurons/features specified by user.
BxTxN(:,:,nvp.dropIdc) = [];
[nBins, ~, ~] = size(BxTxN);


%% Train and test on empirical data

% Preallocate.
performance = cell(nBins, 1);

for iBin = 1:nBins 
    % Obtain slice corresponding to nTrials x nNeurons for current bin.
    TxN = squeeze(BxTxN(iBin,:,:));
    
    % Train and test current slice using cross-validation; attach metrics
    % of signficance.
    performance{iBin} ...
        = classifier_significance(TxN, groundTruth, ...
                                  'classifier', nvp.classifier, ...
                                  'cvFun', nvp.cvFun, ...
                                  'useGpu', nvp.useGpu, ...
                                  'oversampleTrain', nvp.oversampleTrain, ...
                                  'oversampleTest', nvp.oversampleTest, ...
                                  'condLabels', nvp.condLabels, ...
                                  'nResamples', nvp.nResamples, ...
                                  'kFold', nvp.kFold, ...
                                  'nReps', nvp.nReps, ...
                                  'nPerms', nvp.nPerms, ...
                                  'permute', nvp.permute, ...
                                  'saveLabels', nvp.saveLabels, ...
                                  'saveLabels', nvp.saveLabels, ...
                                  'saveConfMat', nvp.saveConfMat, ...
                                  'saveScores', nvp.saveScores, ...
                                  'pval', nvp.pval, 'nullInt', nvp.nullInt);
end

% Optionally concatenate fields to return a single struct with
% multi-dimensional array fields, rather than a cell array of structs.
if nvp.concatenate, performance = combine_bins(performance); end

end


% --------------------------------------------------
function performanceCat = combine_bins(performance)
% As of this writing, MATLAB treates an indexing request into a.x, as a
% index into a; as such, even though the parfor iteration loops are clearly
% order-independent, they cannot be proven so to MATLAB. Hence, the
% combination across bins (run in parallel) is messy and is handled
% separately here, though there are surely better solutions than this one.
% This helper function takes in an nBins x 1 cell array, where each cell
% contains the performance output of a call to classifier_significance on
% one dataset (bin). Returns a struct containing the same field names as
% each performance struct, but with results across all bins concatenated in
% a single numeric array.


nBins = length(performance);
performanceCat = struct();


% Get metric field names. metricNames refer to fields storing performance
% metrics e.g. accuracy, fmeasure, etc. Thus, we exlude the field 'sig',
% which contains substructures with significance measures. Do not modify
% the iterable inside the loop.
iRemove = [];
metricNames = fieldnames(performance{1});
for iMetric = 1:length(metricNames) % remove 'sig' field if it exists
    if strcmp(metricNames{iMetric}, 'sig')
        iRemove = iMetric;
    end
end
metricNames(iRemove) = [];

% Get significance field names. sigNames refers to fields storing
% significance info, e.g. p, nullInt, etc (if significance was assessed,
% otherwise sigNames will be empty).
if isfield(performance{1}, 'sig')
    sigNames = fieldnames(performance{1}.sig);
else
    sigNames = [];
end


% Handle all fields storing metric info.
for iMetric = 1:length(metricNames)
    % Get size of all nonsingleton dims, append dim of size nBins.
    currSize = size(performance{1}.(metricNames{iMetric})); 
    performanceCat.(metricNames{iMetric}) = NaN( [nBins currSize] );

    % Concatenate across bins. % Three ':' because confusion matrix is
    % inherently 2D and the array storing confusion matrices across
    % repetitions will thus be 3D. MATLAB ignores extra ':'s. Note that
    % this is where classes go from being in rows to being in columns.
    for iBin = 1:nBins
        performanceCat.(metricNames{iMetric})(iBin,:,:,:) ...
            = performance{iBin}.(metricNames{iMetric});     
    end

    % Squeeze, in case there is only one dimension other than nBins. 
    performanceCat.(metricNames{iMetric}) ...
        = squeeze(performanceCat.(metricNames{iMetric}));
end


% Handle fields for signficance. First, remove confusionMat, labels, and
% scores fields from metricNames, if they exist (i.e., if user wishes to
% retain them for the original data), as their significance was not
% assessed. 
iRemove = [];
for iMetric = 1:length(metricNames)
    if any(strcmp(metricNames{iMetric}, {'confusionMat', 'labels', 'scores'}))
        iRemove = [iRemove; iMetric];
    end
end
metricNames(iRemove) = [];

% Concatenate significance values.
if ~isempty(sigNames)
for iSig = 1:length(sigNames)
for iMetric = 1:length(metricNames)
    
    % Get size of all nonsingleton dims, append dim of size nBins.
    currSize = size(performance{1}.sig.(sigNames{iSig}).(metricNames{iMetric})); 
    performanceCat.sig.(sigNames{iSig}).(metricNames{iMetric}) ...
        = NaN( [nBins currSize] );

    % Concatenate across bins. % Two ':' because some
    % significance-associate measures, like an interval on the null
    % distribution have two values per class, hence the extra dimension.
    % MATLAB ignores extra ':'s. Note that this is where classes go from
    % being in rows to being in columns.
    for iBin = 1:nBins
        performanceCat.sig.(sigNames{iSig}).(metricNames{iMetric})(iBin,:,:) ...
            = performance{iBin}.sig.(sigNames{iSig}).(metricNames{iMetric});
    end  

    % Squeeze, in case there is only one dimension other than
    % nBins. 
    performanceCat.sig.(sigNames{iSig}).(metricNames{iMetric}) ...
        = squeeze(performanceCat.sig.(sigNames{iSig}).(metricNames{iMetric}));
end
end
end

end
