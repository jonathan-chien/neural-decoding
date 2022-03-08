function performance = classifier_significance(X,groundTruth,nvp)
% Wrapper function accepting an nObservations x nFeatures (e.g. trials =
% observations, features = neurons) predictor matrix and an nObservations x
% 1 vector of class labels (elements correspond one-to-one to rows of
% predictors) and estimates performance of specified classifier in a
% cross-validated manner. Significance measures can be attached to all
% performance metrics via permutation. Oversampling of either or both train
% and test partitions (performed separately for each) within each iteration
% over k folds is also supported through the 'cvFun' name-value pair.
%
% PARAMETERS
% ----------
% X           -- nObservations x nFeatures predictor matrix. 
% groundTruth -- nObservations x 1 vector of integer class labels.
% Name-Value Pairs (nvp)
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
% 
% RETURNS
% -------
% performance -- scalar struct with the following fields:
%   .accuracy  -- Scalar value that is the micro-averaged accuracy across
%                 classes (for an aggregation across folds, this is the sum
%                 of true positive for each class, divided by the total
%                 number of observations; this is also equivalent to
%                 micro-averaged precision, recall, and f-measure),
%                 averaged across repetitions.
%   .precision -- nClasses x 1 vector of precision values (number of true
%                 positives divided by number of predicted positives, where
%                 the i_th element defines positive as membership in the
%                 i_th class), averaged across repetitions.
%   .recall    -- nClasses x 1 vector of recall values (number of true
%                 positives divided by number of actual positives, where
%                 the i_th element defines positive as membership in the
%                 i_th class), averaged across repetitions.
%   .fmeasure  -- nClasses x 1 vector of f-measure scores, calculated as
%                 the elementwise harmonic mean of the precision and recall
%                 vectors, averaged across repetitions.
%   .aucroc    -- nClasses x 1 vector of AUC ROC values (area under the ROC
%                 curve), averaged across repetitions, where the i_th
%                 element regards the i_th class as positive.
%   .aucpr     -- nClasses x 1 vector of AUC PR values (area under the
%                 Precision-Recall curve), averaged across repetitions, ,
%                 where the i_th element regards the i_th class as
%                 positive.
%   .confMat   -- Optionally returned field (if 'saveConfMat' = true)
%                 consisting of an nReps x nClasses x nClasses array where
%                 the (i,:,:) slice contains the nClasses x nClasses
%                 confusion matrix (rows: true class, columns: predicted
%                 class) for the i_th repetition.
%   .labels    -- Optionally returned field (if 'saveLabels' = true and
%                 'oversampleTest' = false) consisting of an nReps x
%                 nObservations numeric array, where the i_th row contains
%                 the predicted labels (aggregated across folds) for the
%                 i_th repetition.
%   .scores    -- Optionally returned field (if 'saveScores' = true and
%                 'oversampleTest' = false) consisting of an nReps x
%                 nObservations x nClasses numeric array, where the (i,:,:)
%                 slice contains the nObservations x nClasses matrix for
%                 the i_th repetition, where the j_th k_th element (of this
%                 matrix) is the score for the j_th observation being
%                 classified into the k_th class.
%   .sig       -- Scalar struct with the following fields (all individual
%                 null performance values, from whose aggregate the
%                 following measures are calculated are generated from
%                 only one repetition of a given permutation).
%       .p        -- Scalar struct with the following fields (note: these p
%                    values are calculated using an observed performance
%                    values that are the average across repetitions).
%           .accuracy  -- p value for micro-averaged accuracy.
%           .precision -- nClasses x 1 vector of p values for precision.
%           .recall    -- nClasses x 1 vector of p values for recall. 
%           .fmeasure  -- nClasses x 1 vector of p values for f-measure.
%           .aucroc    -- nClasses x 1 vector of p values for AUC ROC.
%           .aucpr     -- nClasses x 1 vector of p values for AUC PR.
%       .nullInt  -- Scalar struct with the following fields (size of
%                    interval dictated by the 'nullInt' name-value pair.
%           .accuracy  -- 2-vector (row) whose 1st and 2nd elements are the
%                         lower and upper bounds of the interval on the
%                         accuracy null distribution.
%           .precision -- nClasses x 2 array whose i_th row has as its 1st
%                         and 2nd elements the lower and upper interval
%                         bounds on the precision null distribution, where
%                         the i_th class is regarded as positive.
%           .recall    -- nClasses x 2 array whose i_th row has as its 1st
%                         and 2nd elements the lower and upper interval
%                         bounds on the recall null distribution, where
%                         the i_th class is regarded as positive.
%           .fmeasure  -- nClasses x 2 array whose i_th row has as its 1st
%                         and 2nd elements the lower and upper interval
%                         bounds on the f-measure null distribution, where
%                         the i_th class is regarded as positive.
%           .aucroc    -- nClasses x 2 array whose i_th row has as its 1st
%                         and 2nd elements the lower and upper interval
%                         bounds on the AUC ROC null distribution, where
%                         the i_th class is regarded as positive.
%           .aucpr     -- nClasses x 2 array whose i_th row has as its 1st
%                         and 2nd elements the lower and upper interval
%                         bounds on the AUC PR null distribution, where
%                         the i_th class is regarded as positive.
%       .obsStdev -- Scalar struct with the following fields:
%           .accuracy  -- Scalar value that is the number of standard
%                         deviations from the mean of the null accuracy
%                         distribution that the observed accuracy (averaged
%                         across repetitions) lies.
%           .precision -- nClasses x 1 vector whose i_th element is the
%                         number of standard deviations from the mean of
%                         the null precision distribution (for the i_th
%                         class as positive) that the observed precision
%                         (averaged across repetitions) lies.
%           .recall    -- nClasses x 1 vector whose i_th element is the
%                         number of standard deviations from the mean of
%                         the null recall distribution (for the i_th
%                         class as positive) that the observed recall
%                         (averaged across repetitions) lies.
%           .fmeasure  -- nClasses x 1 vector whose i_th element is the
%                         number of standard deviations from the mean of
%                         the null f-measure distribution (for the i_th
%                         class as positive) that the observed f-measure
%                         (averaged across repetitions) lies.
%           .aucroc    -- nClasses x 1 vector whose i_th element is the
%                         number of standard deviations from the mean of
%                         the null AUC ROC distribution (for the i_th
%                         class as positive) that the observed AUC ROC
%                         (averaged across repetitions) lies.
%           .aucpr     -- nClasses x 1 vector whose i_th element is the
%                         number of standard deviations from the mean of
%                         the null AUC PR distribution (for the i_th
%                         class as positive) that the observed AUC PR
%                         (averaged across repetitions) lies.
% 
% Author: Jonathan Chien.

arguments
    X
    groundTruth
    nvp.classifier = @fitclinear
    nvp.cvFun = 1
    nvp.kFold = 10
    nvp.nReps = 1;
    nvp.oversampleTrain = false
    nvp.oversampleTest = false
    nvp.condLabels = []
    nvp.nResamples = 100
    nvp.useGpu = false
    nvp.pval = 'right-tailed'
    nvp.nullInt = 95
    nvp.nPerms = 1000
    nvp.permute = 'labels'
    nvp.saveLabels = false
    nvp.saveScores = false
    nvp.saveConfMat = false
end

[nObs, nFeatures] = size(X);


%% Cross-validate on observed data

if nvp.cvFun == 1
    [performance, labels, scores] ...
        = crossval_model_1(X, groundTruth, ...
                           'classifier', nvp.classifier, 'nReps', nvp.nReps, ...
                           'kFold', nvp.kFold, 'useGpu', nvp.useGpu);
    if nvp.oversampleTrain | nvp.oversampleTest
        warning("Oversampling will not be performed here. If " + ...
                "desired, use 'cvFun' = 1.")
    end
elseif nvp.cvFun == 2
    [performance, labels, scores] ...
        = crossval_model_2(X, groundTruth, ...
                           'classifier', nvp.classifier, 'nReps', nvp.nReps, ...
                           'kFold', nvp.kFold, 'useGpu', nvp.useGpu, ...
                           'oversampleTrain', nvp.oversampleTrain, ...
                           'oversampleTest', nvp.oversampleTest, ...
                           'condLabels', nvp.condLabels, ...
                           'nResamples', nvp.nResamples);
end

% Optionally save scores and labels (if test set is oversampled, labels and
% scores are returned in cell arrays across reps and do not have guaranteed
% shapes and thus cannot be saved here). Confusion matrices are stored by
% default, but can set them to not be returned to save storage space.
if nvp.saveLabels & ~nvp.oversampleTest, performance.labels = labels; end
if nvp.saveScores & ~nvp.oversampleTest, performance.scores = scores; end
if ~nvp.saveConfMat, performance = rmfield(performance, 'confusionMat'); end


%% Generate and compute on null datasets

if nvp.pval

% Preallocate across permutations.
nullMetrics = cell(nvp.nPerms, 1);

% Calculate set of null values for each metric.
parfor iPerm = 1:nvp.nPerms
    
    % Generate null dataset. Note that for cvFun 2, the condition labels
    % must be permuted to match the trial labels.
    if strcmp(nvp.permute, 'features')
        for iFeature = 1:nFeatures
            nullX(:,iFeature) = X(randperm(nObs),iFeature);
        end
    elseif strcmp(nvp.permute, 'labels')
        nullX = X;
    else
        error("Invalid value for 'permute'.")
    end
    labelPerm = randperm(nObs);
    nullGroundTruth = groundTruth(labelPerm);
    nullCondLabels = nvp.condLabels(labelPerm);

    % Cross-validate on current permuted (null) model. 
    if nvp.cvFun == 1
        [nullMetrics{iPerm},~,~] ...
            = crossval_model_1(nullX, nullGroundTruth, ...
                               'classifier', nvp.classifier, ...
                               'kFold', nvp.kFold, 'useGpu', nvp.useGpu, ...
                               'nReps', 1);
    elseif nvp.cvFun == 2
        [nullMetrics{iPerm},~,~] ...
            = crossval_model_2(nullX, nullGroundTruth, ...
                               'classifier', nvp.classifier, ...
                               'kFold', nvp.kFold, 'useGpu', nvp.useGpu, ...
                               'oversampleTrain', nvp.oversampleTrain, ...
                               'oversampleTest', nvp.oversampleTest, ...
                               'condLabels', nullCondLabels, ...
                               'nResamples', nvp.nResamples, ...
                               'nReps', 1);
    end

    % Remove confusion matrix from struct for current permutation so that
    % downstream processes do not try to calculate significance meausures
    % for it.
    nullMetrics{iPerm} = rmfield(nullMetrics{iPerm}, 'confusionMat');
end

% Combine across permutations. 
nullPerf = combine_perms(nullMetrics);


%% Calculate signficance metrics

metricNames = fieldnames(nullPerf);
for iMetric = 1:length(metricNames)
    currMetric = metricNames{iMetric};
    performance.sig.p.(currMetric) = tail_prob(performance.(currMetric), ...
                                               nullPerf.(currMetric), ...
                                               'exact', false, 'type', nvp.pval); 
    performance.sig.nullInt.(currMetric) ...
        = interval_bounds(nullPerf.(currMetric), nvp.nullInt);
    performance.sig.obsStdev.(currMetric) ...
        = get_num_stdev(performance.(currMetric), nullPerf.(currMetric));
end

end

end


% --------------------------------------------------
function nullPerformance = combine_perms(nullMetrics)
% As of this writing, MATLAB treates an indexing request into a.x, as a
% index into a; as such, even though the parfor iteration loops are clearly
% order-indenpendent, they cannot be proven so to MATLAB. Hence, the
% combination across permutations (run in parallel) is messy and is handled
% separately here, though there are surely better solutions than this one.
% This helper function takes in an nPerms x 1 cell array, where each cell
% contains the metrics output of a call to one of the crossval_model
% functions on one permuted dataset. Returns a struct containing the same
% field names as each metrics struct, but with results across all
% permutations concatenated in a single numeric array.
    
nPerms = length(nullMetrics);

% For each field in each cell of nullMetrics (all cells have same fields),
% create a field in nullPerformance of the same name containing a flexibly
% preallocated matrix (1st dim size = nClasses, which can vary (unless the
% field is accuracy, in which case the 1st dim size is 1), 2nd dim size =
% nPerms). 
nullPerformance = struct();
metricNames = fieldnames(nullMetrics{1});
for iMetric = 1:length(metricNames)
    currMetric = metricNames{iMetric};
    currSize = size(nullMetrics{1}.(currMetric));
    nullPerformance.(currMetric) = NaN( [currSize nPerms] );

    % For current field, concatenate across permutations.
    for iPerm = 1:nPerms
        nullPerformance.(currMetric)(:,iPerm) = nullMetrics{iPerm}.(currMetric);
    end

    % Squeeze, also transpose if column vector so that permutations are
    % along 2nd dimension, consistent across metrics.
    nullPerformance.(currMetric) = squeeze(nullPerformance.(currMetric));
    if iscolumn(nullPerformance.(currMetric))
        nullPerformance.(currMetric) = nullPerformance.(currMetric)';
    end
end

end
