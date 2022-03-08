function [metrics,labels,scores] = crossval_model_2(X,groundTruth,nvp)
% Accepts an nObservations x nFeatures (e.g. trials = observations,
% features = neurons) predictor matrix and an nObservations x 1 vector of
% class labels (elements correspond one-to-one to rows of predictors) and
% estimates performance of specified classifier in a cross-validated
% manner. Oversampling of either or both train and test partitions
% (performed separately for each) within each iteration over k folds is
% supported.
%
% PARAMETERS
% ----------
% X           -- nObservations x nFeatures predictor matrix. 
% groundTruth -- nObservations x 1 vector of integer class labels.
% Name-Value Pairs (nvp)
%   'classifier'      -- Function handle specifying the classifier to be
%                        used. Default = @fitclinear for binary
%                        classification and @fitcecoc for nClasses > 2.
%   'kFold'           -- Integer value specifying the number of
%                        partitions/folds for cross-validation (default =
%                        10).
%   'nReps'           -- Integer number of repetitions of entire
%                        cross-validation process (performance metrics are
%                        calculated for each repetition and then averaged).
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
% 
% RETURNS
% -------
% metrics -- Scalar struct with the following fields:
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
%   .confMat   -- nReps x nClasses x nClasses array where the (i,:,:) slice
%                 contains the nClasses x nClasses confusion matrix
%                 (rows: true class, columns: predicted class) for the i_th
%                 repetition.
% labels -- nReps x nObservations numeric array (if 'oversampleTest' =
%           false (see PARAMETERS)), where the i_th row contains the
%           predicted labels (aggregated across folds) for the i_th
%           repetition. If 'oversampleTest' evaluates to true, labels is an
%           nReps x 1 cell array, where each cell contains a row vector of
%           length nTestingLabels, where nTestingLabels = the number of
%           predicted labels aggregated across folds for that repetition.
%           Note that when oversampling the test set, the size of the test
%           set can vary if a condition or class is not captured in that
%           test fold (if a condition or class is quite rare). Thus,
%           storage in a cell array is necessary.
% scores -- nReps x nObservations x nClasses numeric array (if
%           'oversampleTest' = false (see PARAMETERS)), where the (i,:,:)
%           slice contains the nObservations x nClasses matrix for the i_th
%           repetition, where the j_th k_th element (of this matrix) is the
%           score for the j_th observation being classified into the k_th
%           class. If 'oversampleTest' evaluates to true, scores is
%           returned as an nReps x 1 cell array, where the i_th cell
%           contains the nTestingLabels x nClasses matrix of scores for the
%           i_th repetition (cell array used for the same reason as with
%           labels above).
%
% Author: Jonathan Chien

arguments
    X
    groundTruth
    nvp.classifier = @fitclinear
    nvp.kFold
    nvp.nReps = 1
    nvp.useGpu = false
    nvp.oversampleTrain = false
    nvp.oversampleTest = false
    nvp.condLabels = []
    nvp.nResamples = 100
end

% Option to use GPU.
if nvp.useGpu, X = gpuArray(X); end

% Check number of classes. If multiclass, use fitcecoc and issue warning to
% user if another function handle was passed in.
nClasses = length(unique(groundTruth));
if nClasses > 2
    nvp.classifier = @fitcecoc;
    if ~isequal(nvp.classifier, @fitcecoc)
        disp('@fitcecoc will be used for this nonbinary multi-class problem.')
    end
end

% Prepare for cross-validation. Preallocate containers for confusion
% matrices, labels, and scores acorss repetitions. 
nObs = length(groundTruth);
confusionMats = NaN(nvp.nReps, nClasses, nClasses);
labels = cell(nvp.nReps, 1);
scores = cell(nvp.nReps, 1);
allTestLabels = cell(nvp.nReps, 1);

parfor iRep = 1:nvp.nReps
        
    % Set up cross-validation indices.
    cvIdc = crossval_indices(nObs, nvp.kFold, 'permute', true);

    % Prepare for accumulation across folds. If no oversampling of test
    % observations are performed, repTestLabels will be identical to
    % repGroundTruth.
    repPredLabels = [];
    repScores = [];
    repTestLabels = [];

    % Cross-validate.
    for k = 1:nvp.kFold

        train = X(cvIdc~=k, :);
        test = X(cvIdc==k, :);
        trainLabels = groundTruth(cvIdc~=k);
        testLabels = groundTruth(cvIdc==k);
        
        % Option to oversample within train set.
        if strcmp(nvp.oversampleTrain, 'byClass')
            [train, trainLabels] ...
                = oversample_by_class(train, trainLabels, groundTruth, nvp.nResamples);
        elseif strcmp(nvp.oversampleTrain, 'byCond')
            assert(~isempty(nvp.condLabels), ...
                   "'condLabels' was passed in empty, but condition labels " + ...
                   "must be supplied if oversampling within condition is desired.")
            [train, ~, trainLabels] ...
                = oversample_by_cond(train, trainLabels, groundTruth, ...
                                     nvp.condLabels(cvIdc~=k), nvp.nResamples);
        elseif ischar(nvp.oversampleTrain)
            error("Invalid value for 'oversample'.")
        end
    
        % Fit classifier on train set. If 'classifier' is @fitclinear
        % or @fitcecoc, transpose feature matrix for possible speed up.
        if isequal(nvp.classifier, @fitclinear) || isequal(nvp.classifier, @fitcecoc)
            train = train'; 
            model = nvp.classifier(train, trainLabels, 'ObservationsIn', 'columns');
        else
            model = nvp.classifier(train, trainLabels);
        end

        % Option to oversample test set.
        if strcmp(nvp.oversampleTest, 'byClass')
            [test, testLabels] ...
                = oversample_by_class(test, testLabels, groundTruth, nvp.nResamples);
        elseif strcmp(nvp.oversampleTest, 'byCond')
            assert(~isempty(nvp.condLabels), ...
                   "'condLabels' was passed in empty, but condition labels " + ...
                   "must be supplied if oversampling within condition is desired.")
            [test, ~, testLabels] ...
                = oversample_by_cond(test, testLabels, groundTruth,  ...
                                     nvp.condLabels(cvIdc==k), nvp.nResamples);
        elseif ischar(nvp.oversampleTest)
            error("Invalid value for 'oversampleTest'.")
        end
    
        % Test on held out test set.
        [foldPredLabels, foldScores] = predict(model, test);
        
        % Aggregate predicted labels, scores, and test labels from current
        % fold.
        repPredLabels = [repPredLabels; foldPredLabels];
        repScores = [repScores; foldScores];
        repTestLabels = [repTestLabels; testLabels];
        
    end

    % Calculate confusion matrix. 
    confusionMats(iRep,:,:) = confusionmat(repPredLabels, repTestLabels);
    
    % Store labels and scores from current rep.
    labels{iRep} = repPredLabels';
    scores{iRep} = repScores;
    allTestLabels{iRep} = repTestLabels;

end

% Calculate performance metrics.
metrics = average_repetitions(confusionMats, allTestLabels, scores);

% Convert to numeric array if no oversampling of test (if there was
% oversampling test, some conditions may not have been captured in a given
% fold, resulting in a different number of test labels for that fold).
if ~nvp.oversampleTest
    labels = to_numeric_array(labels);
    scores = to_numeric_array(scores);
end

% Return averaged confusion matrix (note: this is NOT what we calculated
% the metrics from). Squeeze so that matrices will be returned for scores
% if there is only one rep.
metrics.confusionMat = squeeze(confusionMats);
scores = squeeze(scores);

end


% --------------------------------------------------
function numericArray = to_numeric_array(cellArray)
% Accepts as input cellArray, an m x 1 cell array, where the i_th cell
% contains a numeric array of size n x p, and returns a numeric array of
% squeeze(m x n x p).

nCells = length(cellArray);
numArraySize = size(cellArray{1});

numericArray = NaN([nCells numArraySize]);

for iCell = 1:nCells
    numericArray(iCell,:,:) = cellArray{iCell};
end

numericArray = squeeze(numericArray);

end

% --------------------------------------------------
function [overPred,overClassLabels] ...
    = oversample_by_class(pred,classLabels,groundTruth,nResamples)
% For a vector of class labels whose i_th element denotes the class
% membership (index) of the i_th obvservation, and a corresponding
% predictor matrix, this function oversamples (resamples observations)
% within classes for each feature/neuron independently. Note that if a
% class consists of two or more conditions, this difference between
% conditions will not be recognized; all conditions within a class will be
% treated the same; if this behavior is undesired, oversample_by_cond
% should be called instead.


% Derive class indices from the groundTruth vector, as there is technically
% no guarantee all classes will be represented in current train set. There
% is no other use for groundTruth in this function.
classIdc = unique(groundTruth); 
nClasses = length(classIdc);

% Preallocate.
byClass = cell(nClasses, 1);
nNeurons = size(pred, 2);
overPred = NaN(nResamples * nClasses, nNeurons);

% For each class, first extract an nObs x nFeatures/Neurons matrix.
% Store these in the corresponding cell, since the 1st dim sizes
% may be uneven across classes if class sizes are imbalanced. Then
% resample for each feature independently, within current class.
for iClass = 1:nClasses
    % Extract observations from current class.
    byClass{iClass} = pred(classLabels==classIdc(iClass),:);

    % Resample for each feature/neuron indpendently.
    for iNeuron = 1:nNeurons
        overPred((iClass-1)*nResamples+1 : iClass*nResamples, iNeuron) ...
            = datasample(byClass{iClass}(:,iNeuron), nResamples, 'Replace', true);
    end
end

% Assign trainLabels to match oversampled data.
overClassLabels = repelem(classIdc, nResamples);

end


% --------------------------------------------------
function [overPred,overCondLabels,overClassLabels] ...
    = oversample_by_cond(pred,classLabels,groundTruth,condLabels,nResamples)
% For a vector of class labels whose i_th element denotes the class
% membership (index) of the i_th ovservations, a vector of condition labels
% whose i_th element denotes the condition index of the i_th observation,
% and a corresponding predictor matrix, this function oversamples
% (resamples observations) within conditions for each feature/neuron
% independently. Note that if a class consists of two or more conditions,
% each condition will still be independently oversampled so that the number
% of observations from each condition is the same. As a final step,
% however, this function will randomly subsample (if necessary) so that the
% number of observations representing each class is the same.


% If true, ensure that class sizes are the same (regardless, each class is
% guaranteed to consist of an equal number of trials from each of its
% constituent conditions).
EVEN_OUT = true;


% Get indices of classes, conditions. Derive class indices from the
% groundTruth vector, as there is technically no guarantee all classes will
% be represented in current train set. There is no other use for
% groundTruth in this function.
classIdc = unique(groundTruth);
nClasses = length(classIdc);
condIdc = unique(condLabels);
nConds = length(condIdc);

% Determine membership of conditions in classes. classMembership is a
% vector of length = nConds whose i_th element is the label of the class to
% which the i_th condition belongs.
classMembership = NaN(nConds, 1);
for iCond = 1:nConds
for iClass = 1:nClasses
    if any(condLabels == condIdc(iCond) & classLabels == classIdc(iClass))
        classMembership(iCond) = classIdc(iClass);
    end
end
end

% Preallocate for oversampling.
byCond = cell(nConds, 1);
nNeurons = size(pred, 2);
overPred = NaN(nResamples * nConds, nNeurons);
overClassLabels = NaN(nConds * nResamples, 1);

% For each condition, extract an nTrials x nNeurons matrix (with rows being
% all trials deriving from the current condition) and use it to oversample.
for iCond = 1:nConds
    % Extract observations from current condition.
    byCond{iCond} = pred(condLabels==condIdc(iCond),:);

    % For each neuron/feature, independently resample trials/observations.
    for iNeuron = 1:nNeurons
        overPred((iCond-1)*nResamples+1 : iCond*nResamples, iNeuron) ...
            = datasample(byCond{iCond}(:,iNeuron), nResamples, 'Replace', true);
    end

    % Add to trainLabels.
    overClassLabels((iCond-1)*nResamples+1 : iCond*nResamples) = classMembership(iCond);
end

% Assign condition labels to match the oversampled data.
overCondLabels = repelem(condIdc, nResamples);


%---------- Optional balancing of class sizes via subsampling-------------%

% Determime number of trials/observations in each class.
nTrialsByClass= sum(overClassLabels == classIdc');

% If classes are of different sizes, subsample observations (from train and
% cond labels, as well as rows of predictor matrix) so that the number of
% observations representing each condition are the same. The number
% subsampled is equal to the number of obserations representing the
% smallest class after oversampling.
if range(nTrialsByClass) > 0 && EVEN_OUT
    minClassSize = min(nTrialsByClass);
    subTrain = NaN(minClassSize * nClasses, nNeurons);
    subCondLabels = NaN(minClassSize * nClasses, 1);
    for iClass = 1:nClasses
        % First sample indices, then apply them to predictor matrix and
        % labels vectors.
        selectionVec = datasample(find(overClassLabels==classIdc(iClass)), ...
                                  minClassSize, 'Replace', false);
        subTrain((iClass-1)*minClassSize+1 : iClass*minClassSize, :) ...
            = overPred(selectionVec,:);
        subCondLabels((iClass-1)*minClassSize+1 : iClass*minClassSize) ...
            = overCondLabels(selectionVec);
    end

    % Assign subTrainLabels to reflect subsampling.
    subTrainLabels = repelem(classIdc, minClassSize);

    % Reassign function outputs using subsampled values.
    overPred = subTrain;
    overCondLabels = subCondLabels;
    overClassLabels = subTrainLabels;
end

end
