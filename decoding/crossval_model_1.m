function [metrics,labels,scores] = crossval_model_1(X,groundTruth,nvp)
% Accepts an nObservations x nFeatures (e.g. trials = observations,
% features = neurons) predictor matrix and an nObservations x 1 vector of
% class labels (elements correspond one-to-one to rows of predictors) and
% estimates performance of specified classifier in a cross-validated manner
% using MATLAB's built in routines. Oversampling within folds is not
% supported; for this, see crossval_model_2.
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
%           repetition. 
% scores -- nReps x nObservations x nClasses numeric array (if
%           'oversampleTest' = false (see PARAMETERS)), where the (i,:,:)
%           slice contains the nObservations x nClasses matrix for the i_th
%           repetition, where the j_th k_th element (of this matrix) is the
%           score for the j_th observation being classified into the k_th
%           class. 
%
% Author: Jonathan Chien

arguments
    X
    groundTruth
    nvp.classifier = @fitclinear
    nvp.kFold = 10
    nvp.nReps = 3  
    nvp.useGpu = false
end


% Check number of classes. If multiclass, use fitcecoc and issue warning to
% user if another function handle was passed in.
nClasses = length(unique(groundTruth));
if nClasses > 2
    nvp.classifier = @fitcecoc;
    if ~isequal(nvp.classifier, @fitcecoc)
        disp('@fitcecoc will be used for this nonbinary multi-class problem.')
    end
end

% Option to use GPU.
if nvp.useGpu, X = gpuArray(X); end

% Preallocate containers for confusion matrices, labels, and scores.
confusionMats = NaN(nvp.nReps, nClasses, nClasses);
labels = NaN(nvp.nReps, length(groundTruth));
scores = NaN(nvp.nReps, length(groundTruth), nClasses);
nObs = length(groundTruth);

% Must check this here to perform transposition outside of parfor, else
% "Uninitialized Temporaries' error will be thrown.
if isequal(nvp.classifier, @fitclinear) || isequal(nvp.classifier, @fitcecoc)
    X = X';
end

% Note that MATLAB will randomly partition data into folds each time a CV
% run is called. Hence, no need to permute observations/labels together
% manually.
parfor iRep = 1:nvp.nReps
    
    % Cross-validated model. If 'classifier' is @fitclinear or @fitcecoc,
    % transpose feature matrix for possible speed up.
    if isequal(nvp.classifier, @fitclinear) || isequal(nvp.classifier, @fitcecoc)
        cvModel = nvp.classifier(X, groundTruth, ...
                                 'CrossVal', 'on', 'KFold', nvp.kFold, ...
                                 'ObservationsIn', 'columns');
    else
        cvModel = nvp.classifier(X, groundTruth, ...
                                 'CrossVal', 'on', 'KFold', nvp.kFold);
    end
    
    % Get predicted labels/scores based on cross-validated models.
    [labels(iRep,:), scores(iRep,:,:)] = kfoldPredict(cvModel);
    
    % Calculate confusion matrix.
    confusionMats(iRep,:,:) = confusionmat(labels(iRep,:), groundTruth);

end

% Calculate performance metrics.
metrics = average_repetitions(confusionMats, groundTruth, scores);

% Return averaged confusion matrix (note: this is NOT what we calculated
% the metrics from). Squeeze so that matrices will be returned for scores
% if there is only one rep.
metrics.confusionMat = squeeze(confusionMats);
scores = squeeze(scores);

end
