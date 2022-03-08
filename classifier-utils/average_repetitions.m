function metrics = average_repetitions(confusionMats,groundTruth,scores)
% For a series of repetitions of k-fold cross-validation, compute
% performance metrics on each repetition and average results across
% repetitions.
%
% PARAMETERS
% ----------
% confusionMats -- nReps x nClasses x nClasses array of confusion matrices.
%                  The slice (i,:,:) contains the confusion matrix for the
%                  i_th repetition. 
% groundTruth   -- nReps x 1 cell array, where each cell contains an
%                  nObservations x 1 vector (aggregated across folds) of
%                  labels.
% scores        -- nReps x 1 cell array, where each cell contain an
%                  nObservations x nClasses array of scores, where the i_th
%                  j_th element is the score for the i_th observation being
%                  classified into the j_th class.
%
% RETURNS
% metrics -- Scalar struct witht the following fields:
%   .accuracy  -- Scalar value, accuracy averaged across repetitions.
%   .precision -- nClasses x 1 vector of precision values, averaged across
%                 repetitions, where the i_th element has the i_th class as
%                 positive.
%   .recall    -- nClasses x 1 vector of recall values, averaged across
%                 repetitions, where the i_th element has the i_th class as
%                 positive.
%   .fmeasure  -- nClasses x 1 vector of f-measure values, averaged across
%                 repetitions, where the i_th element has the i_th class as
%                 positive.
%   .aucroc    -- nClasses x 1 vector of AUC ROC values, averaged across
%                 repetitions, where the i_th element has the i_th class as
%                 positive.
%   .aucpr     -- nClasses x 1 vector of AUC PR values, averaged across
%                 repetitions, where the i_th element has the i_th class as
%                 positive.
%
% Author: Jonathan Chien.


[nReps, nClasses, ~] = size(confusionMats);

metrics = struct('accuracy', NaN(1, nReps), ...
                 'precision', NaN(nClasses, nReps), ...
                 'recall', NaN(nClasses, nReps), ...
                 'fmeasure', NaN(nClasses, nReps), ...
                 'aucroc', NaN(nClasses, nReps), ...
                 'aucpr', NaN(nClasses, nReps));

% Calculate metrics for each repetition.
for iRep = 1:nReps
    % Get i_th confusion matrix.
    current = squeeze(confusionMats(iRep,:,:));

    % Accuracy.
    metrics.accuracy(iRep) = calc_accuracy(current);

    % Precision, recall, and f-measure.
    [metrics.precision(:,iRep), metrics.recall(:,iRep), metrics.fmeasure(:,iRep)] ...
        = calc_by_class(current);

    % AUCROC and AUCPR.
    [metrics.aucroc(:,iRep), metrics.aucpr(:,iRep)] ...
            = calc_auc(groundTruth{iRep}, scores{iRep});
end

% Average across repetitions.
fnames = fieldnames(metrics);
for iField = 1:length(fnames)
    metrics.(fnames{iField}) = mean(metrics.(fnames{iField}), 2);
end

end
