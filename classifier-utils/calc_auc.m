function [aucroc,aucpr] = calc_auc(groundTruth,scores)
% Calculate AUC for each class (that is, letting each class be the positive
% class) based on both the ROC curve and the Precision-Recall (PR) curve.
% Note that AUC ROC is generally sensitive to algorithms biased toward the
% majority class, but under extreme class rarity, it may be unreliable, and
% AUC PR may be preferrable.
%
% PARAMETERS
% ----------
% groundTruth -- n-vector of ground truth labels over observations, where n
%                = number of observations.
% scores      -- nObs x 2 matrix of scores, where the i_th j_th element
%                contains the score for classifying observation i into
%                class j.
% 
% RETURNS
% -------
% aucroc -- c-vector of AUC ROC values, where c = nClasses.
% aucpr  -- c-vecotr of AUC PR values, where c = nClasses.
%
% Author: Jonathan Chien

% Get class indices and number of classes.
classIdc = unique(groundTruth);
nClasses = length(classIdc);  

% Calculate AUC based on both ROC and Precision-Recall (PR) curves.
aucroc = NaN(nClasses, 1);
aucpr = NaN(nClasses, 1);
for iClass = 1:nClasses
    [~,~,~,aucroc(iClass)] ...
        = perfcurve(groundTruth, scores(:,iClass), classIdc(iClass));
    [~,~,~,aucpr(iClass)] ...
        = perfcurve(groundTruth, scores(:,iClass), classIdc(iClass), ...
                    'XCrit', 'reca', 'YCrit', 'prec');
end

end
