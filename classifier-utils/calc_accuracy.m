function accuracy = calc_accuracy(confusionMat)
% Calculate multi-class (including binary) accuracy = micro-precision,
% micro-recall, micro-fmeasure.
% 
% PARAMETERS
% ----------
% confusionMat -- nClasses x nClasses confusion matrix.
%
% RETURNS
% -------
% meanAccuracy -- Accuray as a proportion, between 0 and 1, of true
%                 positives (for each class) over total number of
%                 observations.
%
% Author: Jonathan Chien

accuracy = trace(confusionMat) / sum(confusionMat, 'all');

end
