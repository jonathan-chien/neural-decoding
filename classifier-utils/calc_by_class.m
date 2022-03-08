function [precision,recall,fmeasure] = calc_by_class(confusionMat)
% Calculate precision, recall, and f-measure for each class individually.
% The f-measure is the harmonic mean of precision and recall. For positive
% datasets where not all values are the same, the harmonic mean is the
% lowest of the three Pythagorean means and will thus be biased toward the
% lower of the two of precision and recall (making it a more conservative
% measure in this sense).
%
% PARAMETERS
% ----------
% confMat - nClasses x nClasses confusion matrix. The i_th j_th element
%           belongs to class i and was predicted to be a member of class j.
%
% RETURNS
% -------
% precision -- c-vector of precision values, where c = nClasses and the
%              i_th element is the precision of the i_th class.
% recall    -- c-vetor of recall values, where c = nClasses and the
%              i_th element is the recall of the i_th class.
% fmeasure  -- c-vector of f-measure values, where c = nClasses and the
%              i_th element is the recall of the i_th class.

truePos = diag(confusionMat); 
if isrow(truePos), truePos = truePos'; end % in case future MATLAB versions change orientation

precision = truePos ./ sum(confusionMat, 1)';
recall = truePos ./ sum(confusionMat, 2);
fmeasure = harmmean([precision recall], 2);

end