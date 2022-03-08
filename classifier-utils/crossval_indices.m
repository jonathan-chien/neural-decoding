function indices = crossval_indices(nObs,kFold,nvp)
% Returns a vector of class indices whose elements are integers from
% 1:kFold, where kFold = the number of folds to be used in
% cross-validation. The number of observations in each fold will be as
% balanced as possible, though if the total number of observations is not
% evenly divisible by the number of desired folds, the extra observations
% will be assigned beginning with the classes whose indices come first
% ordinally.
% 
% PARAMETERS
% ----------
% nObs  -- Scalar integer that is the number of total observations.
% kFold -- Scalar integer that is the number of desired cross-validation
%          folds.
%
% RETURNS
% -------
% indices -- Vector of indices for cross-validation equal in length to the
%            number of observations. The i_th element is a value k that is 
%            a member of the set {1:nFolds} and assigns the i_th element to
%            the k_th fold.
%       
% Author: Jonathan Chien

arguments
    nObs
    kFold
    nvp.permute = false
end

% Preallocate.
indices = NaN(nObs, 1);

% Assign each observation to a fold.
iFold = 0;
for iObs = 1:nObs
    if iFold < kFold
        iFold = iFold + 1;
    else
        iFold = 1;
    end
    
    indices(iObs) = iFold;
end

% Option to permute indices.
if nvp.permute, indices = indices(randperm(nObs)); end

end
