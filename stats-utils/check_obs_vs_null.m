function [observed,null,nullDistrSize] = check_obs_vs_null(observed,null)
% This is a preprocessing function intended to be called before array
% operations in functions like tail_prob and montecarlo_conf_int.
% Essentially, we want to ensure that the arrays that these functions
% operate on are standardized (correct dim sizes and orientation) etc. to
% make the code more robust against errors or subtler bugs/unexpected
% behavior due to unanticipated singleton expansions, etc. For a single
% random variable, we want: observed (1,1) vs null (1, nNullValues). For a
% vector of random variables, we want: observed (nVars, 1) vs null (nVars,
% nNullValues). The reason that this is standardized (instead of allowing
% the user to set "dim" as in many MATLAB functions) is that deep inside of
% a calling function we know the shape of arrays but may not always be
% exactly certain of their orientations, especially if they have passed
% through many other processes; this function ensures that unexpected
% orientations don't cause issues (it's not intended to allow the user to
% lazily not track the orientation of arrays, but rather acts as an extra
% safeguard against unanticipated behavior).
% 
% PARAMETERS
% ----------
% observed -- Either a scalar (one measurement for one random variable) or
%             an m-vector of single measurements over m random variables.
%             If a vector, "observed" may be passed in as either a row or a
%             column vector, and the function will ensure that it is
%             returned with the proper orientation (as a column).
% null     -- Either a vector (corresponding to cases where "observed" is a
%             scalar) or a matrix (corresponding to those cases where
%             "observed" is a vector). If a vector, null should be of
%             length n, where n is the number of values from the null
%             distribution that we are working with. If a matrix, null
%             should be either m x n or n x m, where m and n are as defined
%             above (m = nVars, n = nNull). Note that this means that one
%             of the dim sizes of "null" must match the length of
%             "observed". Errors will be thrown if violations of these
%             conditions are detected amongst the input arguments. Note as
%             well that the case where m = n will result in a warning to
%             the user that nVars and nNull cannot be safely inferred via a
%             match to the length of the vector observed.
%
% RETURNS
% -------
% observed      -- This returned variable corresponds exactly to the input
%                  "observed"; however, if "observed" was passed in as a
%                  vector, it will be returned here as a column vector.
% null          -- This returned variable corresponds exactly to the input
%                  "null". However, if "null" was passed in as a vector, it
%                  is returned here as a row vector, and if "null" was
%                  passed in as a matrix, it is returned here such that the
%                  null values correspond to columns (i.e., null is nVars x
%                  nNullValues).
% nullDistrSize -- Scalar value that is the number of null values
%                  (corresponds to length of "null" if "null" is a vector
%                  and to the second dimension of "null" if "null" is a
%                  matrix).
%
% Author: Jonathan Chien 8/20/21


% Ensure inputs are arrays of correct sizes.
assert(length(size(observed)) <= 2, ...
       '"observed" must be either a scalar or a vector.')
assert(length(size(null)) == 2 || length(size(null)) == 3, ...
       '"null" must be either a vector or a matrix.')

   
% Handle case where obs is scalar.
if all(size(observed)==1) 
    
    % Ensure that "null" is vector.
    assert(~all(size(null)>1), ...
           "If 'observed' passed in as a scalar, 'null' must be a vector.")
    
    % Ensure that "null" is row vector.
    if iscolumn(null), null = null'; end
    
    % Determine number of observations in "null".
    nullDistrSize = length(null);
end


% Handle case where "observed" is a vector.
if any(size(observed)>1) 
    
    % Ensure that "observed" is a column vector.
    if isrow(observed), observed = observed'; end
    
    % Ensure that "null" is a matrix.
    assert(all(size(null)>1), ...
           "If 'observed' passed in as a vector, 'null' must be a matrix.")
    
    % Ensure that "observed" is a column vector.
    if size(null, 1) == size(null, 2) && size(null, 1) == length(observed)
        warning(['"null" is a square matrix. The first array dimension ' ...
                 'will be regarded as corresponding to the random ' ...
                 'variables, but there is no guarantee that this is ' ...
                 'correct, as the number of variables and the number of ' ...
                 'null values are equal and there is thus no way to infer ' ...
                 'which is which based on matching to the length of ' ...
                 '"observed". It may be advisable to double check these inputs.'])
    elseif length(observed) == size(null, 2)
        null = null'; 
    elseif length(observed) ~= size(null, 1)
        error(['If "observed" is a vector, its length must match one of ' ...
               'the dimension sizes of the matrix "null".'])
    end
        
    % Determine number of observations in "null".
    nullDistrSize = length(null);
end

end