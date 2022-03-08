function obsStdev = get_num_stdev(observed,null)
% Calculates the number of standard deviations from the mean of a null
% distirbution that an observation/set of observations lie/lies.
%
% PARAMETERS
% ----------
% observed -- Either a scalar (one observation under one null distirbution)
%             or m-vector of m observations (one each for m null
%             distributions). If a vector, "observed" may be passed as a
%             row or column.
% null     -- If "observed" is a scalar and "null" a vector, the elements
%             of null will be considered draws from a null distribution. If
%             "observed is a vector and "null" an m x n matrix, the m rows
%             of "null" will be regarded as n draws from m distributions
%             (so that the i_th row j_th column element is the j_th draw
%             from the i_th distribution), and the number of standard
%             deviations corresponding to the i_th element of "observed"
%             will be calculated based on the i_th row of "null." Note,
%             however, that "null" may be passed either as m x n or n x m;
%             the function takes the dimension whose size (m) matches the
%             length of "observed" (m) to be the one that corresponds to
%             the various distributions (with a warning issued if "null" is
%             square).
% 
% RETURNS
% -------
% obsStdev -- The number of standard deviations away from the mean of the
%             null distribution(s) that lie(s) the observed value(s). Array
%             size of "obsStdev" matches that of "observed", and elements
%             correspond one to one.
%
% Author: Jonathan Chien 8/20/21


[observed, null, ~] = check_obs_vs_null(observed, null);

obsStdev = (observed - mean(null, 2)) ./ std(null, 0, 2);

end
