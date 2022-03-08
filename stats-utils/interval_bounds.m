function interval = interval_bounds(distr,varargin)
% Calculate percentile interval(s) of specified size on null distribution.
%
% Example syntaxes (see PARAMETERS for arg info):
% -----------------
% interval = montecarlo_int(distr)
% interval = montecarlo_int(distr, interval);
% interval = montecarlo_int(distr, interval, dim);
% interval = montecarlo_int(distr, [], dim);
% 
% PARAMETERS
% ----------
% distr    -- If distr is passed as an n-vector, this function will regard
%             its elements as n draws from a distribution and calculate and
%             return a 2-vector corresponding to the interval
%             bounds (with the interval size specified through the
%             "interval" argument). If distr is an m x n matrix, by default
%             the m rows are regarded as a series of n draws each from m
%             distributions, and interval bounds will be
%             calculated and returned for each of the m rows corresponding
%             to m distributions (this is equivalent to setting dim = 2
%             (see "dim" below)).
% interval -- Optional scalar argument specifying the size of the
%             interval to be calculated.
% dim      -- If "distr" is a matrix, this is an optional scalar argument
%             specifying the array dimension along which are the draws from
%             each distribution. For distr as an m x n matrix, the default
%             value for "dim" is 2, with the m rows being regarded as
%             corresponding to m distributions, and the n columns as draws
%             from each of the m distributions (so that the i_th row j_th
%             column element is the j_th draw for the i_th distribution).
%             If dim = 1 for "distr" as an m x n matrix, the n columns are
%             taken as coresponding to distributions, with the m rows
%             regarded as corresponding to draws from the n distributions.
%             
% RETURNS
% -------
% interval -- If distr was passed in as a vector, interval is a 2-vector
%             whose first and second elements are, respectively, the lower
%             and upper bound of the interval of specified size. If distr
%             was passed in as an m x n or n x m matrix (see distr under
%             PARAMETERS), interval is an m x 2 or 2 x m, matrix where the
%             i_th row or column (of length 2), respectively, correspond to
%             the lower and upper bound of the i_th distribution, for i =
%             1:m.
%
% Author: Jonathan Chien 8/20/21


% Set "interval" and "dim" as specified user passed in values, otherwise
% set to 95.
if nargin == 1
    interval = 95;
    dim = 2;
elseif nargin == 2
    interval = varargin{1};
    dim = 2;
elseif nargin == 3
    if ~isempty(varargin{1})
        interval = varargin{1};
    else
        interval = 95;
    end
    dim = varargin{2};
elseif nargin > 3 
    error('Too many input arguments.')
end

% Check if distr is vector or matrix. If matrix, use "dim" to specify which
% dimension to regard as draws for one random variable.
if any(size(distr)==1) 
    interval = prctile(distr, [50-interval/2 50+interval/2]);
elseif all(size(distr)>1)
    interval = prctile(distr, [50-interval/2 50+interval/2], dim);
end

end
