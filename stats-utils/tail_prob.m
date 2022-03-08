function p = tail_prob(observed,null,nvp)
% Calculates the tail probability for an observation or set of observations
% under a null distribution or set of null distributions. Note that for the
% two-sided case, at least one source (An Introduction to Probability and
% Statistics, Second Edition, Rohatgi & Saleh) takes p values to be poorly
% defined for asymmetric null distributions, though this source states that
% doubling of the smaller tail is usually recommended by many authors in
% that case (also seems to be slightly more conservative than the absolute
% value method). Doubling may also perhaps be interpreted as a correction
% for two one-tailed tests.
%
% PARAMETERS
% ----------
% observed -- Either a scalar (one observation under one null distirbution)
%             or m-vector of m observations (one each for m null
%             distributions). If a vector, "observed" may be passed as a
%             row or column.
% null     -- If "observed" is a scalar and "null" a vector, the elements
%             of null will be considered draws from a null distribution,
%             and the tail probability, as specified through the 'type'
%             name-value pair will be computed for "observed". If "observed
%             is a vector and "null" an m x n matrix, the m rows of "null"
%             will be regarded as n draws each for m distributions (so that
%             the i_th row j_th column element is the j_th draw from the
%             i_th distribution), and the tail probability of the i_th
%             element of "observed" will be calculated based on the i_th
%             row of "null". Note, however, that "null" may be passed
%             either as m x n or n x m; the function takes the dimension
%             whose size (m) matches the length of "observed" (m) to be the
%             one that corresponds to the various distributions (with a
%             warning issued if "null" is square).
% Name-Value Pairs (nvp)
%   'type'  -- String value, either "two-tailed" (default), "right-tailed",
%              or "left-tailed" specifying direction of test. Note that
%              some authors consider the p value to be poorly defined for
%              asymmetric null distributions, though doubling of the
%              smaller tail is usually recommended in that case (which may
%              also perhaps be viewed as a correction for testing two
%              tails); this is the method adopted here.
%   'exact' -- (1|0 default = 0), specifies whether to use calculation for
%              an exact test or to correct for Monte Carlo simulation.
%
% RETURNS
% -------
% p -- Tail probability/ies of observation(s) under their respective null
%      distribution(s). Array size of "p" matches that of "observed", and
%      elements correspond one to one.
%
% Author: Jonathan Chien 8/20/21

arguments
    observed
    null
    nvp.type = 'two-tailed'
    nvp.exact = false
end

%% Check inputs

[observed, null, nullDistrSize] = check_obs_vs_null(observed, null);


%% Calculate p value(s)

% Calculate both right and left tails first. 
if ~nvp.exact 
    % Calculate both right and left tails with correction for
    % random permutations.
    rightTail = (sum(observed <= null, 2) + 1) / (nullDistrSize + 1);
    leftTail = (sum(observed >= null, 2) + 1) / (nullDistrSize + 1);
            
elseif nvp.exact
    % Calculate both right and left tails without correction, if
    % exact permutations used.
    rightTail = mean(observed <= null, 2);
    leftTail = mean(observed >= null, 2);
    
else
    error("Invalid value for 'exact'.")
end

% Apply desired sidedness.
switch nvp.type
    case 'two-tailed'    
        % Double tails for all distributions and preallocate p.
        rightTailDoubled = rightTail*2;
        leftTailDoubled = leftTail*2;
        p = NaN(length(rightTail), 1);
        
        % For each distribution, take smaller tail and double it. If tails
        % have equal mass, assign 1, as doubling in Monte Carlo case will
        % yield p > 1 due to correction.
        p(rightTail > leftTail) = leftTailDoubled(rightTail > leftTail);
        p(rightTail < leftTail) = rightTailDoubled(rightTail < leftTail);
        p(rightTail == leftTail) = ones(sum(rightTail==leftTail), 1); 
        
    case 'right-tailed'
        p = rightTail;
        
    case 'left-tailed'
        p = leftTail;

    otherwise
        error("Invalid value for 'type'.")
end

end
