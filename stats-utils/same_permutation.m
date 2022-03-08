function varargout = same_permutation(varargin)
% Accepts a variable number of vectors and matrices. All vectors must be of
% length n, and the first dimension size of all matrices must also be n. A
% single permutation will be calculated and applied to the elements of the
% vectors and the rows of the matrices.
%
% PARAMETERS
% varargin  -- A series of n-vectors or matrices with 1st dim size = n.
%
% RETURNS
% varargout -- A series of one-to-one match of the inputs, but with the
%              same permutation applied to the elements of all vectors and
%              the rows of all matrices.
%
% Author: Jonathan Chien

% Determine number of elements needed in permutation.
if isvector(varargin{1})
    len = length(varargin{1});
elseif ismatrix(varargin{1})
    len = size(varargin{1}, 1);
end

% Get single permutation.
permutation = randperm(len);

% Apply permutation to inputs.
for iArg = 1:nargin
    if isrow(varargin{iArg}), varargin{iArg} = varargin{iArg}'; end
    if ismatrix(varargin{iArg})
        assert(size(varargin{iArg}, 1), ...
               ['First dimension size of a matrix input must match length' ...
                'of vector inputs.'])
    end
    varargout{iArg} = varargin{iArg}(permutation,:);
end

end
