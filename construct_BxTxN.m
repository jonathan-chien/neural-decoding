function [pseudopopBxTxN,trialLabels] = construct_BxTxN(tMetadata,nvp)
% Accepts as input the 1 x nSessions cell array tMetadata returned by
% find_cond_indices.m and returns pseudopopBxTxN, a 3D array whose
% dimensions, respectively, correspond bins, trials, and neurons, with time
% collapsed over each bin for each single trial spiketrain for each neuron
% within the specified analysis window. Also returns a 1D vector
% trialsLabels, whose elements are the ordered condition codes of each
% trial (corresponding to each element along the second dimension of
% pseudopopBxTxN).
%
% PARAMETERS
% ----------
% tMetadata      -- 1 x nSessions cell array, where each element contains a
%                   a 1 x 1 struct with trimmed metadata for each session.
% Name-Value Pairs (nvp) 
%   'smooth'    -- Passed to local function single_session_BxTxN.m. 
%   'window'    -- Passed to local function single_session_BxTxN.m.
%   'binWidth'  -- Passed to local function single_session_BxTxN.m.
%   'sliding'   -- Passed to local function single_session_BxTxN.m.
%   'step'      -- Passed to local function single_session_BxTxN.m.
%   'normalize' -- Passed to local function single_session_BxTxN.m.
%   'resample'  -- Passed to local function single_session_BxTxN.m.
%
% RETURNS
% -------
% pseudopopBxTxN -- A 3D array whose elements along the 1st, 2nd, and 3rd
%                   dimensions correspond, respectively, to bins, trials,
%                   and neurons. Among the trials, an equal number of each
%                   condition is represented (this number is equal to
%                   overallMin if 'resample' is false, else it is equal to
%                   the scalar value of 'resample'). Note that these
%                   conditions are grouped in order. For example, if
%                   nTrialsPerCond = 100, and nConds = 2, then the second
%                   dimension of pseudopopBxTxN will be of length 200, with
%                   the first 100 elements corresponding to condition 1 and
%                   the latter 100 to condition 2.
% trialLabels    -- A 1D array containing the condition labels of the
%                   trials in the second dimension of pseudopopBxTxN.
%                   Elements of this array are ordered as described above.
%
% Author: Jonathan Chien 4/9/21. Version 2.0. Last edit: 7/19/21.

arguments
    tMetadata
    nvp.smooth {mustBeInteger} = 100
    nvp.window {mustBeInteger} = [101 600]
    nvp.binWidth {mustBeInteger} = 200
    nvp.sliding = true
    nvp.step {mustBeInteger} = 25
    nvp.normalize = false % 'spikes', 'bins', or false 
    nvp.resample = 100
end

% Initialize pseudopopBxTxN and waitbar.
pseudopopBxTxN = [];
w = waitbar(0, '');

% Iterate over all sessions and add current session data to build
% pseudopopulation. 
for iSession = 1:length(tMetadata)
    
    % Update waitbar.
    waitbar(iSession./length(tMetadata), w, ...
            sprintf('Loading and adding data from %s...', ...
                    tMetadata{iSession}.filename));
                
    % Load current session data.
    load(tMetadata{iSession}.filename); 
    
    % Build pseudopopulation by concatening neurons from current session to
    % growing pseudopopulation. Note that trialLabels will be overwritten
    % with each iteration of the iSession loop, but it should be identical
    % in each loop, and so taking its value from the final iteration should
    % be sufficient.
    [newSessionBxTxN, trialLabels] ...
        = single_session_BxTxN(data, tMetadata, ...
                               'smooth', nvp.smooth, ...
                               'window', nvp.window, ...
                               'binWidth', nvp.binWidth, ...
                               'sliding', nvp.sliding, ...
                               'step', nvp.step, ...
                               'normalize', nvp.normalize, ...
                               'resample', nvp.resample);     
    pseudopopBxTxN = cat(3, pseudopopBxTxN, newSessionBxTxN);      
end
close(w)

% Allow construction of TxN matrix at single point in time by elimnating
% temporal (first) dimension of tensor if size of that dimension is one.
if size(pseudopopBxTxN, 1) == 1
    disp('Only one bin detected. Eliminating temporal dimension of tensor.')
    pseudopopBxTxN = squeeze(pseudopopBxTxN);
end

end


%% Local function called by construct_BxTxN
function [singleSessionBxTxN,trialLabels] = single_session_BxTxN(data,tMetadata,nvp)
% Accepts as input data, a 1 x 1 struct containing one recording session's
% data, tMetadata, a 1 x nSessions cell array containing trimmed metadata
% for all recording sessions, and nTrialsPerCond, the number of trials
% desired from each condition. Returns singleSessionBxTxN, a 3D array whose
% dimensions correspond, respectively, to bins, trials, and neurons, where
% the neurons come from the single session corresponding to data. Also
% returns trialLabels, an ordered 1D array of condition labels
% corresponding to the trials in singleSessionBxTxN. In the main function,
% singleSessionBxTxN is concatenated along the third dimension to the
% growing array pseudopopBxTxN. trialLabels overwrites the variable
% trialLabels in the main function workspace each time this local function
% is called (see note at end of this function for more on this).
%
% PARAMETERS
% ----------
% data           -- 1 x 1 struct containing the data from a single
%                   recording session (e.g., M0003).
% tMetadata      -- 1 x nSessions cell array containing trimmed metadata
%                   for all sessions.
% Name-Value Pair (nvp)
%   'smooth'    -- Specifies whether to apply a gaussian smoothing kernel
%                  to the spiketrains. If smoothing is desired, set this
%                  parameter to be the kernel width (positive integer). If
%                  smoothing is not desired, set this parameter to be 0 or
%                  logical false.
%   'window'    -- 1D array of length 2, whose first element is the first
%                  timepoint within the desired analysis window and whose
%                  second element is the final timepoint in the analysis
%                  window. Note that these timepoints are with respect to
%                  0, where 0 corresponds to the epoch-defining event
%                  (e.g., stimulus on for the stimulus epoch).
%   'binWidth'  -- Positive integer scalar specifying the width (in ms) of
%                  each bin over which time will be collapsed. 
%   'sliding'   -- Logical true (default) or false. Specify whether
%                  or not to slide bins with overlap (if true) or to create
%                  discrete bins (if false).
%   'step'      -- How much to slide the bin forward from the last
%                  bin when bins overlap. Will remain unused if 'sliding'
%                  set to false.
%   'normalize' -- May have string value of 'spikes', 'bins', or logical
%                  false. Specifies whether to normalize, and if so, how.
%                  If 'spikes', spiketrains will be normalized within each
%                  neuron across the entire session. If 'bins', the bins x
%                  trials matrix for each neuron (within
%                  singleSessionBxTxN) will be normalized (note that this
%                  means that normalization is only across the specified
%                  analysis window and not across the entire session). If
%                  false, no normalization is performed. 
%   'resample'  -- Option to resample trials in each condition to generate
%                  as many pseudotrials for each condition as desired. Note
%                  that the original method of generating pseudotrials
%                  already destroyed within condition correlation by
%                  sampling a trial of a given condition independently for
%                  each neuron (and assembling the independently sampled
%                  trials into a pseudo population vector), so this
%                  resampling method does not destroy any additional
%                  correlation but merely (re)samples more trials than the
%                  original method (where each trial was sampled once).
%                  Note that this means it may be more advisable to retain
%                  all trials from all conditions by setting the
%                  'subsample' nvp in find_cond_indices to false, as we are
%                  simply tossing trials out otherwise (similar reasoning
%                  to retaining all trials for calculating condition means
%                  in the CxN pipeline). If resampling is desired, set
%                  'resample' to the number of trials desired for each
%                  condition. If resampling is undesired, set 'resample' to
%                  false, or 0.
%
% RETURNS
% -------
% singleSessionBxTxN -- 3D array whose elements along its 3 dimensions
%                       correspond, respectively, to bins, trials, and
%                       neurons, with trials ordered (conditions are
%                       grouped together and ordered by ascending condition
%                       number).
% trialLabels        -- 1D array whose elements are the condition numbers
%                       corresponding to the trials in the second dimension
%                       of singleSessionBxTxN. These elements are ordered
%                       as described above.

arguments
    data
    tMetadata
    nvp.smooth
    nvp.window
    nvp.binWidth
    nvp.sliding
    nvp.step
    nvp.normalize % 'spikes', 'bins', or false
    nvp.resample 
end

%% Construct timestamp matrix.

% Determine current session index within tMetadata.
for iSession = 1:length(tMetadata)
    if strcmp(tMetadata{iSession}.filename, data.BHV.DataFileName)
        break
    end
end

% Determine number of conditions in each epoch for current session.
nStimCond = size(tMetadata{iSession}.stimCondIndices, 2);
nRespCond = size(tMetadata{iSession}.respCondIndices, 2);
nFdbackCond = size(tMetadata{iSession}.fdbackCondIndices, 2);

% Construct timestamps matrix for each epoch. Note that output of
% construct_timestamps is a column vector.
stimTimestamps = [];
if nStimCond > 0
    stimTimestamps = construct_timestamps(data, 'stimulus');
    stimTimestamps = repmat(stimTimestamps, 1, nStimCond);
    for iCond = 1:nStimCond
        stimTimestamps(tMetadata{iSession}.stimCondIndices(:,iCond)~=1,iCond) = NaN;
    end
end

respTimestamps = [];
if nRespCond > 0
    respTimestamps = construct_timestamps(data, 'response');
    respTimestamps = repmat(respTimestamps, 1, nRespCond);
    for iCond = 1:nRespCond
        respTimestamps(tMetadata{iSession}.respCondIndices(:,iCond)~=1,iCond) = NaN;
    end
end

fdbackTimestamps = [];
if nFdbackCond > 0
    fdbackTimestamps = construct_timestamps(data, 'feedback');
    fdbackTimestamps = repmat(fdbackTimestamps, 1, nFdbackCond);
    for iCond = 1:nFdbackCond
        fdbackTimestamps(tMetadata{iSession}.fdbackCondIndices(:,iCond)~=1,iCond) = NaN;
    end
end

% Concatenate timestamp matrices for each epoch into one timestamp matrix.
% Get number of conditions.
timestampsByCond = [stimTimestamps respTimestamps fdbackTimestamps];
nConds = size(timestampsByCond, 2);

% Resample trials with replacement if desired.
if nvp.resample
    % If user has subsampled trials already, suggest resampling from all
    % available trials.
    if length(unique(sum(~isnan(timestampsByCond))))==1 && iSession == 1
        warning(['Trials have already been subsampled. If resampling is ' ...
                 'desired here, consider retaining all trials by setting ' ...
                 'the subsample nvp of find_cond_indices to false. See ' ...
                 'documentation for more info.'])
    end
    
    % Resample trials to obtain nTrials = nvp.resample.
    newTimestampsByCond = NaN(nvp.resample, nConds);
    for iCond = 1:nConds
        newTimestampsByCond(:,iCond) ...
            = datasample(timestampsByCond(~isnan(timestampsByCond(:,iCond)),iCond), ...
                         nvp.resample, 'Replace', true);
    end
    
    % Reassign timestampsByCond as new nTrials x nConds matrix.
    timestampsByCond = newTimestampsByCond;
end


%% Construct singleSessionBxTxN matrix

% Construct spiketrains (spiketrains are rows of matrix).
if strcmp(nvp.normalize, 'spikes')
    spiketrains = construct_spiketrains(data, tMetadata, ...
                                            'drop', true, 'normalize', true, ...
                                            'smooth', nvp.smooth, ...
                                            'convertToHz', true);
    if ~nvp.smooth
        disp('Normalization will be performed on raw, unsmoothed spikes.')
    end
else
    spiketrains = construct_spiketrains(data, tMetadata, ...
                                        'drop', true, 'normalize', false, ...
                                        'smooth', nvp.smooth, ...
                                        'convertToHz', true);
end
nNeurons = size(spiketrains,1);

% Define analysis window (timepoints are wrt to the defining event of each
% epoch (i.e. stim on for stimulus epoch, response for response epoch,
% feedback for feedback epoch). Window bounds ('window' nvp) should be set
% precisely, such as [101 600] if looking from 101st ms to 600th ms.
windowStart = nvp.window(1);
windowEnd = nvp.window(2);
if windowStart > 0 && windowEnd > 0
    windowWidth = windowEnd - windowStart + 1;
elseif windowStart < 0 && windowEnd > 0
    windowWidth = abs(windowStart) + windowEnd + 1;
elseif windowStart < 0 && windowEnd < 0
    windowWidth = abs(windowStart) -  abs(windowEnd) + 1;
end

% Determine number of bins and timepoints for start of each bin (timepoints
% are wrt the analysis window and not to the epoch).
if nvp.sliding
    assert(mod((windowWidth-nvp.binWidth),nvp.step)==0, ...
           ['Step size, bin width, or both are invalid. Window width ' ...
            'minus bin width must be evenly divisible by step size.'])
    binStarts = 1 : nvp.step : windowWidth - nvp.binWidth + 1;
    nBins = length(binStarts);
else
    assert(mod(windowWidth, nvp.binWidth) == 0, ...
           ['Discrete bins requested, but specified window width ' ...
            'cannot be evenly divided by specified bin width.'])
    binStarts = 1 : nvp.binWidth : windowWidth;
    nBins = windowWidth / nvp.binWidth;
    assert(length(binStarts) == nBins)
end

% Get number of trials and number of conditions.
nTrials = size(timestampsByCond, 1);
nConds = size(timestampsByCond, 2);

% Iterate through all neurons in session. Meaning of singleSessionTxSxN is
% thus: T = nTrials, S = nTimepoints in window, and N = nNeurons. 
nTrialsPerCond = unique(sum(~isnan(timestampsByCond)));
nTrials2 = nTrialsPerCond*nConds;
singleSessionTxSxN = NaN(nTrials2, windowWidth, nNeurons);
trialLabels = [];
for iNeuron = 1:nNeurons
    
    % Iterate through conditions and trials. As with construct_BxCxN, it is
    % the fact that iCond iterates from 1 to nConds in ascending order that
    % guarantees that the conditions will be ordered in the same manner
    % (along the second dim of the function output, singleSessionBxTxN). By
    % the same token, this fact is what causes trialLabels to be ordered
    % correctly.
    trials = NaN(nTrials, nConds, windowWidth);
    trialLabels = NaN(nTrials, nConds);
    for iCond = 1:nConds
        for iTrial = 1:nTrials
            if ~isnan(timestampsByCond(iTrial,iCond))
                trials(iTrial, iCond, :)...
                = spiketrains(iNeuron,...
                              timestampsByCond(iTrial,iCond) + windowStart :...
                              timestampsByCond(iTrial,iCond) + windowEnd);
                trialLabels(iTrial, iCond) = iCond;
            end
        end
    end
    
    % Throw error if there are an unequal number of trials from each
    % condition. Probably redundant since assignment into
    % singleSessionTxSxN will most likely fail if the number of trials do
    % not match nTrialsPerCond*nConds = nTrials2.
    assert(length(unique(sum(~isnan(trials), [1 3]))) == 1, ...
           'Unequal number of trials from each condition.')
    
    % Stack trials from all conditions along the same dimension of array.
    trialsReshaped = reshape(trials, [nTrials*nConds, windowWidth]);
    trialsReshaped = trialsReshaped(~isnan(trialsReshaped(:,1)),:);
    trialLabels = reshape(trialLabels, [nTrials*nConds, 1]);
    trialLabels = trialLabels(~isnan(trialLabels));
    
    % Add this neuron's data to singleSessionTxSxN array.
    singleSessionTxSxN(:,:,iNeuron) = trialsReshaped;  
end

% Throw error if NaNs present in firingRates matrix.
if sum(isnan(singleSessionTxSxN), 'all') > 0
    error('NaNs present in data matrix.')
end

% Put time in first dimension of array.
singleSessionSxTCxN = permute(singleSessionTxSxN, [2 1 3]);

% Bin it up.
singleSessionBxTxN = NaN(nBins, nTrials2, nNeurons);
for iBin = 1:nBins
    % binWidth x trials x neurons.
    currentBin = singleSessionSxTCxN(binStarts(iBin) : ...
                                     binStarts(iBin)-1 + nvp.binWidth, :,:);
    % Collapse temporal dimension within bin and store for output. Note
    % that the T (second dim) here is really still TC in the sense that its
    % length = nTrials2 = nTrialsPerCond*nConds, but we write T for the
    % sake of concision and simplicity in the final output of the function.
    singleSessionBxTxN(iBin,:,:) = mean(currentBin, 1);
end

% If opted not to normalize spikes for each neuron earlier, option to
% normalize bins x trials within each neuron here. Note that when
% size(singleSessionBxTxN) = 1, this code will be equivalent to zscoring
% across all trials (2nd dim of singleSessionBxTxN) for a given neuron (3rd
% dim of singleSessionBxTxN).
if strcmp(nvp.normalize, 'bins')
    singleSessionBxTxN = zscore(singleSessionBxTxN, 0, [1 2]);
    if nvp.smooth
        warning(['Spikes were smoothed within each neuron by sliding ' ...
                 'across the entire session. Consider normalizing within ' ...
                 'each neuron across the entire session as well, instead ' ...
                 'instead of across bins x trials?'])
    end
    if nBins == 1 && iSession == 1
        disp(['There is only one bin. Normalization will be across trials ' ...
              'for each neuron.'])
    end
end

% Output of this function (singleSessionBxTxN and trialLabels) will both be
% empty in the event that no neurons from the current session are
% ultimately used; this may occur when all retained neurons (i.e. neurons
% from the current brain region) end up being dropped in the
% construct_spiketrains functions due to a mean FR across the entire
% session of < 1 Hz. This is not a problem in constructing the
% pseudopopulation, as en empty array will simply be concatenated along the
% third (N) dimension of the growing pseudopopBxTxN for that iteration,
% with no effect, and the function will iterate on to the next session.
% trialLabels, on the other hand, is overwritten upon every iteration of
% iSession. This is usually not an issue, as trialLabels should be the same
% for every session, and if it is empty during one session (i.e. if the
% above scenario occurs where all neurons from the current session are
% ultimately dropped), it will be assigned with a vector of labels again
% upon processing the next session with neurons. A problem, however, does
% arise if such a no-neuron session is the last session in tMetadata, as in
% that case, trialLabels will be empty, with no subsequent sessions to fill
% it. In that case, throw a warning to notify the user and explain the
% empty output.
if isempty(trialLabels) && iSession == length(tMetadata)
    warning(["'trialLabels' is empty. This is because no neurons " ...
             "were ultimately retained from the final session (likely " ...
             "due to their firing rate being too low (< 1Hz). See " ...
             "comment above the final warning in construct_BxTxN" ...
             ">single_session_BxTxN for more information."])
end
    
end
