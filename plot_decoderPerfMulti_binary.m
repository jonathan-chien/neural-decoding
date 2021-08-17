function plot_decoderPerfMulti_binary(decoderPerf, nvp)
% Takes as input decoderPerfMulti, the output of decode_BxTxN_multi.m, and
% plots precision, recall, F-measure, and AUCPR for each of the binary
% one-vs-all decoders constituting the multiclass decoder. Also displays p
% values generated via permutation and marks time bins with significant p
% values.
%
% PARAMETERS
% ----------
% decoderPerf -- 1 x b struct returned from decode_BxTxN_multi.m. Do not
%                attempt to pass in decoderPerfBinary.
% Name-Value Pairs (nvp)
%   'threshold'     -- P value threshold for significance. Bins whose p
%                      values for a given metric are below this threshold
%                      are said to be significant for that metric.
%   'binWidth'      -- Scalar width in ms of a time bin. This value is
%                      needed for calculation of timepoints (which are
%                      placed at the center of bins) and is not otherwise
%                      used.
%   'window'        -- 1D array of length 2 whose elements are the first
%                      and last timepoints of the analysis window
%                      (timepoints are wrt not to the window but to the
%                      epoch, with 0 ms locked to the epoch-defining
%                      event). This value is unused except for in
%                      calculation of bin centers (timepoints).
%   'step'          -- Scalar value (in ms) by which a bin is advanced
%                      relative to its predecessor. This value is needed
%                      for calculation of timepoints (which are placed at
%                      the center of bins) and is not otherwise used.
%   'taskVar'       -- String with name of task variable that was decoded.
%                      This value is used to title the plot and is
%                      otherwise unused.
%   'taskVarValues' -- 1 x nClasses cell array where each cell contains a
%                      string value, to be used in the figure legend,
%                      corresponding to one of the classes.
%   'area'          -- String with name of the brain region whose data were
%                      decoded. This value is used to title the plot and is
%                      otherwise unused.
%   'cmap'          -- Colormap to be used for the different performance
%                      metrics. See MATLAB's colormap function for more
%                      information and to visualize the various options.
%
% RETURNS
% -------
% 2x2 subplot -- Each subplot contains the timeseries of all binary
%                one-vs-all decoders for one of the four performance
%                metrics (precision, recall, F-measure, and AUCPR) with p
%                values via permutation and significant bins marked.
%
% Author: Jonathan Chien Version 1.0. 6/3/21. Last edit: 6/5/21.

arguments
    decoderPerf % must pass in decoderPerfBinary, not decoderPerfMulti 
    nvp.threshold {mustBeNumeric}
    nvp.binWidth {mustBeInteger} = 150 % needed to calculate timepoints
    nvp.window % needed to calculate timepoints
    nvp.step = 25 % needed to calculate timepoints
    nvp.taskVar {string} = [] % for plotting purposes
    nvp.taskVarValues % must be cell array; for plotting purposes 
    nvp.area {string} = [] % for plotting purposes
    nvp.cmap = 'cool' % for plotting purposes
end

% Calculate timepoints.
nBins = length(decoderPerf);
windowStart = nvp.window(1); % Careful, 'window' is the name of a MATLAB function!
windowEnd = nvp.window(2);
assert(mod(nvp.binWidth, 2) == 0, ...
           'Must enter a positive even integer for binWidth.')
binCenters = windowStart - 1 + nvp.binWidth/2 ...
            : nvp.step ...
            : windowEnd - nvp.binWidth/2;
assert(length(binCenters) == nBins);

% Create arrays of performance metrics from structure fields for easier
% handling.
perfArrays(1,:,:) = horzcat(decoderPerf.binary_prec)'; % precision 
perfArrays(2,:,:) = horzcat(decoderPerf.binary_reca)'; % recall
perfArrays(3,:,:) = horzcat(decoderPerf.binary_fmea)'; % F-measure
perfArrays(4,:,:) = vertcat(decoderPerf.binary_AUCPR); % AUCPR
nClasses = size(perfArrays, 3);

% Create arrays of p values from structure fields for easier handling.
pvalArrays(1,:,:) = vertcat(decoderPerf.p_binary_prec);
pvalArrays(2,:,:) = vertcat(decoderPerf.p_binary_reca);
pvalArrays(3,:,:) = vertcat(decoderPerf.p_binary_fmea);
pvalArrays(4,:,:) = vertcat(decoderPerf.p_binary_AUCPR);

% Get colormap.
switch nvp.cmap
    case 'cool'
        cmap = cool(nClasses);
    case 'autumn'
        cmap = autumn(nClasses);
    case 'spring'
        cmap = spring(nClasses);
    case 'parula'
        cmap = parula(nClasses);
    case 'turbo'
        cmap = turbo(nClasses);
end

% Plot performance metrics against time.
metricNames = {'Precision', 'Recall', 'F-measure', 'AUCPR'};
for iMetric = 1:4
    for iClass = 1:nClasses
        
        % Plot current performance metric.
        subplot(2,2,iMetric)
        hold on
        plot(binCenters, perfArrays(iMetric,:,iClass), 'Color', cmap(iClass,:),...
        'LineWidth', 1.5, 'DisplayName', nvp.taskVarValues{iClass})
    
        % Add filled circles for significant bins.
        scatter(binCenters(pvalArrays(iMetric,:,iClass)<nvp.threshold), ...
                perfArrays(iMetric, ...
                           pvalArrays(iMetric,:,iClass)<nvp.threshold, ...
                           iClass), ...
                'o', 'MarkerEdgeColor', cmap(iClass,:), ...
                'MarkerFaceColor', cmap(iClass,:), ...
                'DisplayName', 'Significant bin');
            
        % Add title, axes labels, legend, set axes etc.
        title(metricNames{iMetric})
        xlabel('Time (ms)')
        ylabel('Decoder performance')
        xlim([binCenters(1) binCenters(end)])
        ylim([0 1])
    end
    
    % Add vertical line at 0 ms, and horizontal line to denote expected
    % performance level of no-skill classifier. 
    plot([0 0], [0 1], ...
         '--', 'Color', 'k', 'DisplayName', 'Vertical line = epoch event')
    plot([binCenters(1) binCenters(end)], [1/nClasses 1/nClasses], ...
         '--', 'Color', 'k', 'DisplayName', 'Horizontal line = no-skill decoder')
    legend('Location', 'southeastoutside', 'FontSize', 5)
end

% Add title to entire subplot.
sgtitle(sprintf('One-vs-all binary decomposition of %s decoder in %s', ...
                nvp.taskVar, nvp.area))
            
end
