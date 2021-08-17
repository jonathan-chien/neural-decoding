function plot_decoderPerfBinary(decoderPerf, nvp)
% Plots binary decoder performance against time using accuracy and AUC as
% metrics. Also displays p values generated via permutation and
% marks time bins with significant p values.
%
% RGB triplet codes can be found here (divide triplet vector by 255 for use
% in MATLAB): https://www.rapidtables.com/web/color/RGB_Color.html
%
% PARAMETERS
% ----------
% decoderPerf -- 1 x b struct, where b is the number of bins, that contains
%                information on decoder performance. This is the output of
%                decode_BxTxN_binary.m.
% Name-Value Pairs (nvp)
%   'threshold' -- P value threshold for significance. Bins whose p values
%                  for a given metric (accuracy or AUC) are below this
%                  threshold are said to be significant for that metric.
%   'binWidth'  -- Scalar width in ms of a time bin. This value is needed
%                  for calculation of timepoints (which are placed at the
%                  center of bins) and is not otherwise used.
%   'window'    -- 1D array of length 2 whose elements are the first and
%                  last timepoints of the analysis window (timepoints are
%                  wrt not to the window but to the epoch, with 0 ms locked
%                  to the epoch-defining event). This value is unused
%                  except for in calculation of bin centers (timepoints).
%   'step'      -- Scalar value (in ms) by which a bin is advanced relative
%                  to its predecessor. This value is needed for calculation
%                  of timepoints (which are placed at the center of bins)
%                  and is not otherwise used.
%   'taskVar'   -- String with name of task variable that was decoded.
%                  This value is used to title the plot and is otherwise
%                  unused.
%   'area'      -- String with name of the brain region whose data were
%                  decoded. This value is used to title the plot and is
%                  otherwise unused.
%
% RETURNS
% -------
% 2D plot -- Timeseries of decoder performance measured via accuracy and
%            AUC with p values via bootstrapping (permutation).
%
% Author: Jonathan Chien Version 1.0. 6/3/21. Last edit: 6/4/21.

arguments
    decoderPerf % must pass in decoderPerfBinary, not decoderPerfMulti 
    nvp.threshold {mustBeNumeric}
    nvp.binWidth {mustBeInteger} = 150 % needed to calculate timepoints
    nvp.window % needed to calculate timepoints
    nvp.step = 25 % needed to calculate timepoints
    nvp.taskVar {string} = [] % for plotting purposes
    nvp.area {string} = [] % for plotting purposes
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

% Set color scheme.
COLOR1 = [0 255 255]/255;
COLOR2 = [178 102 255]/255;

% Plot accuracy and AUC vs time.
figure
hold on
plot(binCenters, horzcat(decoderPerf.acc), ...
     'Color', COLOR1, 'LineWidth', 1.5, 'DisplayName', 'Accuracy');
plot(binCenters, horzcat(decoderPerf.AUC), ...
     'Color', COLOR2, 'LineWidth', 1.5, 'DisplayName', 'AUC');
xlim([binCenters(1) binCenters(end)])
ylim([0 1])

% Add vertical line at epoch event and horizontal line for no-skill
% classifier performance.
plot([0 0], [0 1], ...
     '--', 'Color', 'k', 'DisplayName', 'Vertical line = epoch event')
plot([binCenters(1) binCenters(end)], [0.5 0.5], ...
     '--', 'Color', 'k', 'DisplayName', 'Horizontal line = no-skill decoder')

% Add closed circle to timepoints corresponding to significant bins.
scatter(binCenters(vertcat(decoderPerf.p_acc)<nvp.threshold), ...
        horzcat(decoderPerf(vertcat(decoderPerf.p_acc)<nvp.threshold).acc), ...
        'o', 'MarkerEdgeColor', COLOR1, 'MarkerFaceColor', COLOR1, ...
        'DisplayName', 'Significant accuracy bin');
scatter(binCenters(vertcat(decoderPerf.p_AUC)<nvp.threshold), ...
        horzcat(decoderPerf(vertcat(decoderPerf.p_AUC)<nvp.threshold).AUC), ...
        'o', 'MarkerEdgeColor', COLOR2, 'MarkerFaceColor', COLOR2, ...
        'DisplayName', 'Significant AUC bin');
    
% Plot p values for accuracy and AUC on same plot.
scatter(binCenters, horzcat(decoderPerf.p_acc), ...
        'o', 'MarkerEdgeColor', COLOR1, 'LineWidth', 1, ...
        'DisplayName', 'Accuracy p value'); 
scatter(binCenters, horzcat(decoderPerf.p_AUC), ...
        'o', 'MarkerEdgeColor', COLOR2, 'LineWidth', 1, ...
        'DisplayName', 'AUC p value'); 
    
% Title, label axes, etc.
assert(~isempty(nvp.taskVar), 'Must specify task variable decoded.')
assert(~isempty(nvp.area), 'Must specify brain region.')
title(sprintf('Decoding of %s in %s', nvp.taskVar, nvp.area))
xlabel('Time (ms)')
ylabel('Decoder performance')
legend('Location', 'eastoutside') 

end
