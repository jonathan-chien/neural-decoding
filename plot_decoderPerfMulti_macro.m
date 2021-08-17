function plot_decoderPerfMulti_macro(decoderPerf,nvp)
% Takes as input decoderPerfMulti, a 1 x b struct returned by
% decode_BxTxN_multi.m, where b = nBins. Plots accuracy (equivalent to
% micro-precision, micro-recall, and thus micro-F1-score), macro-precision,
% macro-recall, macro-F1-score, and macro-AUCPR. Note that all class sizes
% are equal, and the weighted macro metrics are equivalent to the
% unweighted metrics. Also displays p values generated via permutation and
% marks time bins with significant p values.
%
% PARAMETERS
% ----------
% decoderPerf -- 1 x b struct returned from decode_BxTxN_multi.m. Do not
%                attempt to pass in decoderPerfBinary.
% Name-Value Pairs (nvp)
%   'threshold' -- P value threshold for significance. Bins whose p values
%                  for a given metric are below this threshold are said to
%                  be significant for that metric.
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
%   'nClasses'  -- Number of classes/trial types (must be greater than 2)
%                  on which the decoder is trained.
%   'area'      -- String with name of the brain region whose data were
%                  decoded. This value is used to title the plot and is
%                  otherwise unused.
%   'cmap'      -- Colormap to be used for the different performance
%                  metrics. See MATLAB's colormap function for more
%                  information and to visualize the various options.
%
% RETURNS
% -------
% 2D plot -- Timeseries of multiclass decoder macro performance measured
%            via accuracy, macro-precision, macro-recall, macro-F-measure,
%            and macro-AUCPR, with p values via permutation.            
%
% Author: Jonathan Chien Version 1.0. 6/4/21. Last edit: 6/5/21.

arguments
    decoderPerf % must pass in decoderPerfBinary, not decoderPerfMulti 
    nvp.threshold {mustBeNumeric}
    nvp.binWidth {mustBeInteger} = 150 % needed to calculate timepoints
    nvp.window % needed to calculate timepoints
    nvp.step = 25 % needed to calculate timepoints
    nvp.taskVar {string} = [] % for plotting purposes
    nvp.nClasses = [] % needed to plot no-skill decoder performance
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

% Get colormap.
switch nvp.cmap
    case 'cool'
        cmap = cool(5);
    case 'autumn'
        cmap = autumn(5);
    case 'spring'
        cmap = spring(5);
    case 'parula'
        cmap = parula(5);
    case 'turbo'
        cmap = turbo(5);
end

% Plot accuracy (micro precision), macro precision, macro recall, macro
% F-measure, and macro AUCPR.
figure
hold on
plot(binCenters, horzcat(decoderPerf.acc), ...
     'Color', cmap(1,:), 'LineWidth', 1.5, 'DisplayName', 'Accuracy')
plot(binCenters, horzcat(decoderPerf.macro_prec), ...
     'Color', cmap(2,:), 'LineWidth', 1.5, 'DisplayName', 'Precision')
plot(binCenters, horzcat(decoderPerf.macro_reca), ...
     'Color', cmap(3,:), 'LineWidth', 1.5, 'DisplayName', 'Recall')
plot(binCenters, horzcat(decoderPerf.macro_fmea), ...
     'Color', cmap(4,:), 'LineWidth', 1.5, 'DisplayName', 'F-measure')
plot(binCenters, horzcat(decoderPerf.macro_AUCPR), ...
     'Color', cmap(5,:), 'LineWidth', 1.5, 'DisplayName', 'AUCPR')
 
% Add closed circle to timepoints corresponding to significant bins.
scatter(binCenters(vertcat(decoderPerf.p_acc)<nvp.threshold), ...
        horzcat(decoderPerf(vertcat(decoderPerf.p_acc)<nvp.threshold).acc), ...
        'o', 'MarkerEdgeColor', cmap(1,:), 'MarkerFaceColor', cmap(1,:), ...
        'DisplayName', 'Significant accuracy bin');
scatter(binCenters(vertcat(decoderPerf.p_macro_prec)<nvp.threshold), ...
        horzcat(decoderPerf(vertcat(decoderPerf.p_macro_prec) ...
                            <nvp.threshold).macro_prec), ...
        'o', 'MarkerEdgeColor', cmap(2,:), 'MarkerFaceColor', cmap(2,:), ...
        'DisplayName', 'Significant precision bin');
scatter(binCenters(vertcat(decoderPerf.p_macro_reca)<nvp.threshold), ...
        horzcat(decoderPerf(vertcat(decoderPerf.p_macro_reca) ...
                            <nvp.threshold).macro_reca), ...
        'o', 'MarkerEdgeColor', cmap(3,:), 'MarkerFaceColor', cmap(3,:), ...
        'DisplayName', 'Significant recall bin');
scatter(binCenters(vertcat(decoderPerf.p_macro_fmea)<nvp.threshold), ...
        horzcat(decoderPerf(vertcat(decoderPerf.p_macro_fmea) ...
                            <nvp.threshold).macro_fmea), ...
        'o', 'MarkerEdgeColor', cmap(4,:), 'MarkerFaceColor', cmap(4,:), ...
        'DisplayName', 'Significant F-measure bin');
scatter(binCenters(vertcat(decoderPerf.p_macro_AUCPR)<nvp.threshold), ...
        horzcat(decoderPerf(vertcat(decoderPerf.p_macro_AUCPR) ...
                            <nvp.threshold).macro_AUCPR), ...
        'o', 'MarkerEdgeColor', cmap(5,:), 'MarkerFaceColor', cmap(5,:), ...
        'DisplayName', 'Significant AUCPR bin');
    
% Plot p values for accuracy and AUC on same plot.
scatter(binCenters, horzcat(decoderPerf.p_acc), ...
        'o', 'MarkerEdgeColor', cmap(1,:), 'LineWidth', 1, ...
        'DisplayName', 'Accuracy p value'); 
scatter(binCenters, horzcat(decoderPerf.p_macro_prec), ...
        'o', 'MarkerEdgeColor', cmap(2,:), 'LineWidth', 1, ...
        'DisplayName', 'Precision p value'); 
scatter(binCenters, horzcat(decoderPerf.p_macro_reca), ...
        'o', 'MarkerEdgeColor', cmap(3,:), 'LineWidth', 1, ...
        'DisplayName', 'Recall p value'); 
scatter(binCenters, horzcat(decoderPerf.p_macro_fmea), ...
        'o', 'MarkerEdgeColor', cmap(4,:), 'LineWidth', 1, ...
        'DisplayName', 'F-measure p value'); 
scatter(binCenters, horzcat(decoderPerf.p_macro_AUCPR), ...
        'o', 'MarkerEdgeColor', cmap(5,:), 'LineWidth', 1, ...
        'DisplayName', 'AUCPR p value'); 
    
% Add vertical line at epoch event and horizontal line for no-skill
% classifier performance.
assert(~isempty(nvp.nClasses), 'Must supply value for nClasses.')
plot([0 0], [0 1], ...
     '--', 'Color', 'k', 'DisplayName', 'Vertical line = epoch event')
plot([binCenters(1) binCenters(end)], [1/nvp.nClasses 1/nvp.nClasses], ...
     '--', 'Color', 'k', 'DisplayName', 'Horizontal line = no-skill decoder')
    
% Title, label axes, etc.
assert(~isempty(nvp.taskVar), 'Must specify task variable decoded.')
assert(~isempty(nvp.area), 'Must specify brain region.')
title(sprintf('Decoding of %s in %s (macro metrics)', nvp.taskVar, nvp.area))
xlabel('Time (ms)')
ylabel('Macro decoder performance')
xlim([binCenters(1) binCenters(end)])
ylim([0 1])
legend('Location', 'eastoutside') 

end
