function thesisFinalizeFigure(fig, S)
% thesisFinalizeFigure Apply thesis styling after plotting is complete.
%
% This post-processing helper enforces thesis-wide typography and visual
% conventions once all plot elements have been created. It is intended to be
% called immediately before exporting a figure.
%
% The routine handles:
%   - Standard axes formatting (fonts, label/title styling)
%   - Tick label interpreter selection based on presence of LaTeX math ('$...$')
%     (useful for boxplots and categorical tick labels)
%   - Boxplot restyling using MATLAB's tagged Line objects (R2022b conventions)
%   - HeatmapChart objects (fonts + a white-to-cyan colormap)
%   - Math-aware heatmap Title/XLabel/YLabel using an overlay axes for LaTeX,
%     including a vertically oriented Y label
%
% INPUTS
%   fig (optional)
%       Figure handle. If omitted or empty, uses gcf.
%
%   S (optional)
%       Thesis style struct returned by thesisStyle(). If omitted or empty,
%       thesisStyle() is called to obtain defaults.
%
% OUTPUTS
%   (none)
%
    if nargin < 1 || isempty(fig), fig = gcf; end
    if nargin < 2 || isempty(S), S = thesisStyle(); end

    % ------------------------------------------------------------------
    % 1) Standard axes formatting + tick labels + boxplot styling
    % ------------------------------------------------------------------
    ax = findall(fig, 'Type', 'axes');
    for k = 1:numel(ax)
        set(ax(k), 'FontName', S.fontBody, 'FontSize', S.fontSizeAx);

        % Apply fonts and interpreters to labels/titles after the plot is built.
        applyText(ax(k).XLabel, S.fontBody,  S.fontSizeLbl, S);
        applyText(ax(k).YLabel, S.fontBody,  S.fontSizeYlb, S);
        applyText(ax(k).Title,  S.fontTitle, S.fontSizeTit, S);

        % Tick labels: choose interpreter based on whether any label contains '$'.
        % This supports math notation in category labels and boxplot labels.
        try
            xt = string(ax(k).XTickLabel);
        catch
            xt = strings(0,1);
        end
        try
            yt = string(ax(k).YTickLabel);
        catch
            yt = strings(0,1);
        end

        tickStrs = [xt(:); yt(:)];
        if any(contains(tickStrs, "$"))
            set(ax(k), 'TickLabelInterpreter', S.interpMath);  % e.g., 'latex'
        else
            set(ax(k), 'TickLabelInterpreter', S.interpText);  % e.g., 'tex'
        end

        % If the axes contains a boxplot, restyle it via tagged primitives.
        applyBoxplotStyle(ax(k), S);
    end

    % Styled title textboxes (created elsewhere) are detected by tag.
    tb = findall(fig,'Type','textboxshape','Tag','thesisTitle');
    for k = 1:numel(tb)
        set(tb(k), 'FontName', S.fontTitle, ...
                   'FontSize', S.fontSizeTit, ...
                   'FontWeight','bold', ...
                   'Interpreter', S.interpText);
    end

    % ------------------------------------------------------------------
    % 2) Legends
    % ------------------------------------------------------------------
    lg = findall(fig, 'Type', 'legend');
    for k = 1:numel(lg)
        set(lg(k), 'FontName', S.fontBody, 'FontSize', S.legendFont);

        strs = lg(k).String;
        strs = string(strs);

        if any(contains(strs, "$"))
            set(lg(k), 'Interpreter', S.interpMath);   % e.g., 'latex'
        else
            set(lg(k), 'Interpreter', S.interpText);   % e.g., 'tex'
        end
    end

    % ------------------------------------------------------------------
    % 3) Heatmap charts (MATLAB heatmap / HeatmapChart)
    % ------------------------------------------------------------------
    hm = findall(fig, 'Type', 'Heatmap');
    for k = 1:numel(hm)
        hm(k).FontName = S.fontBody;
        hm(k).FontSize = S.fontSizeAx;

        % Use a consistent thesis colormap: white -> cyan.
        hm(k).Colormap = whiteToColorMap(S.cyan, 256);

        % Heatmap labels have limited interpreter support; overlay LaTeX when needed.
        applyHeatmapLabelsWithMathOverlay(fig, hm(k), S, k);
    end
end

% ======================================================================
% Helpers
% ======================================================================
function applyText(h, fontName, fontSize, S)
% applyText Apply font, size, and a math-aware interpreter to a text object.
%
% If the text string contains '$', the LaTeX interpreter is used; otherwise
% the default text interpreter is used (typically 'tex' to honor system fonts).
%
    if isempty(h) || ~isgraphics(h), return; end
    set(h, 'FontName', fontName, 'FontSize', fontSize);

    str = string(h.String);
    if contains(str, "$")
        set(h, 'Interpreter', S.interpMath);
    else
        set(h, 'Interpreter', S.interpText);
    end
end

function applyBoxplotStyle(ax, S)
% applyBoxplotStyle Restyle MATLAB boxplots to match the thesis palette.
%
% Boxplot primitives are drawn as Line objects with Tag values such as:
%   'Box', 'Whisker', 'Cap', 'Median', 'Outliers', 'Adjacent Value'
% This helper detects those tagged objects and applies consistent color/width.
%
% Palette mapping:
%   - Boxes/whiskers/caps/adjacent-value markers: cyan
%   - Median line: yellow
%   - Outliers: yellow markers (no connecting line)
%
    if isempty(ax) || ~isgraphics(ax), return; end

    % Detect whether this axes actually contains a boxplot.
    hb = findobj(ax, 'Tag', 'Box');
    if isempty(hb)
        return;
    end

    % Cyan elements: box structure (boxes, whiskers, caps).
    cyanTags = {'Box','Whisker','Cap'};
    for t = 1:numel(cyanTags)
        h = findobj(ax, 'Tag', cyanTags{t});
        if ~isempty(h)
            set(h, 'Color', S.cyan, 'LineWidth', S.lineWidth);
        end
    end

    % Yellow median line.
    hmed = findobj(ax, 'Tag', 'Median');
    if ~isempty(hmed)
        set(hmed, 'Color', S.yellow, 'LineWidth', S.lineWidth);
    end

    % Yellow outliers (marker-only lines).
    hout = findobj(ax, 'Tag', 'Outliers');
    if ~isempty(hout)
        set(hout, ...
            'MarkerEdgeColor', S.yellow, ...
            'MarkerFaceColor', 'none', ...
            'LineStyle', 'none', ...
            'MarkerSize', max(4, round(0.6 * S.markerSize)));
    end

    % Adjacent value markers (rare): use cyan to match the box structure.
    hav = findobj(ax, 'Tag', 'Adjacent Value');
    if ~isempty(hav)
        set(hav, 'Color', S.cyan, 'LineWidth', S.lineWidth);
    end
end

function cmap = whiteToColorMap(rgb, n)
% whiteToColorMap Create a colormap that linearly blends white -> rgb.
%
% INPUTS
%   rgb (1x3 numeric)
%       Target color in [0,1].
%   n (optional)
%       Number of colormap entries. Default: 256
%
% OUTPUTS
%   cmap (n x 3 numeric)
%       Colormap with cmap(1,:) = [1 1 1] and cmap(end,:) = rgb.
%
    if nargin < 2 || isempty(n), n = 256; end
    rgb = double(rgb(:))';
    rgb = max(0, min(1, rgb));
    t = linspace(0, 1, n)';        % 0 = white, 1 = target color
    cmap = (1 - t) * [1 1 1] + t * rgb;
end

function applyHeatmapLabelsWithMathOverlay(fig, hmObj, S, idx)
% applyHeatmapLabelsWithMathOverlay Add LaTeX-capable labels for heatmaps when needed.
%
% HeatmapChart title/xlabel/ylabel have limited interpreter support. When any
% of these strings contains '$', this function hides the corresponding built-in
% label and draws an overlay axes with LaTeX text objects instead.
%
% INPUTS
%   fig   - figure handle
%   hmObj - HeatmapChart object
%   S     - thesis style struct
%   idx   - index used to generate a unique overlay Tag per heatmap
%
    tStr = string(hmObj.Title);
    xStr = string(hmObj.XLabel);
    yStr = string(hmObj.YLabel);

    if ~(contains(tStr,"$") || contains(xStr,"$") || contains(yStr,"$"))
        return;
    end

    hmObj.Units = 'normalized';
    pos = hmObj.Position;

    % Remove any previous overlay axes for this heatmap.
    tag = sprintf("thesis_heatmap_overlay_%d", idx);
    oldAx = findall(fig, 'Type', 'axes', 'Tag', tag);
    delete(oldAx);

    % Create an invisible overlay axes aligned to the heatmap position.
    axO = axes(fig, ...
        'Units', 'normalized', ...
        'Position', pos, ...
        'Color', 'none', ...
        'XTick', [], 'YTick', [], ...
        'XColor', 'none', 'YColor', 'none', ...
        'Box', 'off', ...
        'HandleVisibility', 'off', ...
        'HitTest', 'off', ...
        'Tag', tag);

    uistack(axO, 'top');

    % Title overlay (slightly above the heatmap).
    if contains(tStr, "$")
        hmObj.Title = "";
        tt = title(axO, char(tStr), ...
            'FontName', S.fontTitle, ...
            'FontSize', S.fontSizeTit, ...
            'Interpreter', S.interpMath);
        tt.Units = 'normalized';
        tt.Position(2) = 1.08;
    end

    % X label overlay (slightly below the heatmap).
    if contains(xStr, "$")
        hmObj.XLabel = "";
        xl = xlabel(axO, char(xStr), ...
            'FontName', S.fontBody, ...
            'FontSize', S.fontSizeLbl, ...
            'Interpreter', S.interpMath);
        xl.Units = 'normalized';
        xl.Position(2) = -0.10;
    end

    % Y label overlay (left of the heatmap; vertical orientation).
    if contains(yStr, "$")
        hmObj.YLabel = "";
        yl = ylabel(axO, char(yStr), ...
            'FontName', S.fontBody, ...
            'FontSize', S.fontSizeLbl, ...
            'Interpreter', S.interpMath, ...
            'Rotation', 90, ...
            'HorizontalAlignment', 'center', ...
            'VerticalAlignment', 'middle');
        yl.Units = 'normalized';
        yl.Position(1) = -0.10;
    end
end
