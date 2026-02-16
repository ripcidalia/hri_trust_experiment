function S = thesisStyle(fig)
% thesisStyle Configure thesis-wide MATLAB plotting defaults and return a style struct.
%
% This utility centralizes plot styling for the thesis figures, including:
%   - A consistent TU Delft-inspired color palette
%   - Font families for body text and titles
%   - Default interpreter choices (TeX for normal text; LaTeX for math strings)
%   - Global graphics defaults via groot for consistent appearance across figures
%   - Optional per-figure sizing and background configuration
%
% INPUTS
%   fig (optional)
%       Figure handle. If provided and valid, the function applies figure-level
%       defaults (units, background color, and export-oriented size) to that figure.
%
% OUTPUTS
%   S (struct)
%       Style configuration struct with fields:
%         - colors.*         : named RGB colors in [0,1]
%         - cyan, yellow     : backward-compatible aliases
%         - colorOrder6      : recommended multiline plotting order
%         - interpText       : default text interpreter ('tex')
%         - interpMath       : math interpreter ('latex')
%         - fontBody         : body font name
%         - fontTitle        : title font name
%         - fontSize*        : font sizes for axes/labels/titles/legend
%         - lineWidth        : default line width
%         - markerSize       : default marker size
%         - fig*             : figure sizing parameters (centimeters)
%
% NOTES
%   - TeX is used as the default interpreter so system fonts (e.g., Arial,
%     Roboto Slab) are honored. Use S.interpMath ('latex') for strings that
%     explicitly contain LaTeX math (e.g., '$...$').
%   - This function sets groot defaults, affecting subsequent figures created
%     in the current MATLAB session.
%

    % ------------------------------------------------------------------
    % Thesis color palette
    % ------------------------------------------------------------------
    S.colors = struct();

    % Core thesis colors
    S.colors.cyan   = (1/255)*[0, 166, 214];   % primary
    S.colors.yellow = (1/255)*[255, 184, 28];  % secondary

    % TU Delft extra accents
    S.colors.green    = (1/255)*[0 155 119];
    S.colors.burgundy = (1/255)*[165 0 52];
    S.colors.red      = (1/255)*[224 60 49];
    S.colors.blue     = (1/255)*[0 118 194];
    S.colors.purple   = (1/255)*[111 29 119];

    % Softer cyan variants (useful for shading/bands)
    S.colors.cyanSoft1 = (1/255)*[128 212 235];
    S.colors.cyanSoft2 = (1/255)*[230 246 250];

    % Backward-compatible aliases (used by older plotting helpers)
    S.cyan   = S.colors.cyan;
    S.yellow = S.colors.yellow;

    % Multi-line default ordering (useful for multi-series plots)
    S.colorOrder6 = [
        S.colors.cyan
        S.colors.yellow
        S.colors.green
        S.colors.burgundy
        S.colors.purple
        S.colors.blue
    ];

    % Optionally make this the default for all normal axes.
    % Note: some plot types (e.g., heatmaps) may be handled separately elsewhere.
    set(groot, 'defaultAxesColorOrder', S.colorOrder6);

    % ------------------------------------------------------------------
    % Interpreters
    % ------------------------------------------------------------------
    S.interpText = 'tex';    % enables system fonts and basic TeX formatting
    S.interpMath = 'latex';  % for strings containing $...$ math markup

    % ------------------------------------------------------------------
    % Fonts
    % ------------------------------------------------------------------
    S.fontBody  = 'Arial';
    S.fontTitle = 'Roboto Slab';

    % ------------------------------------------------------------------
    % Sizes (typography)
    % ------------------------------------------------------------------
    S.fontSizeAx  = 13;
    S.fontSizeLbl = 15;
    S.fontSizeYlb = 15;
    S.fontSizeTit = 16;
    S.legendFont  = 13;

    % ------------------------------------------------------------------
    % Lines and markers
    % ------------------------------------------------------------------
    S.lineWidth  = 1.4;
    S.markerSize = 10;

    % ------------------------------------------------------------------
    % Figure sizing (export-oriented, in centimeters)
    % ------------------------------------------------------------------
    S.figUnits    = 'centimeters';
    S.figWidthCm  = 16;
    S.figHeightCm = 10;
    S.figSizeTrajectoryGrid  = [25, 20];  % cm
    S.figSizeTrajectoryCombo = [32, 15];  % cm
    S.figSizeBIC             = [25, 15];  % cm
    S.figSizeGridMap         = [16, 14];  % cm

    % ------------------------------------------------------------------
    % Global defaults (use TeX so fonts are honored)
    % ------------------------------------------------------------------
    set(groot, ...
        'defaultTextInterpreter',          S.interpText, ...
        'defaultAxesTickLabelInterpreter', S.interpText, ...
        'defaultLegendInterpreter',        S.interpText, ...
        'defaultAxesFontName',             S.fontBody, ...
        'defaultTextFontName',             S.fontBody, ...
        'defaultAxesFontSize',             S.fontSizeAx, ...
        'defaultLineLineWidth',            S.lineWidth, ...
        'defaultLineMarkerSize',           S.markerSize, ...
        'defaultAxesBox',                  'on', ...
        'defaultAxesLineWidth',            1.0, ...
        'defaultAxesColorOrder',           [S.cyan; S.yellow] ...
    );

    % ------------------------------------------------------------------
    % Apply to a specific figure (optional)
    % ------------------------------------------------------------------
    if nargin >= 1 && ishghandle(fig) && strcmp(get(fig,'Type'),'figure')
        set(fig, 'Units', S.figUnits, 'Color', 'w');
        set(fig, 'Position', [2 2 S.figWidthCm S.figHeightCm]);
    end
end
