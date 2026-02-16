function reputation_bias(timeMatPath, saveFigDir)
% stepM6_reputation_bias  Diagnostics for reputation / negativity bias.
%
%   stepM6_reputation_bias(timeMatPath)
%   stepM6_reputation_bias(timeMatPath, saveFigDir)
%
% Step M6 quantifies how strongly participants update their opinion about the
% robot's reputation as a function of the valence of the initial online
% reviews (expected reputation) versus their own post-review rating
% (response).
%
% For each participant, this step:
%   1) Extracts a scalar "expected" value E (review valence) and a scalar
%      "response" value R from the reviews phase.
%   2) Computes:
%        - Raw negativity bias: mean |R| for negative vs positive E.
%        - Normalized bias: E-specific influence index I_i = |R_i| / |E_i|.
%        - Neutral-condition diagnostics for E = 0.
%   3) Produces several summary plots and writes a metrics struct to disk.
%
% Inputs (optional):
%   timeMatPath : Path to the MAT file produced after time enrichment
%                 (Step T1). This file must contain either:
%                   - participants_with_time, or
%                   - participants_clean
%                 where each participant has a .reviews field.
%                 Default:
%                     "derived/participants_time_stepT1.mat"
%
%   saveFigDir  : (Optional) directory in which to save vector PDF figures.
%                 If empty or omitted, figures are shown but not saved.
%
% Output (file):
%   Writes "derived/measurement_reputation.mat" containing struct repMetrics:
%       repMetrics.expected_all        - vector of E for all used participants
%       repMetrics.response_all        - vector of R for all used participants
%       repMetrics.participant_id      - participant IDs for each (E,R) pair
%       repMetrics.unique_expected     - unique E values (conditions)
%       repMetrics.counts_per_cond     - counts per E condition
%       repMetrics.mean_expected       - mean(E_all)
%       repMetrics.mean_response       - mean(R_all)
%       repMetrics.N_pos, N_neg, N_neutral
%       repMetrics.M_pos_absR, M_neg_absR, B_raw
%       repMetrics.I_pos, I_neg, B_norm
%       repMetrics.R_neutral_mean
%       repMetrics.R_neutral_abs_mean
%       repMetrics.bias_reliable       - reliability flag based on N_pos/N_neg
%       repMetrics.N_min_per_group     - threshold used for reliability
%       repMetrics.created             - timestamp
%
% Assumptions:
%   - Each participant P has a .reviews field populated by the preprocessing
%     steps (Step 2 / extract_reviews).
%   - For the first reputation item, P.reviews.items(1).response_struct is
%     a 1-element cell array containing a struct with numeric fields:
%         expected : scalar review-valence score (E)
%         response : scalar rating given by the participant (R)
%   - E and R are finite and on a symmetric scale around zero. Negativity
%     bias is interpreted as stronger reactions (larger |R| or |R|/|E|)
%     for negative E than for positive E.

    if nargin < 1 || isempty(timeMatPath)
        timeMatPath = "derived/participants_time_stepT1.mat";
    end
    if nargin < 2
        saveFigDir = "";
    end

    if ~isfile(timeMatPath)
        error("Time-enriched participants file not found: %s", timeMatPath);
    end

    % Load time-enriched participants; support both naming conventions to
    % remain compatible with earlier pipeline versions.
    S = load(timeMatPath);
    if isfield(S, "participants_with_time")
        participants = S.participants_with_time;
    elseif isfield(S, "participants_clean")
        participants = S.participants_clean;
    else
        error("File %s does not contain 'participants_with_time' or 'participants_clean'.", timeMatPath);
    end

    N = numel(participants);
    if N == 0
        error("No participants found in %s.", timeMatPath);
    end

    % ------------------------------------------------------------
    % 1) Extract expected (E) and response (R) for all participants
    % ------------------------------------------------------------
    E_all   = [];
    R_all   = [];
    pid_all = strings(0,1);

    for i = 1:N
        P = participants(i);

        [Ei, Ri, ok] = extract_expected_and_response(P);
        if ok
            E_all(end+1,1)   = Ei; %#ok<AGROW>
            R_all(end+1,1)   = Ri; %#ok<AGROW>
            pid_all(end+1,1) = string(P.participant_id); %#ok<AGROW>
        end
    end

    nUsed = numel(E_all);
    fprintf('[Step M6] Found %d/%d participants with usable reputation data.\n', ...
            nUsed, N);

    if nUsed == 0
        warning('[Step M6] No participants with usable expected/response values. Nothing to compute.');
        return;
    end

    % ------------------------------------------------------------
    % 2) Condition counts (distribution of expected values)
    % ------------------------------------------------------------
    % Compute how many participants fall into each E condition
    [uniqueE, ~, idxE] = unique(E_all);
    counts = accumarray(idxE, 1);

    % Sort by expected value for consistent negative → zero → positive
    [uniqueE, order] = sort(uniqueE);
    counts = counts(order);

    fprintf('[Step M6] Review condition counts (expected values):\n');
    for k = 1:numel(uniqueE)
        fprintf('   E = %+4.1f : %d participants\n', uniqueE(k), counts(k));
    end

    % ------------------------------------------------------------
    % 3) Basic means: E and R
    % ------------------------------------------------------------
    meanE = mean(E_all);
    meanR = mean(R_all);

    fprintf('[Step M6] Mean expected E = %+0.3f\n', meanE);
    fprintf('[Step M6] Mean response R = %+0.3f\n', meanR);

    % ------------------------------------------------------------
    % 4) Raw negativity bias: |R| for positive vs negative E
    % ------------------------------------------------------------
    % Positive / negative / neutral expected reputation conditions
    mask_pos = (E_all > 0);
    mask_neg = (E_all < 0);
    mask_neu = (E_all == 0);

    N_pos = sum(mask_pos);
    N_neg = sum(mask_neg);
    N_neu = sum(mask_neu);

    % Magnitude of response deviation from zero
    absR = abs(R_all);

    M_pos = NaN;
    M_neg = NaN;
    B_raw = NaN;

    if N_pos > 0
        M_pos = mean(absR(mask_pos));
    end
    if N_neg > 0
        M_neg = mean(absR(mask_neg));
    end
    % Raw bias: ratio of mean |R| for negative vs positive E
    if N_pos > 0 && N_neg > 0 && M_pos > 0
        B_raw = M_neg / M_pos;
    end

    fprintf('[Step M6] Raw bias (using |R|):\n');
    fprintf('          N_pos = %d, mean|R|_pos = %.3f\n', N_pos, safeNaN(M_pos));
    fprintf('          N_neg = %d, mean|R|_neg = %.3f\n', N_neg, safeNaN(M_neg));
    fprintf('          B_raw = mean|R|_neg / mean|R|_pos = %.3f\n', safeNaN(B_raw));

    % ------------------------------------------------------------
    % 5) Normalized negativity bias:
    %       I_i = |R_i| / |E_i|, for E_i ≠ 0
    % ------------------------------------------------------------
    I_all        = NaN(size(E_all));
    nonZeroMask  = (E_all ~= 0);
    I_all(nonZeroMask) = absR(nonZeroMask) ./ abs(E_all(nonZeroMask));

    I_pos  = NaN;
    I_neg  = NaN;
    B_norm = NaN;

    if N_pos > 0
        I_pos = mean(I_all(mask_pos), 'omitnan');
    end
    if N_neg > 0
        I_neg = mean(I_all(mask_neg), 'omitnan');
    end
    % Normalized bias: ratio of mean influence index for negative vs positive E
    if N_pos > 0 && N_neg > 0 && ~isnan(I_pos) && I_pos > 0
        B_norm = I_neg / I_pos;
    end

    fprintf('[Step M6] Normalized bias (I_i = |R| / |E| for E≠0):\n');
    fprintf('          mean I_pos = %.3f\n', safeNaN(I_pos));
    fprintf('          mean I_neg = %.3f\n', safeNaN(I_neg));
    fprintf('          B_norm = I_neg / I_pos = %.3f\n', safeNaN(B_norm));

    % ------------------------------------------------------------
    % 6) Neutral (E=0) diagnostics
    % ------------------------------------------------------------
    R_neu_mean      = NaN;
    R_neu_abs_mean  = NaN;

    if N_neu > 0
        R_neutral      = R_all(mask_neu);
        R_neu_mean     = mean(R_neutral);
        R_neu_abs_mean = mean(abs(R_neutral));
    end

    fprintf('[Step M6] Neutral (E=0) diagnostics:\n');
    fprintf('          N_neutral = %d\n', N_neu);
    fprintf('          mean R_neutral = %.3f\n', safeNaN(R_neu_mean));
    fprintf('          mean |R_neutral| = %.3f\n', safeNaN(R_neu_abs_mean));

    % ------------------------------------------------------------
    % 7) Small-sample / reliability flag
    % ------------------------------------------------------------
    % Require minimum counts in both positive and negative groups to regard
    % bias metrics as reasonably stable.
    N_min = 3;  % threshold for "acceptable" group size
    bias_reliable = (N_pos >= N_min) && (N_neg >= N_min);

    if ~bias_reliable
        fprintf('[Step M6] WARNING: small N_pos or N_neg (pos=%d, neg=%d). Bias metrics are unreliable.\n', ...
                N_pos, N_neg);
    end

    % ------------------------------------------------------------
    % 8) Diagnostic plots (counts, raw bias, normalized bias, neutral)
    % ------------------------------------------------------------
    make_reputation_plots(uniqueE, counts, ...
                          M_pos, M_neg, ...
                          I_pos, I_neg, ...
                          N_pos, N_neg, ...
                          N_neu, R_neu_mean, ...
                          saveFigDir);

    % ------------------------------------------------------------
    % 9) Save metrics to MAT file
    % ------------------------------------------------------------
    if ~isfolder("derived")
        mkdir("derived");
    end

    repMetrics = struct();
    repMetrics.expected_all         = E_all;
    repMetrics.response_all         = R_all;
    repMetrics.participant_id       = pid_all;
    repMetrics.unique_expected      = uniqueE;
    repMetrics.counts_per_cond      = counts;
    repMetrics.mean_expected        = meanE;
    repMetrics.mean_response        = meanR;
    repMetrics.N_pos                = N_pos;
    repMetrics.N_neg                = N_neg;
    repMetrics.N_neutral            = N_neu;
    repMetrics.M_pos_absR           = M_pos;
    repMetrics.M_neg_absR           = M_neg;
    repMetrics.B_raw                = B_raw;
    repMetrics.I_pos                = I_pos;
    repMetrics.I_neg                = I_neg;
    repMetrics.B_norm               = B_norm;
    repMetrics.R_neutral_mean       = R_neu_mean;
    repMetrics.R_neutral_abs_mean   = R_neu_abs_mean;
    repMetrics.bias_reliable        = bias_reliable;
    repMetrics.N_min_per_group      = N_min;
    repMetrics.created              = char(datetime('now','Format','yyyy-MM-dd HH:mm:ss'));

    outPath = "derived/measurement_reputation.mat";
    save(outPath, "repMetrics", "-v7.3");

    fprintf('[Step M6] Reputation / negativity bias metrics saved to %s\n', outPath);
end

% =====================================================================
% Helper: extract expected (E) and response (R) for one participant
% =====================================================================
function [E, R, ok] = extract_expected_and_response(P)
% Extract scalar expected reputation (E) and response (R) from the first
% reputation item of participant P.
%
% E and R are taken from:
%   P.reviews.items(1).response_struct{1}.expected
%   P.reviews.items(1).response_struct{1}.response
%
% Returns:
%   E, R - numeric scalars (NaN if unavailable or invalid)
%   ok   - logical flag indicating successful extraction

    E  = NaN;
    R  = NaN;
    ok = false;

    try
        if ~isfield(P, "reviews") || isempty(P.reviews)
            return;
        end

        Rv = P.reviews;

        if ~isfield(Rv, "items") || isempty(Rv.items)
            return;
        end

        item1 = Rv.items(1);

        if ~isfield(item1, "response_struct") || isempty(item1.response_struct) ...
                || ~iscell(item1.response_struct)
            return;
        end

        s = item1.response_struct{1};
        if ~isstruct(s)
            return;
        end

        if ~isfield(s, "expected") || ~isfield(s, "response")
            return;
        end

        E = double(s.expected);
        R = double(s.response);

        if ~isfinite(E) || ~isfinite(R)
            return;
        end

        ok = true;
    catch
        % Any parsing failure simply results in ok=false, E/R left as NaN.
        ok = false;
    end
end

% =====================================================================
% Helper: safe printing / handling of NaN
% =====================================================================
function v = safeNaN(x)
% Return NaN if x is empty or non-finite; otherwise return x unchanged.
    if isempty(x) || ~isfinite(x)
        v = NaN;
    else
        v = x;
    end
end

% =====================================================================
% Helper: make simple diagnostic plots
% =====================================================================
function make_reputation_plots(uniqueE, counts, ...
                               M_pos, M_neg, ...
                               I_pos, I_neg, ...
                               N_pos, N_neg, ...
                               N_neu, R_neu_mean, ...
                               saveFigDir)
% Create summary figures for:
%   - Count per expected reputation condition.
%   - Raw and normalized bias for negative vs positive conditions.
%   - Neutral-condition mean response (if present).

    % Color definitions for negative, positive, and neutral conditions
    col_neg = [0.80 0.35 0.35];  % reddish
    col_pos = [0.35 0.55 0.80];  % bluish
    col_neu = [0.50 0.50 0.50];  % grey

    % Determine whether to save figures
    doSave = (isstring(saveFigDir) || ischar(saveFigDir)) && strlength(string(saveFigDir)) > 0;
    if doSave
        saveFigDir = string(saveFigDir);
        if ~isfolder(saveFigDir)
            mkdir(saveFigDir);
        end
    end

    % -------- Plot 1: condition counts (negative → zero → positive) --------
    f1 = figure('Name','Reputation Conditions - Counts', ...
                'NumberTitle','off', ...
                'Color','w', ...
                'MenuBar','none', ...
                'ToolBar','none');
    set_fullscreen(f1);

    % Build categorical labels preserving numeric order
    E_labels = arrayfun(@(x) sprintf('%+0.1f', x), uniqueE, 'UniformOutput', false);
    catX = categorical(E_labels, E_labels);  % categorical axis in desired order

    b = bar(catX, counts);

    % Color each bar according to sign of expected value
    C = zeros(numel(uniqueE), 3);
    for k = 1:numel(uniqueE)
        if uniqueE(k) < 0
            C(k,:) = col_neg;
        elseif uniqueE(k) > 0
            C(k,:) = col_pos;
        else
            C(k,:) = col_neu;
        end
    end
    b.FaceColor = 'flat';
    b.CData     = C;

    xlabel('Expected reputation E');
    ylabel('Number of participants');
    title('Distribution of expected reputation conditions');
    grid on;
    set(gca, 'FontName','Times', 'FontSize', 12);

    strip_axes_toolbars(f1);

    if doSave
        exportgraphics(f1, fullfile(saveFigDir, 'reputation_conditions_counts.pdf'), ...
                       'ContentType','vector');
    end

    % -------- Plot 2: raw vs normalized bias (neg on left, pos on right) -----
    f2 = figure('Name','Reputation Bias - Raw and Normalized', ...
                'NumberTitle','off', ...
                'Color','w', ...
                'MenuBar','none', ...
                'ToolBar','none');
    set_fullscreen(f2);

    % Left subplot: raw mean |R|
    subplot(1,2,1);
    vals_raw = [safeNaN(M_neg), safeNaN(M_pos)];
    bar(1:2, vals_raw);
    set(gca, 'XTick', 1:2, 'XTickLabel', {'neg','pos'});
    ylabel('mean |R|');
    title(sprintf('Raw |R| (Nneg=%d, Npos=%d)', N_neg, N_pos));
    grid on;
    set(gca, 'FontName','Times', 'FontSize', 12);

    ch = get(gca, 'Children');
    if isa(ch, 'matlab.graphics.chart.primitive.Bar')
        ch.FaceColor = 'flat';
        ch.CData     = [col_neg; col_pos];
    end

    % Right subplot: normalized influence index
    subplot(1,2,2);
    vals_norm = [safeNaN(I_neg), safeNaN(I_pos)];
    bar(1:2, vals_norm);
    set(gca, 'XTick', 1:2, 'XTickLabel', {'neg','pos'});
    ylabel('mean I = |R| / |E|');
    title('Normalized influence index');
    grid on;
    set(gca, 'FontName','Times', 'FontSize', 12);

    ch2 = get(gca, 'Children');
    if isa(ch2, 'matlab.graphics.chart.primitive.Bar')
        ch2.FaceColor = 'flat';
        ch2.CData     = [col_neg; col_pos];
    end

    strip_axes_toolbars(f2);

    if doSave
        exportgraphics(f2, fullfile(saveFigDir, 'reputation_bias_raw_normalized.pdf'), ...
                       'ContentType','vector');
    end

    % -------- Plot 3: neutral condition response -----------------------------
    if N_neu > 0
        f3 = figure('Name','Neutral Condition Responses', ...
                    'NumberTitle','off', ...
                    'Color','w', ...
                    'MenuBar','none', ...
                    'ToolBar','none');
        set_fullscreen(f3);

        bar(1, safeNaN(R_neu_mean));
        set(gca, 'XTick', 1, 'XTickLabel', {'E=0'});
        ylabel('mean R (neutral)');
        title(sprintf('Neutral condition (N=%d)', N_neu));
        grid on;
        set(gca, 'FontName','Times', 'FontSize', 12);

        ch3 = get(gca, 'Children');
        if isa(ch3, 'matlab.graphics.chart.primitive.Bar')
            ch3.FaceColor = col_neu;
        end

        strip_axes_toolbars(f3);

        if doSave
            exportgraphics(f3, fullfile(saveFigDir, 'reputation_neutral_condition.pdf'), ...
                           'ContentType','vector');
        end
    end
end

% =====================================================================
% Helper: remove axes toolbars from all axes in a figure
% =====================================================================
function strip_axes_toolbars(figHandle)
% Remove interactive toolbars and interactions from all axes in figHandle.
    ax = findall(figHandle, 'Type', 'axes');
    for k = 1:numel(ax)
        try
            if isprop(ax(k), 'Toolbar')
                ax(k).Toolbar = [];
            end
            if isprop(ax(k), 'Interactions')
                ax(k).Interactions = [];
            end
        catch
            % Silently ignore axes that do not support these properties
        end
    end
end

% =====================================================================
% Helper: make figure fullscreen / maximized
% =====================================================================
function set_fullscreen(figHandle)
% Maximize the figure window, with a fallback for older MATLAB versions.
    try
        set(figHandle, 'WindowState', 'maximized');
    catch
        set(figHandle, 'Units','normalized', ...
                       'OuterPosition',[0 0 1 1]);
    end
end
