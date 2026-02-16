function summary = analyze_participants_overview(varargin)
% analyze_participants_overview Generate participant overview plots and reporting tables.
%
% This function loads a time-enriched participant struct array produced by the
% preprocessing pipeline and generates descriptive statistics used for reporting
% and dataset characterization:
%   - Demographic distributions (age range, gender)
%   - Door-sequence set membership counts (Set A ... Set H)
%   - Review expectation distribution and expectation-vs-response scatter
%   - Questionnaire score distributions (overall histogram + by-set boxplot)
%   - Reporting tables: device type, browser, emergency choice
%   - Balance matrices (counts by set) for multiple categorical variables
%
% The function can operate on:
%   - The full participant set (default)
%   - A specified split ("train" or "valid") via name-value input
%
% INPUTS (name-value)
%   "Split" (string|char) (optional)
%       Either "train" or "valid". If omitted, the function analyzes the full
%       participant set in derived/participants_time_stepT1.mat.
%
% OUTPUTS
%   summary (struct)
%       Struct containing extracted vectors and key reporting tables, saved to
%       a MAT file in derived/participant_set_analysis/.
%
% FILE OUTPUTS
%   Figures:
%     derived/participant_set_analysis/<scope>/figs/*.png and *.fig
%   Tables:
%     derived/participant_set_analysis/<scope>/tables/*.csv
%   MAT:
%     derived/participant_set_analysis/<scope>_analysis.mat (filename depends on scope)
%
% ASSUMPTIONS / DATA FORMAT
%   participants_clean is a struct array with at least the fields accessed in
%   Section 1 (participant_id, set_id, device_type, browser_name, demographics,
%   reviews, emergency, questionnaires). The nested review/questionnaire fields
%   are assumed to be consistent after preprocessing.
%
% DEPENDENCIES
%   thesisStyle, thesisFinalizeFigure, thesisExport
%   measurementDisplayName (for legend labels in questionnaire plots, if used)
%
% NOTE
%   This function is intended as a descriptive analysis/reporting step; it does
%   not modify participants or compute model metrics.

    %% ------------------------------------------------------------
    % 0) Paths and load
    % ------------------------------------------------------------

    if nargin < 1
        % Default: analyze the full participant set.
        inMat   = "derived/participants_time_stepT1.mat";
        if ~isfile(inMat)
            error("File not found: %s. Run preprocessing (Step T1) first.", inMat);
        end
        outMat  = "derived/participant_set_analysis";
        MatName = "participant_global_analysis.mat";
        outFigs = "derived/participant_set_analysis/global/figs";
        outTabs = "derived/participant_set_analysis/global/tables";
    else
        % Analyze a requested split ("train" or "valid").
        p = inputParser;
        p.addParameter("Split", "valid", @(s) isstring(s) || ischar(s)); % "train" or "valid"
        p.parse(varargin{:});
        args = p.Results;

        splitName = lower(string(args.Split));
        if ~(splitName == "train" || splitName == "valid")
            error("Split must be 'train' or 'valid'. Got: %s", splitName);
        end

        if splitName == "train"
            inMat   = "derived/participants_train_stepV.mat";
            if ~isfile(inMat)
                error("File not found: %s. Run preprocessing (Step V) first.", inMat);
            end
            outMat  = "derived/participant_set_analysis";
            MatName = "participant_train_analysis.mat";
            outFigs = "derived/participant_set_analysis/train/figs";
            outTabs = "derived/participant_set_analysis/train/tables";
        else
            inMat   = "derived/participants_valid_stepV.mat";
            if ~isfile(inMat)
                error("File not found: %s. Run preprocessing (Step V) first.", inMat);
            end
            outMat  = "derived/participant_set_analysis";
            MatName = "participant_valid_analysis.mat";
            outFigs = "derived/participant_set_analysis/valid/figs";
            outTabs = "derived/participant_set_analysis/valid/tables";
        end
    end

    D = load(inMat, "participants_clean");
    if ~isfield(D, "participants_clean")
        error("Variable participants_clean not found in %s.", inMat);
    end
    participants = D.participants_clean;
    N = numel(participants);

    if ~isfolder(outMat), mkdir(outMat); end
    if ~isfolder(outFigs), mkdir(outFigs); end
    if ~isfolder(outTabs), mkdir(outTabs); end

    fprintf('[analyze_participants_overview] Loaded %d participants from %s\n', N, inMat);

    %% ------------------------------------------------------------
    % 1) Extract fields (assumed complete/consistent after preprocessing)
    % ------------------------------------------------------------
    pid      = strings(N,1);
    set_id   = strings(N,1);
    device   = strings(N,1);
    browser  = strings(N,1);

    age_rng  = strings(N,1);
    gender   = strings(N,1);

    rev_exp  = nan(N,1);
    rev_resp = nan(N,1);

    emerg    = strings(N,1);

    q40pre   = nan(N,1);
    q40post  = nan(N,1);
    q14m1    = nan(N,1);
    q14m2    = nan(N,1);

    for i = 1:N
        P = participants(i);

        % Identifiers and platform metadata
        pid(i)     = string(P.participant_id);
        set_id(i)  = string(P.set_id);
        device(i)  = string(P.device_type);
        browser(i) = string(P.browser_name);

        % Demographics
        age_rng(i) = string(P.demographics.age_range);
        gender(i)  = string(P.demographics.gender);

        % Reviews (nested storage as defined by preprocessing)
        rev_exp(i)  = double(P.reviews.items(1).response_struct{1,1}.expected);
        rev_resp(i) = double(P.reviews.items(1).response_struct{1,1}.response);

        % Emergency choice (e.g., "self" or "robot")
        emerg(i) = string(P.emergency.choice);

        % Questionnaire totals (percent scale as used in reporting)
        q40pre(i)  = local_get_questionnaire_total_percent(P.questionnaires.t40_pre);
        q40post(i) = local_get_questionnaire_total_percent(P.questionnaires.t40_post);
        q14m1(i)   = local_get_questionnaire_total_percent(P.questionnaires.t14_mid1);
        q14m2(i)   = local_get_questionnaire_total_percent(P.questionnaires.t14_mid2);
    end

    % Normalize set identifiers to the canonical "Set <A..H>" format.
    set_id = local_normalize_set_id(set_id);

    % Fixed set order used across analyses.
    setCats = "Set " + string(('A':'H')');
    if ~all(ismember(set_id, setCats))
        bad = unique(set_id(~ismember(set_id, setCats)));
        error(['Unexpected set_id values found (expected only SetA..SetH). ' ...
               'Found: %s'], strjoin(bad, ", "));
    end
    setC = categorical(set_id, setCats, 'Ordinal', true);

    %% ------------------------------------------------------------
    % 2) Basic tables for reporting
    % ------------------------------------------------------------
    T_device = local_count_table(device, "device_type");
    writetable(T_device, fullfile(outTabs, "participants_by_device_type.csv"));

    T_browser = local_count_table(browser, "browser_name");
    writetable(T_browser, fullfile(outTabs, "participants_by_browser.csv"));

    T_emerg = local_count_table(emerg, "emergency_choice");
    writetable(T_emerg, fullfile(outTabs, "participants_by_emergency_choice.csv"));

    %% ------------------------------------------------------------
    % 3) Plots
    % ------------------------------------------------------------
    S = thesisStyle(); % set global plotting defaults and retrieve style struct

    local_bar_counts(age_rng, "Age range (years)", ...
        "Participants by age range", fullfile(outFigs, "age_range"), S);
    
    [gender_format, ~] = local_bin_gender(gender);
    local_bar_counts(gender_format, "Gender", ...
        "Participants by gender", fullfile(outFigs, "gender"), S);

    % Participants by set
    f = figure('Name','Participants by set','Color','w');
    thesisStyle(f);

    countsBySet = countcats(setC);
    b = bar(countsBySet);
    b.FaceColor = S.cyan;
    b.EdgeColor = 'none';

    grid on;
    set(gca,'XTick',1:numel(setCats),'XTickLabel',cellstr(setCats));

    xlabel('Door-trial sequence set', 'FontSize', S.fontSizeLbl);
    ylabel('Number of participants',  'FontSize', S.fontSizeYlb);
    title('Participants by door-trial sequence set', 'FontSize', S.fontSizeTit);

    local_save_figure(f, fullfile(outFigs, "participants_by_set"), S);

    % Expected review distribution
    f = figure('Name','Expected review distribution','Color','w');
    thesisStyle(f);

    h = histogram(rev_exp);
    h.FaceColor = S.cyan;
    h.EdgeColor = 'none';

    grid on;
    xlabel('Expected review score', 'FontSize', S.fontSizeLbl);
    ylabel('Count',                'FontSize', S.fontSizeYlb);
    title('Distribution of expected review scores', 'FontSize', S.fontSizeTit);

    local_save_figure(f, fullfile(outFigs, "review_expected_hist"), S);

    % Expected vs response review (scatter + y=x reference line)
    f = figure('Name','Expected vs response review','Color','w');
    thesisStyle(f);

    sc = scatter(rev_exp, rev_resp, 30, 'filled');
    sc.MarkerFaceColor = S.cyan;
    sc.MarkerFaceAlpha = 1.0; % keep EPS-safe (avoid transparency)
    sc.MarkerEdgeColor = 'none';

    grid on;
    xlabel('Expected review score', 'FontSize', S.fontSizeLbl);
    ylabel('Response review score', 'FontSize', S.fontSizeYlb);
    title('Expected vs response review', 'FontSize', S.fontSizeTit);

    xMin = min([rev_exp; rev_resp]);
    xMax = max([rev_exp; rev_resp]);
    hold on;
    plot([xMin xMax], [xMin xMax], '--', 'Color', S.yellow, 'LineWidth', S.lineWidth);
    hold off;

    local_save_figure(f, fullfile(outFigs, "review_expected_vs_response_scatter"), S);

    % Questionnaires (overall histogram + by-set boxplot)
    local_questionnaire_plots(q40pre,  setC, '$Q^{(40)}_{\mathrm{pre}}$',  '40pre',  outFigs, S);
    local_questionnaire_plots(q40post, setC, '$Q^{(40)}_{\mathrm{post}}$', '40post', outFigs, S);
    local_questionnaire_plots(q14m1,   setC, '$Q^{(14)}_{\mathrm{mid1}}$', '14mid1', outFigs, S);
    local_questionnaire_plots(q14m2,   setC, '$Q^{(14)}_{\mathrm{mid2}}$', '14mid2', outFigs, S);

    %% ------------------------------------------------------------
    % 4) Balance matrices (columns = sets; rows = category bins)
    % ------------------------------------------------------------
    % 4.1 Age range x set
    local_balance_heatmap( ...
        categorical(age_rng), setC, ...
        "Age Range", "Door-trial sequence set", ...
        "Balance matrix: age range vs set", ...
        fullfile(outFigs, "balance_age_range_vs_set"), ...
        fullfile(outTabs, "balance_age_range_vs_set.csv"), S);

    % 4.2 Gender x set (robust to free-form entries)
    [gender_bin, gender_bin_labels] = local_bin_gender(gender);
    local_balance_heatmap( ...
        categorical(gender_bin, gender_bin_labels, 'Ordinal', true), setC, ...
        "Gender", "Door-trial sequence set", ...
        "Balance matrix: gender vs set", ...
        fullfile(outFigs, "balance_gender_vs_set"), ...
        fullfile(outTabs, "balance_gender_vs_set.csv"), S);

    % 4.3 Pre-interaction 40-item questionnaire bins (0-100 percent scale)
    [q40pre_bin, q40pre_bin_labels] = local_bin_percent(q40pre);
    local_balance_heatmap( ...
        categorical(q40pre_bin, q40pre_bin_labels, 'Ordinal', true), setC, ...
        "Pre-interaction trust scores", "Door-trial sequence set", ...
        "Balance matrix: pre-interaction trust vs set", ...
        fullfile(outFigs, "balance_40pre_bins_vs_set"), ...
        fullfile(outTabs, "balance_40pre_bins_vs_set.csv"), S);

    % 4.4 Review expected: discrete values if few uniques, else quantile bins
    [revexp_bin, revexp_bin_labels] = local_bin_numeric_discrete_or_quantiles(rev_exp);
    local_balance_heatmap( ...
        categorical(revexp_bin, revexp_bin_labels, 'Ordinal', true), setC, ...
        "Expected review score", "Door-trial sequence set", ...
        "Balance matrix: expected review vs set", ...
        fullfile(outFigs, "balance_review_expected_vs_set"), ...
        fullfile(outTabs, "balance_review_expected_vs_set.csv"), S);

    % 4.5 Emergency choice x set
    local_balance_heatmap( ...
        categorical(emerg), setC, ...
        "Emergency Choice", "Door-trial sequence set", ...
        "Balance matrix: emergency choice vs set", ...
        fullfile(outFigs, "balance_emergency_vs_set"), ...
        fullfile(outTabs, "balance_emergency_vs_set.csv"), S);

    %% ------------------------------------------------------------
    % 5) Save consolidated MAT
    % ------------------------------------------------------------
    summary = struct();
    summary.N = N;

    summary.participant_id = pid;
    summary.set_id = set_id;
    summary.device_type = device;
    summary.browser_name = browser;

    summary.age_range = age_rng;
    summary.gender = gender;
    summary.emergency_choice = emerg;

    summary.review_expected = rev_exp;
    summary.review_response = rev_resp;

    summary.q40pre_percent  = q40pre;
    summary.q40post_percent = q40post;
    summary.q14mid1_percent = q14m1;
    summary.q14mid2_percent = q14m2;

    summary.tables.device = T_device;
    summary.tables.browser = T_browser;
    summary.tables.emergency = T_emerg;

    save(fullfile(outMat, MatName), "summary");

    fprintf('[analyze_participants_overview] Done.\n');
    fprintf('  Figures: %s\n', outFigs);
    fprintf('  Tables : %s\n', outTabs);
    fprintf('  MAT    : %s\n', outMat);

end

%% =====================================================================
% Helper functions (local)
% ======================================================================

function v = local_get_questionnaire_total_percent(Q)
% local_get_questionnaire_total_percent Extract questionnaire total on percent scale.
%
% Supports either:
%   - Numeric scalar (already a percent total)
%   - Struct with field total_percent
%   - Struct with alternate scalar fields (fallback): total, score
%
% Raises an error if the input cannot be interpreted as a scalar percent.

    if isnumeric(Q) && isscalar(Q)
        v = double(Q);
        return;
    end
    if isstruct(Q)
        if isfield(Q, 'total_percent')
            v = double(Q.total_percent);
            return;
        end
        if isfield(Q, 'total') && isnumeric(Q.total) && isscalar(Q.total)
            v = double(Q.total);
            return;
        end
        if isfield(Q, 'score') && isnumeric(Q.score) && isscalar(Q.score)
            v = double(Q.score);
            return;
        end
    end
    error('Questionnaire value could not be read as a scalar percent.');
end

function set_id = local_normalize_set_id(set_id)
% local_normalize_set_id Normalize set identifiers to the canonical "Set <A..H>" form.
%
% Examples of accepted inputs:
%   "A", "SetA", "set a", "Condition_SetB"
%
% For unrecognized formats, the original trimmed string is retained.

    set_id = string(set_id);
    set_id = strip(set_id);

    for i = 1:numel(set_id)
        s = strip(set_id(i));
        s = replace(s, " ", "");
        sUp = upper(s);

        % Strip leading "SET" if present (e.g., "SetA" -> "A")
        if startsWith(sUp, "SET")
            sUp = extractAfter(sUp, 3);
        end

        % If single letter A-H, normalize directly
        if strlength(sUp) == 1 && sUp >= "A" && sUp <= "H"
            set_id(i) = "Set " + sUp;
            continue;
        end

        % If the last character is A-H, normalize (e.g., "Condition_SetB")
        lastChar = extractAfter(sUp, strlength(sUp)-1);
        if strlength(lastChar) == 1 && lastChar >= "A" && lastChar <= "H"
            set_id(i) = "Set " + lastChar;
        else
            set_id(i) = s;
        end
    end
end

function T = local_count_table(strVec, varName)
% local_count_table Count occurrences of categories in a string/categorical vector.
%
% OUTPUT
%   T - table with columns:
%       <varName> (string) : category label
%       count    (double) : occurrence count
%
% The table is sorted by descending count.

    c = categorical(string(strVec));
    cats = categories(c);
    counts = countcats(c);

    % VariableNames must be a cell array of character vectors (R2022b).
    vn1 = char(string(varName));

    T = table(string(cats), counts, ...
        'VariableNames', {vn1, 'count'});
    T = sortrows(T, 'count', 'descend');
end

function local_bar_counts(strVec, xLabelStr, titleStr, outBase, S)
% local_bar_counts Create and save a count bar chart for a categorical vector.

    c = categorical(string(strVec));
    cats = categories(c);
    counts = countcats(c);

    f = figure('Name', titleStr, 'Color', 'w');
    thesisStyle(f);

    b = bar(counts);
    b.FaceColor = S.cyan;
    b.EdgeColor = 'none';

    grid on;
    set(gca,'XTick',1:numel(cats),'XTickLabel',cellstr(cats));

    xlabel(xLabelStr, 'FontSize', S.fontSizeLbl);
    ylabel('Number of participants', 'FontSize', S.fontSizeYlb);
    title(titleStr, 'FontSize', S.fontSizeTit);

    local_save_figure(f, outBase, S);
end

function local_questionnaire_plots(qPercent, setC, tag, filetag, outFigs, S)
% local_questionnaire_plots Plot questionnaire distribution and by-set boxplots.
%
% INPUTS
%   qPercent - numeric vector of questionnaire totals (percent scale)
%   setC     - categorical set assignment (ordinal Set A..H)
%   tag      - display label (often LaTeX math string)
%   filetag  - file stem used for output naming
%   outFigs  - output folder for figures
%   S        - thesisStyle struct

    % Overall distribution histogram
    f = figure('Name', sprintf('%s overall histogram', tag), 'Color', 'w');
    thesisStyle(f);

    h = histogram(qPercent, 'BinWidth', 5);
    h.FaceColor = S.cyan;
    h.EdgeColor = 'none';

    grid on;
    xlabel(sprintf('%s total', tag), 'FontSize', S.fontSizeLbl);
    ylabel('Count', 'FontSize', S.fontSizeYlb);
    title(sprintf('Distribution of %s questionnaire scores', tag), ...
          'FontSize', S.fontSizeTit);

    local_save_figure(f, fullfile(outFigs, sprintf('%s_hist_all', filetag)), S);

    % By-set boxplot
    f = figure('Name', sprintf('%s by set', tag), 'Color', 'w');
    thesisStyle(f);

    boxplot(qPercent, setC);
    grid on;
    xlabel('Door-trial sequence set', 'FontSize', S.fontSizeLbl);
    ylabel(sprintf('%s total (percent)', tag), 'FontSize', S.fontSizeYlb);
    title(sprintf('%s questionnaire scores by set', tag), 'FontSize', S.fontSizeTit);

    local_save_figure(f, fullfile(outFigs, sprintf('%s_box_by_set', filetag)), S);
end

function local_balance_heatmap(rowCat, colCat, rowName, colName, titleStr, outFigBase, outCsvPath, S)
% local_balance_heatmap Build a set-balance matrix and export as heatmap + CSV.
%
% The resulting matrix has:
%   - Columns: categories of colCat (door-trial sets)
%   - Rows   : categories of rowCat (bins/categories)
%
% The CSV export stores the numeric counts with a leading column containing
% the row category labels.

    rowCat = categorical(rowCat);
    colCat = categorical(colCat);

    rowLevels = categories(rowCat);
    colLevels = categories(colCat);

    M = zeros(numel(rowLevels), numel(colLevels));
    for i = 1:numel(rowLevels)
        for j = 1:numel(colLevels)
            M(i,j) = sum(rowCat == rowLevels{i} & colCat == colLevels{j});
        end
    end

    % Export counts table
    T = array2table(M, 'VariableNames', matlab.lang.makeValidName(colLevels));
    T = addvars(T, string(rowLevels), 'Before', 1, 'NewVariableNames', rowName);
    writetable(T, outCsvPath);

    % Heatmap figure
    f = figure('Name', titleStr, 'Color', 'w');
    thesisStyle(f);

    h = heatmap(colLevels, rowLevels, M);
    h.XLabel = colName;
    h.YLabel = rowName;
    h.Title  = titleStr;

    local_save_figure(f, outFigBase, S);
end


function [binStr, binLabels] = local_bin_gender(x)
% local_bin_gender Bin gender values into fixed names.
%
% Bins:
%   Man, Woman, Non-binary, Gender-Fluid, Prefer not to say
%
% OUTPUTS
%   binStr    - per-element bin label (string)
%   binLabels - ordered list of bin labels (string)

    binStr = strings(size(x));
    for i = 1:length(x)
        gender = string(x(i));
        switch gender
            case "man"
                binStr(i) = "Man";
            case "woman"
                binStr(i) = "Woman";
            case "non-binary"
                binStr(i) = "Non-binary";
            case "gender-fluid"
                binStr(i) = "Gender-fluid";
            case "prefer_not_to_say"
                binStr(i) = "Prefer not to say";
          
            otherwise
                % Never blank:
                binStr(i) = gender;  % fallback to gender
        end
        
        binStr(i) = strip(binStr(i));
        if ismissing(binStr(i)) || strlength(binStr(i))==0
            binStr(i) = "UNKNOWN";
        end
    end
    binStr = cellstr(binStr);
    binLabels = ["Man", "Woman", "Gender-fluid", "Non-binary", "Prefer not to say"];
end

function [binStr, binLabels] = local_bin_percent(x)
% local_bin_percent Bin percent-valued scores into fixed ranges.
%
% Bins:
%   [0,20), [20,40), [40,60), [60,80), [80,100]
%
% OUTPUTS
%   binStr    - per-element bin label (string)
%   binLabels - ordered list of bin labels (string)

    edges = [0 20 40 60 80 100.0001];
    labels = ["0–20", "20–40", "40–60", "60–80", "80–100"];

    binIdx = discretize(x, edges);
    binStr = strings(size(x));
    for k = 1:numel(x)
        binStr(k) = labels(binIdx(k));
    end
    binLabels = labels;
end

function [binStr, binLabels] = local_bin_numeric_discrete_or_quantiles(x)
% local_bin_numeric_discrete_or_quantiles Bin numeric values for balance matrices.
%
% If values are effectively discrete (few unique values and nearly integers),
% bins are the unique values. Otherwise, values are binned into quantiles.
%
% OUTPUTS
%   binStr    - per-element bin label (string)
%   binLabels - ordered list of bin labels (string)

    x = x(:);
    xFinite = x(isfinite(x));
    u = unique(xFinite);

    % Discrete heuristic: small number of unique values and close to integers
    isNearlyInteger = all(abs(u - round(u)) < 1e-9);
    if numel(u) <= 12 && isNearlyInteger
        u = sort(u);
        binLabels = string(u);
        binStr = strings(size(x));
        for k = 1:numel(x)
            binStr(k) = string(x(k));
        end
        return;
    end

    % Quantile bins (5)
    q = quantile(xFinite, [0 0.2 0.4 0.6 0.8 1.0]);
    q(1) = q(1) - 1e-9;
    q(end) = q(end) + 1e-9;

    edges = unique(q);
    if numel(edges) < 3
        edges = linspace(min(xFinite), max(xFinite) + 1e-9, 6);
    end

    idx = discretize(x, edges);
    nBins = max(idx);

    binLabels = strings(nBins,1);
    for i = 1:nBins
        a = edges(i);
        b = edges(i+1);
        % Interval notation: [a, b)
        binLabels(i) = sprintf('[%.1f, %.1f)', a, b);
    end

    binStr = strings(size(x));
    for k = 1:numel(x)
        binStr(k) = binLabels(idx(k));
    end
end

function local_save_figure(f, outBase, S)
% local_save_figure Finalize and export a figure using thesis styling helpers.
    thesisFinalizeFigure(f, S);
    thesisExport(f, outBase);
end
