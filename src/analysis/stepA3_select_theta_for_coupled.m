function stepA3_select_theta_for_coupled(run_id, resultsMatPath, varargin)
% stepA3_select_theta_for_coupled  Select the best fitted theta (from Step A2) for coupled-mode analysis.
%
% This step selects a single parameter vector theta_star from the set of fitted
% candidates evaluated in Step A2 (SIMPLE-mode). The selected theta_star is saved
% run-locally for downstream coupled-mode analysis and reporting steps.
%
% If resultsMatPath is omitted or empty, it is resolved from:
%   derived/analysis_runs/<run_id>/stepA2_simple_mode/meta.mat  (meta.results_file)
%
% Selection rule (lexicographic ranking)
%   Primary objective:
%     1) Minimize valid_wRMSE
%   Tie-breakers (applied in order):
%     2) Minimize valid_wMAE
%     3) Minimize abs(valid_wBias) if available, otherwise abs(valid_bias)
%     4) Minimize train_wRMSE
%
% Inputs
%   run_id (string/char)
%       Analysis run identifier used by Step A1/A2, locating Step A2 outputs under:
%       derived/analysis_runs/<run_id>/stepA2_simple_mode/
%
%   resultsMatPath (string/char)
%       Path to fit results MAT file containing theta_hat_* variables. If omitted or
%       empty, this function uses the reference stored in Step A2 meta.mat.
%
% Name-Value Arguments (optional)
%   "TieTol" (numeric scalar >= 0)
%       Optional tolerance used to round ranking keys (valid_wRMSE, valid_wMAE,
%       bias tie metric, train_wRMSE) before sorting. This can reduce numerical
%       flip-flops when optimizers have nearly identical scores.
%       Default: 0.0 (no rounding)
%
% Outputs
%   (none)
%       Selection artifacts are written to:
%         derived/analysis_runs/<run_id>/stepA3_model_selection/
%           - selection.mat / selection.json
%           - theta_star.mat
%           - optimizer_comparison_with_selection.csv
%
% Assumptions / Dependencies
%   - Step A2 produced optimizer_comparison.(mat|csv) in the expected directory.
%   - Optimizer names in optimizer_comparison match theta_hat_* names discovered
%     in resultsMatPath (as produced by Step A2).
%   - Utility functions are available on the MATLAB path:
%       must_exist_file, ensure_dir, save_json, discover_theta_hats

    % -------------------------
    % Parse
    % -------------------------
    if nargin < 1 || isempty(run_id)
        error("stepA3_select_theta_for_coupled: run_id is required.");
    end
    if nargin < 2
        resultsMatPath = "";
    end

    p = inputParser;
    p.addParameter("TieTol", 0.0, @(x) isnumeric(x) && isscalar(x) && x >= 0);
    p.parse(varargin{:});
    tieTol = double(p.Results.TieTol);

    run_id = string(run_id);
    resultsMatPath = string(resultsMatPath);

    % -------------------------
    % Ensure utils available (non-fatal)
    % -------------------------
    % This check is intended for interactive use; missing utilities will
    % cause hard errors later when called.
    if exist("must_exist_file", "file") ~= 2
        warning("Utilities not on path. Consider: addpath('src/utils')");
    end

    % -------------------------
    % 0) Locate Step A2 outputs
    % -------------------------
    a2Dir = fullfile("derived", "analysis_runs", run_id, "stepA2_simple_mode");
    if ~isfolder(a2Dir)
        error("Step A2 output folder not found: %s", a2Dir);
    end

    % -------------------------
    % Resolve resultsMatPath (optional)
    % -------------------------
    % If not provided, use Step A2 meta reference to the fit results file.
    if strlength(resultsMatPath) == 0
        metaPath = fullfile(a2Dir, "meta.mat");
        must_exist_file(metaPath, "Step A2 meta.mat");

        Smeta = load(metaPath, "meta");
        if ~isfield(Smeta, "meta") || ~isfield(Smeta.meta, "results_file")
            error("Step A2 meta.mat does not contain meta.results_file.");
        end

        resultsMatPath = string(Smeta.meta.results_file);
        fprintf("[Step A3] resultsMatPath not provided; using A2 meta reference:\n");
        fprintf("          %s\n\n", resultsMatPath);
    end

    must_exist_file(resultsMatPath, "Fit results MAT (resultsMatPath)");

    % -------------------------
    % Load optimizer comparison
    % -------------------------
    % Step A2 aggregates per-optimizer metrics into this comparison table.
    compMat = fullfile(a2Dir, "optimizer_comparison.mat");
    compCsv = fullfile(a2Dir, "optimizer_comparison.csv");

    if isfile(compMat)
        S = load(compMat, "comp");
        comp = S.comp;
    elseif isfile(compCsv)
        comp = readtable(compCsv);
    else
        error("No optimizer comparison file found in %s.", a2Dir);
    end

    if ~ismember("optimizer", string(comp.Properties.VariableNames))
        error("optimizer_comparison table does not contain variable 'optimizer'.");
    end
    comp.optimizer = string(comp.optimizer);

    % -------------------------
    % 1) Print summary table (best-effort)
    % -------------------------
    % Display a compact subset of commonly used metrics for transparency.
    fprintf("\n[Step A3] Optimizer comparison (from Step A2):\n\n");

    desiredCols = [ ...
        "optimizer", ...
        "train_wRMSE", "valid_wRMSE", ...
        "train_wMAE",  "valid_wMAE", ...
        "train_bias",  "valid_bias", ...
        "train_wBias", "valid_wBias" ...
    ];

    presentCols = desiredCols(ismember(desiredCols, string(comp.Properties.VariableNames)));
    missingCols = desiredCols(~ismember(desiredCols, string(comp.Properties.VariableNames)));

    if isempty(presentCols)
        fprintf("[Step A3] WARNING: None of the desired columns were found.\n");
        fprintf("          Available columns are:\n");
        disp(string(comp.Properties.VariableNames)');
        disp(comp);
    else
        disp(comp(:, cellstr(presentCols)));
        if ~isempty(missingCols)
            fprintf("[Step A3] NOTE: Missing columns not shown: %s\n\n", strjoin(missingCols, ", "));
        end
    end

    fprintf("[Step A3] Selection rule:\n");
    fprintf("  Primary     : minimize valid_wRMSE\n");
    fprintf("  Tie-breakers: valid_wMAE → |valid_wBias| → train_wRMSE\n\n");

    % -------------------------
    % 2) Extract selection metrics and rank optimizers
    % -------------------------
    required = ["valid_wRMSE", "valid_wMAE", "train_wRMSE"];
    have = ismember(required, string(comp.Properties.VariableNames));
    if any(~have)
        error("optimizer_comparison is missing required columns for selection: %s", ...
              strjoin(required(~have), ", "));
    end

    vRMSE = comp.valid_wRMSE;
    vMAE  = comp.valid_wMAE;
    tRMSE = comp.train_wRMSE;

    % Use a bias-based tie-breaker if available:
    %   prefer weighted bias, otherwise fall back to unweighted bias.
    if ismember("valid_wBias", string(comp.Properties.VariableNames))
        vBiasForTie = comp.valid_wBias;
        biasLabel = "valid_wBias";
    elseif ismember("valid_bias", string(comp.Properties.VariableNames))
        vBiasForTie = comp.valid_bias;
        biasLabel = "valid_bias";
    else
        % If neither exists, set to zeros so bias does not influence ranking.
        vBiasForTie = zeros(height(comp),1);
        biasLabel = "(none)";
    end

    % Clean non-finite values to avoid unstable sorting behavior.
    vRMSE(~isfinite(vRMSE)) = inf;
    vMAE(~isfinite(vMAE))   = inf;
    tRMSE(~isfinite(tRMSE)) = inf;
    vBiasForTie(~isfinite(vBiasForTie)) = inf;

    % Optional tolerance rounding to stabilize near-ties.
    if tieTol > 0
        vRMSE = round(vRMSE / tieTol) * tieTol;
        vMAE  = round(vMAE  / tieTol) * tieTol;
        tRMSE = round(tRMSE / tieTol) * tieTol;
        vBiasForTie = round(vBiasForTie / tieTol) * tieTol;
    end

    % Lexicographic ranking key: [valid_wRMSE, valid_wMAE, |wBias|, train_wRMSE].
    key = [vRMSE, vMAE, abs(vBiasForTie), tRMSE];
    [~, order] = sortrows(key, [1 2 3 4], "ascend");
    bestIdx = order(1);

    theta_name_star = comp.optimizer(bestIdx);

    % -------------------------
    % 3) Load selected theta from resultsMatPath
    % -------------------------
    % Support generalized theta_hat_* naming by using discover_theta_hats.
    R = load(resultsMatPath);
    thetaList = discover_theta_hats(R);  % expected vars: name, theta

    hit = find(string(thetaList.name) == theta_name_star, 1, "first");
    if isempty(hit)
        error("Selected optimizer '%s' not found in resultsMatPath. Available theta_hat_* names:\n%s", ...
              theta_name_star, strjoin(string(thetaList.name), ", "));
    end
    theta_star = thetaList.theta{hit}(:);

    % dt is carried through if available (useful for coupled-mode steps).
    dt = NaN;
    if isfield(R, "cfg") && isfield(R.cfg, "dt") && ~isempty(R.cfg.dt)
        dt = double(R.cfg.dt);
    end

    % -------------------------
    % 4) Print selection details
    % -------------------------
    fprintf("[Step A3] Selected theta: %s\n", theta_name_star);
    fprintf("          valid_wRMSE = %.6f\n", comp.valid_wRMSE(bestIdx));
    fprintf("          valid_wMAE  = %.6f\n", comp.valid_wMAE(bestIdx));
    if biasLabel == "valid_wBias"
        fprintf("          valid_wBias = %.6f\n", comp.valid_wBias(bestIdx));
    elseif biasLabel == "valid_bias"
        fprintf("          valid_bias  = %.6f\n", comp.valid_bias(bestIdx));
    else
        fprintf("          bias tie-breaker: not available\n");
    end
    fprintf("          train_wRMSE = %.6f\n\n", comp.train_wRMSE(bestIdx));

    % -------------------------
    % 5) Save outputs
    % -------------------------
    % Save a structured selection record for reproducibility and later reporting.
    outDir = fullfile("derived", "analysis_runs", run_id, "stepA3_model_selection");
    ensure_dir(outDir);

    selection = struct();
    selection.run_id           = char(run_id);
    selection.theta_name_star  = char(theta_name_star);
    selection.theta_star       = theta_star;
    selection.dt               = dt;
    selection.results_file     = char(resultsMatPath);
    selection.a2_dir           = char(a2Dir);
    selection.bias_tie_metric  = char(biasLabel);
    selection.metrics_star     = table2struct(comp(bestIdx,:), "ToScalar", true);
    selection.created          = char(datetime('now','Format','yyyy-MM-dd HH:mm:ss'));
    selection.tieTol           = tieTol;

    save(fullfile(outDir, "selection.mat"), "selection", "-v7.3");
    save_json(fullfile(outDir, "selection.json"), selection);
    save(fullfile(outDir, "theta_star.mat"), "theta_star", "-v7.3");

    % Emit a comparison table annotated with the selected entry.
    comp.is_selected = false(height(comp),1);
    comp.is_selected(bestIdx) = true;
    writetable(comp, fullfile(outDir, "optimizer_comparison_with_selection.csv"));

    fprintf("[Step A3] Outputs saved to:\n");
    fprintf("          %s\n", outDir);
end
