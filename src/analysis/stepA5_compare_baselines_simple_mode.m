function stepA5_compare_baselines_simple_mode(run_id, resultsMatPath, selectedThetaMatPath, varargin)
% stepA5_compare_baselines_simple_mode Compare the selected SIMPLE model against baseline predictors (train + valid).
%
% This step evaluates a selected SIMPLE-mode trust model parameter vector
% (theta_star) against a set of baseline predictors on both TRAIN and VALID
% participant splits. All methods are evaluated using the exact same simulation
% time grid, door-event alignment, and measurement sampling as produced by the
% simulator, so comparisons are measurement-aligned and directly comparable.
%
% For baselines with free parameters, parameters are fitted on TRAIN by minimizing
% the same weighted residual objective used throughout the pipeline:
%   residual r = y_obs - y_hat, with weights w(kind) determined by measurement kind.
%
% Baselines (evaluated on the same grid/measurements as the model)
%   B1.1 const_dispositional
%       T(t) = participant dispositional trust (fallback to global TRAIN mean).
%
%   B1.2 const_global_train_mean
%       T(t) = global weighted mean computed on TRAIN (measurement-aligned).
%
%   B1.3 const_oracle_participant_mean
%       T(t) = per-participant oracle weighted mean within the evaluated split.
%
%   B2   bump_asymmetric (discrete-time bump + saturation)
%       T(k+1) = clip_[0,1]( T(k) + delta_plus*1{success at k} - delta_minus*1{failure at k} )
%       Fit (delta_plus, delta_minus) on TRAIN with delta_plus >= 0, delta_minus >= 0.
%
%   B3   bump_symmetric
%       T <- clip_[0,1]( T + Delta*u ) at door events. Fit Delta on TRAIN.
%
%   B4   OPTIMo-lite baselines (Bayesian filtering baseline)
%       "optimo_lite"              (filtered; probes enabled; scored pre-update)
%       "optimo_lite_outcome_only" (ablation; probes disabled; scored pre-update)
%
% Methodological consistency
%   - All methods rely on the same simulator-provided measurement alignment.
%   - Fit/eval uses the same weighted metrics computed from aligned residuals.
%
% INPUTS (required)
%   run_id (string|char)
%       Analysis run identifier. Step A1 archived inputs are loaded from:
%         derived/analysis_runs/<run_id>/stepA1_prepare_analysis/
%
%   resultsMatPath (string|char)
%       Fit results MAT file containing cfg.dt and theta_hat_* candidates. If empty,
%       inferred from Step A3 selection output when available.
%
%   selectedThetaMatPath (string|char)
%       MAT file containing theta_star. If empty, inferred from Step A3 outputs.
%       If still missing, falls back to selecting the theta_hat_* with best VALID wRMSE.
%
% NAME-VALUE ARGUMENTS (optional)
%   "Clip01" (logical scalar)
%       Whether baseline outputs should be clipped to [0,1] where applicable.
%       Default: true
%
% OUTPUTS
%   (none)
%       Writes reporting artifacts to:
%         derived/analysis_runs/<run_id>/stepA5_baseline_comparison/
%       including residual tables, summaries, participant tables, fitted baseline
%       parameters, improvements tables, and A5_plot_data.mat (used by Step A6).
%
% ASSUMPTIONS / DEPENDENCIES
%   - Step A1 archived run-local files exist in stepA1_prepare_analysis.
%   - Utility functions on the MATLAB path:
%       must_exist_file, ensure_dir, save_json, load_participants_struct,
%       read_participant_ids, discover_theta_hats, find_theta_in_struct,
%       weight_for_kind, compute_weighted_metrics, summarize_residuals
%   - Simulation helpers on the MATLAB path:
%       trust_simulate_or_predict_one_participant, trust_simulate_baseline_one_participant
%
% NOTES
%   - OPTIMo-lite scoring uses pre-update predictions at measurement times, as
%     implemented inside trust_simulate_baseline_one_participant.
%   - This function preserves baseline method IDs used by downstream steps.

    % -------------------------
    % Parse inputs and options
    % -------------------------
    if nargin < 1 || isempty(run_id)
        error("stepA5_compare_baselines_simple_mode: run_id is required.");
    end
    run_id = string(run_id);

    if nargin < 2, resultsMatPath = ""; end
    if nargin < 3, selectedThetaMatPath = ""; end

    p = inputParser;
    p.addParameter("Clip01", true, @(x) islogical(x) && isscalar(x));
    p.parse(varargin{:});

    cfg = struct();
    cfg.clip01 = logical(p.Results.Clip01);

    % ---- OPTIMo-lite defaults (forwarded into baseline simulator) ----
    cfg.optimo = struct();
    cfg.optimo.n_grid    = 101;
    cfg.optimo.sigma_f   = 0.08;     % fixed probe noise (in [0,1] units)
    cfg.optimo.sigma0    = 0.15;     % initial belief width
    cfg.optimo.use_trend = false;    % 2-parameter transition (no omega_td)
    cfg.optimo.boundary  = "truncate";

    % -------------------------
    % Ensure utilities available (non-fatal)
    % -------------------------
    if exist("must_exist_file", "file") ~= 2
        warning("Utilities not on path. Consider: addpath('src/utils')");
    end

    % ------------------------------------------------------------
    % 0) Load Step A1 archived run-local inputs (train/valid/weights)
    % ------------------------------------------------------------
    a1Dir = fullfile("derived", "analysis_runs", run_id, "stepA1_prepare_analysis");
    manifestPath = fullfile(a1Dir, "manifest.mat");
    must_exist_file(manifestPath, "A1 manifest");

    trainSplitPath         = fullfile(a1Dir, "participants_train_stepV.mat");
    validMappedPath        = fullfile(a1Dir, "participants_valid_probes_mapped_stepM4.mat");
    trainMappedProbesPath  = fullfile(a1Dir, "participants_probes_mapped_stepM4.mat");
    weightsPath            = fullfile(a1Dir, "measurement_weights.mat");

    % Prefer probe-mapped TRAIN file if available (otherwise use raw TRAIN split)
    if isfile(trainMappedProbesPath)
        trainMat = trainMappedProbesPath;
    else
        trainMat = trainSplitPath;
        warning("[A5] Train mapped probes not found in A1 dir; using TRAIN split raw file: %s", trainMat);
    end
    validMat = validMappedPath;

    must_exist_file(trainMat, "Train participants");
    must_exist_file(validMat, "Valid participants (mapped probes)");
    must_exist_file(weightsPath, "Weights");

    participants_train = load_participants_struct(trainMat);
    participants_valid = load_participants_struct(validMat);

    W = load(weightsPath, "weights");
    if ~isfield(W, "weights")
        error("Weights file does not contain variable 'weights': %s", weightsPath);
    end
    weights = W.weights;

    % ------------------------------------------------------------
    % Sanity: participant IDs and duplicates (diagnostics only)
    % ------------------------------------------------------------
    trainIDs = read_participant_ids(participants_train);
    validIDs = read_participant_ids(participants_valid);

    if numel(unique(trainIDs)) < numel(trainIDs)
        warning("[A5] Duplicate participant IDs detected in TRAIN split.");
    end
    if numel(unique(validIDs)) < numel(validIDs)
        warning("[A5] Duplicate participant IDs detected in VALID split.");
    end

    % ------------------------------------------------------------
    % 1) Resolve resultsMatPath, load dt, and discover theta_hat_* candidates
    % ------------------------------------------------------------
    resultsMatPath = string(resultsMatPath);

    if strlength(resultsMatPath) == 0
        selDefault = fullfile("derived","analysis_runs",run_id,"stepA3_model_selection","selection.mat");
        if isfile(selDefault)
            Ssel = load(selDefault, "selection");
            if isfield(Ssel, "selection") && isfield(Ssel.selection, "results_file") && ~isempty(Ssel.selection.results_file)
                resultsMatPath = string(Ssel.selection.results_file);
            end
        end
    end

    if strlength(resultsMatPath) == 0
        error("resultsMatPath was not provided and could not be inferred from A3 selection.");
    end
    must_exist_file(resultsMatPath, "Fit results MAT (resultsMatPath)");

    R = load(resultsMatPath);
    if ~isfield(R, "cfg") || ~isstruct(R.cfg) || ~isfield(R.cfg, "dt") || isempty(R.cfg.dt)
        error("resultsMatPath does not contain cfg.dt.");
    end
    dt = double(R.cfg.dt); %#ok<NASGU>

    thetaList = discover_theta_hats(R);
    if height(thetaList) == 0
        error("No theta_hat_* vectors found in %s.", resultsMatPath);
    end

    % ------------------------------------------------------------
    % 2) Load selected theta_star (Step A3 outputs preferred)
    % ------------------------------------------------------------
    theta_star = [];
    selected_source = "";

    selectedThetaMatPath = string(selectedThetaMatPath);

    % 2.1) User-provided MAT file
    if strlength(selectedThetaMatPath) > 0
        must_exist_file(selectedThetaMatPath, "selectedThetaMatPath");
        Ssel = load(selectedThetaMatPath);
        theta_star = find_theta_in_struct(Ssel);
        selected_source = "selectedThetaMatPath";
    end

    % 2.2) Step A3 selection output (selection.theta_star)
    if isempty(theta_star)
        selPath = fullfile("derived","analysis_runs",run_id,"stepA3_model_selection","selection.mat");
        if isfile(selPath)
            Ssel = load(selPath, "selection");
            if isfield(Ssel, "selection") && isfield(Ssel.selection, "theta_star") && ~isempty(Ssel.selection.theta_star)
                theta_star = Ssel.selection.theta_star(:);
                selected_source = "A3 selection.mat (selection.theta_star)";
            end
        end
    end

    % 2.3) Step A3 theta_star.mat
    if isempty(theta_star)
        thetaPath = fullfile("derived","analysis_runs",run_id,"stepA3_model_selection","theta_star.mat");
        if isfile(thetaPath)
            Ssel = load(thetaPath);
            theta_star = find_theta_in_struct(Ssel);
            selected_source = "A3 theta_star.mat";
        end
    end

    % 2.4) Fallback: choose theta_hat_* with best VALID wRMSE
    if isempty(theta_star)
        fprintf("[Step A5] No selected theta found; falling back to best-valid wRMSE among theta_hat_*.\n");
        bestIdx = 1;
        bestVal = inf;
        for i = 1:height(thetaList)
            th = thetaList.theta{i}(:);
            mv = local_eval_model_wrmse_only(th, participants_valid, R.cfg.dt, weights, cfg);
            if isfinite(mv) && mv < bestVal
                bestVal = mv;
                bestIdx = i;
            end
        end
        theta_star = thetaList.theta{bestIdx}(:);
        selected_source = "fallback: best-valid among theta_hat_*";
        fprintf("[Step A5] Selected fallback theta: %s (valid wRMSE=%.6g)\n", string(thetaList.name(bestIdx)), bestVal);
    end

    theta_star = theta_star(:);

    % ------------------------------------------------------------
    % 3) Output directory + provenance meta
    % ------------------------------------------------------------
    outDir = fullfile("derived","analysis_runs",run_id,"stepA5_baseline_comparison");
    ensure_dir(outDir);

    meta = struct();
    meta.run_id         = char(run_id);
    meta.results_file   = char(resultsMatPath);
    meta.dt             = double(R.cfg.dt);
    meta.a1_manifest    = char(manifestPath);
    meta.train_file     = char(trainMat);
    meta.valid_file     = char(validMat);
    meta.weights_file   = char(weightsPath);
    meta.theta_star     = theta_star;
    meta.theta_dim      = numel(theta_star);
    meta.theta_source   = char(selected_source);
    meta.created        = char(datetime('now','Format','yyyy-MM-dd HH:mm:ss'));
    meta.cfg            = cfg;

    save(fullfile(outDir,"meta.mat"), "meta");
    save_json(fullfile(outDir,"meta.json"), meta);

    % ------------------------------------------------------------
    % 4) Precompute global constant from TRAIN (measurement-aligned definition)
    % ------------------------------------------------------------
    globalConstTrain = local_compute_global_weighted_mean_from_sim_measurements(participants_train, meta.dt, weights, cfg);

    % ------------------------------------------------------------
    % 5) Fit baseline parameters on TRAIN (weighted objective)
    % ------------------------------------------------------------
    fprintf("\n[Step A5] Fitting literature bump+saturation params (delta_plus,delta_minus) on TRAIN using fminsearch...\n");
    [delta_plus, delta_minus, ode_fit_info] = local_fit_ode_params(participants_train, meta.dt, weights, globalConstTrain, cfg);

    fprintf("[Step A5] Fitting symmetric bump Delta on TRAIN using fminbnd...\n");
    [delta_bump, bump_fit_info] = local_fit_bump_delta(participants_train, meta.dt, weights, globalConstTrain, cfg);

    fprintf("[Step A5] Fitting OPTIMo-lite (omega_tb, omega_tp, sigma_t) on TRAIN using fminsearch...\n");
    [optimoParams, optimo_fit_info] = local_fit_optimo_lite(participants_train, meta.dt, weights, globalConstTrain, cfg);

    fit_params = struct();
    fit_params.globalConstTrain = globalConstTrain;

    fit_params.ode.delta_plus  = delta_plus;
    fit_params.ode.delta_minus = delta_minus;
    fit_params.ode.fit_info    = ode_fit_info;

    fit_params.bump.delta      = delta_bump;
    fit_params.bump.fit_info   = bump_fit_info;

    fit_params.optimo          = optimoParams;
    fit_params.optimo.fit_info = optimo_fit_info;

    save(fullfile(outDir,"fit_params.mat"), "fit_params", "cfg");
    save_json(fullfile(outDir,"fit_params.json"), fit_params);

    % Bundle baseline parameters for trust_simulate_baseline_one_participant()
    baselineParams = struct();
    baselineParams.globalConstTrain = globalConstTrain;
    baselineParams.delta_plus  = delta_plus;
    baselineParams.delta_minus = delta_minus;
    baselineParams.delta       = delta_bump;
    baselineParams.optimo      = optimoParams;

    % ------------------------------------------------------------
    % 6) Evaluate all methods on TRAIN and VALID (measurement-aligned)
    % ------------------------------------------------------------
    methods = ["simple_selected", ...
               "const_dispositional", ...
               "const_global_train_mean", ...
               "const_oracle_participant_mean", ...
               "bump_asymmetric", ...
               "bump_symmetric", ...
               "optimo_lite", ...
               "optimo_lite_outcome_only"];

    splits = struct();
    splits.train = participants_train;
    splits.valid = participants_valid;

    allSummaries     = struct();
    allResiduals     = struct();
    allParticipants  = struct();

    for sname = ["train","valid"]
        Pset = splits.(sname);

        for m = 1:numel(methods)
            method = methods(m);
            fprintf("[Step A5] Evaluating %-28s on %s...\n", method, sname);

            [resTbl, sumTbl, partTbl] = local_eval_method_sim_aligned( ...
                method, Pset, meta.dt, weights, cfg, theta_star, baselineParams);

            base = sprintf("%s_%s", method, sname);

            writetable(resTbl,  fullfile(outDir, base + "_residuals.csv"));
            writetable(sumTbl,  fullfile(outDir, base + "_summary.csv"));
            writetable(partTbl, fullfile(outDir, base + "_participants.csv"));
            save(fullfile(outDir, base + ".mat"), "resTbl", "sumTbl", "partTbl");

            allSummaries.(method).(sname)    = sumTbl;
            allResiduals.(method).(sname)    = resTbl;
            allParticipants.(method).(sname) = partTbl;
        end
    end

    % ------------------------------------------------------------
    % 6.1) Sanity: required residual table columns exist (VALID split)
    % ------------------------------------------------------------
    reqResCols = ["participant_id","kind","t_s","y_obs","y_hat","residual","weight"];
    for m = 1:numel(methods)
        rt = allResiduals.(methods(m)).valid;
        missing = reqResCols(~ismember(reqResCols, string(rt.Properties.VariableNames)));
        if ~isempty(missing)
            error("[A5] Residual table missing required columns for method %s: %s", methods(m), strjoin(missing,", "));
        end
    end

    % ------------------------------------------------------------
    % 7) Improvements table: SIMPLE model vs each baseline (overall + per-kind)
    % ------------------------------------------------------------
    improvements = local_build_improvement_table(allResiduals);
    writetable(improvements, fullfile(outDir, "improvements.csv"));
    save(fullfile(outDir, "improvements.mat"), "improvements");

    % ------------------------------------------------------------
    % 8) Plot-ready tables (VALID overall + VALID per-kind)
    % ------------------------------------------------------------
    validOverall = table();
    validOverall.method = methods(:);
    validOverall.N_overall     = NaN(numel(methods),1);
    validOverall.wRMSE_overall = NaN(numel(methods),1);
    validOverall.wMAE_overall  = NaN(numel(methods),1);
    validOverall.bias_overall  = NaN(numel(methods),1);
    validOverall.wBias_overall = NaN(numel(methods),1);

    for i = 1:numel(methods)
        mth = methods(i);
        Ts = allSummaries.(mth).valid;
        validOverall.N_overall(i)     = Ts.N_overall(1);
        validOverall.wRMSE_overall(i) = Ts.wRMSE_overall(1);
        validOverall.wMAE_overall(i)  = Ts.wMAE_overall(1);
        validOverall.bias_overall(i)  = Ts.bias_overall(1);
        if ismember("wBias_overall", string(Ts.Properties.VariableNames))
            validOverall.wBias_overall(i) = Ts.wBias_overall(1);
        end
    end

    validKind = local_build_kind_table_from_residuals(allResiduals, methods, "valid");

    % Select best baseline on VALID excluding the oracle baseline
    baselineMask = (validOverall.method ~= "simple_selected") & ...
                   (validOverall.method ~= "const_oracle_participant_mean");
    cand = validOverall(baselineMask, :);
    finiteMask = isfinite(cand.wRMSE_overall);
    if ~any(finiteMask)
        warning("[A5] No finite baseline wRMSE found on VALID (excluding oracle). bestBaseline set to missing.");
        bestBaseline = string(missing);
        bestVal = NaN;
    else
        cand2 = cand(finiteMask,:);
        [bestVal, idx] = min(cand2.wRMSE_overall);
        bestBaseline = cand2.method(idx);
    end

    % ------------------------------------------------------------
    % 9) Save plot bundle for Step A6 (single source for A6 reporting)
    % ------------------------------------------------------------
    A5plot = struct();
    A5plot.meta = meta;
    A5plot.cfg  = cfg;
    A5plot.fit_params = fit_params;

    A5plot.methods = methods;
    A5plot.dt = meta.dt;

    A5plot.allSummaries    = allSummaries;
    A5plot.allResiduals    = allResiduals;
    A5plot.allParticipants = allParticipants;

    A5plot.improvements = improvements;

    A5plot.validOverall = validOverall;
    A5plot.validKind    = validKind;

    A5plot.bestBaseline_valid = bestBaseline;
    A5plot.bestBaseline_valid_wRMSE = bestVal;

    save(fullfile(outDir, "A5_plot_data.mat"), "A5plot");

    % ------------------------------------------------------------
    % 10) Print concise validation summary
    % ------------------------------------------------------------
    fprintf("\n[Step A5] VALIDATION overall wRMSE comparison (lower is better):\n");
    fprintf("  %-28s | %-12s\n", "method", "valid_wRMSE");
    fprintf("  %s\n", repmat('-',1,45));
    for m = 1:numel(methods)
        method = methods(m);
        sumV = allSummaries.(method).valid;
        fprintf("  %-28s | %-12.6g\n", method, sumV.wRMSE_overall(1));
    end

    fprintf("\n[Step A5] Fitted literature bump+saturation params (TRAIN): delta_plus=%.6g, delta_minus=%.6g\n", delta_plus, delta_minus);
    fprintf("[Step A5] Fitted bump Delta (TRAIN): Delta=%.6g\n", delta_bump);
    fprintf("[Step A5] Fitted OPTIMo-lite (TRAIN): omega_tb=%.6g, omega_tp=%.6g, sigma_t=%.6g (sigma_f fixed=%.6g)\n", ...
        optimoParams.omega_tb, optimoParams.omega_tp, optimoParams.sigma_t, cfg.optimo.sigma_f);

    fprintf("[Step A5] Best baseline on VALID (overall wRMSE): %s (wRMSE=%.6g)\n", string(bestBaseline), bestVal);

    fprintf("\n[Step A5] Complete.\n");
    fprintf("          Output: %s\n", outDir);
end

% =====================================================================
% Evaluation core: measurement-aligned simulation for all methods
% =====================================================================

function [resTbl, sumTbl, partTbl] = local_eval_method_sim_aligned(method, participants, dt, weights, cfg, theta_star, baselineParams)
% local_eval_method_sim_aligned Evaluate one method on a participant set using simulator-aligned measurements.
%
% This helper simulates one method for each participant and collects aligned
% measurement residuals into a long-form residual table. Participant-level and
% aggregate metrics are computed using weighted residuals.
%
% INPUTS
%   method        - method identifier (string/char)
%   participants  - participant struct array (TRAIN or VALID)
%   dt            - simulation time step [s]
%   weights       - measurement weight configuration used by weight_for_kind()
%   cfg           - configuration struct forwarded to baseline simulator
%   theta_star    - SIMPLE model parameter vector (used when method is "simple_selected")
%   baselineParams- fitted baseline parameter bundle (used for baseline methods)
%
% OUTPUTS
%   resTbl  - long-form residual table with one row per measurement:
%             participant_id, kind, t_s, y_obs, y_hat, residual, weight
%   sumTbl  - one-row summary table produced by summarize_residuals(resTbl)
%   partTbl - participant-level metrics table (N, wRMSE, wMAE, bias, wBias)

    method = string(method);

    % Long-form accumulation across all participants/measurements
    pid_all   = strings(0,1);
    kind_all  = strings(0,1);
    t_all     = zeros(0,1);
    y_all     = zeros(0,1);
    yhat_all  = zeros(0,1);
    r_all     = zeros(0,1);
    w_all     = zeros(0,1);

    pid_p = read_participant_ids(participants);

    % Participant-level summaries
    n_p     = zeros(numel(participants),1);
    wrmse_p = NaN(numel(participants),1);
    wmae_p  = NaN(numel(participants),1);
    bias_p  = NaN(numel(participants),1);
    wbias_p = NaN(numel(participants),1);

    for i = 1:numel(participants)
        P   = participants(i);
        pid = pid_p(i);

        % Simulate method and obtain aligned measurements/predictions
        sim = local_simulate_method(method, P, dt, weights, cfg, theta_star, baselineParams);

        meas = sim.measurements;
        yh   = sim.y_hat(:);

        if isempty(meas) || isempty(yh)
            continue;
        end
        if numel(yh) ~= numel(meas)
            error("[A5] Method %s returned y_hat length %d but measurements length %d (pid=%s).", method, numel(yh), numel(meas), pid);
        end

        M = numel(meas);

        % Extract measurement vectors and weights
        y    = NaN(M,1);
        t    = NaN(M,1);
        kind = strings(M,1);
        w    = NaN(M,1);

        for m = 1:M
            y(m)    = double(meas(m).y);
            t(m)    = double(meas(m).t);
            kind(m) = string(meas(m).kind);
            w(m)    = weight_for_kind(kind(m), weights);
        end

        % Residual definition: observed minus predicted
        r = y - yh;

        % Keep only finite rows with positive weights
        ok = isfinite(y) & isfinite(yh) & isfinite(r) & isfinite(w) & (w > 0);
        y = y(ok); t = t(ok); kind = kind(ok); yh = yh(ok); r = r(ok); w = w(ok);

        % Append to global long-form containers
        pid_all  = [pid_all;  repmat(pid, numel(y), 1)]; %#ok<AGROW>
        kind_all = [kind_all; kind]; %#ok<AGROW>
        t_all    = [t_all;    t]; %#ok<AGROW>
        y_all    = [y_all;    y]; %#ok<AGROW>
        yhat_all = [yhat_all; yh]; %#ok<AGROW>
        r_all    = [r_all;    r]; %#ok<AGROW>
        w_all    = [w_all;    w]; %#ok<AGROW>

        % Participant-level weighted metrics
        n_p(i) = numel(r);
        if n_p(i) > 0
            met = compute_weighted_metrics(r, w);
            wrmse_p(i) = met.wRMSE;
            wmae_p(i)  = met.wMAE;
            bias_p(i)  = met.bias;
            wbias_p(i) = met.wBias;
        end
    end

    % Long-form residual table (one row per aligned measurement)
    resTbl = table();
    resTbl.participant_id = pid_all;
    resTbl.kind           = kind_all;
    resTbl.t_s            = t_all;
    resTbl.y_obs          = y_all;
    resTbl.y_hat          = yhat_all;
    resTbl.residual       = r_all;
    resTbl.weight         = w_all;

    % Participant-level summary table
    partTbl = table();
    partTbl.participant_id = pid_p;
    partTbl.N              = n_p;
    partTbl.wRMSE          = wrmse_p;
    partTbl.wMAE           = wmae_p;
    partTbl.bias           = bias_p;
    partTbl.wBias          = wbias_p;

    % Aggregate summary table (utility-provided)
    sumTbl = summarize_residuals(resTbl);
end

function sim = local_simulate_method(method, P, dt, weights, cfg, theta_star, baselineParams)
% local_simulate_method Dispatch simulation for a single participant and method.
%
% SIMPLE model:
%   - method "simple_selected": trust_simulate_or_predict_one_participant("simple", theta_star, ...)
%
% Baselines:
%   - other method IDs are routed to trust_simulate_baseline_one_participant(...)
%
% INPUTS
%   method - method identifier (string/char)
%   P      - participant struct (one element)
%   dt     - time step [s]
%   weights- measurement weights config
%   cfg    - baseline configuration struct (e.g., clipping and OPTIMo cfg)
%   theta_star    - SIMPLE model theta vector
%   baselineParams- fitted baseline parameter bundle
%
% OUTPUT
%   sim - simulator output struct expected to contain:
%         measurements (struct array), y_hat (vector)

    method = string(lower(method));

    switch method
        case "simple_selected"
            sim = trust_simulate_or_predict_one_participant("simple", theta_star, P, dt);

        case {"const_dispositional", "const_global_train_mean", "const_oracle_participant_mean", ...
              "bump_asymmetric", "bump_symmetric", "optimo_lite", "optimo_lite_outcome_only"}
            sim = trust_simulate_baseline_one_participant(method, P, dt, weights, baselineParams, cfg);

        otherwise
            error("[A5] Unknown method: %s", method);
    end
end

% =====================================================================
% Fitting: OPTIMo-lite on TRAIN (objective consistent with evaluation)
% =====================================================================

function [optimoParams, info] = local_fit_optimo_lite(participants_train, dt, weights, globalConstTrain, cfg)
% local_fit_optimo_lite Fit OPTIMo-lite baseline parameters on TRAIN via fminsearch.
%
% Parameters are optimized using the same weighted objective used for method
% evaluation, computed from aligned residuals returned by the baseline simulator.
%
% OUTPUT
%   optimoParams - struct with fields omega_tb, omega_tp, sigma_t (and omega_td=0 if enabled)
%   info         - optimization diagnostics (fval, exitflag, output, z0, zhat)

    obj = @(z) local_obj_optimo_lite(z, participants_train, dt, weights, globalConstTrain, cfg);

    % Initial guesses (kept conservative to stabilize fminsearch behavior)
    omega_tb0 = 0.00;
    omega_tp0 = -0.10;
    sigma_t0  = 0.05;

    % Parameterization:
    %   omega_tp is constrained negative via omega_tp = -exp(z2)
    %   sigma_t  is constrained positive via sigma_t  =  exp(z3)
    z0 = [ omega_tb0; log(max(-omega_tp0,1e-8)); log(max(sigma_t0,1e-8)) ];

    opts = optimset('Display','off', 'MaxIter', 400, 'TolX', 1e-6, 'TolFun', 1e-6);
    [zhat, fval, exitflag, output] = fminsearch(obj, z0, opts);

    omega_tb = double(zhat(1));
    omega_tp = -exp(double(zhat(2)));
    sigma_t  = exp(double(zhat(3)));

    optimoParams = struct();
    optimoParams.omega_tb = omega_tb;
    optimoParams.omega_tp = omega_tp;
    optimoParams.sigma_t  = sigma_t;

    % When use_trend is enabled, omega_td is included in downstream codepaths.
    % Here it is set to 0 to preserve the intended reduced model.
    if isfield(cfg,"optimo") && isstruct(cfg.optimo) && isfield(cfg.optimo,"use_trend") && cfg.optimo.use_trend
        optimoParams.omega_td = 0;
    end

    info = struct();
    info.fval = fval;
    info.exitflag = exitflag;
    info.output = output;
    info.z0 = z0;
    info.zhat = zhat;
end

function f = local_obj_optimo_lite(z, participants, dt, weights, globalConstTrain, cfg)
% local_obj_optimo_lite Objective for OPTIMo-lite parameter fitting.
%
% Uses the filtered OPTIMo-lite baseline ("optimo_lite"), while scoring one-step-ahead
% predictions at measurement times as implemented inside the baseline simulator.

    z = z(:);
    if numel(z) < 3
        f = inf; return;
    end

    omega_tb = double(z(1));
    omega_tp = -exp(double(z(2)));
    sigma_t  = exp(double(z(3)));

    baselineParams = struct();
    baselineParams.globalConstTrain = globalConstTrain;
    baselineParams.delta_plus  = NaN;
    baselineParams.delta_minus = NaN;
    baselineParams.delta       = NaN;

    baselineParams.optimo = struct();
    baselineParams.optimo.omega_tb = omega_tb;
    baselineParams.optimo.omega_tp = omega_tp;
    baselineParams.optimo.sigma_t  = sigma_t;

    % Fit to the filtered OPTIMo-lite baseline (probes enabled for filtering),
    % but scored one-step-ahead (pre-update y_hat) via the simulator.
    met = local_collect_wrmse_only_baseline("optimo_lite", participants, dt, weights, baselineParams, cfg);
    f = met.wRMSE;
    if ~isfinite(f), f = inf; end

    % Regularization terms (kept as implemented; discourages extreme diffusion)
    f = f + 1e-4*(omega_tb^2 + omega_tp^2) + 1e-3*(sigma_t^2);
end

% =====================================================================
% Fitting: bump baselines on TRAIN (logic preserved)
% =====================================================================

function [delta_plus, delta_minus, info] = local_fit_ode_params(participants_train, dt, weights, globalConstTrain, cfg)
% local_fit_ode_params Fit asymmetric bump parameters (delta_plus, delta_minus) on TRAIN.
%
% Optimization uses an unconstrained parameterization:
%   delta_plus  = exp(z1)  >= 0
%   delta_minus = exp(z2)  >= 0
% and minimizes TRAIN wRMSE for method "bump_asymmetric".

    obj = @(z) local_obj_ode_params(z, participants_train, dt, weights, globalConstTrain, cfg);

    dp0 = 0.05;
    dm0 = 0.10;

    z0 = [log(max(dp0,1e-12));
          log(max(dm0,1e-12))];

    opts = optimset('Display','off', 'MaxIter', 400, 'TolX', 1e-6, 'TolFun', 1e-6);
    [zhat, fval, exitflag, output] = fminsearch(obj, z0, opts);

    delta_plus  = exp(zhat(1));
    delta_minus = exp(zhat(2));

    info = struct();
    info.fval = fval;
    info.exitflag = exitflag;
    info.output = output;
    info.z0 = z0;
    info.zhat = zhat;
end

function f = local_obj_ode_params(z, participants, dt, weights, globalConstTrain, cfg)
% local_obj_ode_params Objective for asymmetric bump fitting (wRMSE + regularization).

    z = z(:);
    if numel(z) < 2
        f = inf; return;
    end

    delta_plus  = exp(z(1));
    delta_minus = exp(z(2));

    baselineParams = struct();
    baselineParams.globalConstTrain = globalConstTrain;
    baselineParams.delta_plus  = delta_plus;
    baselineParams.delta_minus = delta_minus;
    baselineParams.delta       = NaN;

    met = local_collect_wrmse_only_baseline("bump_asymmetric", participants, dt, weights, baselineParams, cfg);
    f = met.wRMSE;
    if ~isfinite(f), f = inf; end

    % Regularization (kept as implemented)
    f = f + 1e-4 * (delta_plus^2 + delta_minus^2);
end

function [delta, info] = local_fit_bump_delta(participants_train, dt, weights, globalConstTrain, cfg)
% local_fit_bump_delta Fit symmetric bump parameter Delta on TRAIN using fminbnd.

    obj = @(d) local_obj_bump(d, participants_train, dt, weights, globalConstTrain, cfg);

    opts = optimset('Display','off', 'TolX', 1e-4);
    [dhat, fval, exitflag, output] = fminbnd(obj, 0, 1, opts);

    delta = max(dhat, 0);

    info = struct();
    info.fval = fval;
    info.exitflag = exitflag;
    info.output = output;
end

function f = local_obj_bump(delta, participants, dt, weights, globalConstTrain, cfg)
% local_obj_bump Objective for symmetric bump fitting (wRMSE only).

    baselineParams = struct();
    baselineParams.globalConstTrain = globalConstTrain;
    baselineParams.delta_plus  = NaN;
    baselineParams.delta_minus = NaN;
    baselineParams.delta      = max(delta, 0);

    met = local_collect_wrmse_only_baseline("bump_symmetric", participants, dt, weights, baselineParams, cfg);
    f = met.wRMSE;
    if ~isfinite(f), f = inf; end
end

function met = local_collect_wrmse_only_baseline(method, participants, dt, weights, baselineParams, cfg)
% local_collect_wrmse_only_baseline Collect overall wRMSE for a baseline on a participant set.
%
% This helper runs the baseline simulator for each participant, collects aligned
% residuals across all measurements, and returns overall weighted RMSE.

    r_all = zeros(0,1);
    w_all = zeros(0,1);

    for i = 1:numel(participants)
        P = participants(i);

        sim = trust_simulate_baseline_one_participant(method, P, dt, weights, baselineParams, cfg);
        meas = sim.measurements;
        yh   = sim.y_hat(:);

        if isempty(meas) || isempty(yh)
            continue;
        end
        if numel(yh) ~= numel(meas)
            continue;
        end

        M = numel(meas);
        y = NaN(M,1);
        w = NaN(M,1);

        for m = 1:M
            y(m) = double(meas(m).y);
            w(m) = weight_for_kind(string(meas(m).kind), weights);
        end

        r = y - yh;
        ok = isfinite(r) & isfinite(w) & (w > 0);
        r = r(ok); w = w(ok);

        r_all = [r_all; r]; %#ok<AGROW>
        w_all = [w_all; w]; %#ok<AGROW>
    end

    met = struct();
    met.N = numel(r_all);
    if isempty(r_all)
        met.wRMSE = NaN;
    else
        met.wRMSE = compute_weighted_metrics(r_all, w_all).wRMSE;
    end
end

function mu = local_compute_global_weighted_mean_from_sim_measurements(participants, dt, weights, cfg)
% local_compute_global_weighted_mean_from_sim_measurements Compute TRAIN global mean trust (measurement-aligned).
%
% This global constant is computed using the simulator-aligned measurements produced
% for the "const_dispositional" baseline, ensuring the constant is defined on the
% same measurement set and with the same weighting scheme used for evaluation.

    y_all = zeros(0,1);
    w_all = zeros(0,1);

    baselineParams = struct();
    baselineParams.globalConstTrain = NaN;
    baselineParams.delta_plus  = NaN;
    baselineParams.delta_minus = NaN;
    baselineParams.delta      = NaN;

    for i = 1:numel(participants)
        P = participants(i);

        sim = trust_simulate_baseline_one_participant("const_dispositional", P, dt, weights, baselineParams, cfg);
        meas = sim.measurements;

        if isempty(meas), continue; end

        y = NaN(numel(meas),1);
        w = NaN(numel(meas),1);
        for m = 1:numel(meas)
            y(m) = double(meas(m).y);
            w(m) = weight_for_kind(string(meas(m).kind), weights);
        end

        ok = isfinite(y) & isfinite(w) & (w > 0);
        y_all = [y_all; y(ok)]; %#ok<AGROW>
        w_all = [w_all; w(ok)]; %#ok<AGROW>
    end

    if isempty(y_all) || sum(w_all) <= 0
        mu = NaN;
        return;
    end

    % Weighted mean is obtained from compute_weighted_metrics output (wBias)
    mu = compute_weighted_metrics(y_all, w_all).wBias;
end

% =====================================================================
% Improvements table + helpers (logic preserved)
% =====================================================================

function improvements = local_build_improvement_table(allResiduals)
% local_build_improvement_table Compare SIMPLE model to each baseline (train + valid).
%
% Produces a long-form table of absolute and percent improvements for each
% metric and scope (overall and per measurement kind), using residual tables
% already computed for each method and split.

    methods = string(fieldnames(allResiduals));

    modelName = "simple_selected";
    baselines = methods(methods ~= modelName);
    splits = ["train","valid"];

    split_col    = strings(0,1);
    baseline_col = strings(0,1);
    scope_col    = strings(0,1);
    metric_col   = strings(0,1);
    base_val     = zeros(0,1);
    model_val    = zeros(0,1);
    abs_imp      = zeros(0,1);
    pct_imp      = zeros(0,1);
    N_col        = zeros(0,1);

    for s = 1:numel(splits)
        sp = splits(s);

        res_model = allResiduals.(modelName).(sp);
        kinds = unique(res_model.kind);
        scopes = ["overall"; kinds(:)];

        for b = 1:numel(baselines)
            bl = baselines(b);
            res_base = allResiduals.(bl).(sp);

            for sc = 1:numel(scopes)
                scope = scopes(sc);

                for metric = ["wRMSE","wMAE","bias","wBias"]
                    [mv, N] = local_metric_from_residuals(res_model, scope, metric);
                    [bv, ~] = local_metric_from_residuals(res_base,  scope, metric);

                    split_col(end+1,1)    = sp; %#ok<AGROW>
                    baseline_col(end+1,1) = bl; %#ok<AGROW>
                    scope_col(end+1,1)    = scope; %#ok<AGROW>
                    metric_col(end+1,1)   = metric; %#ok<AGROW>
                    base_val(end+1,1)     = bv; %#ok<AGROW>
                    model_val(end+1,1)    = mv; %#ok<AGROW>
                    abs_imp(end+1,1)      = bv - mv; %#ok<AGROW>
                    if isfinite(bv) && abs(bv) > 1e-12
                        pct_imp(end+1,1) = 100*(bv - mv)/bv; %#ok<AGROW>
                    else
                        pct_imp(end+1,1) = NaN; %#ok<AGROW>
                    end
                    N_col(end+1,1) = N; %#ok<AGROW>
                end
            end
        end
    end

    improvements = table();
    improvements.split = split_col;
    improvements.baseline = baseline_col;
    improvements.scope = scope_col;
    improvements.metric = metric_col;
    improvements.baseline_value = base_val;
    improvements.model_value = model_val;
    improvements.abs_improvement = abs_imp;
    improvements.pct_improvement = pct_imp;
    improvements.N = N_col;
end

function [val, N] = local_metric_from_residuals(resTbl, scope, metric)
% local_metric_from_residuals Compute a weighted metric from a residual table subset.
%
% INPUTS
%   resTbl (table) - must contain residual and weight columns, and kind for scoped selection
%   scope (string) - "overall" or a measurement kind label
%   metric(string) - one of {"wRMSE","wMAE","bias","wBias"}
%
% OUTPUTS
%   val (double) - metric value (NaN if unknown metric)
%   N   (double) - number of residuals contributing (from compute_weighted_metrics)

    metric = string(metric);
    scope  = string(scope);

    if scope == "overall"
        mask = true(height(resTbl),1);
    else
        mask = (resTbl.kind == scope);
    end

    r = double(resTbl.residual(mask));
    w = double(resTbl.weight(mask));

    Mmet = compute_weighted_metrics(r, w);
    N = Mmet.N;

    switch metric
        case "wRMSE"
            val = Mmet.wRMSE;
        case "wMAE"
            val = Mmet.wMAE;
        case "bias"
            val = Mmet.bias;
        case "wBias"
            val = Mmet.wBias;
        otherwise
            val = NaN;
    end
end

function kindTbl = local_build_kind_table_from_residuals(allResiduals, methods, splitName)
% local_build_kind_table_from_residuals Build per-kind metric table for a split.
%
% For each method and each measurement kind present in that method's residuals,
% computes weighted metrics and returns a long-form table:
%   method, kind, N, wRMSE, wMAE, bias, wBias

    method_col = strings(0,1);
    kind_col   = strings(0,1);
    N_col      = zeros(0,1);
    wrmse_col  = zeros(0,1);
    wmae_col   = zeros(0,1);
    bias_col   = zeros(0,1);
    wbias_col  = zeros(0,1);

    for i = 1:numel(methods)
        m = methods(i);
        resTbl = allResiduals.(m).(splitName);

        kinds = unique(resTbl.kind);
        for kk = 1:numel(kinds)
            k = kinds(kk);
            mask = (resTbl.kind == k);

            r = double(resTbl.residual(mask));
            w = double(resTbl.weight(mask));

            Mmet = compute_weighted_metrics(r, w);
            if Mmet.N == 0
                continue;
            end

            method_col(end+1,1) = m; %#ok<AGROW>
            kind_col(end+1,1)   = k; %#ok<AGROW>
            N_col(end+1,1)      = Mmet.N; %#ok<AGROW>
            wrmse_col(end+1,1)  = Mmet.wRMSE; %#ok<AGROW>
            wmae_col(end+1,1)   = Mmet.wMAE;  %#ok<AGROW>
            bias_col(end+1,1)   = Mmet.bias;  %#ok<AGROW>
            wbias_col(end+1,1)  = Mmet.wBias; %#ok<AGROW>
        end
    end

    kindTbl = table();
    kindTbl.method = method_col;
    kindTbl.kind   = kind_col;
    kindTbl.N      = N_col;
    kindTbl.wRMSE  = wrmse_col;
    kindTbl.wMAE   = wmae_col;
    kindTbl.bias   = bias_col;
    kindTbl.wBias  = wbias_col;
end

% =====================================================================
% Fallback: evaluate wRMSE for candidate theta_hat_* (SIMPLE model only)
% =====================================================================

function wRMSE = local_eval_model_wrmse_only(theta, participants, dt, weights, cfg) %#ok<INUSD>
% local_eval_model_wrmse_only Compute overall weighted RMSE for a SIMPLE theta candidate.
%
% This helper is used only for selecting a fallback theta_star when Step A3
% selection outputs are unavailable. It simulates SIMPLE mode for each participant
% and computes overall weighted RMSE from aligned residuals.

    r_all = zeros(0,1);
    w_all = zeros(0,1);

    for i = 1:numel(participants)
        P = participants(i);

        sim = trust_simulate_or_predict_one_participant("simple", theta, P, dt);
        meas = sim.measurements;
        yh   = sim.y_hat(:);

        if isempty(meas) || isempty(yh), continue; end
        if numel(meas) ~= numel(yh), continue; end

        M = numel(meas);
        y = NaN(M,1);
        w = NaN(M,1);

        for m = 1:M
            y(m) = double(meas(m).y);
            w(m) = weight_for_kind(string(meas(m).kind), weights);
        end

        r = y - yh;
        ok = isfinite(r) & isfinite(w) & (w > 0);
        r = r(ok); w = w(ok);

        r_all = [r_all; r]; %#ok<AGROW>
        w_all = [w_all; w]; %#ok<AGROW>
    end

    if isempty(r_all)
        wRMSE = NaN;
    else
        wRMSE = compute_weighted_metrics(r_all, w_all).wRMSE;
    end
end
