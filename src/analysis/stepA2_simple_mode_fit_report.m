function stepA2_simple_mode_fit_report(run_id, resultsMatPath)
% stepA2_simple_mode_fit_report Evaluate and report SIMPLE-mode fit on TRAIN and VALID splits.
%
% This run-local, manifest-driven step evaluates one or more fitted parameter
% vectors for the SIMPLE trust model. It:
%   - Loads fitted parameter vectors theta_hat_* from a results MAT file,
%   - Loads TRAIN/VALID participant sets and measurement weights using the
%     Step A1 manifest for the specified run_id,
%   - Simulates SIMPLE-mode trajectories for all participants,
%   - Computes weighted residual tables and summary metrics,
%   - Writes per-optimizer artifacts (MAT + CSV) and diagnostic figures to:
%       derived/analysis_runs/<run_id>/stepA2_simple_mode/
%
% INPUTS
%   run_id (string|char)
%       Analysis run identifier. Must correspond to a Step A1 run directory:
%         derived/analysis_runs/<run_id>/stepA1_prepare_analysis/
%
%   resultsMatPath (string|char)
%       Path to MAT file containing fit outputs. Expected contents:
%         - One or more fields named theta_hat_* (parameter vectors),
%         - Struct cfg with field cfg.dt specifying the simulation time step.
%
% OUTPUTS
%   (none)
%       Files written under:
%         derived/analysis_runs/<run_id>/stepA2_simple_mode/
%           meta.mat
%           <optimizer>_<split>.mat
%           <optimizer>_<split>_residuals.csv
%           <optimizer>_<split>_summary.csv
%           <optimizer>_<split>_participants.csv
%           optimizer_comparison.mat
%           optimizer_comparison.csv
%           Diagnostic figures exported via thesisExport (PDF + FIG)
%
% ASSUMPTIONS / DEPENDENCIES
%   - Step A1 was executed for run_id and produced:
%       derived/analysis_runs/<run_id>/stepA1_prepare_analysis/manifest.mat
%   - Utility functions available on MATLAB path:
%       must_exist_file, resolve_runlocal_or_source, load_participants_struct,
%       discover_theta_hats, weight_for_kind, compute_weighted_metrics,
%       summarize_residuals
%   - Simulation helper available on MATLAB path:
%       trust_simulate_or_predict_one_participant
%   - Plot export helpers available on MATLAB path:
%       thesisStyle, thesisFinalizeFigure, thesisExport,
%       save_residual_diagnostic_figure, save_participant_metric_figure

    % -------------------------
    % Thesis plotting defaults (global)
    % -------------------------
    S = thesisStyle(); %#ok<NASGU> % sets global defaults; struct passed to local plotters

    if nargin < 1 || isempty(run_id)
        error("stepA2_simple_mode_fit_report: run_id is required.");
    end
    if nargin < 2 || isempty(resultsMatPath)
        error("stepA2_simple_mode_fit_report: resultsMatPath is required.");
    end
    run_id = string(run_id);

    if ~isfile(resultsMatPath)
        error("resultsMatPath not found: %s", resultsMatPath);
    end

    % ---------------------------------------------------------------------
    % 0) Utilities availability check (non-fatal)
    % ---------------------------------------------------------------------
    if exist("must_exist_file", "file") ~= 2 || ...
       exist("weight_for_kind", "file") ~= 2 || ...
       exist("compute_weighted_metrics", "file") ~= 2 || ...
       exist("summarize_residuals", "file") ~= 2
        warning("Utilities not found on path. Consider: addpath('src/utils')");
    end

    % ---------------------------------------------------------------------
    % 1) Load Step A1 manifest (run-local inputs and archived file references)
    % ---------------------------------------------------------------------
    a1Dir = fullfile("derived", "analysis_runs", run_id, "stepA1_prepare_analysis");
    manifestPath = fullfile(a1Dir, "manifest.mat");
    must_exist_file(manifestPath, "A1 manifest");

    M = load(manifestPath, "runInfo");
    if ~isfield(M, "runInfo")
        error("A1 manifest.mat does not contain variable 'runInfo'.");
    end
    runInfo = M.runInfo;

    if ~isfield(runInfo, "archive_dir") || isempty(runInfo.archive_dir)
        error("A1 runInfo missing archive_dir. Expected runInfo.archive_dir to exist.");
    end
    archiveDir = string(runInfo.archive_dir);

    if ~isfield(runInfo, "files") || ~isstruct(runInfo.files)
        error("A1 runInfo missing files struct. Expected runInfo.files.* entries.");
    end

    % Resolve run-local copies (preferred) with fallback to original source paths.
    trainMat = resolve_runlocal_or_source(archiveDir, runInfo.files, ...
        "train_mapped_probes_archive", "participants_probes_mapped_stepM4.mat", "Train participants (mapped probes)");

    validMat = resolve_runlocal_or_source(archiveDir, runInfo.files, ...
        "valid_mappedP_out", "participants_valid_probes_mapped_stepM4.mat", "Valid participants (mapped probes)");

    wMat = resolve_runlocal_or_source(archiveDir, runInfo.files, ...
        "weights", "measurement_weights.mat", "Measurement weights");

    % ---------------------------------------------------------------------
    % 2) Load fit results (theta_hat_* vectors and cfg.dt)
    % ---------------------------------------------------------------------
    R = load(resultsMatPath);

    if ~isfield(R, "cfg") || ~isstruct(R.cfg) || ~isfield(R.cfg, "dt") || isempty(R.cfg.dt)
        error("resultsMatPath does not contain cfg.dt.");
    end
    dt = double(R.cfg.dt);

    thetaList = discover_theta_hats(R);
    if height(thetaList) == 0
        error("No theta_hat_* vectors found in %s.", resultsMatPath);
    end

    % ---------------------------------------------------------------------
    % 3) Load participants and weights
    % ---------------------------------------------------------------------
    participants_train = load_participants_struct(trainMat);
    participants_valid = load_participants_struct(validMat);

    W = load(wMat, "weights");
    if ~isfield(W, "weights")
        error("Weights file does not contain variable 'weights': %s", wMat);
    end
    weights = W.weights;

    % ---------------------------------------------------------------------
    % 4) Output directories and meta information
    % ---------------------------------------------------------------------
    outDir = fullfile("derived", "analysis_runs", run_id, "stepA2_simple_mode");
    if ~isfolder(outDir), mkdir(outDir); end

    meta = struct();
    meta.run_id        = char(run_id);
    meta.results_file  = char(resultsMatPath);
    meta.dt            = dt;
    meta.train_file    = char(trainMat);
    meta.valid_file    = char(validMat);
    meta.weights_file  = char(wMat);
    meta.a1_manifest   = char(manifestPath);
    meta.created       = char(datetime('now','Format','yyyy-MM-dd HH:mm:ss'));
    meta.optimizers    = cellstr(thetaList.name);

    save(fullfile(outDir, "meta.mat"), "meta");

    % ---------------------------------------------------------------------
    % 5) Evaluate each optimizer on each split
    % ---------------------------------------------------------------------
    splits = struct();
    splits.train = participants_train;
    splits.valid = participants_valid;

    allResults = struct();

    for o = 1:height(thetaList)
        opt   = string(thetaList.name(o));
        theta = thetaList.theta{o}(:);

        for sname = ["train","valid"]
            Pset = splits.(sname);

            % Core evaluation: simulations + residual assembly + weighted summaries
            [resTbl, sumTbl, partTbl] = eval_simple_mode(theta, Pset, dt, weights);

            base = sprintf("%s_%s", opt, sname);

            % Persist tables and raw artifacts for later reporting/diagnostics
            save(fullfile(outDir, base + ".mat"), "theta", "resTbl", "sumTbl", "partTbl");
            writetable(resTbl,  fullfile(outDir, base + "_residuals.csv"));
            writetable(sumTbl,  fullfile(outDir, base + "_summary.csv"));
            writetable(partTbl, fullfile(outDir, base + "_participants.csv"));

            % Thesis-style diagnostics (PDF + FIG via thesisExport)
            save_residual_diagnostic_figure(resTbl, outDir, base + "_residuals", thesisStyle());
            save_participant_metric_figure(partTbl, outDir, base + "_participant_wrmse", thesisStyle());

            % Retain in-memory for optimizer comparison table
            allResults.(opt).(sname).theta   = theta;
            allResults.(opt).(sname).resTbl  = resTbl;
            allResults.(opt).(sname).sumTbl  = sumTbl;
            allResults.(opt).(sname).partTbl = partTbl;
        end
    end

    % ---------------------------------------------------------------------
    % 6) Optimizer comparison table (decision aid)
    % ---------------------------------------------------------------------
    comp = table();
    comp.optimizer = thetaList.name;

    comp.train_wRMSE = NaN(height(thetaList),1);
    comp.valid_wRMSE = NaN(height(thetaList),1);
    comp.train_wMAE  = NaN(height(thetaList),1);
    comp.valid_wMAE  = NaN(height(thetaList),1);
    comp.train_bias  = NaN(height(thetaList),1);
    comp.valid_bias  = NaN(height(thetaList),1);
    comp.train_wBias = NaN(height(thetaList),1);
    comp.valid_wBias = NaN(height(thetaList),1);
    comp.n_train     = NaN(height(thetaList),1);
    comp.n_valid     = NaN(height(thetaList),1);

    for o = 1:height(thetaList)
        opt = string(thetaList.name(o));
        st = allResults.(opt).train.sumTbl;
        sv = allResults.(opt).valid.sumTbl;

        comp.train_wRMSE(o) = st.wRMSE_overall(1);
        comp.train_wMAE(o)  = st.wMAE_overall(1);
        comp.train_bias(o)  = st.bias_overall(1);
        comp.train_wBias(o) = st.wBias_overall(1);
        comp.n_train(o)     = st.N_overall(1);

        comp.valid_wRMSE(o) = sv.wRMSE_overall(1);
        comp.valid_wMAE(o)  = sv.wMAE_overall(1);
        comp.valid_bias(o)  = sv.bias_overall(1);
        comp.valid_wBias(o) = sv.wBias_overall(1);
        comp.n_valid(o)     = sv.N_overall(1);
    end

    writetable(comp, fullfile(outDir, "optimizer_comparison.csv"));
    save(fullfile(outDir, "optimizer_comparison.mat"), "comp");

    fprintf("[Step A2] SIMPLE-mode evaluation complete.\n");
    fprintf("          Output: %s\n", outDir);
end

% =====================================================================
% Evaluation core (kept in this file; relies on utils for weights/metrics)
% =====================================================================

function [resTbl, sumTbl, partTbl] = eval_simple_mode(theta, participants, dt, weights)
% eval_simple_mode Simulate SIMPLE-mode and compute weighted residual metrics.
%
% This helper:
%   - Runs trust_simulate_or_predict_one_participant for each participant,
%   - Builds a long-form residual table with per-sample weights,
%   - Computes participant-level weighted metrics (wRMSE, wMAE, bias, wBias),
%   - Returns an overall summary table via summarize_residuals(resTbl).
%
% INPUTS
%   theta        - fitted SIMPLE-mode parameter vector
%   participants - struct array of participants for the target split
%   dt           - simulation time step [s]
%   weights      - measurement weight configuration used by weight_for_kind()
%
% OUTPUTS
%   resTbl  - long-form residual table (one row per measurement sample)
%   sumTbl  - overall summary table produced by summarize_residuals()
%   partTbl - participant-level metric table

    mode = "simple";

    resChunks = cell(numel(participants),1);

    % Participant-level summary vectors (aligned to participants input order)
    pid_p   = strings(numel(participants),1);
    n_p     = zeros(numel(participants),1);
    wrmse_p = NaN(numel(participants),1);
    wmae_p  = NaN(numel(participants),1);
    bias_p  = NaN(numel(participants),1);
    wbias_p = NaN(numel(participants),1);

    for i = 1:numel(participants)
        P = participants(i);

        % Participant identifier (fallback to index if missing)
        if isfield(P, "participant_id") && ~isempty(P.participant_id)
            pid = string(P.participant_id);
        else
            pid = "P" + string(i);
        end
        pid_p(i) = pid;

        % Run SIMPLE-mode simulation/prediction
        sim = trust_simulate_or_predict_one_participant(mode, theta, P, dt);

        meas = sim.measurements;
        yhat = sim.y_hat(:);

        % Participants without measurements contribute no rows to resTbl
        if isempty(meas)
            resChunks{i} = table();
            continue;
        end

        M = numel(meas);

        % Defensive check: simulator should return a y_hat at least as long as meas
        if numel(yhat) < M
            error("Simulator returned y_hat shorter than measurements (pid=%s).", pid);
        end

        % Extract aligned observation metadata and weights
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

        % Compare observations to the corresponding simulated outputs
        yhat_m = yhat(1:M);
        r = y - yhat_m;

        % Keep only valid, positively weighted samples
        ok = isfinite(y) & isfinite(yhat_m) & isfinite(r) & isfinite(w) & (w > 0);
        y = y(ok); t = t(ok); kind = kind(ok); yhat_m = yhat_m(ok); r = r(ok); w = w(ok);

        % Long-form residual table for this participant
        resTbl_i = table();
        resTbl_i.participant_id = repmat(pid, numel(y), 1);
        resTbl_i.kind           = kind;
        resTbl_i.t_s            = t;
        resTbl_i.y_obs          = y;
        resTbl_i.y_hat          = yhat_m;
        resTbl_i.residual       = r;
        resTbl_i.weight         = w;

        resChunks{i} = resTbl_i;

        % Participant-level metrics
        n_p(i) = numel(r);
        if n_p(i) > 0
            met = compute_weighted_metrics(r, w);
            wrmse_p(i) = met.wRMSE;
            wmae_p(i)  = met.wMAE;
            bias_p(i)  = met.bias;
            wbias_p(i) = met.wBias;
        end
    end

    % Concatenate residual rows across all participants
    resTbl = vertcat(resChunks{:});

    % Participant-level metric table (kept aligned to input order)
    partTbl = table();
    partTbl.participant_id = pid_p;
    partTbl.N              = n_p;
    partTbl.wRMSE          = wrmse_p;
    partTbl.wMAE           = wmae_p;
    partTbl.bias           = bias_p;
    partTbl.wBias          = wbias_p;

    % Overall summary (by kind and overall aggregates, as implemented in utils)
    sumTbl = summarize_residuals(resTbl);
end
