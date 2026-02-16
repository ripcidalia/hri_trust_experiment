function stepA4_sensitivity_simple_mode(run_id, resultsMatPath, selectedThetaMatPath, varargin)
% stepA4_sensitivity_simple_mode Sensitivity and stability analysis for the SIMPLE trust model.
%
% This step performs a run-local, simulation-only sensitivity analysis in SIMPLE
% mode around a selected fitted parameter vector (theta_star) and across all
% fitted candidates (theta_hat_*). The screening and visualization objective is:
%
%   f(theta) = validation wRMSE (overall), computed using measurement weights.
%
% The analysis comprises:
%   A4.1 Optimizer-to-optimizer stability across all theta_hat_* on TRAIN and VALID
%        (overall metrics and per-measurement-kind metrics).
%   A4.2 Local one-at-a-time (OAT) sensitivity around theta_star across multiple
%        relative perturbation magnitudes and both +/- directions.
%   A4.3 Morris-like local screening within a bounded neighborhood (no refits).
%   A4.4 Pairwise 2D grid map for the top-two parameters (by Morris mu_star).
%
% INPUTS (required)
%   run_id (string|char)
%       Analysis run identifier. Used to locate run-local artifacts under:
%         derived/analysis_runs/<run_id>/
%
%   resultsMatPath (string|char)
%       Fit results MAT file containing cfg.dt and theta_hat_* candidates.
%       If omitted/empty, the function attempts to infer it from Step A3
%       selection output (selection.mat).
%
%   selectedThetaMatPath (string|char)
%       MAT file containing a selected theta vector (theta_star). If omitted/empty,
%       the function attempts to load Step A3 selection output under:
%         derived/analysis_runs/<run_id>/stepA3_model_selection/
%       If no selected theta is found, the function falls back to choosing the
%       theta_hat_* with the best validation wRMSE among candidates.
%
% NAME-VALUE ARGUMENTS (optional)
%   "EpsList" (numeric vector)
%       Relative perturbation magnitudes for OAT sensitivity.
%       Default: [0.01 0.05 0.10]
%
%   "MorrisCfg" (struct)
%       Overrides for Morris-like screening configuration. Supported fields:
%         delta_rel, box_rel, r_paths, seed, zero_scale
%       Default values are defined in-code (see cfgM initialization).
%
%   "GridFactors" (numeric vector)
%       Multiplicative factors for the 2D grid map around theta_star.
%       Default: linspace(0.9, 1.1, 8)
%
% OUTPUTS
%   (none)
%       Results are written to:
%         derived/analysis_runs/<run_id>/stepA4_sensitivity/
%       including meta files, tables (CSV/MAT), and figures (PDF + FIG via thesisExport).
%
% ASSUMPTIONS / DEPENDENCIES
%   - Step A1 run-local inputs exist under:
%       derived/analysis_runs/<run_id>/stepA1_prepare_analysis/
%     specifically archived participants and measurement weights.
%   - Utility functions are available on the MATLAB path:
%       must_exist_file, ensure_dir, save_json, load_participants_struct,
%       discover_theta_hats, weight_for_kind, compute_weighted_metrics
%   - Simulator helper is available on the MATLAB path:
%       trust_simulate_or_predict_one_participant

    % -------------------------
    % Thesis plotting defaults (global)
    % -------------------------
    S = thesisStyle(); %#ok<NASGU> % sets global defaults; struct passed to local plotters

    % -------------------------
    % Parse inputs
    % -------------------------
    if nargin < 1 || isempty(run_id)
        error("stepA4_sensitivity_simple_mode: run_id is required.");
    end
    run_id = string(run_id);

    if nargin < 2
        resultsMatPath = "";
    end
    if nargin < 3
        selectedThetaMatPath = "";
    end

    p = inputParser;
    p.addParameter("EpsList", [0.01 0.05 0.10], @(x) isnumeric(x) && isvector(x) && all(x > 0));
    p.addParameter("GridFactors", linspace(0.9, 1.1, 8), @(x) isnumeric(x) && isvector(x) && numel(x) >= 3);
    p.addParameter("MorrisCfg", struct(), @(s) isstruct(s));
    p.parse(varargin{:});

    epsList     = double(p.Results.EpsList(:))';
    gridFactors = double(p.Results.GridFactors(:))';

    % Default Morris-like configuration (overridable via "MorrisCfg").
    cfgM = struct();
    cfgM.delta_rel  = 0.05;   % relative step along one coordinate
    cfgM.box_rel    = 0.10;   % +/- neighborhood around theta_star (relative; scaled if near-zero)
    cfgM.r_paths    = 15;     % number of random trajectories
    cfgM.seed       = 1;      % RNG seed for reproducibility
    cfgM.zero_scale = 1.0;    % additive scale used for near-zero parameters
    cfgM_in = p.Results.MorrisCfg;
    fn = fieldnames(cfgM_in);
    for k = 1:numel(fn)
        cfgM.(fn{k}) = cfgM_in.(fn{k});
    end

    % -------------------------
    % Ensure utilities available (non-fatal)
    % -------------------------
    if exist("must_exist_file", "file") ~= 2
        warning("Utilities not on path. Consider: addpath('src/utils')");
    end

    % -------------------------
    % 0) Load Step A1 manifest (run-local train/valid/weights)
    % -------------------------
    a1Dir = fullfile("derived", "analysis_runs", run_id, "stepA1_prepare_analysis");
    manifestPath = fullfile(a1Dir, "manifest.mat");
    must_exist_file(manifestPath, "A1 manifest");

    M = load(manifestPath, "runInfo");
    if ~isfield(M, "runInfo")
        error("A1 manifest.mat does not contain variable 'runInfo'.");
    end
    runInfo = M.runInfo; %#ok<NASGU>

    % Archived copies in A1 step directory (preferred for reproducibility).
    trainSplitPath        = fullfile(a1Dir, "participants_train_stepV.mat");
    validMappedPath       = fullfile(a1Dir, "participants_valid_probes_mapped_stepM4.mat");
    trainMappedProbesPath = fullfile(a1Dir, "participants_probes_mapped_stepM4.mat");
    weightsPath           = fullfile(a1Dir, "measurement_weights.mat");

    % Prefer TRAIN probes-mapped file if present; otherwise fall back to raw TRAIN split.
    if isfile(trainMappedProbesPath)
        trainMat = trainMappedProbesPath;
    else
        trainMat = trainSplitPath;
        warning("[A4] Train mapped probes not found in A1 dir; using TRAIN split raw file: %s", trainMat);
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

    % -------------------------
    % 1) Resolve resultsMatPath and load dt + all theta_hat_*
    % -------------------------
    resultsMatPath = string(resultsMatPath);

    if strlength(resultsMatPath) == 0
        % Attempt to infer the fit results file from Step A3 selection output.
        selDefault = fullfile("derived","analysis_runs",run_id,"stepA3_model_selection","selection.mat");
        if isfile(selDefault)
            Ssel = load(selDefault, "selection");
            if isfield(Ssel, "selection") && isfield(Ssel.selection, "results_file")
                resultsMatPath = string(Ssel.selection.results_file);
            end
        end
    end

    if strlength(resultsMatPath) == 0
        error("resultsMatPath was not provided and could not be inferred from A3 selection.");
    end
    must_exist_file(resultsMatPath, "Fit results MAT (resultsMatPath)");

    R = load(resultsMatPath);

    % dt is required to run simulations consistently with fitted results.
    if ~isfield(R, "cfg") || ~isstruct(R.cfg) || ~isfield(R.cfg, "dt") || isempty(R.cfg.dt)
        error("resultsMatPath does not contain cfg.dt.");
    end
    dt = double(R.cfg.dt);

    % Discover all candidate fitted thetas (theta_hat_*) in results file.
    thetaList = discover_theta_hats(R);  % table with columns: name, theta
    if height(thetaList) == 0
        error("No theta_hat_* vectors found in %s.", resultsMatPath);
    end

    % -------------------------
    % 2) Load selected theta (Step A3 output preferred)
    % -------------------------
    theta_star = [];
    selected_source = "";

    selectedThetaMatPath = string(selectedThetaMatPath);

    % If user provides a selected-theta MAT file, try to extract theta from it.
    if strlength(selectedThetaMatPath) > 0
        must_exist_file(selectedThetaMatPath, "selectedThetaMatPath");
        Ssel = load(selectedThetaMatPath);
        theta_star = find_theta_in_struct(Ssel);
        selected_source = "selectedThetaMatPath";
    end

    % Otherwise, attempt Step A3 selection.mat (selection.theta_star).
    if isempty(theta_star)
        selPath = fullfile("derived","analysis_runs",run_id,"stepA3_model_selection","selection.mat");
        if isfile(selPath)
            Ssel = load(selPath, "selection");
            if isfield(Ssel, "selection") && isfield(Ssel.selection, "theta_star")
                theta_star = Ssel.selection.theta_star(:);
                selected_source = "A3 selection.mat (selection.theta_star)";
            end
        end
    end

    % Otherwise, attempt Step A3 theta_star.mat.
    if isempty(theta_star)
        thetaPath = fullfile("derived","analysis_runs",run_id,"stepA3_model_selection","theta_star.mat");
        if isfile(thetaPath)
            Ssel = load(thetaPath);
            theta_star = find_theta_in_struct(Ssel);
            selected_source = "A3 theta_star.mat";
        end
    end

    % As a last resort, choose the candidate theta_hat_* with best validation wRMSE.
    if isempty(theta_star)
        fprintf("[Step A4] No selected theta found; falling back to best-valid wRMSE among theta_hat_*.\n");
        bestIdx = 1;
        bestVal = inf;
        for i = 1:height(thetaList)
            th = thetaList.theta{i}(:);
            metV = local_eval_simple_metrics(th, participants_valid, dt, weights, false);
            if isfinite(metV.wRMSE) && metV.wRMSE < bestVal
                bestVal = metV.wRMSE;
                bestIdx = i;
            end
        end
        theta_star = thetaList.theta{bestIdx}(:);
        selected_source = "fallback: best-valid among theta_hat_*";
        fprintf("[Step A4] Selected fallback theta: %s (valid wRMSE=%.6g)\n", string(thetaList.name(bestIdx)), bestVal);
    end

    theta_star = theta_star(:);
    D = numel(theta_star);

    % -------------------------
    % Kind list (fixed list used for stable by-kind reporting)
    % -------------------------
    kindList = local_kind_list(); % e.g., ["t40_post","t14_mid1","t14_mid2","probe"]

    % -------------------------
    % Parameter labels (used consistently across plots/tables)
    % -------------------------
    paramLabels = local_param_labels(D); % 1..D labels (TeX strings), fallback pN

    % -------------------------
    % 2.1) Bounds for sensitivity perturbations
    % -------------------------
    [lb, ub] = local_theta_bounds(D);
    theta_star = local_clip_theta(theta_star, lb, ub);

    % -------------------------
    % 3) Output directory and provenance meta
    % -------------------------
    outDir = fullfile("derived","analysis_runs",run_id,"stepA4_sensitivity");
    ensure_dir(outDir);

    meta = struct();
    meta.run_id            = char(run_id);
    meta.results_file      = char(resultsMatPath);
    meta.dt                = dt;
    meta.a1_manifest       = char(manifestPath);
    meta.train_file        = char(trainMat);
    meta.valid_file        = char(validMat);
    meta.weights_file      = char(weightsPath);
    meta.theta_star        = theta_star;
    meta.theta_dim         = D;
    meta.theta_source      = char(selected_source);
    meta.lb                = lb;
    meta.ub                = ub;
    meta.created           = char(datetime('now','Format','yyyy-MM-dd HH:mm:ss'));
    meta.theta_names_found = cellstr(thetaList.name);
    meta.epsList           = epsList;
    meta.gridFactors       = gridFactors;
    meta.morris_cfg        = cfgM;
    meta.kind_list         = cellstr(kindList);
    meta.param_labels      = cellstr(paramLabels);

    save(fullfile(outDir,"meta.mat"), "meta");
    save_json(fullfile(outDir,"meta.json"), meta);

    % ------------------------------------------------------------
    % A4.1 Optimizer-to-optimizer stability (train and valid)
    % ------------------------------------------------------------
    % Compute overall weighted metrics and store per-kind metrics in long form.
    stab = table();
    stab.theta_name  = string(thetaList.name);
    stab.theta_dim   = NaN(height(thetaList),1);
    stab.train_wRMSE = NaN(height(thetaList),1);
    stab.train_wMAE  = NaN(height(thetaList),1);
    stab.train_bias  = NaN(height(thetaList),1);
    stab.train_wBias = NaN(height(thetaList),1);
    stab.valid_wRMSE = NaN(height(thetaList),1);
    stab.valid_wMAE  = NaN(height(thetaList),1);
    stab.valid_bias  = NaN(height(thetaList),1);
    stab.valid_wBias = NaN(height(thetaList),1);

    fprintf("\n[Step A4] A4.1 Optimizer stability metrics (train/valid):\n");
    fprintf("  %-20s | %-10s %-10s %-10s %-10s || %-10s %-10s %-10s %-10s\n", ...
        "theta_name","tr_wRMSE","tr_wMAE","tr_bias","tr_wBias","va_wRMSE","va_wMAE","va_bias","va_wBias");
    fprintf("  %s\n", repmat('-',1,112));

    % Long-format by-kind stability table (one row per theta/split/kind).
    sk_theta   = strings(0,1);
    sk_split   = strings(0,1);
    sk_kind    = strings(0,1);
    sk_N       = zeros(0,1);
    sk_wRMSE   = NaN(0,1);
    sk_wMAE    = NaN(0,1);
    sk_bias    = NaN(0,1);
    sk_wBias   = NaN(0,1);

    for i = 1:height(thetaList)
        th = thetaList.theta{i}(:);

        mt = local_eval_simple_metrics(th, participants_train, dt, weights, false);
        mv = local_eval_simple_metrics(th, participants_valid, dt, weights, false);

        stab.theta_dim(i)   = numel(th);

        stab.train_wRMSE(i) = mt.wRMSE;
        stab.train_wMAE(i)  = mt.wMAE;
        stab.train_bias(i)  = mt.bias;
        stab.train_wBias(i) = mt.wBias;

        stab.valid_wRMSE(i) = mv.wRMSE;
        stab.valid_wMAE(i)  = mv.wMAE;
        stab.valid_bias(i)  = mv.bias;
        stab.valid_wBias(i) = mv.wBias;

        fprintf("  %-20s | %-10.6g %-10.6g %-10.6g %-10.6g || %-10.6g %-10.6g %-10.6g %-10.6g\n", ...
            string(thetaList.name(i)), mt.wRMSE, mt.wMAE, mt.bias, mt.wBias, mv.wRMSE, mv.wMAE, mv.bias, mv.wBias);

        % Append by-kind metrics (train)
        for kk = 1:numel(kindList)
            kname = kindList(kk);
            Kt = mt.by_kind.(kname);
            sk_theta(end+1,1) = string(thetaList.name(i)); %#ok<AGROW>
            sk_split(end+1,1) = "train"; %#ok<AGROW>
            sk_kind(end+1,1)  = kname; %#ok<AGROW>
            sk_N(end+1,1)     = Kt.N; %#ok<AGROW>
            sk_wRMSE(end+1,1) = Kt.wRMSE; %#ok<AGROW>
            sk_wMAE(end+1,1)  = Kt.wMAE; %#ok<AGROW>
            sk_bias(end+1,1)  = Kt.bias; %#ok<AGROW>
            sk_wBias(end+1,1) = Kt.wBias; %#ok<AGROW>
        end

        % Append by-kind metrics (valid)
        for kk = 1:numel(kindList)
            kname = kindList(kk);
            Kv = mv.by_kind.(kname);
            sk_theta(end+1,1) = string(thetaList.name(i)); %#ok<AGROW>
            sk_split(end+1,1) = "valid"; %#ok<AGROW>
            sk_kind(end+1,1)  = kname; %#ok<AGROW>
            sk_N(end+1,1)     = Kv.N; %#ok<AGROW>
            sk_wRMSE(end+1,1) = Kv.wRMSE; %#ok<AGROW>
            sk_wMAE(end+1,1)  = Kv.wMAE; %#ok<AGROW>
            sk_bias(end+1,1)  = Kv.bias; %#ok<AGROW>
            sk_wBias(end+1,1) = Kv.wBias; %#ok<AGROW>
        end
    end
    fprintf("\n");

    stabByKind = table(sk_theta, sk_split, sk_kind, sk_N, sk_wRMSE, sk_wMAE, sk_bias, sk_wBias, ...
        'VariableNames', {'theta_name','split','kind','N','wRMSE','wMAE','bias','wBias'});

    % Pairwise L2 distances between theta candidates (only computed when dimensions match).
    distMat = NaN(height(thetaList), height(thetaList));
    for a = 1:height(thetaList)
        for b = 1:height(thetaList)
            va = thetaList.theta{a}(:);
            vb = thetaList.theta{b}(:);
            if numel(va) == numel(vb)
                distMat(a,b) = norm(va - vb, 2);
            end
        end
    end

    writetable(stab,       fullfile(outDir,"optimizer_stability.csv"));
    writetable(stabByKind, fullfile(outDir,"optimizer_stability_by_kind.csv"));
    save(fullfile(outDir,"optimizer_stability.mat"), "stab", "stabByKind", "distMat", "thetaList");

    % Thesis-style figure export (PDF + FIG)
    local_plot_optimizer_stability(stab, outDir, "optimizer_stability_valid", thesisStyle());

    % ------------------------------------------------------------
    % Baseline evaluation at theta_star (validation)
    % ------------------------------------------------------------
    base = local_eval_simple_metrics(theta_star, participants_valid, dt, weights, true);

    % ------------------------------------------------------------
    % A4.2 Local OAT sensitivity (overall + by-kind deltas)
    % ------------------------------------------------------------
    signs = [-1, +1];

    oat_param_index = zeros(0,1);
    oat_epsilon     = zeros(0,1);
    oat_sign        = zeros(0,1);

    oat_d_wRMSE     = zeros(0,1);
    oat_d_wMAE      = zeros(0,1);
    oat_d_bias      = zeros(0,1);
    oat_d_wBias     = zeros(0,1);
    oat_d_pred      = zeros(0,1);

    % Per-kind delta containers (wRMSE only, to keep OAT output compact).
    d_wRMSE_kind = struct();
    for kk = 1:numel(kindList)
        d_wRMSE_kind.(kindList(kk)) = zeros(0,1);
    end

    for j = 1:D
        for e = 1:numel(epsList)
            eps = epsList(e);
            for s = 1:numel(signs)
                sg = signs(s);

                % Apply signed perturbation to one component (relative unless near-zero).
                thp = theta_star;
                thp(j) = local_perturb_component(theta_star(j), sg, eps);
                thp = local_clip_theta(thp, lb, ub);

                met = local_eval_simple_metrics(thp, participants_valid, dt, weights, true);

                % Overall metric deltas relative to baseline at theta_star.
                oat_param_index(end+1,1) = j; %#ok<AGROW>
                oat_epsilon(end+1,1)     = eps; %#ok<AGROW>
                oat_sign(end+1,1)        = sg; %#ok<AGROW>
                oat_d_wRMSE(end+1,1)     = met.wRMSE - base.wRMSE; %#ok<AGROW>
                oat_d_wMAE(end+1,1)      = met.wMAE  - base.wMAE; %#ok<AGROW>
                oat_d_bias(end+1,1)      = met.bias  - base.bias; %#ok<AGROW>
                oat_d_wBias(end+1,1)     = met.wBias - base.wBias; %#ok<AGROW>

                % By-kind wRMSE deltas relative to baseline at theta_star.
                for kk = 1:numel(kindList)
                    kname = kindList(kk);
                    bk = base.by_kind.(kname);
                    mk = met.by_kind.(kname);
                    d_wRMSE_kind.(kname)(end+1,1) = mk.wRMSE - bk.wRMSE;
                end

                % Prediction change proxy (mean absolute change in y_hat concatenation).
                if ~isempty(base.yhat_all) && ~isempty(met.yhat_all) && numel(base.yhat_all) == numel(met.yhat_all)
                    oat_d_pred(end+1,1) = mean(abs(met.yhat_all - base.yhat_all)); %#ok<AGROW>
                else
                    oat_d_pred(end+1,1) = NaN; %#ok<AGROW>
                end
            end
        end
    end

    oat = table(oat_param_index, oat_epsilon, oat_sign, oat_d_wRMSE, oat_d_wMAE, oat_d_bias, oat_d_wBias, oat_d_pred, ...
        'VariableNames', {'param_index','epsilon','sign','delta_wRMSE','delta_wMAE','delta_bias','delta_wBias','delta_pred'});

    % Add per-kind delta columns for wRMSE.
    for kk = 1:numel(kindList)
        kname = kindList(kk);
        vname = "delta_wRMSE_" + kname;
        oat.(vname) = d_wRMSE_kind.(kname);
    end

    % OAT ranking (overall): median absolute impact on validation wRMSE per parameter.
    oatRank = table();
    oatRank.param_index = (1:D).';
    oatRank.param_label = paramLabels(:);
    oatRank.median_abs_delta_wRMSE = NaN(D,1);
    oatRank.median_abs_delta_pred  = NaN(D,1);
    for j = 1:D
        mask = (oat.param_index == j);
        oatRank.median_abs_delta_wRMSE(j) = median(abs(oat.delta_wRMSE(mask)), 'omitnan');
        oatRank.median_abs_delta_pred(j)  = median(abs(oat.delta_pred(mask)),  'omitnan');
    end
    oatRank = sortrows(oatRank, "median_abs_delta_wRMSE", "descend");

    % OAT ranking by kind (median absolute delta per kind).
    ok_param = (1:D).';
    oatRankByKind = table();
    oatRankByKind.param_index = ok_param;
    oatRankByKind.param_label = paramLabels(:);
    for kk = 1:numel(kindList)
        kname = kindList(kk);
        col = NaN(D,1);
        vname = "delta_wRMSE_" + kname;
        for j = 1:D
            mask = (oat.param_index == j);
            col(j) = median(abs(oat.(vname)(mask)), 'omitnan');
        end
        oatRankByKind.("median_abs_"+vname) = col;
    end

    writetable(oat,            fullfile(outDir,"oat_sensitivity_long.csv"));
    writetable(oatRank,        fullfile(outDir,"oat_sensitivity_rank.csv"));
    writetable(oatRankByKind,  fullfile(outDir,"oat_sensitivity_rank_by_kind.csv"));
    save(fullfile(outDir,"oat_sensitivity.mat"), "oat", "oatRank", "oatRankByKind", "epsList", "theta_star", "paramLabels", "kindList");

    % Thesis-style figure export (PDF + FIG)
    local_plot_oat_rank(oatRank, paramLabels, outDir, "oat_rank_abs_delta_wrmse", thesisStyle());
    local_plot_oat_deltas(oat, D, epsList, paramLabels, outDir, "oat_deltas_vs_eps", thesisStyle());

    % ------------------------------------------------------------
    % A4.3 Morris-like screening (local neighborhood)
    % ------------------------------------------------------------
    rng(cfgM.seed);

    [EE, nEE, EE_byKind, nEE_byKind] = local_morris_screen(theta_star, participants_valid, dt, weights, cfgM, lb, ub, kindList);

    morris = table();
    morris.param_index = (1:D).';
    morris.param_label = paramLabels(:);
    morris.mu_star     = NaN(D,1);
    morris.sigma       = NaN(D,1);
    morris.n_samples   = NaN(D,1);

    for j = 1:D
        eej = EE{j};
        morris.mu_star(j)   = mean(abs(eej), 'omitnan');
        morris.sigma(j)     = std(eej,  'omitnan');
        morris.n_samples(j) = nEE(j);
    end

    % By-kind Morris summary in long format.
    mk_param = strings(0,1);
    mk_kind  = strings(0,1);
    mk_mu    = NaN(0,1);
    mk_sig   = NaN(0,1);
    mk_n     = NaN(0,1);

    for kk = 1:numel(kindList)
        kname = kindList(kk);
        EEK = EE_byKind.(kname);
        nK  = nEE_byKind.(kname);
        for j = 1:D
            eej = EEK{j};
            mk_param(end+1,1) = paramLabels(j); %#ok<AGROW>
            mk_kind(end+1,1)  = kname; %#ok<AGROW>
            mk_mu(end+1,1)    = mean(abs(eej), 'omitnan'); %#ok<AGROW>
            mk_sig(end+1,1)   = std(eej, 'omitnan'); %#ok<AGROW>
            mk_n(end+1,1)     = nK(j); %#ok<AGROW>
        end
    end
    morrisByKind = table(mk_param, mk_kind, mk_mu, mk_sig, mk_n, ...
        'VariableNames', {'param_label','kind','mu_star','sigma','n_samples'});

    writetable(morris,       fullfile(outDir,"morris_summary.csv"));
    writetable(morrisByKind, fullfile(outDir,"morris_summary_by_kind.csv"));
    save(fullfile(outDir,"morris_summary.mat"), "morris", "morrisByKind", "cfgM", "kindList", "paramLabels");

    % Thesis-style figure export (PDF + FIG)
    local_plot_morris(morris, paramLabels, outDir, "morris_mu_sigma", thesisStyle());

    % ------------------------------------------------------------
    % A4.4 Pairwise grid map for top-two parameters by overall mu_star
    % ------------------------------------------------------------
    if D >= 2
        [~, ord] = sort(morris.mu_star, "descend", "MissingPlacement","last");
        p1 = ord(1);
        p2 = ord(2);

        [gridTbl, Z_overall, Z_byKind] = local_pairwise_gridmap(theta_star, p1, p2, gridFactors, participants_valid, dt, weights, base, lb, ub, cfgM.zero_scale, kindList);

        writetable(gridTbl, fullfile(outDir, sprintf("pairwise_grid_p%d_p%d.csv", p1, p2)));
        save(fullfile(outDir, sprintf("pairwise_grid_p%d_p%d.mat", p1, p2)), "gridTbl", "Z_overall", "Z_byKind", "p1", "p2", "gridFactors", "kindList", "paramLabels");

        % Thesis-style figure export (PDF + FIG)
        local_plot_gridmap(Z_overall, gridFactors, p1, p2, paramLabels, outDir, sprintf("gridmap_p%d_p%d", p1, p2), thesisStyle());
    else
        fprintf("[Step A4] Theta dimension < 2; skipping pairwise gridmap.\n");
    end

    fprintf("[Step A4] Sensitivity and stability analysis complete.\n");
    fprintf("          Output: %s\n", outDir);
end

% =====================================================================
% Helpers: measurement kinds and parameter labels
% =====================================================================

function kindList = local_kind_list()
% local_kind_list Fixed list of measurement kinds used for by-kind reporting.
% The kinds must match sim.measurements(:).kind values produced by the simulator.
    kindList = ["t40_post","t14_mid1","t14_mid2","probe"];
end

function labels = local_param_labels(D)
% local_param_labels Parameter labels used in plots and tables.
% Provides TeX-formatted labels for common parameters and "pN" fallbacks.
    base = ["\lambda^{rep}", "\alpha^{sit}", "\lambda^{sit}", "\phi^{fail}", ...
            "\phi^{succ}", "a^{succ}", "\lambda^{lat}", "\kappa^{lat}"];
    labels = strings(D,1);
    for j = 1:D
        if j <= numel(base)
            labels(j) = base(j);
        else
            labels(j) = "p" + string(j);
        end
    end
end

function lbl = local_param_label_for_index(j, paramLabels)
% local_param_label_for_index Safe lookup for a parameter label by index.
    j = double(j);
    if j >= 1 && j <= numel(paramLabels)
        lbl = string(paramLabels(j));
    else
        lbl = "p" + string(j);
    end
end

% =====================================================================
% Core objective evaluation (SIMPLE mode), with optional y_hat concatenation
% =====================================================================

function met = local_eval_simple_metrics(theta, participants, dt, weights, return_yhat)
% local_eval_simple_metrics Evaluate SIMPLE-mode weighted metrics on a participant set.
%
% This helper runs SIMPLE-mode simulation for each participant and aggregates
% weighted residuals across all measurements to compute overall metrics and
% per-kind metrics for the fixed kind list.
%
% INPUTS
%   theta       - parameter vector for SIMPLE mode
%   participants- participant struct array (train or valid)
%   dt          - simulation time step [s]
%   weights     - measurement weight configuration used by weight_for_kind()
%   return_yhat - if true, concatenates predicted values into met.yhat_all
%
% OUTPUT
%   met - struct with fields:
%         N, wRMSE, wMAE, bias, wBias, yhat_all, by_kind.(kind).*

    if nargin < 5, return_yhat = false; end
    mode = "simple";

    % Accumulate residuals and weights across all participants/measurements.
    r_all = zeros(0,1);
    w_all = zeros(0,1);
    k_all = strings(0,1);

    % Optionally store concatenated predictions (used for a simple delta proxy).
    if return_yhat
        yhat_all = zeros(0,1);
    else
        yhat_all = [];
    end

    for i = 1:numel(participants)
        P = participants(i);
        sim = trust_simulate_or_predict_one_participant(mode, theta, P, dt);

        meas = sim.measurements;
        yhat = sim.y_hat(:);

        if isempty(meas) || isempty(yhat)
            continue;
        end

        % Align defensively to the shorter of measurements and predictions.
        M = min(numel(meas), numel(yhat));
        for m = 1:M
            y    = double(meas(m).y);
            yh   = double(yhat(m));
            kind = string(meas(m).kind);
            w    = weight_for_kind(kind, weights);

            if ~(isfinite(y) && isfinite(yh) && isfinite(w) && w > 0)
                continue;
            end

            % Residual definition: observed minus predicted.
            r = y - yh;

            r_all(end+1,1) = r; %#ok<AGROW>
            w_all(end+1,1) = w; %#ok<AGROW>
            k_all(end+1,1) = kind; %#ok<AGROW>

            if return_yhat
                yhat_all(end+1,1) = yh; %#ok<AGROW>
            end
        end
    end

    met = struct();
    met.N = numel(r_all);

    % Overall weighted metrics across all kinds.
    if isempty(r_all)
        met.wRMSE = NaN;
        met.wMAE  = NaN;
        met.bias  = NaN;
        met.wBias = NaN;
    else
        Mmet = compute_weighted_metrics(r_all, w_all);
        % Use fields provided by utils if present; otherwise compute directly.
        if isfield(Mmet,"wRMSE"), met.wRMSE = Mmet.wRMSE; else, met.wRMSE = sqrt(sum(w_all.*(r_all.^2))/sum(w_all)); end
        if isfield(Mmet,"wMAE"),  met.wMAE  = Mmet.wMAE;  else, met.wMAE  = sum(w_all.*abs(r_all))/sum(w_all); end
        if isfield(Mmet,"bias"),  met.bias  = Mmet.bias;  else, met.bias  = mean(r_all); end
        if isfield(Mmet,"wBias"), met.wBias = Mmet.wBias; else, met.wBias = sum(w_all.*r_all)/sum(w_all); end
    end

    met.yhat_all = yhat_all;

    % By-kind metrics computed from the same residual/weight accumulation.
    kindList = local_kind_list();
    met.by_kind = struct();
    for kk = 1:numel(kindList)
        kname = kindList(kk);
        mask = (k_all == kname);
        rk = r_all(mask);
        wk = w_all(mask);

        K = struct();
        K.N = numel(rk);
        if isempty(rk) || sum(wk) <= 0
            K.wRMSE = NaN;
            K.wMAE  = NaN;
            K.bias  = NaN;
            K.wBias = NaN;
        else
            Mk = compute_weighted_metrics(rk, wk);
            if isfield(Mk,"wRMSE"), K.wRMSE = Mk.wRMSE; else, K.wRMSE = sqrt(sum(wk.*(rk.^2))/sum(wk)); end
            if isfield(Mk,"wMAE"),  K.wMAE  = Mk.wMAE;  else, K.wMAE  = sum(wk.*abs(rk))/sum(wk); end
            if isfield(Mk,"bias"),  K.bias  = Mk.bias;  else, K.bias  = mean(rk); end
            if isfield(Mk,"wBias"), K.wBias = Mk.wBias; else, K.wBias = sum(wk.*rk)/sum(wk); end
        end
        met.by_kind.(kname) = K;
    end
end

% =====================================================================
% A4.2 Helper: perturbation rule for a single parameter component
% =====================================================================

function xnew = local_perturb_component(x, signDir, epsRel)
% local_perturb_component Apply a signed relative perturbation to one component.
% Uses a relative multiplicative change for nonzero parameters and an additive
% change for near-zero parameters to avoid negligible updates.
    x = double(x);
    signDir = double(signDir);
    epsRel = double(epsRel);

    if ~isfinite(x)
        xnew = x;
        return;
    end

    if abs(x) < 1e-12
        xnew = x + signDir * epsRel;
    else
        xnew = x * (1 + signDir * epsRel);
    end
end

% =====================================================================
% Bounds helpers (parameter-wise clipping for perturbation/screening neighborhoods)
% =====================================================================

function [lb, ub] = local_theta_bounds(D)
% local_theta_bounds Default parameter bounds used to clip perturbations.
% Bounds are defined for the first 8 parameters (if present). Remaining
% parameters are unbounded (+/- Inf).
    lb8 = [0; 0; 0; 0; 0; 0; 0; 0];
    ub8 = [100; 1.0; 1.0; 1.0; 100; 100; 100; 1];

    lb = -inf(D,1);
    ub = +inf(D,1);

    k = min(D, numel(lb8));
    lb(1:k) = lb8(1:k);
    ub(1:k) = ub8(1:k);

    % Defensive correction if any bound pairs are inverted.
    bad = lb > ub;
    if any(bad)
        tmp = lb(bad); lb(bad) = ub(bad); ub(bad) = tmp;
    end
end

function th = local_clip_theta(th, lb, ub)
%LOCAL_CLIP_THETA Clip theta element-wise to [lb, ub] when sizes match.
    th = double(th(:));
    if isempty(lb) || isempty(ub), return; end
    lb = lb(:); ub = ub(:);
    if numel(lb) ~= numel(th) || numel(ub) ~= numel(th)
        return;
    end
    th = min(max(th, lb), ub);
end

% =====================================================================
% A4.3 Morris-like screening (overall + by-kind) using one eval per theta
% =====================================================================

function [EE, nEE, EE_byKind, nEE_byKind] = local_morris_screen(theta_star, participants_valid, dt, weights, cfgM, lb, ub, kindList)
% local_morris_screen Morris-like local screening around theta_star.
% Returns elementary effects (EE) for overall validation wRMSE and per-kind wRMSE.

    theta_star = theta_star(:);
    D = numel(theta_star);

    % Overall elementary effects per parameter.
    EE = cell(D,1);
    for j = 1:D, EE{j} = zeros(0,1); end
    nEE = zeros(D,1);

    % By-kind elementary effects per parameter.
    EE_byKind = struct();
    nEE_byKind = struct();
    for kk = 1:numel(kindList)
        kname = kindList(kk);
        EE_byKind.(kname) = cell(D,1);
        for j = 1:D, EE_byKind.(kname){j} = zeros(0,1); end
        nEE_byKind.(kname) = zeros(D,1);
    end

    delta_rel = cfgM.delta_rel;
    box_rel   = cfgM.box_rel;
    r_paths   = cfgM.r_paths;

    % Define a local sampling box around theta_star.
    lo = theta_star; hi = theta_star;
    for j = 1:D
        if abs(theta_star(j)) < 1e-12
            lo(j) = -cfgM.zero_scale * box_rel;
            hi(j) = +cfgM.zero_scale * box_rel;
        else
            lo(j) = theta_star(j) * (1 - box_rel);
            hi(j) = theta_star(j) * (1 + box_rel);
        end
        if lo(j) > hi(j)
            tmp = lo(j); lo(j) = hi(j); hi(j) = tmp;
        end
    end

    % Intersect neighborhood bounds with global bounds.
    lo = max(lo, lb(:));
    hi = min(hi, ub(:));

    fval = @(th) local_eval_simple_metrics(th, participants_valid, dt, weights, false);

    for r = 1:r_paths
        % Random starting point in neighborhood.
        th = lo + (hi - lo) .* rand(D,1);
        th = local_clip_theta(th, lb, ub);

        f0 = fval(th);
        if ~isfinite(f0.wRMSE), continue; end

        % Random parameter order per trajectory.
        order = randperm(D);
        for k = 1:D
            j = order(k);

            th2 = th;

            % Random signed step direction; uses delta_rel as nominal scaling.
            stepSign = sign(randn());
            if stepSign == 0, stepSign = 1; end

            if abs(theta_star(j)) < 1e-12
                th2(j) = th2(j) + stepSign * (delta_rel * cfgM.zero_scale);
            else
                th2(j) = th2(j) * (1 + stepSign * delta_rel);
            end

            % Keep within neighborhood and global bounds.
            th2(j) = min(max(th2(j), lo(j)), hi(j));
            th2 = local_clip_theta(th2, lb, ub);

            f1 = fval(th2);
            if isfinite(f1.wRMSE)
                % Elementary effect normalized by delta_rel (as implemented).
                EE{j}(end+1,1) = (f1.wRMSE - f0.wRMSE) / delta_rel; %#ok<AGROW>
                nEE(j) = nEE(j) + 1;

                % By-kind elementary effects on wRMSE.
                for kk = 1:numel(kindList)
                    kname = kindList(kk);
                    a0 = f0.by_kind.(kname).wRMSE;
                    a1 = f1.by_kind.(kname).wRMSE;
                    if isfinite(a0) && isfinite(a1)
                        EE_byKind.(kname){j}(end+1,1) = (a1 - a0) / delta_rel; %#ok<AGROW>
                        nEE_byKind.(kname)(j) = nEE_byKind.(kname)(j) + 1;
                    end
                end

                % Accept the move along trajectory.
                th = th2;
                f0 = f1;
            end
        end
    end
end

% =====================================================================
% A4.4 Pairwise grid map (overall + by-kind) using one eval per theta
% =====================================================================

function [gridTbl, Z_overall, Z_byKind] = local_pairwise_gridmap(theta_star, p1, p2, factors, participants_valid, dt, weights, baseMet, lb, ub, zero_scale, kindList)
% local_pairwise_gridmap Evaluate a 2D multiplicative grid around theta_star for two parameters.
% Returns matrix form (for plotting) and a long-form table (for CSV export).

    theta_star = theta_star(:);

    fval = @(th) local_eval_simple_metrics(th, participants_valid, dt, weights, false);

    n = numel(factors);
    Z_overall = NaN(n,n);
    Z_byKind = struct();
    for kk = 1:numel(kindList)
        Z_byKind.(kindList(kk)) = NaN(n,n);
    end

    idx = 0;
    rows = n*n;

    p1_idx     = zeros(rows,1);
    p2_idx     = zeros(rows,1);
    p1_factor  = zeros(rows,1);
    p2_factor  = zeros(rows,1);
    wr_overall = NaN(rows,1);
    d_overall  = NaN(rows,1);

    wr_kind = struct();
    d_kind  = struct();
    for kk = 1:numel(kindList)
        kname = kindList(kk);
        wr_kind.(kname) = NaN(rows,1);
        d_kind.(kname)  = NaN(rows,1);
    end

    for i = 1:n
        for j = 1:n
            th = theta_star;

            % Apply multiplicative factors (with near-zero handling).
            th(p1) = local_apply_factor(theta_star(p1), factors(i), zero_scale);
            th(p2) = local_apply_factor(theta_star(p2), factors(j), zero_scale);

            th = local_clip_theta(th, lb, ub);

            met = fval(th);

            % Store matrix form for plotting.
            Z_overall(i,j) = met.wRMSE;
            for kk = 1:numel(kindList)
                kname = kindList(kk);
                Z_byKind.(kname)(i,j) = met.by_kind.(kname).wRMSE;
            end

            % Store long form for CSV/MAT outputs.
            idx = idx + 1;

            p1_idx(idx)     = p1;
            p2_idx(idx)     = p2;
            p1_factor(idx)  = factors(i);
            p2_factor(idx)  = factors(j);

            wr_overall(idx) = met.wRMSE;
            d_overall(idx)  = met.wRMSE - baseMet.wRMSE;

            for kk = 1:numel(kindList)
                kname = kindList(kk);
                wr_kind.(kname)(idx) = met.by_kind.(kname).wRMSE;
                d_kind.(kname)(idx)  = met.by_kind.(kname).wRMSE - baseMet.by_kind.(kname).wRMSE;
            end
        end
    end

    gridTbl = table();
    gridTbl.p1_index = p1_idx;
    gridTbl.p2_index = p2_idx;
    gridTbl.p1_factor = p1_factor;
    gridTbl.p2_factor = p2_factor;

    gridTbl.valid_wRMSE = wr_overall;
    gridTbl.delta_valid_wRMSE = d_overall;

    for kk = 1:numel(kindList)
        kname = kindList(kk);
        gridTbl.("valid_wRMSE_" + kname) = wr_kind.(kname);
        gridTbl.("delta_valid_wRMSE_" + kname) = d_kind.(kname);
    end
end

function x = local_apply_factor(x0, f, zero_scale)
% local_apply_factor Apply a multiplicative factor with near-zero handling.
% For near-zero parameters, uses an additive convention based on (f-1)*zero_scale
% to keep perturbations meaningful.
    x0 = double(x0);
    f  = double(f);

    if abs(x0) < 1e-12
        x = (f - 1.0) * zero_scale;
    else
        x = x0 * f;
    end
end

% =====================================================================
% Plotting (thesis-style finalize + export)
% =====================================================================

function local_save_figure(f, outBase, S)
% local_save_figure Finalize and export a figure using thesis styling helpers.
    thesisFinalizeFigure(f, S);
    thesisExport(f, outBase);
end

function local_plot_optimizer_stability(stab, outDir, outBaseName, S)
% local_plot_optimizer_stability Bar chart of validation wRMSE across theta candidates.
    f = figure('Visible','off','Color','w','Name','Optimizer stability (valid)');
    thesisStyle(f);

    b = bar(stab.valid_wRMSE);
    b.EdgeColor = 'none';

    grid on;
    ax = gca;
    ax.XTick = 1:height(stab);
    ax.XTickLabel = optimizerDisplayName(stab.theta_name);
    ax.XTickLabelRotation = 25;

    ylabel('Validation wRMSE', 'FontSize', S.fontSizeYlb);
    title('Optimizer stability: validation wRMSE', 'FontSize', S.fontSizeTit);

    local_save_figure(f, fullfile(outDir, outBaseName), S);
end

function local_plot_oat_rank(oatRank, paramLabels, outDir, outBaseName, S) %#ok<INUSD>
% local_plot_oat_rank Rank parameters by median absolute delta in validation wRMSE.
    f = figure('Visible','off','Color','w','Name','OAT ranking');
    thesisStyle(f);

    b = bar(oatRank.median_abs_delta_wRMSE);
    b.EdgeColor = 'none';

    grid on;
    ax = gca;
    ax.XTick = 1:height(oatRank);
    ax.XTickLabel = cellstr(oatRank.param_label);

    xlabel('Parameter (sorted by sensitivity)', 'FontSize', S.fontSizeLbl);
    ylabel('Median $\left| \Delta \mathrm{wRMSE} \right|$', 'FontSize', S.fontSizeYlb);
    title('OAT sensitivity ranking (validation)', 'FontSize', S.fontSizeTit);

    local_save_figure(f, fullfile(outDir, outBaseName), S);
end

function local_plot_oat_deltas(oat, D, epsList, paramLabels, outDir, outBaseName, S)
% local_plot_oat_deltas Visualize OAT delta(wRMSE) versus epsilon for each parameter.
    f = figure('Visible','off','Color','w','Name','OAT deltas vs epsilon');
    thesisStyle(f);

    set(f,'Units','centimeters');
    set(f,'Position',[2 2 S.figSizeTrajectoryGrid]);

    tiledlayout(ceil(D/2), 2, 'Padding','compact', 'TileSpacing','compact');

    for j = 1:D
        nexttile;
        hold on;

        for e = 1:numel(epsList)
            eps = epsList(e);

            % Separate +/- perturbations for the same epsilon.
            mpos = (oat.param_index==j) & (oat.epsilon==eps) & (oat.sign==+1);
            mneg = (oat.param_index==j) & (oat.epsilon==eps) & (oat.sign==-1);

            yneg = oat.delta_wRMSE(mneg);
            ypos = oat.delta_wRMSE(mpos);

            if ~isempty(yneg)
                plot(eps, mean(yneg,'omitnan'), 'o', 'MarkerSize', 5);
            end
            if ~isempty(ypos)
                plot(eps, mean(ypos,'omitnan'), 'o', 'MarkerSize', 5);
            end
        end

        yline(0,'-');
        grid on;
        xlabel('\epsilon', 'FontSize', S.fontSizeLbl);
        ylabel('$\Delta \mathrm{wRMSE}$', 'FontSize', S.fontSizeYlb);
        title(local_param_label_for_index(j, paramLabels), 'FontSize', S.fontSizeTit);
        hold off;
    end

    local_save_figure(f, fullfile(outDir, outBaseName), S);
end

function local_plot_morris(morris, paramLabels, outDir, outBaseName, S)
% local_plot_morris Scatter plot of Morris mu_star versus sigma with labels.
    f = figure('Visible','off','Color','w','Name','Morris screening');
    thesisStyle(f);

    plot(morris.mu_star, morris.sigma, '.', 'MarkerSize', 14);
    grid on;
    xlabel('$\mu^*$ (mean $|EE|$)', 'FontSize', S.fontSizeLbl);
    ylabel('$\sigma$ (std $EE$)', 'FontSize', S.fontSizeYlb);
    title('Morris-like screening (validation wRMSE objective)', 'FontSize', S.fontSizeTit);

    % Add text labels (may be visually dense for larger parameter counts).
    hold on;
    for i = 1:height(morris)
        if isfinite(morris.mu_star(i)) && isfinite(morris.sigma(i))
            text(morris.mu_star(i), morris.sigma(i), " " + local_param_label_for_index(i, paramLabels), ...
                'Interpreter', S.interpText, 'FontSize', 9);
        end
    end
    hold off;

    local_save_figure(f, fullfile(outDir, outBaseName), S);
end

function local_plot_gridmap(Z, factors, p1, p2, paramLabels, outDir, outBaseName, S)
% local_plot_gridmap Visualize a 2D wRMSE surface for two parameters (grid factors).
    f = figure('Visible','off','Color','w','Name','Pairwise gridmap');
    thesisStyle(f);

    set(f,'Units','centimeters');
    set(f,'Position',[2 2 S.figSizeGridMap]);

    imagesc(Z);
    clim([min(Z(:)) max(Z(:))]);

    % Thesis colormap: cyan to yellow
    n = 256;
    gamma = 0.6;
    t = linspace(0,1,n)'.^gamma;
    cmap = (1-t).*S.cyan + t.*S.yellow;
    colormap(gca, cmap);

    axis tight;
    cb = colorbar; %#ok<NASGU>
    grid on;

    ax = gca;
    ax.XTick = 1:numel(factors);
    ax.YTick = 1:numel(factors);
    ax.XTickLabel = compose("%.3f", factors);
    ax.YTickLabel = compose("%.3f", factors);
    ax.XTickLabelRotation = 25;

    p1lab = local_param_label_for_index(p1, paramLabels);
    p2lab = local_param_label_for_index(p2, paramLabels);

    xlabel(sprintf('Factor on %s', p2lab), 'FontSize', S.fontSizeLbl);
    ylabel(sprintf('Factor on %s', p1lab), 'FontSize', S.fontSizeYlb);
    title(sprintf('Pairwise gridmap: validation wRMSE (%s vs %s)', p1lab, p2lab), 'FontSize', S.fontSizeTit);

    local_save_figure(f, fullfile(outDir, outBaseName), S);
end
