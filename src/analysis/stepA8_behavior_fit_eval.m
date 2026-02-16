function stepA8_behavior_fit_eval(run_id, varargin)
% stepA8_behavior_fit_eval Fit behavioral choice models on TRAIN and evaluate on VALID.
%
% This step fits simple behavioral mappings from trust-related decision variables
% to the probability of following the robot recommendation, using door-level
% datasets produced by Step A7. The fitted models are then evaluated on the
% held-out VALID set, with metrics reported using "override" as the positive class.
%
% Behavioral model variants (follow probability):
%   Model 0 (direct trust-as-probability; no fitting):
%       p_follow = clamp(tau_decision, 0, 1)
%
%   Model 1 (baseline threshold / logistic on margin):
%       p_follow = sigmoid( k * (tau_decision - self_confidence) )
%
%   Model 2 (offset + lapse):
%       z        = k * tau_decision + beta * (self_confidence - 0.5)
%       p*_follow= sigmoid(z)
%       p_follow = (1 - eps) * p*_follow + eps * 0.5
%
% IMPORTANT: "override" (i.e., NOT follow) is treated as the POSITIVE class for
% override-focused metrics. Probabilities for override are:
%       p_override = 1 - p_follow
%
% INPUTS (required)
%   run_id (string|char)
%       Analysis run identifier. Used to locate A7 artifacts under:
%         derived/analysis_runs/<run_id>/stepA7_behavior_dataset/
%
% NAME-VALUE ARGUMENTS (optional)
%   "OutDir"        (string|char)  Output directory. Default: derived/.../stepA8_behavior_fit_eval
%   "Overwrite"     (logical)      If false, error if outputs already exist. Default: false
%   "NBins"         (scalar)       Number of calibration bins. Default: 10
%   "BootstrapN"    (scalar)       Number of participant-level bootstrap resamples. Default: 2000
%   "DoAUC"         (logical)      Compute ROC-AUC for follow classification. Default: true
%   "DoPRAUC"       (logical)      Compute PR-AUC for override (positive) if feasible. Default: true
%   "ProfileGridK"  (vector)       Grid for profile likelihood over k. Default: linspace(0,40,401)
%   "ProfileGridBeta"(vector)      Grid for profile likelihood over beta. Default: linspace(-10,10,401)
%   "ProfileGridEps"(vector)       Grid for profile likelihood over eps. Default: linspace(0,0.5,251)
%   "RandomSeed"    (scalar)       RNG seed for reproducibility. Default: 1
%
% OUTPUTS
%   (none)
%       Artifacts are written to:
%         derived/analysis_runs/<run_id>/stepA8_behavior_fit_eval/
%       including:
%         - fit_params.mat              Fitted parameters, profile grids, bootstrap samples
%         - valid_metrics.csv/.mat      Pooled metrics (including override-focused)
%         - valid_metrics_delta.csv     Metric deltas vs baselines on VALID
%         - calibration_bins_valid.csv  Calibration curves (follow probability)
%         - A8_plot_data.mat            Plot data bundle for downstream reporting
%         - figures/*.pdf (+ .fig)      Figures exported via thesisExport
%
% ASSUMPTIONS / DEPENDENCIES
%   - Step A7 has produced behavior_dataset_train.mat and behavior_dataset_valid.mat
%     containing a table T with the required variables listed below.
%   - Utility functions are available on the MATLAB path:
%       must_exist_file, ensure_dir, save_json
%   - Plot/format helpers are available on the MATLAB path:
%       thesisStyle, thesisFinalizeFigure, thesisExport
%       behavioralDisplayName
%
% NOTES
%   - "baseline_global" uses TRAIN follow rate (mean(y_train)) as constant p_follow on VALID.
%   - "random_guesser_50_50" uses p_follow = 0.5 on VALID (uninformed predictor).
%   - Delta tables include deltas vs both baselines.

    % ------------------------------------------------------------------
    % Parse and validate inputs
    % ------------------------------------------------------------------
    if nargin < 1 || isempty(run_id)
        error("stepA8_behavior_fit_eval: run_id is required.");
    end
    run_id = string(run_id);

    % Thesis plotting defaults (sets global graphics defaults)
    S = thesisStyle(); %#ok<NASGU> % keep S available for local plotters

    p = inputParser;
    p.addParameter("OutDir", "", @(s) isstring(s) || ischar(s));
    p.addParameter("Overwrite", false, @(x) islogical(x) && isscalar(x));
    p.addParameter("NBins", 10, @(x) isnumeric(x) && isscalar(x) && x>=2);
    p.addParameter("BootstrapN", 2000, @(x) isnumeric(x) && isscalar(x) && x>=0);
    p.addParameter("DoAUC", true, @(x) islogical(x) && isscalar(x));
    p.addParameter("DoPRAUC", true, @(x) islogical(x) && isscalar(x)); % PR-AUC for override
    p.addParameter("ProfileGridK", linspace(0, 40, 401), @(x) isnumeric(x) && isvector(x));
    p.addParameter("ProfileGridBeta", linspace(-10, 10, 401), @(x) isnumeric(x) && isvector(x));
    p.addParameter("ProfileGridEps", linspace(0, 0.5, 251), @(x) isnumeric(x) && isvector(x));
    p.addParameter("RandomSeed", 1, @(x) isnumeric(x) && isscalar(x));
    p.parse(varargin{:});
    args = p.Results;

    rng(args.RandomSeed);

    % ------------------------------------------------------------------
    % Load Step A7 door-level datasets (TRAIN / VALID)
    % ------------------------------------------------------------------
    a7Dir = fullfile("derived","analysis_runs",run_id,"stepA7_behavior_dataset");
    trainMat = fullfile(a7Dir, "behavior_dataset_train.mat");
    validMat = fullfile(a7Dir, "behavior_dataset_valid.mat");
    must_exist_file(trainMat, "A7 TRAIN dataset");
    must_exist_file(validMat, "A7 VALID dataset");

    S_tr = load(trainMat, "T");
    S_va = load(validMat, "T");
    if ~isfield(S_tr,"T") || ~istable(S_tr.T), error("[A8] TRAIN mat missing table T."); end
    if ~isfield(S_va,"T") || ~istable(S_va.T), error("[A8] VALID mat missing table T."); end
    Ttr = S_tr.T;
    Tva = S_va.T;

    % Keep only rows with usable ground-truth choice labels
    Ttr = Ttr(Ttr.is_valid_label==1, :);
    Tva = Tva(Tva.is_valid_label==1, :);

    % Verify required variables exist (dataset contract from Step A7)
    reqCols = ["participant_id","tau_decision","self_confidence","sc_centered", ...
               "margin_treshold","followed","block_index","door_index"];
    assert(all(ismember(reqCols, string(Ttr.Properties.VariableNames))), "[A8] TRAIN missing required columns.");
    assert(all(ismember(reqCols, string(Tva.Properties.VariableNames))), "[A8] VALID missing required columns.");

    % Remove rows with non-finite predictors (defensive data cleaning)
    Ttr = Ttr(isfinite(Ttr.tau_decision) & isfinite(Ttr.self_confidence) & ...
              isfinite(Ttr.sc_centered) & isfinite(Ttr.margin_treshold), :);
    Tva = Tva(isfinite(Tva.tau_decision) & isfinite(Tva.self_confidence) & ...
              isfinite(Tva.sc_centered) & isfinite(Tva.margin_treshold), :);

    % ------------------------------------------------------------------
    % Output directory layout and overwrite policy
    % ------------------------------------------------------------------
    outDir = string(args.OutDir);
    if strlength(outDir)==0
        outDir = fullfile("derived","analysis_runs",run_id,"stepA8_behavior_fit_eval");
    end
    ensure_dir(outDir);

    figDir = fullfile(outDir, "figures");
    ensure_dir(figDir);

    fitMat     = fullfile(outDir, "fit_params.mat");
    plotMat    = fullfile(outDir, "A8_plot_data.mat");
    metricsCsv = fullfile(outDir, "valid_metrics.csv");
    metricsDeltaCsv = fullfile(outDir, "valid_metrics_delta.csv");
    calibCsv   = fullfile(outDir, "calibration_bins_valid.csv");
    metaMat    = fullfile(outDir, "meta.mat");
    metaJson   = fullfile(outDir, "meta.json");

    if ~args.Overwrite
        if isfile(fitMat) || isfile(metricsCsv) || isfile(plotMat)
            error("[A8] Outputs exist. Set Overwrite=true to replace. (%s)", outDir);
        end
    end

    % ------------------------------------------------------------------
    % Extract arrays used for model fitting and evaluation
    % ------------------------------------------------------------------
    % Binary label: 1=follow, 0=override
    ytr = double(Ttr.followed(:));
    yva = double(Tva.followed(:));

    % Override label (positive class for override-focused metrics): 1=override
    ytr_ov = 1 - ytr;
    yva_ov = 1 - yva;

    % Margin used by the threshold model(s) as provided by A7:
    % margin_treshold = tau_decision - self_confidence
    m_tr = double(Ttr.margin_treshold(:));
    m_va = double(Tva.margin_treshold(:));

    tau_tr = double(Ttr.tau_decision(:));
    tau_va = double(Tva.tau_decision(:));

    % Centered self-confidence: sc_centered = self_confidence - 0.5
    scC_tr = double(Ttr.sc_centered(:));
    scC_va = double(Tva.sc_centered(:));

    pid_tr = string(Ttr.participant_id); %#ok<NASGU> % kept for participant-level bootstrap resampling
    pid_va = string(Tva.participant_id);

    % ------------------------------------------------------------------
    % Model 0: direct mapping p_follow = tau_decision (no fitting)
    % ------------------------------------------------------------------
    p0_va_follow = clamp01(tau_va);
    p0_va_ov     = 1 - p0_va_follow;

    % ------------------------------------------------------------------
    % Fit Model 1: single-parameter k (MLE)
    % ------------------------------------------------------------------
    % Parameterization: k = exp(z) to enforce k >= 0 during optimization.
    f1 = @(z) nll_model1(exp(z), m_tr, ytr);
    z0_1 = log(10);
    zhat1 = fminsearch(f1, z0_1, optimset('Display','off'));
    k1_hat = exp(zhat1);

    % Profile likelihood over k for diagnostics and reporting
    kGrid = args.ProfileGridK(:);
    nll1_grid = NaN(numel(kGrid),1);
    for i = 1:numel(kGrid)
        nll1_grid(i) = nll_model1(kGrid(i), m_tr, ytr);
    end

    % ------------------------------------------------------------------
    % Fit Model 2: (k, beta, eps) (MLE)
    % ------------------------------------------------------------------
    % k constrained to be nonnegative via exp(z1)
    % eps constrained to [0, epsMax] via epsMax*sigmoid(z3)
    epsMax = 0.5;
    f2 = @(z) nll_model2(exp(z(1)), z(2), epsMax*sigmoid(z(3)), tau_tr, scC_tr, ytr);

    z0_2 = [log(10); 0; logit(0.05/epsMax)];
    zhat2 = fminsearch(f2, z0_2, optimset('Display','off'));
    k2_hat    = exp(zhat2(1));
    beta_hat  = zhat2(2);
    eps_hat   = epsMax*sigmoid(zhat2(3));

    % Profile likelihood grids for Model 2 parameters (1D slices)
    betaGrid = args.ProfileGridBeta(:);
    epsGrid  = args.ProfileGridEps(:);

    nll2_kgrid = NaN(numel(kGrid),1);
    for i = 1:numel(kGrid)
        nll2_kgrid(i) = nll_model2(kGrid(i), beta_hat, eps_hat, tau_tr, scC_tr, ytr);
    end

    nll2_betagrid = NaN(numel(betaGrid),1);
    for i = 1:numel(betaGrid)
        nll2_betagrid(i) = nll_model2(k2_hat, betaGrid(i), eps_hat, tau_tr, scC_tr, ytr);
    end

    nll2_epsgrid = NaN(numel(epsGrid),1);
    for i = 1:numel(epsGrid)
        nll2_epsgrid(i) = nll_model2(k2_hat, beta_hat, epsGrid(i), tau_tr, scC_tr, ytr);
    end

    % ------------------------------------------------------------------
    % Bootstrap uncertainty (participant-level resampling on TRAIN)
    % ------------------------------------------------------------------
    B = args.BootstrapN;
    boot = struct();
    boot.B = B;

    if B > 0
        uniqP = unique(string(Ttr.participant_id));
        nP = numel(uniqP);

        boot.k1 = NaN(B,1);
        boot.k2 = NaN(B,1);
        boot.beta = NaN(B,1);
        boot.eps = NaN(B,1);

        for b = 1:B
            % Resample participants with replacement, then include all their trials.
            sampP = uniqP(randi(nP, nP, 1));

            % Build row indices with replication (preserves bootstrap weighting)
            idx = [];
            for sp = 1:numel(sampP)
                idx_sp = find(string(Ttr.participant_id) == sampP(sp));
                idx = [idx; idx_sp]; %#ok<AGROW>
            end

            yb = ytr(idx);
            mb = m_tr(idx);
            taub = tau_tr(idx);
            scCb = scC_tr(idx);

            % Fit Model 1 on bootstrap sample
            f1b = @(z) nll_model1(exp(z), mb, yb);
            zb = fminsearch(f1b, z0_1, optimset('Display','off'));
            boot.k1(b) = exp(zb);

            % Fit Model 2 on bootstrap sample
            f2b = @(z) nll_model2(exp(z(1)), z(2), epsMax*sigmoid(z(3)), taub, scCb, yb);
            zb2 = fminsearch(f2b, z0_2, optimset('Display','off'));
            boot.k2(b)   = exp(zb2(1));
            boot.beta(b) = zb2(2);
            boot.eps(b)  = epsMax*sigmoid(zb2(3));
        end

        % Simple percentile intervals (reported for quick uncertainty summaries)
        boot.ci95_k1   = prctile(boot.k1,   [2.5 97.5]);
        boot.ci95_k2   = prctile(boot.k2,   [2.5 97.5]);
        boot.ci95_beta = prctile(boot.beta, [2.5 97.5]);
        boot.ci95_eps  = prctile(boot.eps,  [2.5 97.5]);
    end

    % ------------------------------------------------------------------
    % VALID predictions for each model
    % ------------------------------------------------------------------
    % Model 1: logistic on margin
    p1_va_follow = sigmoid(k1_hat .* m_va);
    p1_va_ov     = 1 - p1_va_follow;

    % Model 2: logistic on linear predictor z with lapse mixing
    z2_va = k2_hat.*tau_va + beta_hat.*scC_va;
    p2_va_follow = (1-eps_hat).*sigmoid(z2_va) + eps_hat.*0.5;
    p2_va_follow = clamp01(p2_va_follow);
    p2_va_ov     = 1 - p2_va_follow;

    % Baseline predictors on VALID
    p_base_global_follow = mean(ytr); % TRAIN follow-rate baseline
    p_base_global_va = p_base_global_follow * ones(size(yva));
    p_base_global_ov = 1 - p_base_global_va;

    % Random guesser baseline (uninformed)
    p_rand_follow_va = 0.5 * ones(size(yva));
    p_rand_ov_va     = 0.5 * ones(size(yva));

    % ------------------------------------------------------------------
    % VALID pooled metrics (override-focused metrics included)
    % ------------------------------------------------------------------
    rows = {
        "random_guesser_50_50", metrics_all_with_override(yva, yva_ov, p_rand_follow_va, p_rand_ov_va, args.DoAUC, args.DoPRAUC);
        "baseline_global", metrics_all_with_override(yva, yva_ov, p_base_global_va, p_base_global_ov, args.DoAUC, args.DoPRAUC);
        "model0_trust_as_probability", metrics_all_with_override(yva, yva_ov, p0_va_follow, p0_va_ov, args.DoAUC, args.DoPRAUC);
        "model1_threshold", metrics_all_with_override(yva, yva_ov, p1_va_follow, p1_va_ov, args.DoAUC, args.DoPRAUC);
        "model2_offset_lapse", metrics_all_with_override(yva, yva_ov, p2_va_follow, p2_va_ov, args.DoAUC, args.DoPRAUC);
    };

    validOverall = cell2table(rows, 'VariableNames', {'method','metrics'});
    validOverall = unpack_metrics(validOverall);

    % Participant-level metrics on VALID (same methods, per participant_id)
    validByP = metrics_by_participant(pid_va, yva, yva_ov, ...
        p0_va_follow, p0_va_ov, ...
        p1_va_follow, p1_va_ov, ...
        p2_va_follow, p2_va_ov, ...
        p_base_global_va, p_base_global_ov, ...
        p_rand_follow_va, p_rand_ov_va, ...
        args.DoAUC, args.DoPRAUC);

    % Calibration bins on VALID (follow-probability calibration)
    calib0 = calibration_bins(yva, p0_va_follow, args.NBins);
    calib0.method = repmat("model0_trust_as_probability", height(calib0), 1);

    calib1 = calibration_bins(yva, p1_va_follow, args.NBins);
    calib1.method = repmat("model1_threshold", height(calib1), 1);

    calib2 = calibration_bins(yva, p2_va_follow, args.NBins);
    calib2.method = repmat("model2_offset_lapse", height(calib2), 1);

    calib = [calib0; calib1; calib2];

    writetable(validOverall, metricsCsv);
    writetable(calib, calibCsv);

    % ------------------------------------------------------------------
    % Metric deltas vs baselines (VALID)
    % ------------------------------------------------------------------
    validDelta = make_delta_table_multi(validOverall, ["baseline_global","random_guesser_50_50"]);
    writetable(validDelta, metricsDeltaCsv);

    % ------------------------------------------------------------------
    % Save MAT bundles and provenance metadata
    % ------------------------------------------------------------------
    fit = struct();

    fit.model0 = struct();
    fit.model0.description = "p_follow = tau_decision (clamped to [0,1])";

    fit.model1.k_hat = k1_hat;
    fit.model1.k_grid = kGrid;
    fit.model1.nll_grid = nll1_grid;

    fit.model2.k_hat = k2_hat;
    fit.model2.beta_hat = beta_hat;
    fit.model2.eps_hat = eps_hat;
    fit.model2.k_grid = kGrid;
    fit.model2.nll_kgrid = nll2_kgrid;
    fit.model2.beta_grid = betaGrid;
    fit.model2.nll_betagrid = nll2_betagrid;
    fit.model2.eps_grid = epsGrid;
    fit.model2.nll_epsgrid = nll2_epsgrid;

    fit.bootstrap = boot;

    save(fitMat, "fit", "validOverall", "validDelta", "validByP", "calib", "-v7.3");

    % Single plot-data bundle used by downstream steps for reporting
    A8plot = struct();
    A8plot.run_id = char(run_id);
    A8plot.fit = fit;
    A8plot.validOverall = validOverall;
    A8plot.validDelta = validDelta;
    A8plot.validByParticipant = validByP;
    A8plot.calibration = calib;

    A8plot.pred_valid = struct( ...
        "y_follow",yva, ...
        "y_override",yva_ov, ...
        "p_random_follow",p_rand_follow_va, ...
        "p_random_override",p_rand_ov_va, ...
        "p_model0_follow",p0_va_follow, ...
        "p_model0_override",p0_va_ov, ...
        "p_model1_follow",p1_va_follow, ...
        "p_model1_override",p1_va_ov, ...
        "p_model2_follow",p2_va_follow, ...
        "p_model2_override",p2_va_ov, ...
        "z_model2", z2_va, ...
        "p_base_global_follow",p_base_global_va, ...
        "p_base_global_override",p_base_global_ov, ...
        "pid",pid_va, ...
        "block", double(Tva.block_index(:)), ...
        "door", double(Tva.door_index(:)), ...
        "tau", tau_va, ...
        "sc_centered", scC_va, ...
        "margin_treshold", m_va);

    save(plotMat, "A8plot", "-v7.3");

    meta = struct();
    meta.run_id = char(run_id);
    meta.created = char(datetime('now','Format','yyyy-MM-dd HH:mm:ss'));
    meta.train_rows = height(Ttr);
    meta.valid_rows = height(Tva);
    meta.bootstrapN = B;
    meta.nbins = args.NBins;
    meta.random_seed = args.RandomSeed;
    meta.override_positive = true;
    meta.do_auc = args.DoAUC;
    meta.do_prauc = args.DoPRAUC;
    meta.dropped_participant_trainrate_baseline = true;
    save(metaMat, "meta");
    save_json(metaJson, meta);

    % ------------------------------------------------------------------
    % Figures (exported via thesisExport)
    % ------------------------------------------------------------------
    % Profile likelihoods (TRAIN)
    make_fig_profile_k(fullfile(figDir, "train_profile_model1_k"), kGrid, nll1_grid, k1_hat, ...
        "Thresholded reliance: profile NLL(k)", thesisStyle());
    make_fig_profile_k(fullfile(figDir, "train_profile_model2_k"), kGrid, nll2_kgrid, k2_hat, ...
        "Threshold + modulation + lapse: profile NLL(k)", thesisStyle());
    make_fig_profile_1d(fullfile(figDir, "train_profile_model2_beta"), betaGrid, nll2_betagrid, beta_hat, ...
        "Threshold + modulation + lapse: profile NLL(beta)", thesisStyle());
    make_fig_profile_1d(fullfile(figDir, "train_profile_model2_eps"), epsGrid, nll2_epsgrid, eps_hat, ...
        "Threshold + modulation + lapse: profile NLL(eps)", thesisStyle());

    % Calibration curves (VALID) for follow probability
    make_fig_calibration(fullfile(figDir, "valid_calibration_model0"), calib0, ...
        "Trust-as-probability calibration", thesisStyle());
    make_fig_calibration(fullfile(figDir, "valid_calibration_model1"), calib1, ...
        "Thresholded reliance calibration", thesisStyle());
    make_fig_calibration(fullfile(figDir, "valid_calibration_model2"), calib2, ...
        "Threshold + modulation + lapse calibration", thesisStyle());

    % Predicted probability vs predictor (VALID)
    make_fig_prob_vs_x(fullfile(figDir, "valid_prob_vs_tau_model0"), tau_va, p0_va_follow, yva, ...
        "Trust-as-probability", "$\tau_{\mathrm{decision}}$", thesisStyle());

    make_fig_prob_vs_x(fullfile(figDir, "valid_prob_vs_margin_model1"), m_va, p1_va_follow, yva, ...
        "Thresholded reliance", "$\tau-\beta$", thesisStyle());

    make_fig_prob_vs_x(fullfile(figDir, "valid_prob_vs_margin_model2"), m_va, p2_va_follow, yva, ...
        "Threshold + modulation + lapse", "$\tau-\beta$", thesisStyle());

    % Model 2 diagnostic: p_follow vs linear predictor z
    make_fig_prob_vs_x(fullfile(figDir, "valid_prob_vs_z_model2"), z2_va, p2_va_follow, yva, ...
        "Threshold + modulation + lapse", "$z = k\tau + \alpha*\left(\beta-0.5\right)$", thesisStyle());

    % PR curve for override (positive class) for Model 0/1/2
    if args.DoPRAUC
        make_fig_pr_curve(fullfile(figDir, "valid_prcurve_override_models012"), ...
            yva_ov, ...
            {p0_va_ov, p1_va_ov, p2_va_ov}, ...
            ["trust-as-probability","thresholded reliance","threshold + modulation + lapse"], ...
            thesisStyle());
    end

    % Class-conditional NLL plot (follow vs override)
    make_fig_class_nll(fullfile(figDir, "valid_nll_by_class_models012"), validOverall, thesisStyle());

    % Bootstrap distributions (TRAIN)
    if B > 0
        make_fig_bootstrap_hists(fullfile(figDir, "train_bootstrap_params_model1"), ...
            boot.k1, "Bootstrap thresholded reliance (k)", "$k$", thesisStyle());
        make_fig_bootstrap_hists(fullfile(figDir, "train_bootstrap_params_model2"), ...
            [boot.k2, boot.beta, boot.eps], "Bootstrap threshold + modulation + lapse ($k$, $\beta$, $\varepsilon$)", ...
            ["$k$","$\beta$","$\varepsilon$"], thesisStyle());
    end

    fprintf("[Step A8] Done.\n");
    fprintf("  Model0: p_follow=tau (no fit)\n");
    fprintf("  Model1: k=%.4g\n", k1_hat);
    fprintf("  Model2: k=%.4g, beta=%.4g, eps=%.4g\n", k2_hat, beta_hat, eps_hat);
    fprintf("  Output dir: %s\n", outDir);

    % Optional console summaries (non-fatal)
    try
        show_delta_summary_multi(validDelta, "baseline_global", "random_guesser_50_50");
        show_override_quick_summary(validOverall);
    catch
        % no-op
    end
end

% ======================================================================
% Helpers: negative log-likelihood (NLL)
% ======================================================================

function nll = nll_model1(k, margin, y_follow)
% nll_model1 NLL for Model 1: sigmoid(k * margin) with k >= 0.
    if ~isfinite(k) || k < 0, nll = Inf; return; end
    p = sigmoid(k .* margin);
    nll = bernoulli_nll(y_follow, p);
end

function nll = nll_model2(k, beta, eps, tau, scC, y_follow)
% nll_model2 NLL for Model 2 with lapse mixing and centered self-confidence.
    if ~isfinite(k) || k < 0 || ~isfinite(beta) || ~isfinite(eps) || eps < 0 || eps > 1
        nll = Inf; return;
    end
    z = k .* tau + beta .* scC;
    pstar = sigmoid(z);
    p = (1-eps).*pstar + eps.*0.5;
    nll = bernoulli_nll(y_follow, p);
end

function nll = bernoulli_nll(y, p)
% bernoulli_nll Stable Bernoulli negative log-likelihood.
    p = min(max(p, 1e-12), 1-1e-12);
    nll = -sum(y .* log(p) + (1-y).*log(1-p));
end

function p = clamp01(x)
% clamp01 Clamp values elementwise to [0, 1].
    p = min(max(x, 0), 1);
end

% ======================================================================
% Helpers: metrics (pooled)
% ======================================================================

function M = metrics_all_with_override(y_follow, y_override, p_follow, p_override, doAUC, doPRAUC)
% metrics_all_with_override Compute pooled metrics, including override-positive metrics.
%
% Inputs:
%   y_follow    - binary follow label (1=follow)
%   y_override  - binary override label (1=override), typically 1-y_follow
%   p_follow    - predicted probability of follow
%   p_override  - predicted probability of override (=1-p_follow)
%
% Notes:
%   - Threshold for hard decisions is 0.5 on the corresponding probability.
%   - PR-AUC is computed on the override task when requested and both classes exist.

    p_follow   = min(max(p_follow, 1e-12), 1-1e-12);
    p_override = min(max(p_override, 1e-12), 1-1e-12);

    n = numel(y_follow);

    nll_total = -sum(y_follow .* log(p_follow) + (1-y_follow).*log(1-p_follow));
    brier = mean((p_follow - y_follow).^2);

    acc_follow = mean((p_follow >= 0.5) == (y_follow==1));

    % Override-positive hard predictions use p_override thresholding.
    pred_ov = (p_override >= 0.5);
    [prec_ov, rec_ov, f1_ov, bal_acc, acc_ovpos] = override_metrics_from_predictions(y_override, pred_ov);

    % Class-conditional decompositions for diagnostics
    [nll_mean_follow, nll_mean_override] = nll_by_class(y_follow, p_follow);
    [brier_follow, brier_override] = brier_by_class(y_follow, p_follow);

    out = struct();
    out.N = n;

    out.NLL_total = nll_total;
    out.NLL_mean  = nll_total / max(1,n);
    out.Brier     = brier;
    out.Acc       = acc_follow;

    if doAUC
        out.AUC = auc_roc(y_follow, p_follow);
    else
        out.AUC = NaN;
    end

    out.OverrideRate = mean(y_override==1);
    out.OverridePrecision = prec_ov;
    out.OverrideRecall = rec_ov;
    out.OverrideF1 = f1_ov;
    out.BalancedAcc = bal_acc;
    out.OverrideAcc_at_0_5 = acc_ovpos;

    if doPRAUC
        out.OverridePRAUC = pr_auc(y_override, p_override);
    else
        out.OverridePRAUC = NaN;
    end

    out.NLL_mean_follow = nll_mean_follow;
    out.NLL_mean_override = nll_mean_override;
    out.Brier_follow = brier_follow;
    out.Brier_override = brier_override;

    M = out;
end

function [prec_ov, rec_ov, f1_ov, bal_acc, acc_ovpos] = override_metrics_from_predictions(y_override, pred_override)
% override_metrics_from_predictions Compute override-positive precision/recall/F1 and balanced accuracy.
    y = (y_override==1);
    p = (pred_override==1);

    TP = sum(p & y);
    FP = sum(p & ~y);
    FN = sum(~p & y);
    TN = sum(~p & ~y);

    prec_ov = TP / max(1, (TP+FP));
    rec_ov  = TP / max(1, (TP+FN));
    f1_ov   = (2*prec_ov*rec_ov) / max(1e-12, (prec_ov+rec_ov));

    % Follow recall is recall of the negative class under override-positive framing
    rec_follow = TN / max(1, (TN+FP));

    bal_acc = 0.5*(rec_ov + rec_follow);

    acc_ovpos = (TP + TN) / max(1, (TP+FP+FN+TN));
end

function [nll_f, nll_o] = nll_by_class(y_follow, p_follow)
% nll_by_class Class-conditional mean NLL for follow and override classes.
    p_follow = min(max(p_follow, 1e-12), 1-1e-12);
    maskF = (y_follow==1);
    maskO = (y_follow==0);

    if any(maskF)
        nll_f = mean(-log(p_follow(maskF)));
    else
        nll_f = NaN;
    end
    if any(maskO)
        nll_o = mean(-log(1 - p_follow(maskO)));
    else
        nll_o = NaN;
    end
end

function [b_f, b_o] = brier_by_class(y_follow, p_follow)
% brier_by_class Class-conditional mean Brier score contributions.
    maskF = (y_follow==1);
    maskO = (y_follow==0);
    if any(maskF)
        b_f = mean((p_follow(maskF) - 1).^2);
    else
        b_f = NaN;
    end
    if any(maskO)
        b_o = mean((p_follow(maskO) - 0).^2);
    else
        b_o = NaN;
    end
end

% ======================================================================
% Helpers: unpack metrics structs into a flat table
% ======================================================================

function T = unpack_metrics(Tin)
% unpack_metrics Expand a table column of metric structs into scalar columns.
    methods = Tin.method;
    S = Tin.metrics;

    if iscell(S)
        s1 = S{1};
        getField = @(r,f) S{r}.(f);
    elseif isstruct(S)
        s1 = S(1);
        getField = @(r,f) S(r).(f);
    else
        error("unpack_metrics: Unexpected type for Tin.metrics: %s", class(S));
    end

    fn = fieldnames(s1);

    T = table();
    T.method = methods;

    for i = 1:numel(fn)
        f = fn{i};
        vals = NaN(height(Tin),1);
        for r = 1:height(Tin)
            vals(r) = getField(r,f);
        end
        T.(f) = vals;
    end
end

% ======================================================================
% Helpers: participant-level metrics (VALID)
% ======================================================================

function Tp = metrics_by_participant(pid, y_follow, y_override, ...
    p0_follow, p0_ov, ...
    p1_follow, p1_ov, ...
    p2_follow, p2_ov, ...
    pb_follow, pb_ov, ...
    pr_follow, pr_ov, ...
    doAUC, doPRAUC)
% metrics_by_participant Compute per-participant metrics for all methods on VALID.
%
% Notes:
%   - This function intentionally does not implement a participant-specific
%     train-rate baseline; only global and random baselines are included.

    uniqP = unique(pid);
    Tp = table();
    Tp.participant_id = uniqP;
    Tp.N = zeros(numel(uniqP),1);

    cols = [ ...
        "OverrideRate", ...
        "NLL_mean_random","Brier_random","Acc_random","AUC_random","OverrideRecall_random","OverridePrecision_random","OverrideF1_random","BalancedAcc_random","OverridePRAUC_random","NLL_mean_override_random", ...
        "NLL_mean_base_global","Brier_base_global","Acc_base_global","AUC_base_global","OverrideRecall_base_global","OverridePrecision_base_global","OverrideF1_base_global","BalancedAcc_base_global","OverridePRAUC_base_global","NLL_mean_override_base_global", ...
        "NLL_mean_model0","Brier_model0","Acc_model0","AUC_model0","OverrideRecall_model0","OverridePrecision_model0","OverrideF1_model0","BalancedAcc_model0","OverridePRAUC_model0","NLL_mean_override_model0", ...
        "NLL_mean_model1","Brier_model1","Acc_model1","AUC_model1","OverrideRecall_model1","OverridePrecision_model1","OverrideF1_model1","BalancedAcc_model1","OverridePRAUC_model1","NLL_mean_override_model1", ...
        "NLL_mean_model2","Brier_model2","Acc_model2","AUC_model2","OverrideRecall_model2","OverridePrecision_model2","OverrideF1_model2","BalancedAcc_model2","OverridePRAUC_model2","NLL_mean_override_model2" ...
    ];

    for c = 1:numel(cols)
        Tp.(cols(c)) = NaN(numel(uniqP),1);
    end

    for i = 1:numel(uniqP)
        mask = (pid==uniqP(i));
        Tp.N(i) = sum(mask);

        Tp.OverrideRate(i) = mean(y_override(mask)==1);

        mr  = metrics_all_with_override(y_follow(mask), y_override(mask), pr_follow(mask), pr_ov(mask), doAUC, doPRAUC);
        mbg = metrics_all_with_override(y_follow(mask), y_override(mask), pb_follow(mask), pb_ov(mask), doAUC, doPRAUC);

        mm0 = metrics_all_with_override(y_follow(mask), y_override(mask), p0_follow(mask), p0_ov(mask), doAUC, doPRAUC);
        mm1 = metrics_all_with_override(y_follow(mask), y_override(mask), p1_follow(mask), p1_ov(mask), doAUC, doPRAUC);
        mm2 = metrics_all_with_override(y_follow(mask), y_override(mask), p2_follow(mask), p2_ov(mask), doAUC, doPRAUC);

        Tp.NLL_mean_random(i) = mr.NLL_mean;
        Tp.Brier_random(i)    = mr.Brier;
        Tp.Acc_random(i)      = mr.Acc;
        Tp.AUC_random(i)      = mr.AUC;
        Tp.OverrideRecall_random(i)    = mr.OverrideRecall;
        Tp.OverridePrecision_random(i) = mr.OverridePrecision;
        Tp.OverrideF1_random(i)        = mr.OverrideF1;
        Tp.BalancedAcc_random(i)       = mr.BalancedAcc;
        Tp.OverridePRAUC_random(i)      = mr.OverridePRAUC;
        Tp.NLL_mean_override_random(i)  = mr.NLL_mean_override;

        Tp.NLL_mean_base_global(i) = mbg.NLL_mean;
        Tp.Brier_base_global(i)    = mbg.Brier;
        Tp.Acc_base_global(i)      = mbg.Acc;
        Tp.AUC_base_global(i)      = mbg.AUC;
        Tp.OverrideRecall_base_global(i)    = mbg.OverrideRecall;
        Tp.OverridePrecision_base_global(i) = mbg.OverridePrecision;
        Tp.OverrideF1_base_global(i)        = mbg.OverrideF1;
        Tp.BalancedAcc_base_global(i)       = mbg.BalancedAcc;
        Tp.OverridePRAUC_base_global(i)      = mbg.OverridePRAUC;
        Tp.NLL_mean_override_base_global(i)  = mbg.NLL_mean_override;

        Tp.NLL_mean_model0(i) = mm0.NLL_mean;
        Tp.Brier_model0(i)    = mm0.Brier;
        Tp.Acc_model0(i)      = mm0.Acc;
        Tp.AUC_model0(i)      = mm0.AUC;
        Tp.OverrideRecall_model0(i)    = mm0.OverrideRecall;
        Tp.OverridePrecision_model0(i) = mm0.OverridePrecision;
        Tp.OverrideF1_model0(i)        = mm0.OverrideF1;
        Tp.BalancedAcc_model0(i)       = mm0.BalancedAcc;
        Tp.OverridePRAUC_model0(i)      = mm0.OverridePRAUC;
        Tp.NLL_mean_override_model0(i)  = mm0.NLL_mean_override;

        Tp.NLL_mean_model1(i) = mm1.NLL_mean;
        Tp.Brier_model1(i)    = mm1.Brier;
        Tp.Acc_model1(i)      = mm1.Acc;
        Tp.AUC_model1(i)      = mm1.AUC;
        Tp.OverrideRecall_model1(i)    = mm1.OverrideRecall;
        Tp.OverridePrecision_model1(i) = mm1.OverridePrecision;
        Tp.OverrideF1_model1(i)        = mm1.OverrideF1;
        Tp.BalancedAcc_model1(i)       = mm1.BalancedAcc;
        Tp.OverridePRAUC_model1(i)      = mm1.OverridePRAUC;
        Tp.NLL_mean_override_model1(i)  = mm1.NLL_mean_override;

        Tp.NLL_mean_model2(i) = mm2.NLL_mean;
        Tp.Brier_model2(i)    = mm2.Brier;
        Tp.Acc_model2(i)      = mm2.Acc;
        Tp.AUC_model2(i)      = mm2.AUC;
        Tp.OverrideRecall_model2(i)    = mm2.OverrideRecall;
        Tp.OverridePrecision_model2(i) = mm2.OverridePrecision;
        Tp.OverrideF1_model2(i)        = mm2.OverrideF1;
        Tp.BalancedAcc_model2(i)       = mm2.BalancedAcc;
        Tp.OverridePRAUC_model2(i)      = mm2.OverridePRAUC;
        Tp.NLL_mean_override_model2(i)  = mm2.NLL_mean_override;
    end
end

% ======================================================================
% Helpers: calibration (follow probability)
% ======================================================================

function C = calibration_bins(y, p, nbins)
% calibration_bins Bin-wise calibration summary and expected calibration error (ECE).
    p = min(max(p, 0), 1);

    edges = linspace(0,1,nbins+1);
    bin = discretize(p, edges);
    bin(isnan(bin)) = nbins;

    C = table();
    C.bin = (1:nbins)';
    C.p_lo = edges(1:end-1)';
    C.p_hi = edges(2:end)';

    C.n = zeros(nbins,1);
    C.p_mean = NaN(nbins,1);
    C.y_mean = NaN(nbins,1);
    C.ci_lo = NaN(nbins,1);
    C.ci_hi = NaN(nbins,1);

    for b = 1:nbins
        mask = (bin==b);
        C.n(b) = sum(mask);
        if C.n(b) > 0
            C.p_mean(b) = mean(p(mask));
            C.y_mean(b) = mean(y(mask));
            [lo, hi] = wilson_ci(sum(y(mask)), C.n(b), 0.05);
            C.ci_lo(b) = lo;
            C.ci_hi(b) = hi;
        end
    end

    w = C.n / max(1,sum(C.n));
    C.abs_gap = abs(C.p_mean - C.y_mean);
    ece = nansum(w .* C.abs_gap);
    C.ECE = repmat(ece, nbins, 1);
end

function [lo, hi] = wilson_ci(k, n, alpha)
% wilson_ci Wilson score interval for a binomial proportion.
    if n <= 0
        lo = NaN; hi = NaN; return;
    end
    z = norminv(1 - alpha/2);
    phat = k/n;
    denom = 1 + z^2/n;
    center = (phat + z^2/(2*n)) / denom;
    half = (z/denom) * sqrt((phat*(1-phat) + z^2/(4*n))/n);
    lo = max(0, center - half);
    hi = min(1, center + half);
end

% ======================================================================
% Helpers: ROC-AUC and PR-AUC
% ======================================================================

function a = auc_roc(y, p)
% auc_roc ROC-AUC using rank statistic (Mann-Whitney U).
    y = y(:);
    p = p(:);
    if numel(unique(y)) < 2
        a = NaN; return;
    end
    r = tiedrank(p);
    n1 = sum(y==1);
    n0 = sum(y==0);
    a = (sum(r(y==1)) - n1*(n1+1)/2) / (n1*n0);
end

function ap = pr_auc(y_pos, p_pos)
% pr_auc Area under the precision-recall curve (average precision style).
    y_pos = y_pos(:) == 1;
    p_pos = p_pos(:);

    if numel(unique(y_pos)) < 2
        ap = NaN; return;
    end

    [~, idx] = sort(p_pos, 'descend');
    y_sorted = y_pos(idx);

    tp = cumsum(y_sorted==1);
    fp = cumsum(y_sorted==0);

    prec = tp ./ max(1, (tp + fp));
    rec  = tp ./ max(1, sum(y_sorted==1));

    drec = [rec(1); diff(rec)];
    ap = sum(prec .* drec);
    ap = min(max(ap, 0), 1);
end

% ======================================================================
% Helpers: sigmoid/logit
% ======================================================================

function s = sigmoid(z)
% sigmoid Logistic sigmoid function.
    s = 1 ./ (1 + exp(-z));
end

function z = logit(p)
% logit Inverse sigmoid with numerical clipping.
    p = min(max(p, 1e-12), 1-1e-12);
    z = log(p./(1-p));
end

% ======================================================================
% Delta metrics table helper (multiple baselines)
% ======================================================================

function Td = make_delta_table_multi(T, baselineMethods)
% make_delta_table_multi Add delta columns vs each baseline method.
    if ~ismember("method", string(T.Properties.VariableNames))
        error("make_delta_table_multi: T must include 'method'.");
    end

    Td = T;
    metricNames = ["NLL_total","NLL_mean","Brier","Acc","AUC", ...
                   "OverridePrecision","OverrideRecall","OverrideF1","BalancedAcc","OverridePRAUC", ...
                   "NLL_mean_override","NLL_mean_follow", "Brier_override","Brier_follow", ...
                   "OverrideAcc_at_0_5"];

    baselineMethods = string(baselineMethods(:))';
    for bm = baselineMethods
        baseIdx = find(string(Td.method) == bm, 1);
        if isempty(baseIdx)
            error("make_delta_table_multi: baseline method '%s' not found.", bm);
        end
        base = Td(baseIdx, :);

        for mn = metricNames
            if ismember(mn, string(Td.Properties.VariableNames))
                Td.("d_" + mn + "_vs_" + bm) = Td.(mn) - base.(mn);
            end
        end
    end
end

function show_delta_summary_multi(Td, baselineGlobalName, randomName)
% show_delta_summary_multi Console summary of delta NLL_mean vs the specified baselines.
    keyG = "d_NLL_mean_vs_" + string(baselineGlobalName);
    keyR = "d_NLL_mean_vs_" + string(randomName);

    haveG = ismember(keyG, string(Td.Properties.VariableNames));
    haveR = ismember(keyR, string(Td.Properties.VariableNames));

    rows = ["model0_trust_as_probability","model1_threshold","model2_offset_lapse"];
    labels = ["Model 0","Model 1","Model 2"];

    for i = 1:numel(rows)
        r = Td(string(Td.method)==rows(i), :);
        if isempty(r), continue; end
        if haveG
            fprintf("  Delta NLL_mean vs %s (%s): %+0.4f\n", baselineGlobalName, labels(i), r.(keyG));
        end
        if haveR
            fprintf("  Delta NLL_mean vs %s (%s): %+0.4f\n", randomName, labels(i), r.(keyR));
        end
    end
end

function show_override_quick_summary(validOverall)
% show_override_quick_summary Console summary of key override-positive metrics.
    fprintf("  Override-focused (positive=override): Recall / Precision / F1 / BalancedAcc\n");
    rows = ["random_guesser_50_50","baseline_global","model0_trust_as_probability","model1_threshold","model2_offset_lapse"];
    for i = 1:numel(rows)
        r = validOverall(string(validOverall.method)==rows(i), :);
        if isempty(r), continue; end
        fprintf("    %-28s: R=%.3f  P=%.3f  F1=%.3f  BalAcc=%.3f\n", ...
            rows(i), r.OverrideRecall, r.OverridePrecision, r.OverrideF1, r.BalancedAcc);
    end
end

% ======================================================================
% Plot helpers (thesis-style finalize + export)
% ======================================================================

function local_save_figure(f, outBase, S)
% local_save_figure Finalize and export a figure using thesis styling helpers.
    thesisFinalizeFigure(f, S);
    thesisExport(f, outBase);
end

function make_fig_profile_k(outBase, x, nll, xhat, titleStr, S)
% make_fig_profile_k Plot a 1D profile NLL curve with the MLE marked.
    f = figure('Visible','off','Color','w','Name','Profile NLL');
    thesisStyle(f);

    plot(x, nll);
    hold on;
    yl = ylim;
    plot([xhat xhat], yl, '--');
    grid on;

    xlabel('Parameter', 'FontSize', S.fontSizeLbl);
    ylabel('NLL', 'FontSize', S.fontSizeYlb);
    title(titleStr, 'FontSize', S.fontSizeTit); % uses TeX unless "$" triggers LaTeX

    hold off;
    local_save_figure(f, outBase, S);
end

function make_fig_profile_1d(outBase, x, nll, xhat, titleStr, S)
% make_fig_profile_1d Generic 1D profile plotter (used for beta and eps).
    f = figure('Visible','off','Color','w','Name','Profile NLL');
    thesisStyle(f);

    plot(x, nll);
    hold on;
    yl = ylim;
    plot([xhat xhat], yl, '--');
    grid on;

    xlabel('Parameter', 'FontSize', S.fontSizeLbl);
    ylabel('NLL', 'FontSize', S.fontSizeYlb);
    title(titleStr, 'FontSize', S.fontSizeTit);

    hold off;
    local_save_figure(f, outBase, S);
end

function make_fig_calibration(outBase, C, titleStr, S)
% make_fig_calibration Reliability diagram with Wilson intervals and ECE summary.
    f = figure('Visible','off','Color','w','Name','Calibration');
    thesisStyle(f);

    plot([0 1],[0 1], '--');
    hold on;

    mask = C.n > 0 & isfinite(C.p_mean) & isfinite(C.y_mean);
    x = C.p_mean(mask);
    y = C.y_mean(mask);
    lo = C.ci_lo(mask);
    hi = C.ci_hi(mask);
    errLow = y - lo;
    errHigh = hi - y;
    errorbar(x, y, errLow, errHigh, 'o');

    grid on;
    xlim([0 1]); ylim([0 1]);

    xlabel('Predicted probability (bin mean)', 'FontSize', S.fontSizeLbl);
    ylabel('Empirical follow rate', 'FontSize', S.fontSizeYlb);
    title(sprintf('%s (ECE=%.3f)', titleStr, C.ECE(1)), 'FontSize', S.fontSizeTit);

    hold off;
    local_save_figure(f, outBase, S);
end

function make_fig_prob_vs_x(outBase, x, p, y_follow, titleStr, xlabStr, S)
% make_fig_prob_vs_x Scatter plot of predicted follow probability vs a predictor.
    f = figure('Visible','off','Color','w','Name','Probability vs predictor');
    thesisStyle(f);

    % Downsample for readability in dense datasets (keeps behavior deterministic via RNG seed)
    n = numel(x);
    idx = 1:n;
    if n > 800
        idx = idx(randperm(n, 800));
    end

    scatter(x(idx), p(idx), 12, y_follow(idx), 'filled');
    grid on;

    xlabel(xlabStr, 'FontSize', S.fontSizeLbl);
    ylabel('Predicted $p(\mathrm{follow})$', 'FontSize', S.fontSizeYlb);
    title(titleStr, 'FontSize', S.fontSizeTit);

    local_save_figure(f, outBase, S);
end

function make_fig_bootstrap_hists(outBase, X, titleStr, paramNames, S)
% make_fig_bootstrap_hists Histogram(s) of bootstrap parameter estimates.
    f = figure('Visible','off','Color','w','Name','Bootstrap distributions');
    thesisStyle(f);

    if isvector(X)
        x = X(:);
        x = x(isfinite(x));
        histogram(x, 40);
        grid on;

        xlabel(string(paramNames), 'FontSize', S.fontSizeLbl);
        ylabel('Count', 'FontSize', S.fontSizeYlb);
        title(titleStr, 'FontSize', S.fontSizeTit);

        local_save_figure(f, outBase, S);
        return;
    end

    set(f,'Units','centimeters');
    set(f,'Position',[2 2 S.figSizeTrajectoryGrid]);

    [~,D] = size(X);
    if ischar(paramNames) || isstring(paramNames)
        paramNames = string(paramNames);
    end
    if numel(paramNames) ~= D
        paramNames = "param" + (1:D);
    end

    % Keep subplot structure (minimal change), then finalize/export once.
    for d = 1:D
        subplot(D,1,d);
        xd = X(:,d);
        xd = xd(isfinite(xd));
        histogram(xd, 40);
        grid on;
        xlabel(paramNames(d), 'FontSize', S.fontSizeLbl);
        ylabel('Count', 'FontSize', S.fontSizeYlb);
        if d == 1
            title(titleStr, 'FontSize', S.fontSizeTit);
        end
    end

    local_save_figure(f, outBase, S);
end

function make_fig_pr_curve(outBase, y_pos, ppos_list, label_list, S)
% make_fig_pr_curve Plot precision-recall curves for override-positive prediction.
    f = figure('Visible','off','Color','w','Name','PR curves (override)');
    thesisStyle(f);
    hold on; grid on;

    % Prevalence reference line
    prev = mean(y_pos==1);
    plot([0 1], [prev prev], '--');

    for i = 1:numel(ppos_list)
        p = ppos_list{i}(:);
        y = (y_pos(:)==1);

        if numel(unique(y)) < 2
            continue;
        end

        [~, idx] = sort(p, 'descend');
        y_sorted = y(idx);

        tp = cumsum(y_sorted==1);
        fp = cumsum(y_sorted==0);

        prec = tp ./ max(1, (tp + fp));
        rec  = tp ./ max(1, sum(y_sorted==1));

        plot(rec, prec);
    end

    xlabel('Recall (override)', 'FontSize', S.fontSizeLbl);
    ylabel('Precision (override)', 'FontSize', S.fontSizeYlb);
    title('PR curves (positive = override)', 'FontSize', S.fontSizeTit);

    lg = ["prevalence "] + string(label_list);
    legend(lg, 'Location','best');

    xlim([0 1]); ylim([0 1]);
    hold off;

    local_save_figure(f, outBase, S);
end

function make_fig_class_nll(outBase, validOverall, S)
% make_fig_class_nll Plot mean class-conditional NLL for the fitted models.
    f = figure('Visible','off','Color','w','Name','Class-conditional NLL');
    thesisStyle(f);
    grid on; hold on;

    methods = ["model0_trust_as_probability","model1_threshold","model2_offset_lapse"];
    xs = 1:numel(methods);

    nllF = NaN(size(xs));
    nllO = NaN(size(xs));

    for i = 1:numel(methods)
        r = validOverall(string(validOverall.method)==methods(i), :);
        if isempty(r), continue; end
        nllF(i) = r.NLL_mean_follow;
        nllO(i) = r.NLL_mean_override;
    end

    plot(xs, nllF, 'o-');
    plot(xs, nllO, 's-');

    set(gca,'XTick',xs,'XTickLabel',behavioralDisplayName(methods));
    xtickangle(20);

    ylabel('Mean NLL (lower is better)', 'FontSize', S.fontSizeYlb);
    title('Class-conditional NLL (follow vs override)', 'FontSize', S.fontSizeTit);
    legend(["follow","override"], 'Location','best');

    hold off;
    local_save_figure(f, outBase, S);
end
