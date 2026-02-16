function stepA10_behavior_fit_by_participant(run_id, varargin)
% stepA10_behavior_fit_by_participant Fit per-participant behavioral models on VALID data only.
%
% This step performs a mechanistic, within-participant analysis of reliance
% decisions in the SAR interaction. For each VALID participant, it fits three
% behavioral models to all available door trials (typically ~60) and reports
% likelihood-based fit metrics and simple uncertainty estimates:
%
%   - Model 0: Trust-as-probability baseline (uses tau_decision directly).
%   - Model 1: Thresholded reliance as a logistic function of decision margin.
%   - Model 2: Threshold + modulation + lapse (tau and centered self-confidence).
%
% For each participant and model, the code computes:
%   - Negative log-likelihood (NLL) of binary follow/override decisions
%   - Brier score (calibration / probability accuracy)
%   - AIC and BIC (with complexity penalties)
%
% Parameter uncertainty is quantified via bootstrap resampling over doors
% (sampling with replacement). The best model per participant is selected by
% BIC with a parsimony rule: if a more complex model is within DeltaBIC of a
% simpler one, the simpler model is chosen.
%
% INPUTS (required)
%   run_id (string|char)
%       Analysis run identifier. Used to locate the A7 VALID behavior dataset:
%         derived/analysis_runs/<run_id>/stepA7_behavior_dataset/behavior_dataset_valid.mat
%
% NAME-VALUE ARGUMENTS (optional)
%   "OutDir" (string|char)
%       Output directory. Default:
%         derived/analysis_runs/<run_id>/stepA10_behavior_fit_by_participant
%
%   "Overwrite" (logical scalar)
%       If false (default), the function errors if outputs already exist.
%
%   "BootstrapN" (numeric scalar)
%       Number of bootstrap resamples per participant (default: 1000).
%       Set to 0 to disable bootstrapping.
%
%   "EpsMax" (numeric scalar)
%       Maximum lapse probability for Model 2, with eps constrained to [0, EpsMax]
%       via eps = EpsMax * sigmoid(z3). Default: 0.5.
%
%   "RandomSeed" (numeric scalar)
%       RNG seed used for bootstrap resampling. Default: 1.
%
%   "DeltaBIC_Parsimony" (numeric scalar)
%       Parsimony threshold for model selection. Default: 2.
%
%   "KHuge" (numeric scalar)
%       Diagnostic threshold for "huge" k estimates. Default: 100.
%
%   "MinN" (numeric scalar)
%       Diagnostic threshold for low number of doors (still fits, but flags).
%       Default: 40.
%
% OUTPUTS
%   (none)
%       Writes artifacts to:
%         derived/analysis_runs/<run_id>/stepA10_behavior_fit_by_participant/
%           - A10_params_by_participant.csv
%           - A10_params_by_participant.mat   (Tsum + details)
%           - meta.mat / meta.json
%           - figures/*.png
%
% ASSUMPTIONS / REQUIRED TABLE COLUMNS
%   The loaded VALID behavior table T must contain at least:
%     participant_id, door_index, block_index,
%     tau_decision, self_confidence, sc_centered, margin_treshold,
%     followed, is_valid_label
%   where followed is a binary indicator (1=follow drone, 0=override).
%
% DEPENDENCIES (assumed on MATLAB path)
%   must_exist_file, ensure_dir, save_json, thesisStyle, thesisFinalizeFigure, thesisExport

    % ------------------------------------------------------------------
    % Parse required input and name-value arguments
    % ------------------------------------------------------------------
    if nargin < 1 || isempty(run_id)
        error("stepA10_behavior_fit_by_participant: run_id is required.");
    end
    run_id = string(run_id);

    p = inputParser;
    p.addParameter("OutDir", "", @(s) isstring(s) || ischar(s));
    p.addParameter("Overwrite", false, @(x) islogical(x) && isscalar(x));
    p.addParameter("BootstrapN", 1000, @(x) isnumeric(x) && isscalar(x) && x>=0);
    p.addParameter("EpsMax", 0.5, @(x) isnumeric(x) && isscalar(x) && x>0 && x<=1);
    p.addParameter("RandomSeed", 1, @(x) isnumeric(x) && isscalar(x));
    p.addParameter("DeltaBIC_Parsimony", 2, @(x) isnumeric(x) && isscalar(x) && x>=0);
    p.addParameter("KHuge", 100, @(x) isnumeric(x) && isscalar(x) && x>0);
    p.addParameter("MinN", 40, @(x) isnumeric(x) && isscalar(x) && x>=1);
    p.parse(varargin{:});
    args = p.Results;

    rng(args.RandomSeed);

    % Thesis plotting defaults + style struct (used by local plotters)
    Sth = thesisStyle();

    % ------------------------------------------------------------------
    % Load A7 VALID dataset
    % ------------------------------------------------------------------
    a7Dir = fullfile("derived","analysis_runs",run_id,"stepA7_behavior_dataset");
    validMat = fullfile(a7Dir, "behavior_dataset_valid.mat");
    must_exist_file(validMat, "A7 VALID dataset");

    S = load(validMat, "T");
    if ~isfield(S,"T") || ~istable(S.T)
        error("[A10] VALID mat missing table T.");
    end
    T = S.T;

    % Validate required columns (data contract with Step A7)
    reqCols = ["participant_id","door_index","block_index", ...
               "tau_decision","self_confidence","sc_centered","margin_treshold", ...
               "followed","is_valid_label"];
    assert(all(ismember(reqCols, string(T.Properties.VariableNames))), ...
        "[A10] A7 VALID missing required columns.");

    % Keep only rows marked as valid labels (analysis subset)
    T = T(T.is_valid_label==1, :);

    % Drop NaNs in predictors used by the models (prevents NaNs in likelihood)
    okPred = isfinite(T.tau_decision) & isfinite(T.self_confidence) & ...
             isfinite(T.sc_centered) & isfinite(T.margin_treshold);
    T = T(okPred, :);

    % ------------------------------------------------------------------
    % Output directories and overwrite guard
    % ------------------------------------------------------------------
    outDir = string(args.OutDir);
    if strlength(outDir)==0
        outDir = fullfile("derived","analysis_runs",run_id,"stepA10_behavior_fit_by_participant");
    end
    ensure_dir(outDir);

    figDir = fullfile(outDir, "figures");
    ensure_dir(figDir);

    outCsv  = fullfile(outDir, "A10_params_by_participant.csv");
    outMat  = fullfile(outDir, "A10_params_by_participant.mat");
    metaMat = fullfile(outDir, "meta.mat");
    metaJson= fullfile(outDir, "meta.json");

    if ~args.Overwrite
        if isfile(outCsv) || isfile(outMat) || isfile(metaMat)
            error("[A10] Outputs exist. Set Overwrite=true to replace. (%s)", outDir);
        end
    end

    % ------------------------------------------------------------------
    % Per-participant loop
    %   - Stores sequences in door_index order for traceability
    %   - Fits Model 0/1/2 and bootstraps parameter CIs
    % ------------------------------------------------------------------
    pid_all = string(T.participant_id);
    uniqP = unique(pid_all);
    nP = numel(uniqP);

    details = struct();
    details.run_id = char(run_id);
    details.participants = cell(nP,1);

    % Summary table (one row per participant)
    Tsum = table();
    Tsum.participant_id = uniqP;
    Tsum.N = zeros(nP,1);

    % Model 0 metrics
    Tsum.m0_NLL = NaN(nP,1);
    Tsum.m0_Brier = NaN(nP,1);
    Tsum.m0_AIC = NaN(nP,1);
    Tsum.m0_BIC = NaN(nP,1);

    % Model 1 params + metrics + CI + flags
    Tsum.m1_k_hat = NaN(nP,1);
    Tsum.m1_NLL = NaN(nP,1);
    Tsum.m1_Brier = NaN(nP,1);
    Tsum.m1_AIC = NaN(nP,1);
    Tsum.m1_BIC = NaN(nP,1);
    Tsum.m1_k_ci_lo = NaN(nP,1);
    Tsum.m1_k_ci_hi = NaN(nP,1);
    Tsum.m1_boot_fail_rate = NaN(nP,1);
    Tsum.m1_flag_k_huge = false(nP,1);
    Tsum.m1_flag_ci_wide = false(nP,1);

    % Model 2 params + metrics + CI + flags
    Tsum.m2_k_hat = NaN(nP,1);
    Tsum.m2_beta_hat = NaN(nP,1);
    Tsum.m2_eps_hat = NaN(nP,1);
    Tsum.m2_NLL = NaN(nP,1);
    Tsum.m2_Brier = NaN(nP,1);
    Tsum.m2_AIC = NaN(nP,1);
    Tsum.m2_BIC = NaN(nP,1);
    Tsum.m2_k_ci_lo = NaN(nP,1);
    Tsum.m2_k_ci_hi = NaN(nP,1);
    Tsum.m2_beta_ci_lo = NaN(nP,1);
    Tsum.m2_beta_ci_hi = NaN(nP,1);
    Tsum.m2_eps_ci_lo = NaN(nP,1);
    Tsum.m2_eps_ci_hi = NaN(nP,1);
    Tsum.m2_boot_fail_rate = NaN(nP,1);
    Tsum.m2_flag_k_huge = false(nP,1);
    Tsum.m2_flag_eps_at_bound = false(nP,1);
    Tsum.m2_flag_ci_wide = false(nP,1);

    % Best model selection outputs
    Tsum.best_model = strings(nP,1);
    Tsum.best_model_idx = NaN(nP,1);
    Tsum.best_model_BIC = NaN(nP,1);
    Tsum.best_model_is_suspect = false(nP,1);
    Tsum.deltaBIC_m1_minus_m0 = NaN(nP,1);
    Tsum.deltaBIC_m2_minus_m1 = NaN(nP,1);
    Tsum.flag_low_N = false(nP,1);

    epsMax = double(args.EpsMax);
    B = double(args.BootstrapN);

    % Simple thresholds for "wide CI" diagnostics (absolute widths)
    CIW_K = 50;
    CIW_BETA = 10;
    CIW_EPS = 0.3;

    for i = 1:nP
        pid = uniqP(i);
        mask = (pid_all == pid);
        Tp = T(mask, :);

        % Preserve temporal order by door_index when fitting/storing sequences
        [~,ord] = sort(double(Tp.door_index(:)));
        Tp = Tp(ord,:);

        % Observed decisions (binary) and predictors used by the models
        y   = double(Tp.followed(:));          % 1=follow, 0=override
        tau = double(Tp.tau_decision(:));      % trust at decision time
        m   = double(Tp.margin_treshold(:));   % decision margin (e.g., tau - sc)
        scC = double(Tp.sc_centered(:));       % centered self-confidence (e.g., sc - 0.5)

        N = numel(y);
        Tsum.N(i) = N;
        if N < args.MinN
            Tsum.flag_low_N(i) = true;
        end

        % ----------------------------------------------------------
        % Model 0: Trust-as-probability baseline
        %   p0 = clamp01(tau_decision)
        % ----------------------------------------------------------
        p0 = clamp01(tau);
        nll0 = bernoulli_nll(y, p0);
        brier0 = mean((p0 - y).^2);

        Tsum.m0_NLL(i) = nll0;
        Tsum.m0_Brier(i) = brier0;
        Tsum.m0_AIC(i) = aic_from_nll(nll0, 0);
        Tsum.m0_BIC(i) = bic_from_nll(nll0, 0, N);

        % ----------------------------------------------------------
        % Model 1: Thresholded reliance (logistic in margin)
        %   p1 = sigmoid(k * margin), with k >= 0 enforced by k = exp(z)
        % ----------------------------------------------------------
        z0_1 = log(10);
        f1 = @(z) nll_model1(exp(z), m, y);
        [k1_hat, nll1, ok1] = safe_fit_1d_exp(f1, z0_1);

        p1 = sigmoid(k1_hat .* m);
        brier1 = mean((p1 - y).^2);

        Tsum.m1_k_hat(i) = k1_hat;
        Tsum.m1_NLL(i) = nll1;
        Tsum.m1_Brier(i) = brier1;
        Tsum.m1_AIC(i) = aic_from_nll(nll1, 1);
        Tsum.m1_BIC(i) = bic_from_nll(nll1, 1, N);

        % ----------------------------------------------------------
        % Model 2: Offset/lapse model with modulation
        %   z = k * tau + beta * sc_centered
        %   p2 = (1-eps)*sigmoid(z) + eps*0.5, with:
        %       k >= 0 via exp(z1)
        %       eps in [0, epsMax] via epsMax*sigmoid(z3)
        % ----------------------------------------------------------
        z0_2 = [log(10); 0; logit(0.05/epsMax)];
        f2 = @(z) nll_model2(exp(z(1)), z(2), epsMax*sigmoid(z(3)), tau, scC, y);
        [k2_hat, beta_hat, eps_hat, nll2, ok2] = safe_fit_model2(f2, z0_2, epsMax);

        z2 = k2_hat.*tau + beta_hat.*scC;
        p2 = (1-eps_hat).*sigmoid(z2) + eps_hat.*0.5;
        p2 = clamp01(p2);
        brier2 = mean((p2 - y).^2);

        Tsum.m2_k_hat(i) = k2_hat;
        Tsum.m2_beta_hat(i) = beta_hat;
        Tsum.m2_eps_hat(i) = eps_hat;
        Tsum.m2_NLL(i) = nll2;
        Tsum.m2_Brier(i) = brier2;
        Tsum.m2_AIC(i) = aic_from_nll(nll2, 3);
        Tsum.m2_BIC(i) = bic_from_nll(nll2, 3, N);

        % ----------------------------------------------------------
        % Bootstrap uncertainty via door-resampling (with replacement)
        %   - Refit Model 1 and Model 2 on each resample
        %   - Report percentile 95% CIs (2.5%, 97.5%)
        % ----------------------------------------------------------
        boot = struct();
        boot.B = B;

        if B > 0
            boot.m1_k = NaN(B,1);
            boot.m2_k = NaN(B,1);
            boot.m2_beta = NaN(B,1);
            boot.m2_eps = NaN(B,1);

            fail1 = 0;
            fail2 = 0;

            for b = 1:B
                idx = randi(N, N, 1);  % resample doors with replacement

                yb   = y(idx);
                mb   = m(idx);
                taub = tau(idx);
                scCb = scC(idx);

                % Model 1 bootstrap fit
                f1b = @(z) nll_model1(exp(z), mb, yb);
                [k1b, ~, okb1] = safe_fit_1d_exp(f1b, z0_1);
                if okb1
                    boot.m1_k(b) = k1b;
                else
                    fail1 = fail1 + 1;
                end

                % Model 2 bootstrap fit
                f2b = @(z) nll_model2(exp(z(1)), z(2), epsMax*sigmoid(z(3)), taub, scCb, yb);
                [k2b, betab, epsb, ~, okb2] = safe_fit_model2(f2b, z0_2, epsMax);
                if okb2
                    boot.m2_k(b) = k2b;
                    boot.m2_beta(b) = betab;
                    boot.m2_eps(b) = epsb;
                else
                    fail2 = fail2 + 1;
                end
            end

            boot.m1_fail_rate = fail1 / max(1,B);
            boot.m2_fail_rate = fail2 / max(1,B);

            Tsum.m1_boot_fail_rate(i) = boot.m1_fail_rate;
            Tsum.m2_boot_fail_rate(i) = boot.m2_fail_rate;

            % Percentile CI (ignoring NaNs from failed fits)
            [lo,hi] = ci95_percentile(boot.m1_k);
            Tsum.m1_k_ci_lo(i) = lo;
            Tsum.m1_k_ci_hi(i) = hi;

            [lo,hi] = ci95_percentile(boot.m2_k);
            Tsum.m2_k_ci_lo(i) = lo;
            Tsum.m2_k_ci_hi(i) = hi;

            [lo,hi] = ci95_percentile(boot.m2_beta);
            Tsum.m2_beta_ci_lo(i) = lo;
            Tsum.m2_beta_ci_hi(i) = hi;

            [lo,hi] = ci95_percentile(boot.m2_eps);
            Tsum.m2_eps_ci_lo(i) = lo;
            Tsum.m2_eps_ci_hi(i) = hi;
        end

        % ----------------------------------------------------------
        % Diagnostic flags (non-fatal; used for interpretation)
        % ----------------------------------------------------------
        Tsum.m1_flag_k_huge(i) = isfinite(k1_hat) && (k1_hat > args.KHuge);
        Tsum.m2_flag_k_huge(i) = isfinite(k2_hat) && (k2_hat > args.KHuge);

        % Lapse at/near bounds suggests weak identifiability or local optimum
        Tsum.m2_flag_eps_at_bound(i) = isfinite(eps_hat) && (eps_hat < 1e-3 || eps_hat > (epsMax - 1e-3));

        % CI width flags (mark as wide if CI missing/NaN)
        if isfinite(Tsum.m1_k_ci_lo(i)) && isfinite(Tsum.m1_k_ci_hi(i))
            Tsum.m1_flag_ci_wide(i) = (Tsum.m1_k_ci_hi(i) - Tsum.m1_k_ci_lo(i)) > CIW_K;
        else
            Tsum.m1_flag_ci_wide(i) = true;
        end

        wide2 = false;
        if isfinite(Tsum.m2_k_ci_lo(i)) && isfinite(Tsum.m2_k_ci_hi(i))
            wide2 = wide2 || ((Tsum.m2_k_ci_hi(i) - Tsum.m2_k_ci_lo(i)) > CIW_K);
        else
            wide2 = true;
        end
        if isfinite(Tsum.m2_beta_ci_lo(i)) && isfinite(Tsum.m2_beta_ci_hi(i))
            wide2 = wide2 || ((Tsum.m2_beta_ci_hi(i) - Tsum.m2_beta_ci_lo(i)) > CIW_BETA);
        else
            wide2 = true;
        end
        if isfinite(Tsum.m2_eps_ci_lo(i)) && isfinite(Tsum.m2_eps_ci_hi(i))
            wide2 = wide2 || ((Tsum.m2_eps_ci_hi(i) - Tsum.m2_eps_ci_lo(i)) > CIW_EPS);
        else
            wide2 = true;
        end
        Tsum.m2_flag_ci_wide(i) = wide2;

        % Additional suspect signals: optimizer failure or frequent bootstrap failures
        suspect1 = (~ok1) || (isfinite(Tsum.m1_boot_fail_rate(i)) && Tsum.m1_boot_fail_rate(i) > 0.2) || ...
                   Tsum.m1_flag_ci_wide(i) || Tsum.m1_flag_k_huge(i);
        suspect2 = (~ok2) || (isfinite(Tsum.m2_boot_fail_rate(i)) && Tsum.m2_boot_fail_rate(i) > 0.2) || ...
                   Tsum.m2_flag_ci_wide(i) || Tsum.m2_flag_k_huge(i) || Tsum.m2_flag_eps_at_bound(i);

        % ----------------------------------------------------------
        % Best model selection: BIC + parsimony rule
        % ----------------------------------------------------------
        bic0 = Tsum.m0_BIC(i);
        bic1 = Tsum.m1_BIC(i);
        bic2 = Tsum.m2_BIC(i);

        Tsum.deltaBIC_m1_minus_m0(i) = bic1 - bic0;
        Tsum.deltaBIC_m2_minus_m1(i) = bic2 - bic1;

        [bestIdx, bestName] = select_best_model_bic_parsimony([bic0 bic1 bic2], args.DeltaBIC_Parsimony);
        Tsum.best_model_idx(i) = bestIdx;
        Tsum.best_model(i) = bestName;

        bicVec = [bic0 bic1 bic2];
        Tsum.best_model_BIC(i) = bicVec(bestIdx);

        % Best-model suspect flag (Model 0 has no fitted parameters)
        switch bestIdx
            case 1
                Tsum.best_model_is_suspect(i) = false;
            case 2
                Tsum.best_model_is_suspect(i) = suspect1;
            case 3
                Tsum.best_model_is_suspect(i) = suspect2;
        end

        % ----------------------------------------------------------
        % Store per-participant details (including ordered sequences)
        % ----------------------------------------------------------
        D = struct();
        D.participant_id = char(pid);
        D.N = N;
        D.seq = struct( ...
            "door_index", double(Tp.door_index(:)), ...
            "block_index", double(Tp.block_index(:)), ...
            "y_follow", y(:), ...
            "tau", tau(:), ...
            "margin_treshold", m(:), ...
            "sc_centered", scC(:));

        D.model0 = struct("nll",nll0,"brier",brier0);
        D.model1 = struct("k_hat",k1_hat,"nll",nll1,"brier",brier1, ...
                          "k_ci95",[Tsum.m1_k_ci_lo(i) Tsum.m1_k_ci_hi(i)], ...
                          "boot_fail_rate",Tsum.m1_boot_fail_rate(i));
        D.model2 = struct("k_hat",k2_hat,"beta_hat",beta_hat,"eps_hat",eps_hat,"nll",nll2,"brier",brier2, ...
                          "k_ci95",[Tsum.m2_k_ci_lo(i) Tsum.m2_k_ci_hi(i)], ...
                          "beta_ci95",[Tsum.m2_beta_ci_lo(i) Tsum.m2_beta_ci_hi(i)], ...
                          "eps_ci95",[Tsum.m2_eps_ci_lo(i) Tsum.m2_eps_ci_hi(i)], ...
                          "boot_fail_rate",Tsum.m2_boot_fail_rate(i));

        D.flags = struct( ...
            "low_N", Tsum.flag_low_N(i), ...
            "m1_k_huge", Tsum.m1_flag_k_huge(i), ...
            "m1_ci_wide", Tsum.m1_flag_ci_wide(i), ...
            "m2_k_huge", Tsum.m2_flag_k_huge(i), ...
            "m2_eps_at_bound", Tsum.m2_flag_eps_at_bound(i), ...
            "m2_ci_wide", Tsum.m2_flag_ci_wide(i), ...
            "best_model_is_suspect", Tsum.best_model_is_suspect(i));

        D.best = struct("model_idx",bestIdx,"model_name",char(bestName),"bic",Tsum.best_model_BIC(i));

        if B > 0
            % Includes bootstrap arrays (can be large but is run-local and manageable)
            D.bootstrap = boot;
        end

        details.participants{i} = D;
    end

    % ------------------------------------------------------------------
    % Save artifacts (CSV + MAT + run metadata)
    % ------------------------------------------------------------------
    writetable(Tsum, outCsv);
    save(outMat, "Tsum", "details", "-v7.3");

    meta = struct();
    meta.run_id = char(run_id);
    meta.created = char(datetime('now','Format','yyyy-MM-dd HH:mm:ss'));
    meta.valid_participants = nP;
    meta.bootstrapN = B;
    meta.epsMax = epsMax;
    meta.random_seed = args.RandomSeed;
    meta.deltaBIC_parsimony = args.DeltaBIC_Parsimony;
    meta.k_huge_threshold = args.KHuge;
    meta.minN_flag_threshold = args.MinN;
    save(metaMat, "meta");
    save_json(metaJson, meta);

    % ------------------------------------------------------------------
    % Figures (minimal, thesis-friendly)
    % ------------------------------------------------------------------
    make_fig_bic_by_participant(fullfile(figDir, "bic_by_participant"), Tsum, Sth);
    make_fig_best_model_counts(fullfile(figDir, "best_model_counts"), Tsum, Sth);

    fprintf("[Step A10] Done.\n");
    fprintf("  VALID participants: %d\n", nP);
    fprintf("  BootstrapN: %d\n", B);
    fprintf("  Output dir: %s\n", outDir);
    fprintf("  Wrote: %s\n", outCsv);
end

% ======================================================================
% Selection rule: BIC + parsimony (DeltaBIC <= delta => choose simpler)
% Models are ordered by simplicity: 0 < 1 < 2
% ======================================================================
function [bestIdx, bestName] = select_best_model_bic_parsimony(bicVec012, delta)
% select_best_model_bic_parsimony Choose best model using BIC with parsimony rule.
%
% INPUTS
%   bicVec012 - 1x3 vector [bic0 bic1 bic2]
%   delta     - nonnegative threshold; if complex model is within delta of a
%               simpler model, the simpler model is selected.
%
% OUTPUTS
%   bestIdx   - model index in {1,2,3} corresponding to {0,1,2}
%   bestName  - string model identifier used in summary tables

    % bicVec012: [bic0 bic1 bic2]
    bic0 = bicVec012(1);
    bic1 = bicVec012(2);
    bic2 = bicVec012(3);

    % Start with raw BIC minimizer
    [~, idxMin] = min([bic0 bic1 bic2]);
    bestIdx = idxMin;

    % Apply parsimony along simplicity chain 0 -> 1 -> 2:
    % - If Model 2 wins but is within delta of Model 1, choose Model 1.
    if bestIdx == 3 && isfinite(bic2) && isfinite(bic1) && (bic1 - bic2) <= delta
        bestIdx = 2;
    end
    % - If Model 1 wins but is within delta of Model 0, choose Model 0.
    if bestIdx == 2 && isfinite(bic1) && isfinite(bic0) && (bic0 - bic1) <= delta
        bestIdx = 1;
    end
    % - If Model 2 wins but is within delta of Model 0 directly, choose Model 0.
    if bestIdx == 3 && isfinite(bic2) && isfinite(bic0) && (bic0 - bic2) <= delta
        bestIdx = 1;
    end

    switch bestIdx
        case 1, bestName = "model0_trust_as_probability";
        case 2, bestName = "model1_threshold";
        case 3, bestName = "model2_offset_lapse";
        otherwise, bestName = "unknown";
    end
end

% ======================================================================
% Model likelihoods (same structure as Step A8; adapted for per-participant)
% ======================================================================
function nll = nll_model1(k, margin, y_follow)
% nll_model1 Negative log-likelihood for Model 1 (thresholded reliance).
%
% Model 1:
%   p = sigmoid(k * margin), with k >= 0.
%
% INPUTS
%   k        - nonnegative gain parameter
%   margin   - vector predictor (decision margin)
%   y_follow - binary outcomes (1=follow, 0=override)
%
% OUTPUT
%   nll      - Bernoulli negative log-likelihood

    if ~isfinite(k) || k < 0
        nll = Inf; return;
    end
    p = sigmoid(k .* margin);
    nll = bernoulli_nll(y_follow, p);
end

function nll = nll_model2(k, beta, eps, tau, scC, y_follow)
% nll_model2 Negative log-likelihood for Model 2 (modulation + lapse).
%
% Model 2:
%   z = k * tau + beta * sc_centered
%   p = (1-eps)*sigmoid(z) + eps*0.5
%
% INPUTS
%   k        - nonnegative gain parameter
%   beta     - modulation coefficient on centered self-confidence
%   eps      - lapse probability in [0,1] (constrained externally to [0, epsMax])
%   tau      - vector trust-at-decision predictor
%   scC      - vector centered self-confidence predictor
%   y_follow - binary outcomes (1=follow, 0=override)
%
% OUTPUT
%   nll      - Bernoulli negative log-likelihood

    if ~isfinite(k) || k < 0 || ~isfinite(beta) || ~isfinite(eps) || eps < 0 || eps > 1
        nll = Inf; return;
    end
    z = k .* tau + beta .* scC;
    pstar = sigmoid(z);
    p = (1-eps).*pstar + eps.*0.5;
    nll = bernoulli_nll(y_follow, p);
end

function nll = bernoulli_nll(y, p)
% bernoulli_nll Bernoulli negative log-likelihood with probability clipping.
%
% INPUTS
%   y - binary outcomes (0/1)
%   p - predicted probabilities
%
% OUTPUT
%   nll - -sum(y*log(p) + (1-y)*log(1-p))

    p = min(max(p, 1e-12), 1-1e-12);
    nll = -sum(y .* log(p) + (1-y).*log(1-p));
end

% ======================================================================
% Fitting wrappers (guards against optimizer failures)
% ======================================================================
function [k_hat, nll_hat, ok] = safe_fit_1d_exp(f, z0)
% safe_fit_1d_exp Fit a 1D objective with positivity enforced by exp transform.
%
% Uses unconstrained fminsearch over z with k = exp(z).
%
% INPUTS
%   f   - function handle in z-space returning objective value
%   z0  - initial guess for z
%
% OUTPUTS
%   k_hat   - fitted k in original space
%   nll_hat - objective value at solution (as returned by f)
%   ok      - true if solution is finite

    ok = true;
    try
        zhat = fminsearch(f, z0, optimset('Display','off'));
        k_hat = exp(zhat);
        nll_hat = f(zhat);
        if ~isfinite(k_hat) || ~isfinite(nll_hat)
            ok = false;
        end
    catch
        ok = false;
        k_hat = NaN;
        nll_hat = Inf;
    end
end

function [k_hat, beta_hat, eps_hat, nll_hat, ok] = safe_fit_model2(f, z0, epsMax)
% safe_fit_model2 Fit Model 2 objective with parameter transforms.
%
% Parameterization:
%   k   = exp(z1)            (enforces k >= 0)
%   beta= z2                (unconstrained)
%   eps = epsMax*sigmoid(z3) (enforces eps in [0, epsMax])
%
% INPUTS
%   f      - function handle in z-space returning objective value
%   z0     - 3x1 initial guess for z
%   epsMax - upper bound for eps
%
% OUTPUTS
%   k_hat, beta_hat, eps_hat - fitted parameters in original space
%   nll_hat                  - objective value at solution (as returned by f)
%   ok                       - true if solution is finite

    ok = true;
    try
        zhat = fminsearch(f, z0, optimset('Display','off'));
        k_hat    = exp(zhat(1));
        beta_hat = zhat(2);
        eps_hat  = epsMax * sigmoid(zhat(3));
        nll_hat  = f(zhat);
        if ~isfinite(k_hat) || ~isfinite(beta_hat) || ~isfinite(eps_hat) || ~isfinite(nll_hat)
            ok = false;
        end
    catch
        ok = false;
        k_hat = NaN; beta_hat = NaN; eps_hat = NaN; nll_hat = Inf;
    end
end

% ======================================================================
% Information criteria helpers
% ======================================================================
function a = aic_from_nll(nll, k_params)
% aic_from_nll Compute Akaike Information Criterion from NLL.
    a = 2*k_params + 2*nll;
end

function b = bic_from_nll(nll, k_params, N)
% bic_from_nll Compute Bayesian Information Criterion from NLL.
    b = k_params*log(max(1,N)) + 2*nll;
end

function [lo,hi] = ci95_percentile(x)
% ci95_percentile Percentile-based 95% CI (2.5%, 97.5%) ignoring NaNs/Infs.
    x = x(:);
    x = x(isfinite(x));
    if isempty(x)
        lo = NaN; hi = NaN; return;
    end
    qs = prctile(x, [2.5 97.5]);
    lo = qs(1); hi = qs(2);
end

% ======================================================================
% Plots (minimal outputs consistent with the analysis pipeline style)
% ======================================================================
function make_fig_bic_by_participant(pathPdf, Tsum, S)
% make_fig_bic_by_participant Bar plot of per-participant BIC for Models 0/1/2.
%
% INPUTS
%   pathPdf - output base path for thesisExport (extension handled there)
%   Tsum    - summary table produced by stepA10_behavior_fit_by_participant
%   S       - thesis style struct

    f = figure('Visible','off');
    thesisStyle(f);

    set(f,'Units','centimeters');
    set(f,'Position',[2 2 S.figSizeBIC]);

    % -------------------------------
    % Create tiled layout (single axis)
    % -------------------------------
    tl = tiledlayout(f, 1, 1, 'TileSpacing','compact', 'Padding','compact');

    % Reserve space above/below tiledlayout for title/legend/xlabel
    tl.Units = 'normalized';
    pos = tl.Position;

    titleStripH  = 0.00;
    legendStripH = 0.04;
    gapTop = 0.035;

    bottomStripH = 0.045;
    gapBot = 0.010;

    topReserve    = titleStripH + legendStripH + 2*gapTop;
    bottomReserve = bottomStripH + gapBot;

    pos(2) = pos(2) + bottomReserve;
    pos(4) = max(0.10, pos(4) - topReserve - bottomReserve);
    tl.Position = pos;

    % (Retained) second adjustment block as implemented in the original code
    titleStripH  = 0.00;
    legendStripH = 0.04;
    gap = 0.035;

    topReserve = titleStripH + legendStripH + 2*gap;
    pos(4) = pos(4) - topReserve;
    tl.Position = pos;

    % -------------------------------
    % Title annotation (kept consistent with thesis export style)
    % -------------------------------
    titleY = pos(2) + pos(4) + legendStripH + 2*gap;

    annotation(f,'textbox', ...
        [pos(1), titleY, pos(3), titleStripH], ...
        'String','BIC by participant', ...
        'Interpreter','none', ...
        'EdgeColor','none', ...
        'HorizontalAlignment','center', ...
        'VerticalAlignment','middle', ...
        'FontName', S.fontTitle, ...
        'FontSize', S.fontSizeTit, ...
        'Tag','thesisTitle');

    % -------------------------------
    % Legend host axes (hidden; legend is placed above main plot)
    % -------------------------------
    legendY = pos(2) + pos(4) + gap;

    axLegendHost = axes(f,'Units','normalized', ...
        'Position',[pos(1), legendY, pos(3), legendStripH], ...
        'Visible','off');
    hold(axLegendHost,'on');

    % -------------------------------
    % Main plot
    % -------------------------------
    ax = nexttile(tl,1);
    hold(ax,'on');

    x = 1:height(Tsum);

    bic0 = Tsum.m0_BIC;
    bic1 = Tsum.m1_BIC;
    bic2 = Tsum.m2_BIC;

    M = [bic0 bic1 bic2];
    b = bar(ax, M);

    % Thesis palette (colors defined in thesisStyle())
    b(1).FaceColor = S.colors.cyan;
    b(2).FaceColor = S.colors.yellow;
    b(3).FaceColor = S.colors.green;
    [b.EdgeColor] = deal('none');

    grid(ax,'on');
    xlabel(ax,'Participant index');
    ylabel(ax,'BIC');
    set(ax,'XTick',x,'XTickLabel',x);

    hold(ax,'off');

    % -------------------------------
    % Legend (proxies to match bar colors)
    % -------------------------------
    p1 = plot(axLegendHost, NaN, NaN, ...
        's', 'MarkerFaceColor', S.colors.cyan, ...
        'MarkerEdgeColor','none', ...
        'DisplayName','Trust-as-probability');

    p2 = plot(axLegendHost, NaN, NaN, ...
        's', 'MarkerFaceColor', S.colors.yellow, ...
        'MarkerEdgeColor','none', ...
        'DisplayName','Thresholded reliance');

    p3 = plot(axLegendHost, NaN, NaN, ...
        's', 'MarkerFaceColor', S.colors.green, ...
        'MarkerEdgeColor','none', ...
        'DisplayName','Threshold + modulation + lapse');

    lgd = legend(axLegendHost,[p1 p2 p3], ...
        'Orientation','horizontal', ...
        'Location','north');

    lgd.Box = 'on';
    lgd.FontName = S.fontBody;
    lgd.FontSize = S.legendFont;
    lgd.Interpreter = 'none';

    % Force single-row legend where supported
    try, lgd.NumColumns = 3; end

    % Center legend in its strip
    lgd.Units = 'normalized';
    lp = lgd.Position;
    lp(1) = 0.5 - lp(3)/2;
    lgd.Position = lp;

    % -------------------------------
    % Finalize/export
    % -------------------------------
    thesisFinalizeFigure(f, S);
    thesisExport(f, string(pathPdf));
end

function make_fig_best_model_counts(pathPdf, Tsum, S)
% make_fig_best_model_counts Bar plot of selected best-model counts across participants.
%
% INPUTS
%   pathPdf - output base path for thesisExport (extension handled there)
%   Tsum    - summary table produced by stepA10_behavior_fit_by_participant
%   S       - thesis style struct

    f = figure('Visible','off');
    thesisStyle(f);

    grid on; hold on;

    names = ["model0_trust_as_probability","model1_threshold","model2_offset_lapse"];
    counts = zeros(1,numel(names));
    for i = 1:numel(names)
        counts(i) = sum(string(Tsum.best_model)==names(i));
    end

    bar(counts);
    set(gca,'XTick',1:numel(names),'XTickLabel', ...
        {'Trust-as-probability','Thresholded reliance','Threshold + modulation + lapse'});
    xtickangle(20);
    ylabel('Count');
    title('Best model counts (BIC + parsimony rule)', 'Interpreter','none');

    local_save_figure(f, string(pathPdf), S);
end

function local_save_figure(f, outBase, S)
% local_save_figure Finalize and export a figure using thesis styling helpers.
    thesisFinalizeFigure(f, S);
    thesisExport(f, outBase);
end

% ======================================================================
% Math helpers
% ======================================================================
function p = clamp01(x)
% clamp01 Clip values element-wise to [0, 1].
    p = min(max(x, 0), 1);
end

function s = sigmoid(z)
% sigmoid Logistic sigmoid.
    s = 1 ./ (1 + exp(-z));
end

function z = logit(p)
% logit Inverse logistic with defensive probability clipping.
    p = min(max(p, 1e-12), 1-1e-12);
    z = log(p./(1-p));
end
