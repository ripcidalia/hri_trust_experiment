function stepA13_trust_divergence_sanity_check(run_id, varargin)
% stepA13_trust_divergence_sanity_check Trust realism sanity check on continuous-time trust trajectories.
%
% Step A13 re-simulates trust trajectories on the full simulation time grid
% (not only door-event times) and quantifies divergence between:
%
%   A) SIMPLE-mode trust replay vs COUPLED-mode trust
%        - COUPLED with GLOBAL behavior parameters (A8 fit)
%        - COUPLED with PERSONALIZED behavior parameters (A10, with A11 guardrails)
%
%   B) COUPLED-mode trust with PERSONALIZED vs GLOBAL behavior parameters
%        - Uses matched RNG seeds so that differences reflect parameter changes,
%          not Monte Carlo sampling noise.
%
% For each rollout, an error curve e(t_k) is defined on the full grid:
%   Option A: e(t_k) = tau_cpl(t_k) - tau_simple(t_k)
%   Option B: e(t_k) = tau_cpl_personal(t_k) - tau_cpl_global(t_k)
%
% Divergence metrics (computed per rollout from e(t_k)):
%   IAD    = sum_k |e(t_k)| * dt                      [trust*s]
%   MAE_t  = (1/T) * sum_k |e(t_k)| * dt              [trust]
%   RMSE_t = sqrt( (1/T) * sum_k e(t_k)^2 * dt )      [trust]
%   MaxAbs = max_k |e(t_k)|                           [trust]
%
% INPUTS (required)
%   run_id (string|char)
%       Analysis run identifier used to locate run-local artifacts under:
%         derived/analysis_runs/<run_id>/
%
% NAME-VALUE ARGUMENTS (optional)
%   "OutDir" (string|char)
%       Output directory. Default:
%         derived/analysis_runs/<run_id>/stepA13_trust_divergence_sanity
%
%   "Overwrite" (logical scalar)
%       If false, abort when outputs already exist. Default: false
%
%   "RolloutsPerParticipant" (numeric scalar >= 1)
%       Number of Monte Carlo rollouts per participant. Default: 1000
%
%   "Quantiles" (1x2 numeric in [0,1])
%       Quantiles reported across rollouts. Default: [0.05 0.95]
%
%   "RandomSeed" (numeric scalar)
%       Base RNG seed used to generate rollout seeds. Default: 1
%
%   "UseA11Guards" (logical scalar)
%       If true and A11 file exists, apply guardrail fallback for under-identified
%       participants (e.g., degrade model 2->1->0). Default: true
%
%   "FallbackStrategy" (string|char)
%       Fallback mode label. Current implementation expects "simple" and applies
%       the 2->1->0 degradation rule when guardrails/invalid params trigger.
%       Default: "simple"
%
%   "EpsMax" (numeric scalar in (0,1])
%       Upper bound for lapse/epsilon parameter validity check (model 2).
%       Default: 0.5
%
%   "GlobalModelIdx" (numeric scalar in {0,1,2})
%       Global behavior model index for A8 parameters. Default: 2
%
%   "PooledTimeGridN" (numeric scalar >= 51)
%       Number of normalized-time points for pooled curve summaries. Default: 201
%
%   "SaveRolloutLevelMat" (logical scalar)
%       If true, also save rollout-level metrics (can be large). Default: true
%
% OUTPUTS
%   (none)
%       Writes artifacts to outDir:
%         - A13_divergence_by_participant.csv
%         - A13_divergence_by_participant.mat (table T + meta + pooled summaries)
%         - A13_rollout_level_metrics.mat (optional, potentially large)
%         - meta.mat / meta.json
%         - figures/*.png and figures/by_participant/*.png (via thesisExport)
%
% ASSUMPTIONS / DEPENDENCIES
%   - Run-local inputs exist:
%       * A1: participants_valid_probes_mapped_stepM4.mat
%       * A3: selection.mat (theta_star and results_file with cfg.dt)
%       * A8: stepA8_behavior_fit_eval/fit_params.mat (global behavior fit)
%       * A10: stepA10_behavior_fit_by_participant/A10_params_by_participant.mat
%       * A11: stepA11_behavior_param_robustness/A11_blockwise_params.mat (optional)
%   - Utilities on path:
%       must_exist_file, ensure_dir, save_json, load_participants_struct, find_theta_in_struct
%   - Simulator on path:
%       trust_simulate_or_predict_one_participant
%
    if nargin < 1 || isempty(run_id)
        error("stepA13_trust_divergence_sanity_check: run_id is required.");
    end
    run_id = string(run_id);

    % ------------------------------------------------------------------
    % Parse inputs
    % ------------------------------------------------------------------
    p = inputParser;
    p.addParameter("OutDir", "", @(s) isstring(s) || ischar(s));
    p.addParameter("Overwrite", false, @(x) islogical(x) && isscalar(x));
    p.addParameter("RolloutsPerParticipant", 1000, @(x) isnumeric(x) && isscalar(x) && x>=1);
    p.addParameter("Quantiles", [0.05 0.95], @(x) isnumeric(x) && numel(x)==2 && all(x>=0) && all(x<=1));
    p.addParameter("RandomSeed", 1, @(x) isnumeric(x) && isscalar(x));
    p.addParameter("UseA11Guards", true, @(x) islogical(x) && isscalar(x));
    p.addParameter("FallbackStrategy", "simple", @(s) isstring(s) || ischar(s));
    p.addParameter("EpsMax", 0.5, @(x) isnumeric(x) && isscalar(x) && x>0 && x<=1);
    p.addParameter("GlobalModelIdx", 2, @(x) isnumeric(x) && isscalar(x) && ismember(round(x), [0 1 2]));
    p.addParameter("PooledTimeGridN", 201, @(x) isnumeric(x) && isscalar(x) && x>=51);
    p.addParameter("SaveRolloutLevelMat", true, @(x) islogical(x) && isscalar(x));
    p.parse(varargin{:});
    args = p.Results;

    % Thesis plotting defaults + style struct (global)
    S = thesisStyle();

    % Base RNG seed for reproducibility (rollout seeds are derived deterministically).
    rng(args.RandomSeed);

    % ------------------------------------------------------------------
    % Locate and validate required inputs
    % ------------------------------------------------------------------
    a8Dir = fullfile("derived","analysis_runs",run_id,"stepA8_behavior_fit_eval");
    fitMatA8 = fullfile(a8Dir, "fit_params.mat");
    must_exist_file(fitMatA8, "A8 fit_params.mat (global behavior params)");

    a10Dir = fullfile("derived","analysis_runs",run_id,"stepA10_behavior_fit_by_participant");
    a10Mat = fullfile(a10Dir, "A10_params_by_participant.mat");
    must_exist_file(a10Mat, "A10_params_by_participant.mat");

    a11Dir = fullfile("derived","analysis_runs",run_id,"stepA11_behavior_param_robustness");
    a11Mat = fullfile(a11Dir, "A11_blockwise_params.mat");
    haveA11 = isfile(a11Mat);

    % ------------------------------------------------------------------
    % Load theta_star, dt, and VALID participants (consistent with A9/A12 loader)
    % ------------------------------------------------------------------
    [theta_star, dt, validParticipants] = local_load_theta_dt_and_valid_participants_like_A5(run_id);

    % ------------------------------------------------------------------
    % Load global behavior fit (A8) and select behavior parameter set
    % ------------------------------------------------------------------
    Sfit = load(fitMatA8, "fit");
    if ~isfield(Sfit,"fit") || ~isstruct(Sfit.fit)
        error("[A13] A8 fit_params.mat missing 'fit' struct.");
    end
    fit = Sfit.fit;

    globalModelIdx = round(args.GlobalModelIdx);
    [globalName, bpar_global] = resolve_behavior_params_global(globalModelIdx, fit);

    % ------------------------------------------------------------------
    % Load personalized behavior fit summary (A10) and optional guardrails (A11)
    % ------------------------------------------------------------------
    Sa10 = load(a10Mat);
    Ta10 = local_find_first_table(Sa10, ["Tsum","T"]);
    if isempty(Ta10)
        error("[A13] Could not find A10 summary table in %s (expected Tsum).", a10Mat);
    end
    if ~ismember("participant_id", string(Ta10.Properties.VariableNames)) || ...
       ~ismember("best_model_idx", string(Ta10.Properties.VariableNames))
        error("[A13] A10 table must contain participant_id and best_model_idx.");
    end

    Tblocks = table();
    if haveA11
        Sa11 = load(a11Mat);
        Tblocks = local_find_first_table(Sa11, ["Tblocks","T"]);
        if isempty(Tblocks)
            Tblocks = table(); % optional; absence is acceptable
        end
    end

    % ------------------------------------------------------------------
    % Output directories and overwrite guard
    % ------------------------------------------------------------------
    outDir = string(args.OutDir);
    if strlength(outDir)==0
        outDir = fullfile("derived","analysis_runs",run_id,"stepA13_trust_divergence_sanity");
    end
    ensure_dir(outDir);
    figDir = fullfile(outDir, "figures"); ensure_dir(figDir);
    figByP = fullfile(figDir, "by_participant"); ensure_dir(figByP);

    outCsv = fullfile(outDir, "A13_divergence_by_participant.csv");
    outMat = fullfile(outDir, "A13_divergence_by_participant.mat");
    rolloutMat = fullfile(outDir, "A13_rollout_level_metrics.mat");
    metaMat = fullfile(outDir, "meta.mat");
    metaJson= fullfile(outDir, "meta.json");

    if ~args.Overwrite
        if isfile(outCsv) || isfile(outMat) || (args.SaveRolloutLevelMat && isfile(rolloutMat))
            error("[A13] Outputs exist. Set Overwrite=true to replace. (%s)", outDir);
        end
    end

    % ------------------------------------------------------------------
    % Determine participant set
    % ------------------------------------------------------------------
    uniqP = local_get_participant_ids(validParticipants);
    nP = numel(uniqP);

    R   = double(args.RolloutsPerParticipant);
    qlo = double(args.Quantiles(1));
    qhi = double(args.Quantiles(2));

    fprintf("[A13] Trust divergence sanity check\n");
    fprintf("      VALID participants: %d\n", nP);
    fprintf("      Rollouts/participant: %d\n", R);
    fprintf("      dt: %.6g s\n", dt);
    fprintf("      Global behavior: idx=%d (%s)\n", globalModelIdx, globalName);

    % ------------------------------------------------------------------
    % Allocate participant summary table
    % ------------------------------------------------------------------
    T = table();
    T.participant_id = uniqP;

    % A) Simple vs Coupled (GLOBAL) metrics summary
    T.A_simple_vs_cpl_global_MAE_mean     = NaN(nP,1);
    T.A_simple_vs_cpl_global_MAE_qlo      = NaN(nP,1);
    T.A_simple_vs_cpl_global_MAE_qhi      = NaN(nP,1);
    T.A_simple_vs_cpl_global_RMSE_mean    = NaN(nP,1);
    T.A_simple_vs_cpl_global_RMSE_qlo     = NaN(nP,1);
    T.A_simple_vs_cpl_global_RMSE_qhi     = NaN(nP,1);
    T.A_simple_vs_cpl_global_IAD_mean     = NaN(nP,1);
    T.A_simple_vs_cpl_global_IAD_qlo      = NaN(nP,1);
    T.A_simple_vs_cpl_global_IAD_qhi      = NaN(nP,1);
    T.A_simple_vs_cpl_global_MaxAbs_mean  = NaN(nP,1);
    T.A_simple_vs_cpl_global_MaxAbs_qlo   = NaN(nP,1);
    T.A_simple_vs_cpl_global_MaxAbs_qhi   = NaN(nP,1);

    % A) Simple vs Coupled (PERSONALIZED) metrics summary
    T.A_simple_vs_cpl_person_MAE_mean     = NaN(nP,1);
    T.A_simple_vs_cpl_person_MAE_qlo      = NaN(nP,1);
    T.A_simple_vs_cpl_person_MAE_qhi      = NaN(nP,1);
    T.A_simple_vs_cpl_person_RMSE_mean    = NaN(nP,1);
    T.A_simple_vs_cpl_person_RMSE_qlo     = NaN(nP,1);
    T.A_simple_vs_cpl_person_RMSE_qhi     = NaN(nP,1);
    T.A_simple_vs_cpl_person_IAD_mean     = NaN(nP,1);
    T.A_simple_vs_cpl_person_IAD_qlo      = NaN(nP,1);
    T.A_simple_vs_cpl_person_IAD_qhi      = NaN(nP,1);
    T.A_simple_vs_cpl_person_MaxAbs_mean  = NaN(nP,1);
    T.A_simple_vs_cpl_person_MaxAbs_qlo   = NaN(nP,1);
    T.A_simple_vs_cpl_person_MaxAbs_qhi   = NaN(nP,1);

    % B) Coupled PERSONALIZED vs Coupled GLOBAL (matched seeds)
    T.B_cpl_person_vs_global_MAE_mean     = NaN(nP,1);
    T.B_cpl_person_vs_global_MAE_qlo      = NaN(nP,1);
    T.B_cpl_person_vs_global_MAE_qhi      = NaN(nP,1);
    T.B_cpl_person_vs_global_RMSE_mean    = NaN(nP,1);
    T.B_cpl_person_vs_global_RMSE_qlo     = NaN(nP,1);
    T.B_cpl_person_vs_global_RMSE_qhi     = NaN(nP,1);
    T.B_cpl_person_vs_global_IAD_mean     = NaN(nP,1);
    T.B_cpl_person_vs_global_IAD_qlo      = NaN(nP,1);
    T.B_cpl_person_vs_global_IAD_qhi      = NaN(nP,1);
    T.B_cpl_person_vs_global_MaxAbs_mean  = NaN(nP,1);
    T.B_cpl_person_vs_global_MaxAbs_qlo   = NaN(nP,1);
    T.B_cpl_person_vs_global_MaxAbs_qhi   = NaN(nP,1);

    % Guardrail/fallback audit (personalized resolver)
    T.model_idx_a10      = NaN(nP,1);
    T.model_idx_used     = NaN(nP,1);
    T.fallback_applied   = false(nP,1);
    T.fallback_reason    = strings(nP,1);
    T.model_name_used    = strings(nP,1);

    % ------------------------------------------------------------------
    % Pooled curve storage on normalized time grid s in [0,1]
    %
    % Participants may have different horizon lengths. For pooled summaries, each
    % participant's rollout-mean error curve is resampled onto sgrid and then
    % aggregated across participants.
    % ------------------------------------------------------------------
    Ns = round(args.PooledTimeGridN);
    sgrid = linspace(0,1,Ns)';

    E_A_global_byP   = NaN(Ns, nP);
    E_A_person_byP   = NaN(Ns, nP);
    E_B_byP          = NaN(Ns, nP);

    Acc_A_global_byP = NaN(Ns, nP);
    Acc_A_person_byP = NaN(Ns, nP);
    Acc_B_byP        = NaN(Ns, nP);

    % Optional rollout-level metric arrays (useful for debugging / deep dives)
    rollLevel = struct();
    rollLevel.meta = struct("run_id",char(run_id),"R",R,"dt",dt,"global_model_idx",globalModelIdx, ...
        "global_model_name",char(globalName),"random_seed",args.RandomSeed);
    if args.SaveRolloutLevelMat
        rollLevel.pid = uniqP;
        rollLevel.A_global = init_rollout_metric_store(nP, R);
        rollLevel.A_person = init_rollout_metric_store(nP, R);
        rollLevel.B = init_rollout_metric_store(nP, R);
    end

    % ------------------------------------------------------------------
    % Main loop: per participant, compute rollout distributions and mean curves
    % ------------------------------------------------------------------
    for pi = 1:nP
        pid = uniqP(pi);
        Pp = get_participant_from_collection(validParticipants, pid);

        % Resolve personalized behavior parameters for this participant.
        % This mirrors the A12 resolver and includes optional A11 guardrails.
        [bpar_personal, info] = resolve_personalized_behavior_params( ...
            pid, Ta10, Tblocks, args.UseA11Guards, args.EpsMax, args.FallbackStrategy);

        T.model_idx_a10(pi)        = info.model_idx_a10;
        T.model_idx_used(pi)       = info.model_idx_used;
        T.model_name_used(pi)      = info.model_name_used;
        T.fallback_applied(pi)     = info.fallback_applied;
        T.fallback_reason(pi)      = info.fallback_reason;

        % SIMPLE replay is deterministic given (theta_star, participant, dt).
        simSimple = trust_simulate_or_predict_one_participant("simple", theta_star, Pp, dt);
        tauS = double(simSimple.tau_hist(:));
        tgrid = double(simSimple.t_grid(:));
        if isempty(tgrid) || isempty(tauS) || numel(tgrid) ~= numel(tauS)
            error("[A13] pid=%s returned invalid simple trajectory.", pid);
        end

        % Total horizon length for continuous-time normalized metrics.
        Ttotal = max(tgrid) - min(tgrid);
        if ~isfinite(Ttotal) || Ttotal <= 0
            % Defensive fallback (should not trigger if tgrid is valid).
            Ttotal = (numel(tgrid)-1) * dt;
        end

        % Per-rollout metric arrays: columns are [MAE_t, RMSE_t, IAD, MaxAbs].
        Aglob = NaN(R,4); % A: simple vs coupled (global behavior)
        Apers = NaN(R,4); % A: simple vs coupled (personal behavior)
        Bpg   = NaN(R,4); % B: coupled personal vs coupled global

        % Rollout-mean error and accumulated absolute error curves.
        eAglob_mean = zeros(numel(tgrid),1);
        eApers_mean = zeros(numel(tgrid),1);
        eB_mean     = zeros(numel(tgrid),1);

        accAglob_mean = zeros(numel(tgrid),1);
        accApers_mean = zeros(numel(tgrid),1);
        accB_mean     = zeros(numel(tgrid),1);

        for r = 1:R
            % Matched seed across GLOBAL and PERSONALIZED COUPLED simulations.
            % This isolates parameter effects from sampling noise.
            seed = args.RandomSeed + 100000*pi + r;

            % --- Coupled GLOBAL ---
            rng(seed);
            simG = trust_simulate_or_predict_one_participant("coupled", theta_star, Pp, dt, bpar_global);
            tauG = double(simG.tau_hist(:));

            % --- Coupled PERSONALIZED ---
            rng(seed);
            simP = trust_simulate_or_predict_one_participant("coupled", theta_star, Pp, dt, bpar_personal);
            tauP = double(simP.tau_hist(:));

            % Ensure coupled vectors match SIMPLE length (defensive alignment).
            tauG = local_align_length(tauG, tauS);
            tauP = local_align_length(tauP, tauS);

            % A-global: tau_cpl_global - tau_simple
            eA_g = tauG - tauS;
            [mae, rmse, iad, mx, acc] = divergence_metrics_from_error(eA_g, dt, Ttotal);
            Aglob(r,:) = [mae rmse iad mx];

            % A-personal: tau_cpl_personal - tau_simple
            eA_p = tauP - tauS;
            [mae, rmse, iad, mx, accp] = divergence_metrics_from_error(eA_p, dt, Ttotal);
            Apers(r,:) = [mae rmse iad mx];

            % B: tau_cpl_personal - tau_cpl_global
            eB = tauP - tauG;
            [mae, rmse, iad, mx, accb] = divergence_metrics_from_error(eB, dt, Ttotal);
            Bpg(r,:) = [mae rmse iad mx];

            % Accumulate rollout curves for participant-level means.
            eAglob_mean = eAglob_mean + eA_g;
            eApers_mean = eApers_mean + eA_p;
            eB_mean     = eB_mean     + eB;

            accAglob_mean = accAglob_mean + acc;
            accApers_mean = accApers_mean + accp;
            accB_mean     = accB_mean     + accb;

            % Optional rollout-level metrics storage.
            if args.SaveRolloutLevelMat
                rollLevel.A_global.MAE(pi,r)    = Aglob(r,1);
                rollLevel.A_global.RMSE(pi,r)   = Aglob(r,2);
                rollLevel.A_global.IAD(pi,r)    = Aglob(r,3);
                rollLevel.A_global.MaxAbs(pi,r) = Aglob(r,4);

                rollLevel.A_person.MAE(pi,r)    = Apers(r,1);
                rollLevel.A_person.RMSE(pi,r)   = Apers(r,2);
                rollLevel.A_person.IAD(pi,r)    = Apers(r,3);
                rollLevel.A_person.MaxAbs(pi,r) = Apers(r,4);

                rollLevel.B.MAE(pi,r)           = Bpg(r,1);
                rollLevel.B.RMSE(pi,r)          = Bpg(r,2);
                rollLevel.B.IAD(pi,r)           = Bpg(r,3);
                rollLevel.B.MaxAbs(pi,r)        = Bpg(r,4);
            end
        end

        % Convert accumulated sums to means across rollouts for curve plotting.
        eAglob_mean = eAglob_mean / R;
        eApers_mean = eApers_mean / R;
        eB_mean     = eB_mean     / R;

        accAglob_mean = accAglob_mean / R;
        accApers_mean = accApers_mean / R;
        accB_mean     = accB_mean     / R;

        % Resample participant mean curves to normalized time sgrid for pooling.
        s = (tgrid - min(tgrid)) ./ max(1e-12, (max(tgrid)-min(tgrid)));
        E_A_global_byP(:,pi)   = interp1(s, eAglob_mean, sgrid, "linear", "extrap");
        E_A_person_byP(:,pi)   = interp1(s, eApers_mean, sgrid, "linear", "extrap");
        E_B_byP(:,pi)          = interp1(s, eB_mean,     sgrid, "linear", "extrap");

        % Resample accumulated absolute error curves (units: trust*s).
        Acc_A_global_byP(:,pi) = interp1(s, accAglob_mean, sgrid, "linear", "extrap");
        Acc_A_person_byP(:,pi) = interp1(s, accApers_mean, sgrid, "linear", "extrap");
        Acc_B_byP(:,pi)        = interp1(s, accB_mean,     sgrid, "linear", "extrap");

        % Participant summaries (quantiles across rollouts).
        T = local_write_summary_row(T, pi, Aglob, "A_simple_vs_cpl_global", qlo, qhi);
        T = local_write_summary_row(T, pi, Apers, "A_simple_vs_cpl_person", qlo, qhi);
        T = local_write_summary_row(T, pi, Bpg,   "B_cpl_person_vs_global", qlo, qhi);

        % Per-participant figures on real-time grid.
        make_fig_error_and_accum(fullfile(figByP, sprintf("pid_%s_A_global_error", sanitize_pid(pid))), ...
            pid, tgrid, eAglob_mean, accAglob_mean, "pooled and accumulated error", S);
        make_fig_error_and_accum(fullfile(figByP, sprintf("pid_%s_A_person_error", sanitize_pid(pid))), ...
            pid, tgrid, eApers_mean, accApers_mean, "pooled and accumulated error", S);
        make_fig_error_and_accum(fullfile(figByP, sprintf("pid_%s_B_person_minus_global", sanitize_pid(pid))), ...
            pid, tgrid, eB_mean, accB_mean, "pooled and accumulated error", S);
    end

    % ------------------------------------------------------------------
    % Pooled curves and pooled figures (normalized time)
    % ------------------------------------------------------------------
    pooled = struct();
    pooled.sgrid = sgrid;

    pooled.A_global = summarize_curves(E_A_global_byP, Acc_A_global_byP, qlo, qhi);
    pooled.A_person = summarize_curves(E_A_person_byP, Acc_A_person_byP, qlo, qhi);
    pooled.B        = summarize_curves(E_B_byP,        Acc_B_byP,        qlo, qhi);

    make_fig_pooled_error(fullfile(figDir, "pooled_A_error_simple_vs_coupled_global"), ...
        sgrid, pooled.A_global, "Pooled error over normalized time", S);
    make_fig_pooled_error(fullfile(figDir, "pooled_A_error_simple_vs_coupled_personal"), ...
        sgrid, pooled.A_person, "Pooled error over normalized time", S);
    make_fig_pooled_error(fullfile(figDir, "pooled_B_error_person_minus_global"), ...
        sgrid, pooled.B, "Pooled error over normalized time", S);

    make_fig_pooled_accum(fullfile(figDir, "pooled_A_accum_abs_error_global"), ...
        sgrid, pooled.A_global, "Pooled accumulated error over normalized time", S);
    make_fig_pooled_accum(fullfile(figDir, "pooled_A_accum_abs_error_personal"), ...
        sgrid, pooled.A_person, "Pooled accumulated error over normalized time", S);
    make_fig_pooled_accum(fullfile(figDir, "pooled_B_accum_abs_error_person_minus_global"), ...
        sgrid, pooled.B, "Pooled accumulated error over normalized time", S);

    % ------------------------------------------------------------------
    % Save outputs
    % ------------------------------------------------------------------
    writetable(T, outCsv);

    meta = struct();
    meta.run_id = char(run_id);
    meta.created = char(datetime('now','Format','yyyy-MM-dd HH:mm:ss'));
    meta.valid_participants = nP;
    meta.rollouts_per_participant = R;
    meta.quantiles = [qlo qhi];
    meta.random_seed = args.RandomSeed;
    meta.dt = dt;
    meta.theta_dim = numel(theta_star);
    meta.global_model_idx = globalModelIdx;
    meta.global_model_name = char(globalName);
    meta.use_a11_guards = args.UseA11Guards && haveA11;
    meta.have_a11_file = haveA11;
    meta.a8_fit_file = char(fitMatA8);
    meta.a10_file = char(a10Mat);
    meta.a11_file = char(a11Mat);
    meta.fallback_strategy = char(string(args.FallbackStrategy));
    meta.pooled_time_grid_n = Ns;

    save(outMat, "T", "meta", "pooled", "-v7.3");
    save(metaMat, "meta");
    save_json(metaJson, meta);

    if args.SaveRolloutLevelMat
        save(rolloutMat, "rollLevel", "-v7.3");
    end

    % ------------------------------------------------------------------
    % Terminal summary (quick inspection)
    % ------------------------------------------------------------------
    fprintf("\n[A13] Quick pooled summaries (mean across participants)\n");

    % Pooled means of participant-level rollout-mean metrics (already in T).
    Aglob_MAE = mean(T.A_simple_vs_cpl_global_MAE_mean, "omitnan");
    Apers_MAE = mean(T.A_simple_vs_cpl_person_MAE_mean, "omitnan");
    B_MAE     = mean(T.B_cpl_person_vs_global_MAE_mean, "omitnan");

    Aglob_RMSE = mean(T.A_simple_vs_cpl_global_RMSE_mean, "omitnan");
    Apers_RMSE = mean(T.A_simple_vs_cpl_person_RMSE_mean, "omitnan");
    B_RMSE     = mean(T.B_cpl_person_vs_global_RMSE_mean, "omitnan");

    Aglob_Max = mean(T.A_simple_vs_cpl_global_MaxAbs_mean, "omitnan");
    Apers_Max = mean(T.A_simple_vs_cpl_person_MaxAbs_mean, "omitnan");
    B_Max     = mean(T.B_cpl_person_vs_global_MaxAbs_mean, "omitnan");

    fprintf("  A (simple vs coupled GLOBAL):   MAE_t=%.4g  RMSE_t=%.4g  Max|e|=%.4g\n", Aglob_MAE, Aglob_RMSE, Aglob_Max);
    fprintf("  A (simple vs coupled PERSONAL): MAE_t=%.4g  RMSE_t=%.4g  Max|e|=%.4g\n", Apers_MAE, Apers_RMSE, Apers_Max);
    fprintf("  B (PERSONAL - GLOBAL coupled):  MAE_t=%.4g  RMSE_t=%.4g  Max|e|=%.4g\n", B_MAE, B_RMSE, B_Max);

    % Relative change in divergence-to-simple when using personalized behavior params.
    if isfinite(Aglob_MAE) && Aglob_MAE > 0 && isfinite(Apers_MAE)
        pct = 100*(Apers_MAE - Aglob_MAE)/Aglob_MAE;
        fprintf("  Personalization effect on A (MAE_t): %+0.2f%% relative to global\n", pct);
    end

    % Guardrail fallback audit summary.
    nFallback = sum(T.fallback_applied);
    fprintf("  Guardrail fallback applied for %d/%d participants\n", nFallback, nP);
    if nFallback > 0
        idx = find(T.fallback_applied);
        for k = 1:min(5,numel(idx))
            i = idx(k);
            fprintf("    pid=%s  used=%s  reason=%s\n", T.participant_id(i), T.model_name_used(i), T.fallback_reason(i));
        end
        if numel(idx) > 5
            fprintf("    ... (%d more)\n", numel(idx)-5);
        end
    end

    fprintf("\n[Step A13] Done.\n");
    fprintf("  Output dir: %s\n", outDir);
    fprintf("  Wrote: %s\n", outCsv);
end

% ======================================================================
% Metrics from an error curve e(t) on the simulation grid
% ======================================================================
function [mae_t, rmse_t, iad, maxabs, acc_abs_curve] = divergence_metrics_from_error(e, dt, Ttotal)
    e = double(e(:));
    e(~isfinite(e)) = 0;

    abs_e = abs(e);

    % Continuous-time integral approximations using the uniform step dt.
    iad = sum(abs_e) * dt;
    mae_t = iad / max(1e-12, Ttotal);
    rmse_t = sqrt( (sum(e.^2) * dt) / max(1e-12, Ttotal) );

    maxabs = max(abs_e);

    % Accumulated absolute error curve (units: trust*s), same length as e.
    acc_abs_curve = cumsum(abs_e) * dt;
end

% ======================================================================
% Participant-level summary writing
% ======================================================================
function T = local_write_summary_row(T, rowIdx, X, prefix, qlo, qhi)
    % X: [R x 4] where columns are [MAE RMSE IAD MaxAbs].
    prefix = string(prefix);

    mae  = X(:,1); rmse = X(:,2); iad = X(:,3); mx = X(:,4);

    T.(prefix + "_MAE_mean")(rowIdx)  = mean(mae, "omitnan");
    T.(prefix + "_MAE_qlo")(rowIdx)   = quantile_safe(mae, qlo);
    T.(prefix + "_MAE_qhi")(rowIdx)   = quantile_safe(mae, qhi);

    T.(prefix + "_RMSE_mean")(rowIdx) = mean(rmse, "omitnan");
    T.(prefix + "_RMSE_qlo")(rowIdx)  = quantile_safe(rmse, qlo);
    T.(prefix + "_RMSE_qhi")(rowIdx)  = quantile_safe(rmse, qhi);

    T.(prefix + "_IAD_mean")(rowIdx)  = mean(iad, "omitnan");
    T.(prefix + "_IAD_qlo")(rowIdx)   = quantile_safe(iad, qlo);
    T.(prefix + "_IAD_qhi")(rowIdx)   = quantile_safe(iad, qhi);

    T.(prefix + "_MaxAbs_mean")(rowIdx) = mean(mx, "omitnan");
    T.(prefix + "_MaxAbs_qlo")(rowIdx)  = quantile_safe(mx, qlo);
    T.(prefix + "_MaxAbs_qhi")(rowIdx)  = quantile_safe(mx, qhi);
end

function q = quantile_safe(x, qq)
    % quantile_safe Quantile with NaN/Inf removal; returns NaN if no data remain.
    x = x(isfinite(x));
    if isempty(x), q = NaN; else, q = quantile(x, qq); end
end

% ======================================================================
% Curve summaries (pooling across participants)
% ======================================================================
function S = summarize_curves(E_byP, Acc_byP, qlo, qhi)
    % Each input is [Ns x nP]. Statistics are computed across participants.
    S = struct();
    S.e_mean = mean(E_byP, 2, "omitnan");
    S.e_qlo  = quantile_cols(E_byP, qlo);
    S.e_qhi  = quantile_cols(E_byP, qhi);

    S.acc_mean = mean(Acc_byP, 2, "omitnan");
    S.acc_qlo  = quantile_cols(Acc_byP, qlo);
    S.acc_qhi  = quantile_cols(Acc_byP, qhi);
end

function q = quantile_cols(X, qq)
    % quantile_cols Column-wise quantile per row (robust to NaN/Inf).
    q = NaN(size(X,1),1);
    for i = 1:size(X,1)
        xi = X(i,:);
        xi = xi(isfinite(xi));
        if isempty(xi), q(i) = NaN; else, q(i) = quantile(xi, qq); end
    end
end

% ======================================================================
% Figures (thesis-style export)
% ======================================================================
function make_fig_error_and_accum(pathPdf, pid, t, e, acc, titleStr, Sth)
    % make_fig_error_and_accum Plot mean error and accumulated absolute error on real time grid.
    f = figure('Visible','off');
    thesisStyle(f);
    set(f,'Units','centimeters');
    set(f,'Position',[2 2 Sth.figSizeTrajectoryGrid]);

    % Use two panels within one exported figure file.
    subplot(2,1,1);
    plot(t, e, 'LineWidth', 1.25);
    grid on;
    xlabel('time (s)');
    ylabel('e(t)');
    title(sprintf("Participant %s  %s", string(pid), titleStr), 'Interpreter','none');

    subplot(2,1,2);
    plot(t, acc, 'LineWidth', 1.25);
    grid on;
    xlabel('Time [s]');
    ylabel('$\int |e| \mathrm{d}t$');

    thesisFinalizeFigure(f, Sth);
    thesisExport(f, string(pathPdf));
end

function make_fig_pooled_error(pathPdf, sgrid, S, titleStr, Sth)
    % make_fig_pooled_error Plot pooled error statistics over normalized time.
    f = figure('Visible','off');
    thesisStyle(f);

    h1 = plot(sgrid, S.e_mean, 'LineWidth', 1.5);
    hold on; grid on;
    h2 = plot(sgrid, S.e_qlo, '--', 'LineWidth', 1.0);
    h3 = plot(sgrid, S.e_qhi, '--', 'LineWidth', 1.0);

    % Explicit colors follow existing thesisStyle conventions.
    h1.Color = Sth.colors.cyan;
    h2.Color = Sth.colors.yellow;
    h3.Color = Sth.colors.red;

    xlabel('Normalized time');
    ylabel('Pooled error');
    title(titleStr, 'Interpreter','none');
    legend({'mean','$P_{5}$','$P_{95}$'}, 'Location','best');

    thesisFinalizeFigure(f, Sth);
    thesisExport(f, string(pathPdf));
end

function make_fig_pooled_accum(pathPdf, sgrid, S, titleStr, Sth)
    % make_fig_pooled_accum Plot pooled accumulated absolute error statistics over normalized time.
    f = figure('Visible','off');
    thesisStyle(f);

    h1 = plot(sgrid, S.acc_mean, 'LineWidth', 1.5);
    hold on; grid on;
    h2 = plot(sgrid, S.acc_qlo, '--', 'LineWidth', 1.0);
    h3 = plot(sgrid, S.acc_qhi, '--', 'LineWidth', 1.0);

    h1.Color = Sth.colors.cyan;
    h2.Color = Sth.colors.yellow;
    h3.Color = Sth.colors.red;

    xlabel('Normalized time');
    ylabel('Accumulated pooled error');
    title(titleStr, 'Interpreter','none');
    legend({'mean','$P_{5}$','$P_{95}$'}, 'Location','best');

    thesisFinalizeFigure(f, Sth);
    thesisExport(f, string(pathPdf));
end

% ======================================================================
% Global behavior params (local copy of A9/A12 behavior selection logic)
% ======================================================================
function [name, bpar] = resolve_behavior_params_global(modelIdx, fit)
    % resolve_behavior_params_global Convert an A8 fit struct into behavior params for the simulator.
    bpar = struct();
    bpar.tau_flag = 0;
    bpar.m1_flag  = 0;
    bpar.m2_flag  = 0;

    switch modelIdx
        case 0
            name = "model0_trust_as_probability";
            bpar.tau_flag = 1;

        case 1
            name = "model1_threshold";
            assert(isfield(fit,"model1") && isfield(fit.model1,"k_hat"), "[A13] fit.model1.k_hat missing.");
            bpar.m1_flag = 1;
            bpar.k_m1 = fit.model1.k_hat;

        case 2
            name = "model2_offset_lapse";
            assert(isfield(fit,"model2"), "[A13] fit.model2 missing.");
            bpar.m2_flag = 1;
            bpar.k_m2 = fit.model2.k_hat;
            bpar.beta = fit.model2.beta_hat;
            bpar.eps  = fit.model2.eps_hat;

        otherwise
            error("[A13] Unknown model idx: %d", modelIdx);
    end
end

% ======================================================================
% Personalized behavior params resolver (kept consistent with A12)
% ======================================================================
function [bpar, info] = resolve_personalized_behavior_params(pid, Ta10, Tblocks, useA11Guards, epsMax, fallbackStrategy)
    % resolve_personalized_behavior_params Select per-participant behavior model and parameters.
    %
    % Selection begins from A10 best_model_idx (stored as 1..3, mapped to 0..2).
    % Guardrails and parameter validity checks can trigger fallback degradation:
    %   model 2 -> model 1 -> model 0
    %
    pid = string(pid);
    fallbackStrategy = string(fallbackStrategy);

    idx = find(string(Ta10.participant_id)==pid, 1);
    if isempty(idx)
        error("[A13] pid=%s not found in A10 table.", pid);
    end

    bestIdx_a10 = double(Ta10.best_model_idx(idx));  % A10 uses 1..3
    modelA10 = bestIdx_a10 - 1;                      % internal 0..2

    info = struct();
    info.model_idx_a10  = modelA10;
    info.model_name_a10 = model_name_from_idx(modelA10);

    % Extract model-specific parameters from A10 (with flexible column naming).
    switch modelA10
        case 0
            kA10 = NaN; betaA10 = NaN; epsA10 = NaN;

        case 1
            kA10 = get_first_existing_numeric(Ta10, idx, ["m1_k_hat","k_hat","k_mle","k","best_k","k_best"]);
            betaA10 = NaN; epsA10 = NaN;

        case 2
            kA10    = get_first_existing_numeric(Ta10, idx, ["m2_k_hat","k_hat","k_mle","k","best_k","k_best"]);
            betaA10 = get_first_existing_numeric(Ta10, idx, ["m2_beta_hat","beta_hat","beta_mle","beta","best_beta","beta_best"]);
            epsA10  = get_first_existing_numeric(Ta10, idx, ["m2_eps_hat","eps_hat","eps_mle","eps","epsilon_hat","epsilon","best_eps","eps_best"]);

        otherwise
            error("[A13] Unexpected modelA10=%g after mapping from best_model_idx.", modelA10);
    end

    % Under-identified flag from A11 (if available).
    underIdent = false;
    if useA11Guards && ~isempty(Tblocks) && ismember("participant_id", string(Tblocks.Properties.VariableNames))
        j = find(string(Tblocks.participant_id)==pid, 1);
        if ~isempty(j)
            if ismember("flag_under_identified", string(Tblocks.Properties.VariableNames))
                underIdent = logical(Tblocks.flag_under_identified(j));
            elseif ismember("drift_label", string(Tblocks.Properties.VariableNames))
                underIdent = (string(Tblocks.drift_label(j))=="under_identified");
            end
        end
    end

    % Basic parameter sanity checks (model 2 has additional constraints).
    bad2 = ~isfinite(kA10) || kA10 < 0 || ~isfinite(betaA10) || ~isfinite(epsA10) || epsA10 < 0 || epsA10 > epsMax;
    bad1 = ~isfinite(kA10) || kA10 < 0;

    % Decide model_used with fallback degradation.
    modelUsed = modelA10;
    fallback_applied = false;
    reason = "";

    if fallbackStrategy ~= "simple"
        warning("[A13] Unknown FallbackStrategy=%s. Using 'simple'.", fallbackStrategy);
    end

    if useA11Guards && underIdent
        fallback_applied = true;
        reason = "A11_under_identified";
        if modelUsed == 2, modelUsed = 1; elseif modelUsed == 1, modelUsed = 0; end
    end

    if modelUsed == 2 && bad2
        fallback_applied = true;
        if strlength(reason)==0, reason="A10_params_invalid_model2"; else, reason=reason + ";A10_params_invalid_model2"; end
        modelUsed = 1;
    end
    if modelUsed == 1 && bad1
        fallback_applied = true;
        if strlength(reason)==0, reason="A10_params_invalid_model1"; else, reason=reason + ";A10_params_invalid_model1"; end
        modelUsed = 0;
    end

    % Build behavior parameter struct expected by the simulator.
    bpar = struct();
    bpar.tau_flag = 0;
    bpar.m1_flag  = 0;
    bpar.m2_flag  = 0;

    switch modelUsed
        case 0
            bpar.tau_flag = 1;

        case 1
            bpar.m1_flag = 1;
            bpar.k_m1 = kA10;

        case 2
            bpar.m2_flag = 1;
            bpar.k_m2 = kA10;
            bpar.beta = betaA10;
            bpar.eps  = epsA10;

        otherwise
            error("[A13] Unknown model idx: %d", modelUsed);
    end

    info.model_idx_used     = modelUsed;
    info.model_name_used    = model_name_from_idx(modelUsed);
    info.fallback_applied   = fallback_applied;
    info.fallback_reason    = string(reason);
end

function v = get_first_existing_numeric(T, rowIdx, candidates)
    % get_first_existing_numeric Return the first matching numeric column value from a list of candidates.
    v = NaN;
    candidates = string(candidates);
    for c = candidates
        if ismember(c, string(T.Properties.VariableNames))
            x = T.(c);
            if isnumeric(x)
                v = double(x(rowIdx));
                return;
            end
        end
    end
end

% ======================================================================
% Load theta_star, dt, and VALID participants (run-local A1/A3)
% ======================================================================
function [theta_star, dt, participants_valid] = local_load_theta_dt_and_valid_participants_like_A5(run_id)
    run_id = string(run_id);

    % --- A1 archived inputs (VALID participants) ---
    a1Dir = fullfile("derived", "analysis_runs", run_id, "stepA1_prepare_analysis");
    manifestPath = fullfile(a1Dir, "manifest.mat");
    must_exist_file(manifestPath, "A1 manifest");

    validPath = fullfile(a1Dir, "participants_valid_probes_mapped_stepM4.mat");
    must_exist_file(validPath, "A1 VALID participants (mapped probes)");
    participants_valid = load_participants_struct(validPath);

    % --- A3 selection -> resultsMatPath and theta_star ---
    selPath = fullfile("derived","analysis_runs",run_id,"stepA3_model_selection","selection.mat");
    must_exist_file(selPath, "A3 selection.mat");

    Ssel = load(selPath, "selection");
    if ~isfield(Ssel,"selection") || ~isstruct(Ssel.selection)
        error("[A13] A3 selection.mat missing variable 'selection'.");
    end
    selection = Ssel.selection;

    % theta_star (preferred from selection struct, fallback to theta_star.mat)
    theta_star = [];
    if isfield(selection,"theta_star") && ~isempty(selection.theta_star)
        theta_star = selection.theta_star(:);
    else
        thetaPath = fullfile("derived","analysis_runs",run_id,"stepA3_model_selection","theta_star.mat");
        if isfile(thetaPath)
            Sth = load(thetaPath);
            theta_star = find_theta_in_struct(Sth);
        end
    end
    if isempty(theta_star)
        error("[A13] Could not resolve theta_star from A3 selection.");
    end

    % results file -> cfg.dt (required for consistent simulation timing)
    if ~isfield(selection,"results_file") || isempty(selection.results_file)
        error("[A13] selection.results_file missing. Cannot locate cfg.dt.");
    end
    resultsMatPath = string(selection.results_file);
    must_exist_file(resultsMatPath, "Fit results MAT (selection.results_file)");

    R = load(resultsMatPath);
    if ~isfield(R,"cfg") || ~isstruct(R.cfg) || ~isfield(R.cfg,"dt") || isempty(R.cfg.dt)
        error("[A13] results MAT does not contain cfg.dt: %s", resultsMatPath);
    end
    dt = double(R.cfg.dt);
    if ~isscalar(dt) || ~isfinite(dt) || dt <= 0
        error("[A13] cfg.dt invalid in results MAT: %s", resultsMatPath);
    end
end

% ======================================================================
% Helpers: participants / tables / alignment / rollout store
% ======================================================================
function ids = local_get_participant_ids(validParticipants)
    % local_get_participant_ids Return sorted participant IDs from supported container types.
    if isa(validParticipants, "containers.Map")
        ks = validParticipants.keys;
        ids = string(ks(:));
        ids = sort(ids);
        return;
    end
    if isstruct(validParticipants)
        if ~isfield(validParticipants,"participant_id")
            error("[A13] validParticipants struct missing participant_id field.");
        end
        ids = string({validParticipants.participant_id})';
        ids = sort(ids);
        return;
    end
    if istable(validParticipants)
        if ~ismember("participant_id", string(validParticipants.Properties.VariableNames))
            error("[A13] validParticipants table missing participant_id.");
        end
        ids = string(validParticipants.participant_id);
        ids = sort(ids);
        return;
    end
    error("[A13] Unsupported validParticipants type: %s", class(validParticipants));
end

function T = local_find_first_table(S, preferNames)
    % local_find_first_table Find the first table in struct S, preferring specific field names.
    T = [];
    preferNames = string(preferNames);
    for nm = preferNames
        if isfield(S, nm) && istable(S.(nm))
            T = S.(nm);
            return;
        end
    end
    fn = fieldnames(S);
    for k = 1:numel(fn)
        if istable(S.(fn{k}))
            T = S.(fn{k});
            return;
        end
    end
end

function x = local_align_length(x, ref)
    % local_align_length Align a vector to the length of a reference vector (trim or pad).
    x = double(x(:));
    n = numel(ref);
    if numel(x) == n
        return;
    end
    if numel(x) > n
        x = x(1:n);
    else
        % Pad with last value to maintain continuity for short outputs.
        if isempty(x)
            x = ref; % defensive worst-case fallback
        else
            x(end+1:n,1) = x(end);
        end
    end
end

function M = init_rollout_metric_store(nP, R)
    % init_rollout_metric_store Preallocate rollout metric arrays [nP x R].
    M = struct();
    M.MAE = NaN(nP,R);
    M.RMSE = NaN(nP,R);
    M.IAD = NaN(nP,R);
    M.MaxAbs = NaN(nP,R);
end

% ======================================================================
% Participant collection access (supports containers.Map, struct arrays, and tables)
% ======================================================================
function Pp = get_participant_from_collection(collection, pid)
    pid = string(pid);

    if isa(collection, "containers.Map")
        if ~isKey(collection, char(pid))
            error("[A13] VALID participant '%s' not found in collection Map.", pid);
        end
        Pp = collection(char(pid));
        return;
    end

    if isstruct(collection)
        if ~isfield(collection, "participant_id")
            error("[A13] validParticipants struct must have field participant_id.");
        end
        ids = string({collection.participant_id});
        idx = find(ids==pid, 1);
        if isempty(idx), error("[A13] VALID participant '%s' not found in struct collection.", pid); end
        Pp = collection(idx);
        return;
    end

    if istable(collection)
        if ~ismember("participant_id", string(collection.Properties.VariableNames))
            error("[A13] validParticipants table must have participant_id column.");
        end
        idx = find(string(collection.participant_id)==pid, 1);
        if isempty(idx), error("[A13] VALID participant '%s' not found in table collection.", pid); end
        Pp = collection(idx,:);
        return;
    end

    error("[A13] Unsupported validParticipants type: %s", class(collection));
end

function s = sanitize_pid(pid)
    % sanitize_pid Make participant ID safe for filenames by replacing non-alphanumerics.
    pid = char(string(pid));
    s = regexprep(pid, '[^a-zA-Z0-9_-]', '_');
end

% ======================================================================
% Model name mapping
% ======================================================================
function name = model_name_from_idx(modelIdx)
    % model_name_from_idx Human-readable model name for logging and audit trails.
    switch modelIdx
        case 0, name = "model0_trust_as_probability";
        case 1, name = "model1_threshold";
        case 2, name = "model2_offset_lapse";
        otherwise, name = "unknown";
    end
end
