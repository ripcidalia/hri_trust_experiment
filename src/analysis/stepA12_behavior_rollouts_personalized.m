function stepA12_behavior_rollouts_personalized(run_id, varargin)
% stepA12_behavior_rollouts_personalized Personalized coupled rollouts on VALID participants.
%
% This step evaluates generative realism of the coupled trust+behavior simulator
% using per-participant behavioral parameters fitted on VALID data (A10), with
% optional robustness guardrails (A11). For each VALID participant, the function:
%   - Extracts the observed follow/override door sequence from the A7 VALID dataset
%   - Computes override-centric interaction "signature" statistics on the observed
%     binary sequence (and optionally per-block rates if block_index is present)
%   - Resolves a per-participant behavioral model (0/1/2) and parameters from A10,
%     optionally applying A11-based guardrails and sanity checks with a fallback rule
%   - Runs coupled rollouts via trust_simulate_or_predict_one_participant("coupled", ...)
%   - Computes the same signature statistics on each rollout and summarizes the
%     simulated distribution using quantiles
%   - Compares observed vs simulated summary statistics per participant and pooled
%
% Guardrail fallback (enabled by default)
%   If A11 marks a participant as under-identified, or if A10 parameters are invalid,
%   the used model is degraded using the selected strategy (currently "simple"):
%     Model 2 -> Model 1 -> Model 0
%   Fallback decisions are recorded in the output table.
%
% INPUTS (required)
%   run_id (string|char)
%       Analysis run identifier. Used to locate required artifacts under:
%         derived/analysis_runs/<run_id>/
%
% NAME-VALUE ARGUMENTS (optional)
%   "OutDir" (string|char)
%       Output directory. Default:
%         derived/analysis_runs/<run_id>/stepA12_behavior_rollouts_personalized
%
%   "Overwrite" (logical scalar)
%       If false (default), error if outputs already exist.
%
%   "RolloutsPerParticipant" (numeric scalar)
%       Number of coupled rollouts per participant (default: 1000).
%
%   "Quantiles" (numeric 1x2)
%       Lower/upper quantiles for simulated summaries (default: [0.05 0.95]).
%
%   "RandomSeed" (numeric scalar)
%       Base RNG seed used for rollouts (default: 1).
%
%   "UseA11Guards" (logical scalar)
%       If true (default), use A11 under-identification flags when available.
%
%   "FallbackStrategy" (string|char)
%       Fallback policy identifier (default: "simple"). Currently only:
%         "simple" -> 2 -> 1 -> 0
%
%   "EpsMax" (numeric scalar)
%       Maximum lapse probability used for sanity checks in Model 2 (default: 0.5).
%
% INPUT FILES (reads)
%   A7 VALID dataset:
%     derived/analysis_runs/<run_id>/stepA7_behavior_dataset/behavior_dataset_valid.mat
%       - expects table T with at least: participant_id, followed, door_index, is_valid_label
%       - block_index is optional (enables blockwise rates in signatures)
%
%   A10 per-participant fits:
%     derived/analysis_runs/<run_id>/stepA10_behavior_fit_by_participant/A10_params_by_participant.mat
%       - expects a summary table (preferably Tsum) with: participant_id, best_model_idx
%       - expects parameter columns consistent with A10 naming (m1_k_hat, m2_k_hat, m2_beta_hat, m2_eps_hat)
%
%   A11 robustness (optional):
%     derived/analysis_runs/<run_id>/stepA11_behavior_param_robustness/A11_blockwise_params.mat
%       - expects table Tblocks with participant_id and either:
%           flag_under_identified (logical) OR drift_label == "under_identified"
%
% OUTPUTS (writes)
%   derived/analysis_runs/<run_id>/stepA12_behavior_rollouts_personalized/
%     - A12_rollout_stats_personalized.csv
%     - A12_rollout_stats_personalized.mat  (roll + sig + outTable)
%     - meta.mat / meta.json
%     - figures/*.png
%     - figures/by_participant/*.png
%
% NOTES
%   - Signature statistics are computed on sampled binary sequences (follow/override),
%     emphasizing override-centric patterns and transitions.
%   - This step targets generative realism, not predictive generalization.
%
% DEPENDENCIES (assumed on MATLAB path)
%   must_exist_file, ensure_dir, save_json, load_participants_struct, find_theta_in_struct
%   trust_simulate_or_predict_one_participant, thesisStyle, thesisFinalizeFigure, thesisExport

    % ------------------------------------------------------------------
    % Parse required input and name-value arguments
    % ------------------------------------------------------------------
    if nargin < 1 || isempty(run_id)
        error("stepA12_behavior_rollouts_personalized: run_id is required.");
    end
    run_id = string(run_id);

    p = inputParser;
    p.addParameter("OutDir", "", @(s) isstring(s) || ischar(s));
    p.addParameter("Overwrite", false, @(x) islogical(x) && isscalar(x));
    p.addParameter("RolloutsPerParticipant", 1000, @(x) isnumeric(x) && isscalar(x) && x>=1);
    p.addParameter("Quantiles", [0.05 0.95], @(x) isnumeric(x) && numel(x)==2 && all(x>=0) && all(x<=1));
    p.addParameter("RandomSeed", 1, @(x) isnumeric(x) && isscalar(x));
    p.addParameter("UseA11Guards", true, @(x) islogical(x) && isscalar(x));
    p.addParameter("FallbackStrategy", "simple", @(s) isstring(s) || ischar(s));
    p.addParameter("EpsMax", 0.5, @(x) isnumeric(x) && isscalar(x) && x>0 && x<=1);
    p.parse(varargin{:});
    args = p.Results;

    rng(args.RandomSeed);

    % Thesis plotting defaults + style struct
    S = thesisStyle();

    % ------------------------------------------------------------------
    % Locate inputs
    % ------------------------------------------------------------------
    a7Dir = fullfile("derived","analysis_runs",run_id,"stepA7_behavior_dataset");
    validMatA7 = fullfile(a7Dir, "behavior_dataset_valid.mat");
    must_exist_file(validMatA7, "A7 VALID dataset");

    a10Dir = fullfile("derived","analysis_runs",run_id,"stepA10_behavior_fit_by_participant");
    a10Mat = fullfile(a10Dir, "A10_params_by_participant.mat");
    must_exist_file(a10Mat, "A10_params_by_participant.mat");

    a11Dir = fullfile("derived","analysis_runs",run_id,"stepA11_behavior_param_robustness");
    a11Mat = fullfile(a11Dir, "A11_blockwise_params.mat");
    haveA11 = isfile(a11Mat);

    % ------------------------------------------------------------------
    % Load A7 VALID observed door sequences
    % ------------------------------------------------------------------
    Sva = load(validMatA7, "T");
    if ~isfield(Sva,"T") || ~istable(Sva.T)
        error("[A12] VALID mat missing table T.");
    end
    Tva = Sva.T;

    reqCols = ["participant_id","followed","door_index"];
    assert(all(ismember(reqCols, string(Tva.Properties.VariableNames))), ...
        "[A12] A7 VALID missing required columns.");

    haveBlockA7 = ismember("block_index", string(Tva.Properties.VariableNames));

    % Keep only valid-labeled doors
    Tva = Tva(Tva.is_valid_label==1, :);

    pid_va = string(Tva.participant_id);
    uniqP  = unique(pid_va);
    nP     = numel(uniqP);

    % Build observed signatures per participant (sequence -> signature struct)
    obsStats = table();
    obsStats.participant_id = uniqP;

    for i = 1:nP
        pid = uniqP(i);
        mask = (pid_va == pid);
        Tpi = Tva(mask, :);

        % Preserve temporal order by door_index
        [~,ord] = sort(double(Tpi.door_index(:)));
        Tpi = Tpi(ord,:);

        seqFollow = double(Tpi.followed(:));  % 1=follow, 0=override
        if haveBlockA7
            seqBlock = double(Tpi.block_index(:));
        else
            seqBlock = NaN(size(seqFollow));
        end

        st = compute_signatures(seqFollow, seqBlock);
        obsStats = set_row_struct(obsStats, i, "obs_", st);
    end

    % ------------------------------------------------------------------
    % Load A10 summary table (best model + fitted params)
    % ------------------------------------------------------------------
    Sa10 = load(a10Mat);

    % Prefer Tsum if present; otherwise attempt to locate any table in MAT
    if isfield(Sa10,"Tsum") && istable(Sa10.Tsum)
        Ta10 = Sa10.Tsum;
    elseif isfield(Sa10,"T") && istable(Sa10.T)
        Ta10 = Sa10.T;
    else
        Ta10 = [];
        fn = fieldnames(Sa10);
        for k = 1:numel(fn)
            if istable(Sa10.(fn{k}))
                Ta10 = Sa10.(fn{k});
                break;
            end
        end
        if isempty(Ta10)
            error("[A12] Could not find A10 summary table in %s (expected Tsum).", a10Mat);
        end
    end

    if ~ismember("participant_id", string(Ta10.Properties.VariableNames)) || ...
       ~ismember("best_model_idx", string(Ta10.Properties.VariableNames))
        error("[A12] A10 table must contain participant_id and best_model_idx.");
    end

    % ------------------------------------------------------------------
    % Load A11 blockwise table (optional guardrails)
    % ------------------------------------------------------------------
    Tblocks = table();
    if haveA11
        Sa11 = load(a11Mat);
        if isfield(Sa11,"Tblocks") && istable(Sa11.Tblocks)
            Tblocks = Sa11.Tblocks;
        elseif isfield(Sa11,"T") && istable(Sa11.T)
            Tblocks = Sa11.T;
        else
            % If expected table is absent, guardrails fall back to parameter sanity checks
            Tblocks = table();
        end
    end

    % ------------------------------------------------------------------
    % Load theta_star, dt, and VALID participants (consistent with earlier steps)
    % ------------------------------------------------------------------
    [theta_star, dt, validParticipants] = local_load_theta_dt_and_valid_participants_like_A5(run_id);

    % ------------------------------------------------------------------
    % Output directories and overwrite guard
    % ------------------------------------------------------------------
    outDir = string(args.OutDir);
    if strlength(outDir)==0
        outDir = fullfile("derived","analysis_runs",run_id,"stepA12_behavior_rollouts_personalized");
    end
    ensure_dir(outDir);

    figDir = fullfile(outDir, "figures");
    ensure_dir(figDir);

    figByP = fullfile(figDir, "by_participant");
    ensure_dir(figByP);

    statsCsv = fullfile(outDir, "A12_rollout_stats_personalized.csv");
    statsMat = fullfile(outDir, "A12_rollout_stats_personalized.mat");
    metaMat  = fullfile(outDir, "meta.mat");
    metaJson = fullfile(outDir, "meta.json");

    if ~args.Overwrite
        if isfile(statsCsv) || isfile(statsMat)
            error("[A12] Outputs exist. Set Overwrite=true to replace. (%s)", outDir);
        end
    end

    % ------------------------------------------------------------------
    % Rollouts (coupled simulation with personalized behavior)
    % ------------------------------------------------------------------
    R   = double(args.RolloutsPerParticipant);
    qlo = double(args.Quantiles(1));
    qhi = double(args.Quantiles(2));

    roll = struct();
    roll.meta = struct( ...
        "run_id",char(run_id), ...
        "R",R, ...
        "qlo",qlo, ...
        "qhi",qhi, ...
        "random_seed",args.RandomSeed, ...
        "dt",dt, ...
        "use_a11_guards",args.UseA11Guards, ...
        "fallback_strategy",char(string(args.FallbackStrategy)));

    roll.obsStats = obsStats;

    % Store per-rollout signature values (nP x R per field)
    sig = init_sig_store(nP, R);

    % Bookkeeping: which model/params were requested (A10) vs actually used (after guards)
    bk = table();
    bk.participant_id = uniqP;
    bk.model_idx_a10 = NaN(nP,1);
    bk.model_name_a10 = strings(nP,1);
    bk.model_idx_used = NaN(nP,1);
    bk.model_name_used = strings(nP,1);
    bk.k_used = NaN(nP,1);
    bk.beta_used = NaN(nP,1);
    bk.eps_used = NaN(nP,1);
    bk.fallback_applied = false(nP,1);
    bk.fallback_reason = strings(nP,1);

    fprintf("[A12] Personalized coupled rollouts: R=%d, participants=%d\n", R, nP);

    for pi = 1:nP
        pid = uniqP(pi);

        % Retrieve participant struct/table row used by simulator
        Pp  = get_participant_from_collection(validParticipants, pid);

        % Resolve per-participant behavioral model and parameters (with optional guards)
        [bpar, info] = resolve_personalized_behavior_params( ...
            pid, Ta10, Tblocks, args.UseA11Guards, args.EpsMax, args.FallbackStrategy);

        bk.model_idx_a10(pi)    = info.model_idx_a10;
        bk.model_name_a10(pi)   = info.model_name_a10;
        bk.model_idx_used(pi)   = info.model_idx_used;
        bk.model_name_used(pi)  = info.model_name_used;
        bk.k_used(pi)           = info.k_used;
        bk.beta_used(pi)        = info.beta_used;
        bk.eps_used(pi)         = info.eps_used;
        bk.fallback_applied(pi) = info.fallback_applied;
        bk.fallback_reason(pi)  = info.fallback_reason;

        % Run R coupled rollouts with deterministic per-rollout seeding
        for r = 1:R
            seed = args.RandomSeed + 1000*pi + r;
            rng(seed);

            sim = trust_simulate_or_predict_one_participant("coupled", theta_star, Pp, dt, bpar);

            if ~isfield(sim, "coupled") || ~isfield(sim.coupled, "followed_sampled")
                error("[A12] Simulator did not return sim.coupled.followed_sampled in coupled mode (pid=%s).", pid);
            end

            [seqFollow, seqBlock] = coupled_sim_to_sequences(sim);
            st = compute_signatures(seqFollow, seqBlock);

            sig = store_sig(sig, pi, r, st);
        end
    end

    % Summarize simulated signature distributions per participant
    simSumm = summarize_sig_store(sig, qlo, qhi);

    % Combine: simulated summaries + observed stats + bookkeeping
    simSumm.participant_id = uniqP;
    outTable = join(simSumm, obsStats, "Keys","participant_id");
    outTable = join(outTable, bk, "Keys","participant_id");

    % Append pooled row (mean across participant rows for numeric fields)
    pooled = pooled_summary_personalized(outTable);
    outTable = [outTable; pooled];

    % ------------------------------------------------------------------
    % Save artifacts
    % ------------------------------------------------------------------
    writetable(outTable, statsCsv);
    save(statsMat, "roll", "sig", "outTable", "-v7.3");

    meta = struct();
    meta.run_id = char(run_id);
    meta.created = char(datetime('now','Format','yyyy-MM-dd HH:mm:ss'));
    meta.valid_participants = nP;
    meta.rollouts_per_participant = R;
    meta.quantiles = [qlo qhi];
    meta.random_seed = args.RandomSeed;
    meta.dt = dt;
    meta.have_block_index_in_A7 = haveBlockA7;
    meta.used_a11_guards = args.UseA11Guards && haveA11;
    meta.have_a11_file = haveA11;
    meta.a10_file = char(a10Mat);
    meta.a11_file = char(a11Mat);
    meta.fallback_strategy = char(string(args.FallbackStrategy));

    save(metaMat, "meta");
    save_json(metaJson, meta);

    % ------------------------------------------------------------------
    % Figures (pooled + per participant)
    % ------------------------------------------------------------------
    make_fig_pooled_personalized_rates(fullfile(figDir, "pooled_override_rate"), outTable, S);
    make_fig_pooled_personalized_switch(fullfile(figDir, "pooled_switch_rates"), outTable, S);
    make_fig_pooled_personalized_gap(fullfile(figDir, "pooled_inter_override_gap"), outTable, S);
    make_fig_pooled_personalized_streak(fullfile(figDir, "pooled_override_streak"), outTable, S);

    % Per-participant comparison figures for a small set of key metrics
    for pi = 1:nP
        pid = uniqP(pi);
        row = outTable(string(outTable.participant_id)==pid, :);
        if isempty(row), continue; end

        make_fig_participant_errorbar( ...
            fullfile(figByP, sprintf("pid_%s_compare_override_rate", sanitize_pid(pid))), ...
            pid, "override_rate", row, S);

        make_fig_participant_errorbar( ...
            fullfile(figByP, sprintf("pid_%s_compare_inter_override_gap_mean", sanitize_pid(pid))), ...
            pid, "inter_override_gap_mean", row, S);

        make_fig_participant_errorbar( ...
            fullfile(figByP, sprintf("pid_%s_compare_p_follow_to_override", sanitize_pid(pid))), ...
            pid, "p_follow_to_override", row, S);
    end

    fprintf("[Step A12] Done.\n");
    fprintf("  Output dir: %s\n", outDir);
    fprintf("  Wrote: %s\n", statsCsv);
end

% ======================================================================
% Personalized behavior params resolver (A10 + optional A11 guardrails)
% ======================================================================
function [bpar, info] = resolve_personalized_behavior_params(pid, Ta10, Tblocks, useA11Guards, epsMax, fallbackStrategy)
% resolve_personalized_behavior_params Resolve per-participant behavior model and parameters.
%
% This helper selects the behavioral model recommended by A10 (best_model_idx),
% reads its fitted parameters from A10 columns, optionally applies A11-based
% guardrails and parameter sanity checks, and returns the behavior parameter
% struct expected by the coupled simulator.
%
% INPUTS
%   pid              - participant identifier (string/char)
%   Ta10             - A10 summary table containing best_model_idx and parameter columns
%   Tblocks          - (optional) A11 blockwise table with under-identification signals
%   useA11Guards     - whether to use A11 guardrails when available
%   epsMax           - upper bound for Model 2 lapse (used for sanity checks)
%   fallbackStrategy - strategy identifier ("simple" supported)
%
% OUTPUTS
%   bpar - struct passed into trust_simulate_or_predict_one_participant("coupled", ...)
%   info - struct with bookkeeping fields (A10 model, used model, fallback reason)

    pid = string(pid);
    fallbackStrategy = string(fallbackStrategy);

    idx = find(string(Ta10.participant_id)==pid, 1);
    if isempty(idx)
        error("[A12] pid=%s not found in A10 table.", pid);
    end

    bestIdx_a10 = double(Ta10.best_model_idx(idx));  % A10 uses 1..3
    modelA10 = bestIdx_a10 - 1;                      % internal 0..2

    info = struct();
    info.model_idx_a10  = modelA10;
    info.model_name_a10 = model_name_from_idx(modelA10);

    % Read A10 parameters using known column names, with conservative fallbacks.
    % Model 0 has no parameters.
    switch modelA10
        case 0
            kA10 = NaN; betaA10 = NaN; epsA10 = NaN;

        case 1
            kA10 = get_first_existing_numeric(Ta10, idx, ...
                ["m1_k_hat","k_hat","k_mle","k","best_k","k_best"]);
            betaA10 = NaN; epsA10 = NaN;

        case 2
            kA10    = get_first_existing_numeric(Ta10, idx, ...
                ["m2_k_hat","k_hat","k_mle","k","best_k","k_best"]);
            betaA10 = get_first_existing_numeric(Ta10, idx, ...
                ["m2_beta_hat","beta_hat","beta_mle","beta","best_beta","beta_best"]);
            epsA10  = get_first_existing_numeric(Ta10, idx, ...
                ["m2_eps_hat","eps_hat","eps_mle","eps","epsilon_hat","epsilon","best_eps","eps_best"]);

        otherwise
            error("[A12] Unexpected modelA10=%g after mapping from best_model_idx.", modelA10);
    end

    % Under-identified flag from A11 (if available)
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

    % Parameter sanity checks (used to trigger fallbacks)
    bad2 = ~isfinite(kA10) || kA10 < 0 || ~isfinite(betaA10) || ~isfinite(epsA10) || epsA10 < 0 || epsA10 > epsMax;
    bad1 = ~isfinite(kA10) || kA10 < 0;

    % Decide model_used with fallback
    modelUsed = modelA10;
    fallback_applied = false;
    reason = "";

    if fallbackStrategy ~= "simple"
        warning("[A12] Unknown FallbackStrategy=%s. Using 'simple'.", fallbackStrategy);
    end

    % A11 under-identification triggers a degradation by one level (2->1 or 1->0)
    if useA11Guards && underIdent
        fallback_applied = true;
        reason = "A11_under_identified";
        if modelUsed == 2
            modelUsed = 1;
        elseif modelUsed == 1
            modelUsed = 0;
        end
    end

    % Invalid parameters trigger additional degradation
    if modelUsed == 2 && bad2
        fallback_applied = true;
        if strlength(reason)==0
            reason = "A10_params_invalid_model2";
        else
            reason = reason + ";A10_params_invalid_model2";
        end
        modelUsed = 1;
    end
    if modelUsed == 1 && bad1
        fallback_applied = true;
        if strlength(reason)==0
            reason = "A10_params_invalid_model1";
        else
            reason = reason + ";A10_params_invalid_model1";
        end
        modelUsed = 0;
    end

    % Build behavior parameter struct expected by behavioral_model() in coupled mode
    bpar = struct();
    bpar.tau_flag = 0;
    bpar.m1_flag  = 0;
    bpar.m2_flag  = 0;

    k_used = NaN; beta_used = NaN; eps_used = NaN;

    switch modelUsed
        case 0
            bpar.tau_flag = 1;

        case 1
            bpar.m1_flag = 1;
            bpar.k_m1 = kA10; % Model 1 slope parameter
            k_used = kA10;

        case 2
            bpar.m2_flag = 1;
            bpar.k_m2 = kA10;
            bpar.beta = betaA10;
            bpar.eps  = epsA10;
            k_used = kA10; beta_used = betaA10; eps_used = epsA10;

        otherwise
            error("[A12] Unknown model idx: %d", modelUsed);
    end

    info.model_idx_used = modelUsed;
    info.model_name_used = model_name_from_idx(modelUsed);
    info.k_used = k_used;
    info.beta_used = beta_used;
    info.eps_used = eps_used;
    info.fallback_applied = fallback_applied;
    info.fallback_reason = string(reason);
end

function v = get_first_existing_numeric(T, rowIdx, candidates)
% get_first_existing_numeric Return the first finite numeric value among candidate columns.
%
% If none of the candidate variable names exist (or are non-numeric), NaN is returned.
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
% Load theta_star, dt, and VALID participants (consistent with earlier steps)
% ======================================================================
function [theta_star, dt, participants_valid] = local_load_theta_dt_and_valid_participants_like_A5(run_id)
% local_load_theta_dt_and_valid_participants_like_A5 Load coupled simulator inputs.
%
% This helper loads:
%   - participants_valid: archived VALID participant structs from Step A1
%   - theta_star: selected SIMPLE trust parameters from Step A3
%   - dt: simulation time step from the results MAT referenced by Step A3
%
% The intent is to ensure Step A12 uses the same artifacts and time step as the
% fitted trust model pipeline.

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
        error("[A12] A3 selection.mat missing variable 'selection'.");
    end
    selection = Ssel.selection;

    % theta_star (prefer selection.theta_star, otherwise theta_star.mat)
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
        error("[A12] Could not resolve theta_star from A3 selection.");
    end

    % results file -> cfg.dt
    if ~isfield(selection,"results_file") || isempty(selection.results_file)
        error("[A12] selection.results_file missing. Cannot locate cfg.dt.");
    end
    resultsMatPath = string(selection.results_file);
    must_exist_file(resultsMatPath, "Fit results MAT (selection.results_file)");

    R = load(resultsMatPath);
    if ~isfield(R,"cfg") || ~isstruct(R.cfg) || ~isfield(R.cfg,"dt") || isempty(R.cfg.dt)
        error("[A12] results MAT does not contain cfg.dt: %s", resultsMatPath);
    end
    dt = double(R.cfg.dt);
    if ~isscalar(dt) || ~isfinite(dt) || dt <= 0
        error("[A12] cfg.dt invalid in results MAT: %s", resultsMatPath);
    end
end

% ======================================================================
% Convert coupled sim output -> sequences (robust to missing indices)
% ======================================================================
function [seqFollow, seqBlock] = coupled_sim_to_sequences(sim)
% coupled_sim_to_sequences Extract follow sequence (and optional block indices) from coupled sim output.
%
% The coupled simulator is expected to return:
%   sim.coupled.followed_sampled  (binary sequence; 1=follow, 0=override)
% Optional fields are used when present to enforce door ordering:
%   sim.coupled.door_index, sim.coupled.block_index

    seqFollow = double(sim.coupled.followed_sampled(:));  % 1=follow, 0=override

    % If door_index exists, sort by it and reorder all available sequences consistently
    if isfield(sim.coupled, "door_index") && ~isempty(sim.coupled.door_index) && ...
            any(isfinite(double(sim.coupled.door_index(:))))
        doorIdx = double(sim.coupled.door_index(:));
        [~,ord] = sort(doorIdx);
        seqFollow = seqFollow(ord);

        if isfield(sim.coupled, "block_index") && ~isempty(sim.coupled.block_index)
            seqBlock = double(sim.coupled.block_index(:));
            seqBlock = seqBlock(ord);
        else
            seqBlock = NaN(size(seqFollow));
        end
    else
        % Fall back to the natural simulator order
        if isfield(sim.coupled, "block_index") && ~isempty(sim.coupled.block_index)
            seqBlock = double(sim.coupled.block_index(:));
        else
            seqBlock = NaN(size(seqFollow));
        end
    end

    % Defensive: keep the pipeline alive if simulator emits NaNs
    if any(isnan(seqFollow))
        seqFollow(isnan(seqFollow)) = 1;
    end
end

% ======================================================================
% Signatures (override-centric + transitions)
% ======================================================================
function st = compute_signatures(seqFollow, seqBlock)
% compute_signatures Compute interaction signature statistics from a binary sequence.
%
% INPUTS
%   seqFollow - binary vector (1=follow, 0=override)
%   seqBlock  - optional block index vector (NaN if unavailable)
%
% OUTPUT
%   st - struct of scalar signature statistics (rates, transitions, gaps, streaks)

    seqFollow = double(seqFollow(:));
    seqOverride = 1 - seqFollow;

    n = numel(seqFollow);
    if n==0
        st = empty_sig();
        return;
    end

    st.N_doors = n;
    st.follow_rate = mean(seqFollow);
    st.override_rate = mean(seqOverride);

    % Blockwise rates when block indices are available
    if all(isnan(seqBlock))
        st.follow_rate_b1 = NaN; st.follow_rate_b2 = NaN; st.follow_rate_b3 = NaN;
        st.override_rate_b1 = NaN; st.override_rate_b2 = NaN; st.override_rate_b3 = NaN;
    else
        st.follow_rate_b1 = rate_in_block(seqFollow, seqBlock, 1);
        st.follow_rate_b2 = rate_in_block(seqFollow, seqBlock, 2);
        st.follow_rate_b3 = rate_in_block(seqFollow, seqBlock, 3);

        st.override_rate_b1 = rate_in_block(seqOverride, seqBlock, 1);
        st.override_rate_b2 = rate_in_block(seqOverride, seqBlock, 2);
        st.override_rate_b3 = rate_in_block(seqOverride, seqBlock, 3);
    end

    % Switching and directed transitions
    if n >= 2
        sw = (seqFollow(2:end) ~= seqFollow(1:end-1));
        st.p_switch = mean(sw);

        f2o = (seqFollow(1:end-1)==1) & (seqFollow(2:end)==0);
        o2f = (seqFollow(1:end-1)==0) & (seqFollow(2:end)==1);
        st.p_follow_to_override = mean(f2o);
        st.p_override_to_follow = mean(o2f);
    else
        st.p_switch = NaN;
        st.p_follow_to_override = NaN;
        st.p_override_to_follow = NaN;
    end

    % Inter-override gaps (distance between override events)
    ovIdx = find(seqOverride==1);
    if numel(ovIdx) >= 2
        gaps = diff(ovIdx);
        st.inter_override_gap_mean = mean(gaps);
        st.inter_override_gap_med  = median(gaps);
        st.inter_override_gap_p90  = prctile(gaps, 90);
    else
        st.inter_override_gap_mean = NaN;
        st.inter_override_gap_med  = NaN;
        st.inter_override_gap_p90  = NaN;
    end

    % Streak lengths for override and follow runs
    st.override_streak_mean = mean_streak_len(seqOverride);
    st.override_streak_p90  = prctile_streak_len(seqOverride, 90);

    st.follow_streak_mean = mean_streak_len(seqFollow);
    st.follow_streak_p90  = prctile_streak_len(seqFollow, 90);
end

function st = empty_sig()
% empty_sig Return a signature struct populated with NaNs/zeros for empty sequences.
    st = struct( ...
        "N_doors",0, ...
        "follow_rate",NaN,"override_rate",NaN, ...
        "follow_rate_b1",NaN,"follow_rate_b2",NaN,"follow_rate_b3",NaN, ...
        "override_rate_b1",NaN,"override_rate_b2",NaN,"override_rate_b3",NaN, ...
        "p_switch",NaN,"p_follow_to_override",NaN,"p_override_to_follow",NaN, ...
        "inter_override_gap_mean",NaN,"inter_override_gap_med",NaN,"inter_override_gap_p90",NaN, ...
        "override_streak_mean",NaN,"override_streak_p90",NaN, ...
        "follow_streak_mean",NaN,"follow_streak_p90",NaN ...
    );
end

function r = rate_in_block(x, b, blk)
% rate_in_block Mean rate of x within a given block index (blk).
    mask = (double(b(:)) == blk);
    if any(mask), r = mean(double(x(mask))); else, r = NaN; end
end

function m = mean_streak_len(binarySeq)
% mean_streak_len Mean run length of contiguous nonzero elements.
    lens = streak_lengths(binarySeq);
    if isempty(lens), m = 0; else, m = mean(lens); end
end

function p = prctile_streak_len(binarySeq, q)
% prctile_streak_len Percentile of run lengths (q in [0,100]).
    lens = streak_lengths(binarySeq);
    if isempty(lens), p = 0; else, p = prctile(lens, q); end
end

function lens = streak_lengths(binarySeq)
% streak_lengths Return lengths of runs where binarySeq is nonzero.
    x = double(binarySeq(:)~=0);
    if isempty(x), lens = []; return; end
    d = diff([0; x; 0]);
    runStarts = find(d==1);
    runEnds   = find(d==-1) - 1;
    lens = runEnds - runStarts + 1;
end

% ======================================================================
% Storage and summarization
% ======================================================================
function sig = init_sig_store(nP, R)
% init_sig_store Preallocate signature storage as numeric matrices for each field.
    fields = fieldnames(empty_sig());
    for i = 1:numel(fields)
        sig.(fields{i}) = NaN(nP, R);
    end
end

function sig = store_sig(sig, pi, r, st)
% store_sig Store a single rollout's signature struct into the preallocated store.
    fn = fieldnames(st);
    for i = 1:numel(fn)
        sig.(fn{i})(pi, r) = st.(fn{i});
    end
end

function T = summarize_sig_store(sig, qlo, qhi)
% summarize_sig_store Summarize per-participant simulated signature distributions.
%
% For each signature field, produces:
%   <field>_mean, <field>_qlo, <field>_qhi  computed row-wise across rollouts.
    fn = fieldnames(sig);
    T = table();
    for i = 1:numel(fn)
        X = sig.(fn{i}); % [nP x R]
        T.(fn{i} + "_mean") = mean(X, 2, "omitnan");
        T.(fn{i} + "_qlo")  = quantile_rows(X, qlo);
        T.(fn{i} + "_qhi")  = quantile_rows(X, qhi);
    end
end

function q = quantile_rows(X, qq)
% quantile_rows Row-wise quantile (qq in [0,1]) with NaN omission.
    q = NaN(size(X,1),1);
    for i = 1:size(X,1)
        xi = X(i,:);
        xi = xi(isfinite(xi));
        if isempty(xi), q(i) = NaN; else, q(i) = quantile(xi, qq); end
    end
end

function T = set_row_struct(T, rowIdx, prefix, st)
% set_row_struct Create/update prefixed scalar columns in a table from a struct.
    fn = fieldnames(st);
    for i = 1:numel(fn)
        col = prefix + string(fn{i});
        if ~ismember(col, string(T.Properties.VariableNames))
            T.(col) = NaN(height(T),1);
        end
        T.(col)(rowIdx) = st.(fn{i});
    end
end

% ======================================================================
% Pooled row builder for personalized output table
% ======================================================================
function pooled = pooled_summary_personalized(T)
% pooled_summary_personalized Append a pooled row by averaging numeric fields.
%
% The pooled row uses participant_id = "<POOLED>" and sets string fields empty.
    pooled = T(1,:);
    pooled(:,:) = []; % empty but keeps schema

    row = T(1,:);
    row.participant_id = "<POOLED>";

    vars = string(T.Properties.VariableNames);
    for i = 1:numel(vars)
        v = vars(i);
        if v == "participant_id"
            continue;
        end

        x = T.(v);
        if isnumeric(x) || islogical(x)
            row.(v) = mean(x(isfinite(x)), "omitnan");
        elseif isstring(x) || ischar(x)
            if isstring(row.(v))
                row.(v) = "";
            end
        end
    end

    pooled = [pooled; row];
end

% ======================================================================
% Plotting (pooled; minimal + thesis-friendly)
% ======================================================================
function make_fig_pooled_personalized_rates(pathPdf, T, S)
% make_fig_pooled_personalized_rates Compare pooled observed vs simulated override rate.
    f = figure('Visible','off');
    thesisStyle(f);
    hold on; grid on;

    % Participant rows only (exclude pooled row)
    M = T(string(T.participant_id)~="<POOLED>", :);

    x = [1 2];
    y = [mean(M.obs_override_rate,'omitnan'), mean(M.override_rate_mean,'omitnan')];
    eLo = [0, mean(M.override_rate_mean,'omitnan') - mean(M.override_rate_qlo,'omitnan')];
    eHi = [0, mean(M.override_rate_qhi,'omitnan') - mean(M.override_rate_mean,'omitnan')];

    plot(1, y(1), 's', 'MarkerSize', 8, 'LineWidth', 1.5);
    errorbar(2, y(2), eLo(2), eHi(2), 'o', 'LineWidth', 1.5);

    set(gca,'XTick',x,'XTickLabel',{'observed','sim (personalized)'});
    ylabel('Override rate');
    ylim([0 1]);
    title('Pooled override rate: observed vs personalized coupled rollouts');
    thesisFinalizeFigure(f, S);
    thesisExport(f, string(pathPdf));
end

function make_fig_pooled_personalized_switch(pathPdf, T, S)
% make_fig_pooled_personalized_switch Compare pooled observed vs simulated transition probabilities.
    f = figure('Visible','off');
    thesisStyle(f);
    hold on; grid on;
    M = T(string(T.participant_id)~="<POOLED>", :);

    cats_obs = ["obs_p_switch","obs_p_follow_to_override","obs_p_override_to_follow"];
    cats_sim = ["p_switch_mean","p_follow_to_override_mean","p_override_to_follow_mean"];

    x = 1:3;
    yobs = NaN(1,3);
    ysim = NaN(1,3);
    elo  = NaN(1,3);
    ehi  = NaN(1,3);

    for i = 1:3
        yobs(i) = mean(M.(cats_obs(i)),'omitnan');
        ysim(i) = mean(M.(cats_sim(i)),'omitnan');

        % Approximate pooled uncertainty by averaging participant-level quantiles
        qlo = mean(M.(strrep(cats_sim(i),"_mean","_qlo")),'omitnan');
        qhi = mean(M.(strrep(cats_sim(i),"_mean","_qhi")),'omitnan');
        elo(i) = ysim(i) - qlo;
        ehi(i) = qhi - ysim(i);
    end

    plot(x-0.08, yobs, 's-', 'LineWidth', 1.5);
    errorbar(x+0.08, ysim, elo, ehi, 'o-', 'LineWidth', 1.5);

    set(gca,'XTick',x,'XTickLabel',{'P(switch)','P(F→O)','P(O→F)'});
    ylim([0 1]);
    ylabel('Probability');
    legend({'observed','sim (personalized)'}, 'Location','best');
    title('Pooled transition structure: observed vs personalized rollouts');
    thesisFinalizeFigure(f, S);
    thesisExport(f, string(pathPdf));
end

function make_fig_pooled_personalized_gap(pathPdf, T, S)
% make_fig_pooled_personalized_gap Compare pooled observed vs simulated inter-override gap.
    f = figure('Visible','off');
    thesisStyle(f);
    hold on; grid on;
    M = T(string(T.participant_id)~="<POOLED>", :);

    yobs = mean(M.obs_inter_override_gap_mean,'omitnan');
    ysim = mean(M.inter_override_gap_mean_mean,'omitnan');
    qlo = mean(M.inter_override_gap_mean_qlo,'omitnan');
    qhi = mean(M.inter_override_gap_mean_qhi,'omitnan');

    plot(1, yobs, 's', 'MarkerSize', 8, 'LineWidth', 1.5);
    errorbar(2, ysim, ysim-qlo, qhi-ysim, 'o', 'LineWidth', 1.5);

    set(gca,'XTick',[1 2],'XTickLabel',{'observed','sim (personalized)'});
    ylabel('Mean inter-override gap (doors)');
    title('Pooled inter-override gap: observed vs personalized rollouts');
    thesisFinalizeFigure(f, S);
    thesisExport(f, string(pathPdf));
end

function make_fig_pooled_personalized_streak(pathPdf, T, S)
% make_fig_pooled_personalized_streak Compare pooled observed vs simulated override streak length.
    f = figure('Visible','off');
    thesisStyle(f);
    hold on; grid on;
    M = T(string(T.participant_id)~="<POOLED>", :);

    yobs = mean(M.obs_override_streak_mean,'omitnan');
    ysim = mean(M.override_streak_mean_mean,'omitnan');
    qlo = mean(M.override_streak_mean_qlo,'omitnan');
    qhi = mean(M.override_streak_mean_qhi,'omitnan');

    plot(1, yobs, 's', 'MarkerSize', 8, 'LineWidth', 1.5);
    errorbar(2, ysim, ysim-qlo, qhi-ysim, 'o', 'LineWidth', 1.5);

    set(gca,'XTick',[1 2],'XTickLabel',{'observed','sim (personalized)'});
    ylabel('Mean override streak length');
    title('Pooled override streak: observed vs personalized rollouts');
    thesisFinalizeFigure(f, S);
    thesisExport(f, string(pathPdf));
end

% ======================================================================
% Per-participant comparison plots (observed marker vs sim quantile band)
% ======================================================================
function make_fig_participant_errorbar(pathPdf, pid, metricBase, row, S)
% make_fig_participant_errorbar Participant-specific observed vs simulated comparison for one metric.
%
% metricBase naming convention:
%   observed column: "obs_<metricBase>"
%   simulated columns: "<metricBase>_mean", "<metricBase>_qlo", "<metricBase>_qhi"
%
% Some simulated outputs include an additional "_mean" suffix (e.g., "..._mean_mean").
% This function preserves the existing lookup logic used by the pipeline.

    pid = string(pid);
    metricBase = string(metricBase);

    obsName = "obs_" + metricBase;
    simMean = metricBase + "_mean";
    simQlo  = metricBase + "_qlo";
    simQhi  = metricBase + "_qhi";

    % Special case: in sim summary table, some columns use "<metric>_mean_mean"
    if ~ismember(simMean, string(row.Properties.VariableNames))
        if ismember(metricBase + "_mean_mean", string(row.Properties.VariableNames))
            simMean = metricBase + "_mean_mean";
            simQlo  = metricBase + "_qlo";
            simQhi  = metricBase + "_qhi";
        end
    end

    if ~ismember(obsName, string(row.Properties.VariableNames)) || ...
       ~ismember(simMean, string(row.Properties.VariableNames))
        return;
    end

    yobs = row.(obsName);
    ysim = row.(simMean);
    qlo  = row.(simQlo);
    qhi  = row.(simQhi);

    f = figure('Visible','off'); hold on; grid on;
    thesisStyle(f);

    plot(1, yobs, 's', 'MarkerSize', 8, 'LineWidth', 1.5);
    errorbar(2, ysim, ysim-qlo, qhi-ysim, 'o', 'LineWidth', 1.5);

    set(gca,'XTick',[1 2],'XTickLabel',{'observed','sim'});
    title(sprintf("pid=%s  (%s)", pid, metricBase), 'Interpreter','none');
    ylabel(char(metricBase));
    thesisFinalizeFigure(f, S);
    thesisExport(f, string(pathPdf));
end

function s = sanitize_pid(pid)
% sanitize_pid Convert participant ID to a filesystem-friendly token.
    pid = char(string(pid));
    s = regexprep(pid, '[^a-zA-Z0-9_-]', '_');
end

% ======================================================================
% Participant collection access
% ======================================================================
function Pp = get_participant_from_collection(collection, pid)
% get_participant_from_collection Retrieve one participant from a heterogeneous container.
%
% Supported collection types:
%   - containers.Map keyed by participant_id (char keys)
%   - struct array with field 'participant_id'
%   - table with variable 'participant_id'
    pid = string(pid);

    if isa(collection, "containers.Map")
        if ~isKey(collection, char(pid))
            error("[A12] VALID participant '%s' not found in collection Map.", pid);
        end
        Pp = collection(char(pid));
        return;
    end

    if isstruct(collection)
        if ~isfield(collection, "participant_id")
            error("[A12] validParticipants struct must have field participant_id.");
        end
        ids = string({collection.participant_id});
        idx = find(ids==pid, 1);
        if isempty(idx)
            error("[A12] VALID participant '%s' not found in struct collection.", pid);
        end
        Pp = collection(idx);
        return;
    end

    if istable(collection)
        if ~ismember("participant_id", string(collection.Properties.VariableNames))
            error("[A12] validParticipants table must have participant_id column.");
        end
        idx = find(string(collection.participant_id)==pid, 1);
        if isempty(idx)
            error("[A12] VALID participant '%s' not found in table collection.", pid);
        end
        Pp = collection(idx,:);
        return;
    end

    error("[A12] Unsupported validParticipants type: %s", class(collection));
end

% ======================================================================
% Model name mapping
% ======================================================================
function name = model_name_from_idx(modelIdx)
% model_name_from_idx Map model index in {0,1,2} to stable identifier string.
    switch modelIdx
        case 0, name = "model0_trust_as_probability";
        case 1, name = "model1_threshold";
        case 2, name = "model2_offset_lapse";
        otherwise, name = "unknown";
    end
end
