function stepA9_behavior_rollouts(run_id, varargin)
% stepA9_behavior_rollouts Coupled (closed-loop) generative rollout analysis on VALID.
%
% This step evaluates the coupled trust+behavior system on VALID participants by
% running CLOSED-LOOP ("coupled") simulations. In each rollout, door decisions
% are sampled from a behavioral choice model and fed back into the trust dynamics
% using the counterfactual outcome inversion rule implemented inside the
% simulator (trust_simulate_or_predict_one_participant):
%
%   if sampled_follow ~= recorded_follow  => outcome := 1 - recorded_outcome
%   else                                   outcome := recorded_outcome
%
% Unlike Step A8 (pointwise behavioral prediction on recorded data), this step
% targets emergent interaction signatures such as override timing, switching, and
% streak/gap structure under coupled decision-making.
%
% Behavioral models (must match Step A8):
%   Model 0: p_follow = clamp(tau_decision, 0, 1)                           (no fit)
%   Model 1: p_follow = sigmoid( k * (tau_decision - self_confidence) )     (k from A8 TRAIN)
%   Model 2: p_follow = (1-eps)*sigmoid(k*tau_decision + beta*(sc-0.5)) + eps*0.5
%
% INPUTS (required)
%   run_id (string|char)
%       Analysis run identifier. Used to locate run-local artifacts under:
%         derived/analysis_runs/<run_id>/
%
% NAME-VALUE ARGUMENTS (optional)
%   "OutDir"                 (string|char)  Output directory. Default: derived/.../stepA9_behavior_rollouts
%   "Overwrite"              (logical)      If false, error if outputs exist. Default: false
%   "RolloutsPerParticipant" (scalar)       Number of stochastic rollouts per participant. Default: 300
%   "Quantiles"              (1x2 numeric)  Quantiles for rollout summaries per participant. Default: [0.05 0.95]
%   "Models"                 (vector)       Which model indices to run (subset of [0 1 2]). Default: [0 1 2]
%   "RandomSeed"             (scalar)       RNG seed used to derive per-rollout seeds. Default: 1
%
% OUTPUTS
%   (none)
%       Artifacts are written to:
%         derived/analysis_runs/<run_id>/stepA9_behavior_rollouts/
%       including:
%         - A9_rollout_stats.mat   Full bundle incl. per-rollout signature arrays
%         - A9_rollout_stats.csv   Participant-level summaries + pooled rows
%         - figures/*              Pooled diagnostic figures exported via thesisExport
%         - meta.mat / meta.json   Provenance metadata
%
% SIGNATURES (override-centric; robust under high follow base-rate)
%   - follow_rate, override_rate
%   - follow/override rate by block (1..3) if block_index is available
%   - switch probabilities: P(switch), P(follow->override), P(override->follow)
%   - inter-override gaps (mean/median/p90)
%   - override streak lengths (mean/p90)
%   - follow streak lengths (mean/p90) (reported but interpret cautiously)
%
% ASSUMPTIONS / DEPENDENCIES
%   - Simulator:
%       trust_simulate_or_predict_one_participant must support:
%         trust_simulate_or_predict_one_participant("coupled", theta_star, P, dt, behavior_params)
%       and return sim.coupled.followed_sampled, plus (optionally) door_index and block_index.
%   - Behavioral mapping:
%       behavioral_model must interpret the behavior_params struct fields set here.
%   - Utilities:
%       must_exist_file, ensure_dir, save_json
%       load_participants_struct (preferred for loading A1 archived participants)
%       find_theta_in_struct (used as a fallback when theta_star is stored non-canonically)
%   - Plot/style:
%       thesisStyle, thesisFinalizeFigure, thesisExport, behavioralDisplayName
%
% NOTE
%   This implementation does not re-implement coupled dynamics. It delegates
%   rollout execution to trust_simulate_or_predict_one_participant("coupled", ...)
%   and reads sim.coupled outputs.

    % ------------------------------------------------------------------
    % Parse and validate inputs
    % ------------------------------------------------------------------
    if nargin < 1 || isempty(run_id)
        error("stepA9_behavior_rollouts: run_id is required.");
    end
    run_id = string(run_id);

    p = inputParser;
    p.addParameter("OutDir", "", @(s) isstring(s) || ischar(s));
    p.addParameter("Overwrite", false, @(x) islogical(x) && isscalar(x));
    p.addParameter("RolloutsPerParticipant", 300, @(x) isnumeric(x) && isscalar(x) && x>=1);
    p.addParameter("Quantiles", [0.05 0.95], @(x) isnumeric(x) && numel(x)==2 && all(x>=0) && all(x<=1));
    p.addParameter("Models", [0 1 2], @(x) isnumeric(x) && isvector(x));
    p.addParameter("RandomSeed", 1, @(x) isnumeric(x) && isscalar(x));
    p.parse(varargin{:});
    args = p.Results;

    rng(args.RandomSeed);

    % Thesis plotting defaults + style struct
    S = thesisStyle();

    % ------------------------------------------------------------------
    % Load observed VALID door-level behavior from Step A7
    % ------------------------------------------------------------------
    a7Dir = fullfile("derived","analysis_runs",run_id,"stepA7_behavior_dataset");
    validMatA7 = fullfile(a7Dir, "behavior_dataset_valid.mat");
    must_exist_file(validMatA7, "A7 VALID dataset");

    S_va = load(validMatA7, "T");
    if ~isfield(S_va,"T") || ~istable(S_va.T), error("[A9] VALID mat missing table T."); end
    Tva = S_va.T;

    % Keep only rows with usable labels for observed signatures
    Tva = Tva(Tva.is_valid_label==1, :);

    reqCols = ["participant_id","followed","door_index"];
    assert(all(ismember(reqCols, string(Tva.Properties.VariableNames))), "[A9] A7 VALID missing required columns.");
    haveBlockA7 = ismember("block_index", string(Tva.Properties.VariableNames));

    % ------------------------------------------------------------------
    % Load behavioral fit parameters from Step A8 (trained on TRAIN)
    % ------------------------------------------------------------------
    a8Dir = fullfile("derived","analysis_runs",run_id,"stepA8_behavior_fit_eval");
    fitMat = fullfile(a8Dir, "fit_params.mat");
    must_exist_file(fitMat, "A8 fit_params.mat");

    S_fit = load(fitMat, "fit");
    if ~isfield(S_fit,"fit"), error("[A9] fit_params.mat missing 'fit' struct."); end
    fit = S_fit.fit;

    % ------------------------------------------------------------------
    % Load trust model inputs consistent with Step A5:
    %   - VALID participants from A1 archive
    %   - dt from cfg.dt in the results file referenced by A3 selection
    %   - theta_star from A3 selection
    % ------------------------------------------------------------------
    [theta_star, dt, validParticipants] = local_load_theta_dt_and_valid_participants_like_A5(run_id);

    % ------------------------------------------------------------------
    % Output directory and overwrite policy
    % ------------------------------------------------------------------
    outDir = string(args.OutDir);
    if strlength(outDir)==0
        outDir = fullfile("derived","analysis_runs",run_id,"stepA9_behavior_rollouts");
    end
    ensure_dir(outDir);

    figDir = fullfile(outDir, "figures");
    ensure_dir(figDir);

    statsMat = fullfile(outDir, "A9_rollout_stats.mat");
    statsCsv = fullfile(outDir, "A9_rollout_stats.csv");
    metaMat  = fullfile(outDir, "meta.mat");
    metaJson = fullfile(outDir, "meta.json");

    if ~args.Overwrite
        if isfile(statsMat) || isfile(statsCsv)
            error("[A9] Outputs exist. Set Overwrite=true to replace. (%s)", outDir);
        end
    end

    % ------------------------------------------------------------------
    % Build observed per-participant signatures from A7 VALID
    % ------------------------------------------------------------------
    pid_va = string(Tva.participant_id);
    uniqP  = unique(pid_va);
    nP     = numel(uniqP);

    obsStats = table();
    obsStats.participant_id = uniqP;

    for i = 1:nP
        mask = (pid_va == uniqP(i));
        Tpi = Tva(mask, :);

        % Align the observed sequence in door index order
        [~,ord] = sort(double(Tpi.door_index(:)));
        Tpi = Tpi(ord, :);

        % Observed sequence: 1=follow, 0=override
        seqFollow = double(Tpi.followed(:));

        % Optional block index for block-wise rates
        if haveBlockA7
            seqBlock  = double(Tpi.block_index(:));
        else
            seqBlock  = NaN(size(seqFollow));
        end

        st = compute_signatures(seqFollow, seqBlock);
        obsStats = set_row_struct(obsStats, i, "obs_", st);
    end

    % ------------------------------------------------------------------
    % Resolve requested model indices into behavior_params structs
    % ------------------------------------------------------------------
    models = unique(round(args.Models(:)'));
    R   = args.RolloutsPerParticipant;
    qlo = args.Quantiles(1);
    qhi = args.Quantiles(2);

    modelNames   = strings(1,numel(models));
    modelBParams = cell(1,numel(models));
    for mi = 1:numel(models)
        [modelNames(mi), modelBParams{mi}] = resolve_behavior_params(models(mi), fit);
    end

    % ------------------------------------------------------------------
    % Coupled rollouts via trust_simulate_or_predict_one_participant("coupled", ...)
    % ------------------------------------------------------------------
    roll = struct();
    roll.meta = struct("run_id",char(run_id), ...
                       "R",R, ...
                       "models",models, ...
                       "modelNames",modelNames, ...
                       "qlo",qlo,"qhi",qhi, ...
                       "random_seed",args.RandomSeed, ...
                       "dt",dt);
    roll.obsStats = obsStats;

    perModelSummaries = cell(numel(models),1);

    for mi = 1:numel(models)
        mName = modelNames(mi);
        bpar  = modelBParams{mi};

        fprintf("[A9] %s: %d rollouts x %d participants\n", mName, R, nP);

        % Storage for per-rollout signatures (nP x R arrays per field)
        sig = init_sig_store(nP, R);

        for pi = 1:nP
            pid = uniqP(pi);

            % Retrieve participant struct/table row from archived VALID collection
            Pp  = get_participant_from_collection(validParticipants, pid);

            for r = 1:R
                % Deterministic per-rollout seed schedule (reproducible)
                seed = args.RandomSeed + 100000*mi + 1000*pi + r;
                rng(seed);

                sim = trust_simulate_or_predict_one_participant("coupled", theta_star, Pp, dt, bpar);

                if ~isfield(sim, "coupled") || ~isfield(sim.coupled, "followed_sampled")
                    error("[A9] Simulator did not return sim.coupled.followed_sampled in coupled mode (pid=%s).", pid);
                end

                % Convert simulator output to aligned sequences (door-sorted if possible)
                [seqFollow, seqBlock] = coupled_sim_to_sequences(sim);

                % Compute rollout signatures and store
                st = compute_signatures(seqFollow, seqBlock);
                sig = store_sig(sig, pi, r, st);
            end
        end

        % Per-participant rollout summaries (mean + quantile bands)
        summ = summarize_sig_store(sig, qlo, qhi);
        summ.model = repmat(mName, height(summ), 1);
        summ.participant_id = uniqP;

        % Join observed signatures for convenient side-by-side reporting
        summ = join(summ, obsStats, "Keys","participant_id");

        perModelSummaries{mi} = summ;
        roll.(char(mName)).sig = sig;
        roll.(char(mName)).summary = summ;
    end

    % Concatenate per-model participant summaries and add pooled rows
    allSumm = vertcat(perModelSummaries{:});
    pooled  = pooled_summary(allSumm, modelNames);

    pooled.participant_id = repmat("<POOLED>", height(pooled), 1);
    pooled = movevars(pooled, "participant_id", "Before", 1);

    outTable = [allSumm; pooled];
    writetable(outTable, statsCsv);
    save(statsMat, "roll", "outTable", "pooled", "-v7.3");

    % ------------------------------------------------------------------
    % Provenance metadata
    % ------------------------------------------------------------------
    meta = struct();
    meta.run_id = char(run_id);
    meta.created = char(datetime('now','Format','yyyy-MM-dd HH:mm:ss'));
    meta.valid_participants = nP;
    meta.rollouts_per_participant = R;
    meta.models = models;
    meta.model_names = cellstr(modelNames);
    meta.quantiles = [qlo qhi];
    meta.random_seed = args.RandomSeed;
    meta.dt = dt;
    meta.have_block_index_in_A7 = haveBlockA7;
    save(metaMat, "meta");
    save_json(metaJson, meta);

    % ------------------------------------------------------------------
    % Figures (pooled; override-centric)
    % ------------------------------------------------------------------
    make_fig_pooled_block_rates(fullfile(figDir, "pooled_rates_by_block"), outTable, modelNames, S);
    make_fig_pooled_switch_rates(fullfile(figDir, "pooled_switch_rates"), outTable, modelNames, S);
    make_fig_pooled_override_gap(fullfile(figDir, "pooled_inter_override_gaps"), outTable, modelNames, S);
    make_fig_pooled_override_streak(fullfile(figDir, "pooled_override_streaks"), outTable, modelNames, S);

    fprintf("[Step A9] Done.\n");
    fprintf("  Output dir: %s\n", outDir);
    fprintf("  Wrote: %s\n", statsCsv);
end

% ======================================================================
% Load theta_star, dt, and VALID participants consistent with Step A5
% ======================================================================
function [theta_star, dt, participants_valid] = local_load_theta_dt_and_valid_participants_like_A5(run_id)
% local_load_theta_dt_and_valid_participants_like_A5 Resolve theta_star, dt, and VALID participants.
%
% This helper follows the same artifact conventions as Step A5:
%   - VALID participants are loaded from the A1 archive (mapped probes file).
%   - theta_star and results_file are resolved from A3 selection.mat, with an
%     optional fallback to A3 theta_star.mat for theta_star.
%   - dt is read from cfg.dt within the results MAT file.

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
        error("[A9] A3 selection.mat missing variable 'selection'.");
    end
    selection = Ssel.selection;

    % theta_star
    theta_star = [];
    if isfield(selection,"theta_star") && ~isempty(selection.theta_star)
        theta_star = selection.theta_star(:);
    else
        % Fallback: A3 theta_star.mat
        thetaPath = fullfile("derived","analysis_runs",run_id,"stepA3_model_selection","theta_star.mat");
        if isfile(thetaPath)
            Sth = load(thetaPath);
            theta_star = find_theta_in_struct(Sth);
        end
    end
    if isempty(theta_star)
        error("[A9] Could not resolve theta_star from A3 selection (selection.theta_star missing/empty).");
    end

    % results file -> cfg.dt
    if ~isfield(selection,"results_file") || isempty(selection.results_file)
        error("[A9] selection.results_file missing. Cannot locate cfg.dt.");
    end
    resultsMatPath = string(selection.results_file);
    must_exist_file(resultsMatPath, "Fit results MAT (selection.results_file)");

    R = load(resultsMatPath);
    if ~isfield(R,"cfg") || ~isstruct(R.cfg) || ~isfield(R.cfg,"dt") || isempty(R.cfg.dt)
        error("[A9] results MAT does not contain cfg.dt: %s", resultsMatPath);
    end
    dt = double(R.cfg.dt);
    if ~isscalar(dt) || ~isfinite(dt) || dt <= 0
        error("[A9] cfg.dt invalid in results MAT: %s", resultsMatPath);
    end
end

% ======================================================================
% Convert coupled simulator output into aligned sequences
% ======================================================================
function [seqFollow, seqBlock] = coupled_sim_to_sequences(sim)
% coupled_sim_to_sequences Extract follow and block sequences from sim.coupled.
%
% The returned sequences are aligned in door order when sim.coupled.door_index
% is provided; otherwise, the simulator order is used as-is.
%
% Output conventions:
%   seqFollow - binary vector, 1=follow, 0=override
%   seqBlock  - numeric block index vector or NaNs if unavailable

    seqFollow = double(sim.coupled.followed_sampled(:));  % 1=follow, 0=override

    % Sort by door index if present and usable
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
        % No usable door_index: retain original ordering
        if isfield(sim.coupled, "block_index") && ~isempty(sim.coupled.block_index)
            seqBlock = double(sim.coupled.block_index(:));
        else
            seqBlock = NaN(size(seqFollow));
        end
    end

    % Defensive: replace NaNs in sampled decisions to keep rollout alive
    if any(isnan(seqFollow))
        seqFollow(isnan(seqFollow)) = 1; % degenerate default; diagnostics should reflect this
    end
end

% ======================================================================
% Build behavior_params for behavioral_model(state, params)
% ======================================================================
function [name, bpar] = resolve_behavior_params(modelIdx, fit)
% resolve_behavior_params Map a model index to a behavior_params struct.
%
% The returned struct fields are interpreted by the behavioral_model used
% inside the coupled simulator. Flags are used to select the model variant.

    bpar = struct();

    % Flags interpreted by behavioral_model
    bpar.tau_flag = 0;
    bpar.m1_flag  = 0;
    bpar.m2_flag  = 0;

    switch modelIdx
        case 0
            name = "model0_trust_as_probability";
            bpar.tau_flag = 1;

        case 1
            name = "model1_threshold";
            assert(isfield(fit,"model1") && isfield(fit.model1,"k_hat"), "[A9] fit.model1.k_hat missing.");
            k = fit.model1.k_hat;
            bpar.m1_flag = 1;
            bpar.k_m1 = k;

        case 2
            name = "model2_offset_lapse";
            assert(isfield(fit,"model2"), "[A9] fit.model2 missing.");
            k    = fit.model2.k_hat;
            beta = fit.model2.beta_hat;
            eps  = fit.model2.eps_hat;

            bpar.m2_flag = 1;
            bpar.k_m2 = k;
            bpar.beta = beta;
            bpar.eps  = eps;

        otherwise
            error("[A9] Unknown model index: %d", modelIdx);
    end
end

% ======================================================================
% Signature computation (override-centric + transitions)
% ======================================================================
function st = compute_signatures(seqFollow, seqBlock)
% compute_signatures Compute interaction signatures from a follow/override sequence.
%
% Inputs:
%   seqFollow - binary vector, 1=follow, 0=override
%   seqBlock  - numeric block indices aligned to seqFollow, or NaNs if unavailable
%
% Output:
%   st        - struct containing rates, transition statistics, and streak/gap summaries

    seqFollow = double(seqFollow(:));
    seqOverride = 1 - seqFollow;

    n = numel(seqFollow);
    if n==0
        st = empty_sig();
        return;
    end

    st.N_doors = n;

    % Overall rates
    st.follow_rate = mean(seqFollow);
    st.override_rate = mean(seqOverride);

    % Block-wise rates when block annotations are available
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

    % Transition structure (first-order)
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

    % Inter-override gaps (door counts between overrides)
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

    % Streak length summaries (burstiness / persistence)
    st.override_streak_mean = mean_streak_len(seqOverride);
    st.override_streak_p90  = prctile_streak_len(seqOverride, 90);

    st.follow_streak_mean = mean_streak_len(seqFollow);
    st.follow_streak_p90  = prctile_streak_len(seqFollow, 90);
end

function st = empty_sig()
% empty_sig Signature struct with consistent field set and NaN defaults.
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
% rate_in_block Mean of x restricted to entries with block index == blk.
    mask = (double(b(:)) == blk);
    if any(mask), r = mean(double(x(mask))); else, r = NaN; end
end

function m = mean_streak_len(binarySeq)
% mean_streak_len Mean run length of nonzero entries in a binary sequence.
    lens = streak_lengths(binarySeq);
    if isempty(lens), m = 0; else, m = mean(lens); end
end

function p = prctile_streak_len(binarySeq, q)
% prctile_streak_len Percentile of run lengths of nonzero entries in a binary sequence.
    lens = streak_lengths(binarySeq);
    if isempty(lens), p = 0; else, p = prctile(lens, q); end
end

function lens = streak_lengths(binarySeq)
% streak_lengths Compute run lengths of nonzero entries in a binary sequence.
    x = double(binarySeq(:)~=0);
    if isempty(x), lens = []; return; end
    d = diff([0; x; 0]);
    runStarts = find(d==1);
    runEnds   = find(d==-1) - 1;
    lens = runEnds - runStarts + 1;
end

% ======================================================================
% Storage and summarization of rollout signatures
% ======================================================================
function sig = init_sig_store(nP, R)
% init_sig_store Pre-allocate per-field [nP x R] arrays for signature storage.
    fields = fieldnames(empty_sig());
    for i = 1:numel(fields)
        sig.(fields{i}) = NaN(nP, R);
    end
end

function sig = store_sig(sig, pi, r, st)
% store_sig Store one rollout's signature struct into the [nP x R] arrays.
    fn = fieldnames(st);
    for i = 1:numel(fn)
        sig.(fn{i})(pi, r) = st.(fn{i});
    end
end

function T = summarize_sig_store(sig, qlo, qhi)
% summarize_sig_store Compute per-participant mean and quantile bands across rollouts.
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
% quantile_rows Row-wise quantile with NaN removal.
    q = NaN(size(X,1),1);
    for i = 1:size(X,1)
        xi = X(i,:);
        xi = xi(isfinite(xi));
        if isempty(xi), q(i) = NaN; else, q(i) = quantile(xi, qq); end
    end
end

function T = set_row_struct(T, rowIdx, prefix, st)
% set_row_struct Insert fields from struct st into row rowIdx with a name prefix.
    fn = fieldnames(st);
    for i = 1:numel(fn)
        col = prefix + string(fn{i});
        if ~ismember(col, string(T.Properties.VariableNames))
            T.(col) = NaN(height(T),1);
        end
        T.(col)(rowIdx) = st.(fn{i});
    end
end

function pooled = pooled_summary(allSumm, modelNames)
% pooled_summary Compute pooled mean summaries across participants per model.
%
% This returns a table with the same schema as allSumm (except participant_id),
% containing one pooled row per model. Numeric and logical variables are pooled
% via mean(...,'omitnan'); non-numeric variables retain their template values.

    % Variables to keep in pooled output (everything except participant_id)
    varsKeep = allSumm.Properties.VariableNames;
    varsKeep = varsKeep(~strcmp(varsKeep, "participant_id"));

    % Prepare an empty pooled table with identical types to allSumm(varsKeep)
    tmpl = allSumm(1, varsKeep);
    tmpl(1,:) = [];  % empty but preserves schema
    pooled = tmpl;

    % Identify numeric/logical columns to pool (skip "model" and non-numeric variables)
    numericVars = {};
    for i = 1:numel(varsKeep)
        v = varsKeep{i};
        if strcmp(v, "model")
            continue;
        end
        x = allSumm.(v);
        if isnumeric(x) || islogical(x)
            numericVars{end+1} = v; %#ok<AGROW>
        end
    end

    % One pooled row per model
    for mi = 1:numel(modelNames)
        m = string(modelNames(mi));

        % Subset rows for this model and exclude pooled marker rows if present
        M = allSumm(string(allSumm.model)==m & string(allSumm.participant_id)~="<POOLED>", :);

        % Start from a one-row template with correct schema/types
        row = allSumm(1, varsKeep);

        % Set model identifier
        row.model = m;

        % Fill pooled means for numeric/logical columns
        for ci = 1:numel(numericVars)
            c = numericVars{ci};
            row.(c) = mean(M.(c), "omitnan");
        end

        pooled = [pooled; row]; %#ok<AGROW>
    end
end

% ======================================================================
% Plotting (pooled diagnostics; exported via thesisExport)
% ======================================================================
function make_fig_pooled_block_rates(pathPdf, outTable, modelNames, S)
% make_fig_pooled_block_rates Pooled follow rates by block (observed vs rollouts).
    f = figure('Name','Pooled follow rate by block', 'Visible','off', 'Color','w');
    thesisStyle(f);

    x = [1 2 3];

    ax = gca;
    ax.ColorOrder = S.colorOrder6(1:4,:);
    ax.ColorOrderIndex = 1;
    hold on;

    for mi = 1:numel(modelNames)
        m = modelNames(mi);
        M = outTable(string(outTable.model)==string(m) & outTable.participant_id~="<POOLED>", :);

        y = [ ...
            mean(M.follow_rate_b1_mean,'omitnan'), ...
            mean(M.follow_rate_b2_mean,'omitnan'), ...
            mean(M.follow_rate_b3_mean,'omitnan') ];

        if all(~isfinite(y)), continue; end
        plot(x, y, '-o', 'LineWidth', S.lineWidth, 'DisplayName', char(m));
    end

    % Observed reference (uses the same participant subset as modelNames(1))
    Mop = outTable(string(outTable.model)==string(modelNames(1)) & outTable.participant_id~="<POOLED>", :);
    yobs = [ ...
        mean(Mop.obs_follow_rate_b1,'omitnan'), ...
        mean(Mop.obs_follow_rate_b2,'omitnan'), ...
        mean(Mop.obs_follow_rate_b3,'omitnan') ];

    if any(isfinite(yobs))
        plot(x, yobs, '--s', 'LineWidth', S.lineWidth, 'DisplayName', 'observed');
    end

    xlabel('Block');
    xticks([1 2 3])
    ylabel('Follow rate');
    title('Pooled follow rate by block (observed vs coupled rollouts)');
    grid on;
    xlim([0.8 3.2]);
    ylim([0 1]);

    leg = behavioralDisplayName(modelNames);
    leg{end+1} = 'observed';
    legend(leg, 'Location','best');

    local_save_figure(f, string(pathPdf), S);
end

function make_fig_pooled_switch_rates(pathPdf, outTable, modelNames, S)
% make_fig_pooled_switch_rates Pooled transition probabilities (observed vs rollouts).
    f = figure('Name','Pooled transition structure', 'Visible','off', 'Color','w');
    thesisStyle(f);

    cats = ["p_switch_mean","p_follow_to_override_mean","p_override_to_follow_mean"];
    catLabels = {'$p(\mathrm{switch})$','$p(\mathrm{F}\rightarrow\mathrm{O})$','$p(\mathrm{O}\rightarrow\mathrm{F})$'};
    x = 1:numel(cats);

    ax = gca;
    ax.ColorOrder = S.colorOrder6(1:4,:);
    ax.ColorOrderIndex = 1;

    hold on;
    for mi = 1:numel(modelNames)
        m = modelNames(mi);
        M = outTable(string(outTable.model)==string(m) & outTable.participant_id~="<POOLED>", :);

        y = zeros(1,numel(cats));
        for ci = 1:numel(cats)
            y(ci) = mean(M.(cats(ci)),'omitnan');
        end
        plot(x, y, '-o', 'LineWidth', S.lineWidth, 'DisplayName', char(m));
    end

    Mop = outTable(string(outTable.model)==string(modelNames(1)) & outTable.participant_id~="<POOLED>", :);
    yobs = [ ...
        mean(Mop.obs_p_switch,'omitnan'), ...
        mean(Mop.obs_p_follow_to_override,'omitnan'), ...
        mean(Mop.obs_p_override_to_follow,'omitnan') ];
    plot(x, yobs, '--s', 'LineWidth', S.lineWidth, 'DisplayName', 'observed');

    set(gca,'XTick',x,'XTickLabel',catLabels);
    ylabel('Probability ($p$)');
    title('Pooled transition structure (observed vs coupled rollouts)');
    grid on;
    xlim([0.8 3.2]);
    ylim([0 1]);

    leg = behavioralDisplayName(modelNames);
    leg{end+1} = 'observed';
    legend(leg,'Location','best');

    local_save_figure(f, string(pathPdf), S);
end

function make_fig_pooled_override_gap(pathPdf, outTable, modelNames, S)
% make_fig_pooled_override_gap Pooled mean inter-override gap (sparsity/clustering).
    f = figure('Name','Pooled inter-override gap', 'Visible','off', 'Color','w');
    thesisStyle(f);

    ax = gca;
    ax.ColorOrder = S.colorOrder6(1:4,:);
    ax.ColorOrderIndex = 1;

    hold on;

    for mi = 1:numel(modelNames)
        m = modelNames(mi);
        M = outTable(string(outTable.model)==string(m) & outTable.participant_id~="<POOLED>", :);
        y = mean(M.inter_override_gap_mean_mean,'omitnan');
        plot(mi, y, 'o', 'MarkerSize', max(4, round(0.6*S.markerSize)), 'DisplayName', char(m));
    end

    Mop = outTable(string(outTable.model)==string(modelNames(1)) & outTable.participant_id~="<POOLED>", :);
    yobs = mean(Mop.obs_inter_override_gap_mean,'omitnan');
    plot(numel(modelNames)+1, yobs, 's', 'MarkerSize', max(4, round(0.6*S.markerSize)), 'DisplayName', 'observed');

    xtlbl = behavioralDisplayName(modelNames);
    xtlbl = xtlbl(:)'; % force row cell
    set(gca,'XTick',1:(numel(modelNames)+1), 'XTickLabel',[xtlbl {'observed'}]);

    ylabel('Mean gap (doors) between overrides');
    title('Pooled inter-override gap (override sparsity / clustering)');
    grid on;

    leg = behavioralDisplayName(modelNames);
    leg{end+1} = 'observed';
    legend(leg,'Location','best');

    local_save_figure(f, string(pathPdf), S);
end

function make_fig_pooled_override_streak(pathPdf, outTable, modelNames, S)
% make_fig_pooled_override_streak Pooled mean override streak length (burstiness).
    f = figure('Name','Pooled override streak length', 'Visible','off', 'Color','w');
    thesisStyle(f);

    ax = gca;
    ax.ColorOrder = S.colorOrder6(1:4,:);
    ax.ColorOrderIndex = 1;

    hold on;

    for mi = 1:numel(modelNames)
        m = modelNames(mi);
        M = outTable(string(outTable.model)==string(m) & outTable.participant_id~="<POOLED>", :);
        y = mean(M.override_streak_mean_mean,'omitnan');
        plot(mi, y, 'o', 'MarkerSize', max(4, round(0.6*S.markerSize)), 'DisplayName', char(m));
    end

    Mop = outTable(string(outTable.model)==string(modelNames(1)) & outTable.participant_id~="<POOLED>", :);
    yobs = mean(Mop.obs_override_streak_mean,'omitnan');
    plot(numel(modelNames)+1, yobs, 's', 'MarkerSize', max(4, round(0.6*S.markerSize)), 'DisplayName', 'observed');

    xtlbl = behavioralDisplayName(modelNames);
    xtlbl = xtlbl(:)';
    set(gca,'XTick',1:(numel(modelNames)+1), 'XTickLabel',[xtlbl {'observed'}]);

    ylabel('Mean override streak length');
    title('Pooled override streak length (burstiness)');
    grid on;

    leg = behavioralDisplayName(modelNames);
    leg{end+1} = 'observed';
    legend(leg,'Location','best');

    local_save_figure(f, string(pathPdf), S);
end

function local_save_figure(f, outBase, S)
% local_save_figure Finalize and export a figure using thesis styling helpers.
    thesisFinalizeFigure(f, S);
    thesisExport(f, outBase);
end

% ======================================================================
% Participant collection access (struct / table / map)
% ======================================================================
function Pp = get_participant_from_collection(collection, pid)
% get_participant_from_collection Retrieve a participant by participant_id from a collection.
%
% Supported collection types:
%   - containers.Map with keys as participant_id (char)
%   - struct array with field participant_id
%   - table with column participant_id

    if isa(collection, "containers.Map")
        if ~isKey(collection, char(pid))
            error("[A9] VALID participant '%s' not found in collection Map.", pid);
        end
        Pp = collection(char(pid));
        return;
    end

    if isstruct(collection)
        if ~isfield(collection, "participant_id")
            error("[A9] validParticipants struct must have field participant_id.");
        end
        ids = string({collection.participant_id});
        idx = find(ids==pid, 1);
        if isempty(idx)
            error("[A9] VALID participant '%s' not found in struct collection.", pid);
        end
        Pp = collection(idx);
        return;
    end

    if istable(collection)
        if ~ismember("participant_id", string(collection.Properties.VariableNames))
            error("[A9] validParticipants table must have participant_id column.");
        end
        idx = find(string(collection.participant_id)==pid, 1);
        if isempty(idx)
            error("[A9] VALID participant '%s' not found in table collection.", pid);
        end
        Pp = collection(idx,:);
        return;
    end

    error("[A9] Unsupported validParticipants type: %s", class(collection));
end
