function stepA7_build_behavior_dataset(run_id, varargin)
% stepA7_build_behavior_dataset  Build per-door behavioral dataset for follow/override prediction.
%
% This step produces a run-local, door-trial-level dataset to support
% probabilistic behavioral modeling (A8) and evaluation (A9).
%
% Key design choice (causality):
%   - "decision trust" must be evaluated BEFORE the door event update occurs.
%   - Therefore we use tau_decision = tau_hist(k_grid-1), where k_grid is the
%     index in sim.t_grid at which the door event occurs.
%
% Outputs
%   Writes to: derived/analysis_runs/<run_id>/stepA7_behavior_dataset/
%     - meta.mat / meta.json
%     - behavior_dataset_<split>.csv / .mat
%     - behavior_dataset_<split>_summary.csv
%     - behavior_dataset_<split>_by_participant.csv
%
% Assumptions / Dependencies
%   - Step A1 has created: derived/analysis_runs/<run_id>/stepA1_prepare_analysis/manifest.mat
%   - Participant loaders/utilities on MATLAB path:
%       must_exist_file, ensure_dir, load_participants_struct, read_participant_ids,
%       resolve_runlocal_or_source, file_info_struct, save_json
%   - Simulation helper on MATLAB path:
%       trust_simulate_or_predict_one_participant
%
% Notes
%   - Components (tau_lat, tau_rep, tau_sit) are NOT stored (per user request).
%   - Self-confidence sc is stored as a participant-specific constant.
%   - Added derived fields:
%       * sc_centered     = sc - 0.5
%       * margin_treshold = tau_decision - sc   (baseline threshold-model margin)
%     (Spelling kept as "margin_treshold" to match pipeline naming.)
%   - The dataset includes block index (1..3) and within-block trial (1..20),
%     and global trial index (1..60), assuming preprocessing verified 60 door
%     trials split into 3 blocks of 20 for all participants.

    % -------------------------
    % Parse inputs
    % -------------------------
    if nargin < 1 || isempty(run_id)
        error("stepA7_build_behavior_dataset: run_id is required.");
    end
    run_id = string(run_id);

    p = inputParser;
    p.addParameter("Split", "valid", @(s) isstring(s) || ischar(s));  % "train" or "valid"
    p.addParameter("ResultsMatPath", "", @(s) isstring(s) || ischar(s)); % optional (for dt inference)
    p.addParameter("SelectedThetaMatPath", "", @(s) isstring(s) || ischar(s)); % optional
    p.addParameter("Mode", "simple", @(s) isstring(s) || ischar(s)); % typically "simple"
    p.addParameter("OutDir", "", @(s) isstring(s) || ischar(s));
    p.addParameter("Overwrite", false, @(x) islogical(x) && isscalar(x));
    p.parse(varargin{:});
    args = p.Results;

    splitName = lower(string(args.Split));
    if ~(splitName == "train" || splitName == "valid")
        error("Split must be 'train' or 'valid'. Got: %s", splitName);
    end

    mode = string(args.Mode);

    % -------------------------
    % Locate A1 manifest + archived files
    % -------------------------
    a1Dir = fullfile("derived", "analysis_runs", run_id, "stepA1_prepare_analysis");
    manifestPath = fullfile(a1Dir, "manifest.mat");
    must_exist_file(manifestPath, "A1 manifest");

    M = load(manifestPath, "runInfo");
    if ~isfield(M, "runInfo")
        error("A1 manifest.mat missing variable 'runInfo'.");
    end
    runInfo = M.runInfo; %#ok<NASGU>

    % Resolve participant file for requested split
    if splitName == "train"
        % Prefer probes-mapped train file if present in A1 dir; else raw split
        cand1 = fullfile(a1Dir, "participants_probes_mapped_stepM4.mat");
        cand2 = fullfile(a1Dir, "participants_train_stepV.mat");
        if isfile(cand1), partMat = cand1; else, partMat = cand2; end
        must_exist_file(partMat, "TRAIN participants (A1 archive)");
    else
        partMat = fullfile(a1Dir, "participants_valid_probes_mapped_stepM4.mat");
        must_exist_file(partMat, "VALID participants (A1 archive)");
    end

    participants = load_participants_struct(partMat);
    pid_list = read_participant_ids(participants);

    % -------------------------
    % Resolve dt + theta_star (prefer A3 outputs, fallback if needed)
    % -------------------------
    resultsMatPath = string(args.ResultsMatPath);
    selectedThetaMatPath = string(args.SelectedThetaMatPath);

    % Infer resultsMatPath from A3 selection if missing
    if strlength(resultsMatPath) == 0
        selDefault = fullfile("derived","analysis_runs",run_id,"stepA3_model_selection","selection.mat");
        if isfile(selDefault)
            Ssel = load(selDefault, "selection");
            if isfield(Ssel, "selection") && isfield(Ssel.selection, "results_file") && ~isempty(Ssel.selection.results_file)
                resultsMatPath = string(Ssel.selection.results_file);
            end
        end
    end

    dt = NaN;
    if strlength(resultsMatPath) > 0
        must_exist_file(resultsMatPath, "Fit results MAT (for dt/theta discovery)");
        R = load(resultsMatPath);
        if isfield(R, "cfg") && isfield(R.cfg, "dt") && ~isempty(R.cfg.dt)
            dt = double(R.cfg.dt);
        end
    end
    if ~isfinite(dt)
        error("[A7] Could not resolve dt. Provide ResultsMatPath or ensure A3 selection.mat has results_file with cfg.dt.");
    end

    % Load theta_star
    theta_star = [];

    % 1) User-specified theta mat
    if strlength(selectedThetaMatPath) > 0
        must_exist_file(selectedThetaMatPath, "SelectedThetaMatPath");
        Sth = load(selectedThetaMatPath);
        theta_star = find_theta_in_struct(Sth);
    end

    % 2) A3 selection.mat
    if isempty(theta_star)
        selPath = fullfile("derived","analysis_runs",run_id,"stepA3_model_selection","selection.mat");
        if isfile(selPath)
            Ssel = load(selPath, "selection");
            if isfield(Ssel, "selection") && isfield(Ssel.selection, "theta_star") && ~isempty(Ssel.selection.theta_star)
                theta_star = Ssel.selection.theta_star(:);
            end
        end
    end

    % 3) A3 theta_star.mat
    if isempty(theta_star)
        thPath = fullfile("derived","analysis_runs",run_id,"stepA3_model_selection","theta_star.mat");
        if isfile(thPath)
            Sth = load(thPath);
            theta_star = find_theta_in_struct(Sth);
        end
    end

    if isempty(theta_star)
        error("[A7] theta_star not found. Provide SelectedThetaMatPath or run Step A3.");
    end
    theta_star = theta_star(:);

    % -------------------------
    % Output directory
    % -------------------------
    outDir = string(args.OutDir);
    if strlength(outDir) == 0
        outDir = fullfile("derived", "analysis_runs", run_id, "stepA7_behavior_dataset");
    end
    ensure_dir(outDir);

    outMat  = fullfile(outDir, sprintf("behavior_dataset_%s.mat", splitName));
    outCsv  = fullfile(outDir, sprintf("behavior_dataset_%s.csv", splitName));
    sumCsv  = fullfile(outDir, sprintf("behavior_dataset_%s_summary.csv", splitName));
    bypCsv  = fullfile(outDir, sprintf("behavior_dataset_%s_by_participant.csv", splitName));
    metaMat = fullfile(outDir, sprintf("meta_%s.mat", splitName));
    metaJson= fullfile(outDir, sprintf("meta_%s.json", splitName));

    if ~args.Overwrite
        if isfile(outMat) || isfile(outCsv)
            error("[A7] Output exists (set Overwrite=true to replace) in %s", outDir);
        end
    end

    % -------------------------
    % Build dataset (one row per door trial)
    % -------------------------
    pid_all   = strings(0,1);
    trial_g   = zeros(0,1);   % 1..60
    block     = zeros(0,1);   % 1..3
    trial_b   = zeros(0,1);   % 1..20
    t_door    = zeros(0,1);   % door time [s]
    risk      = NaN(0,1);     % risk at door
    tau_dec   = NaN(0,1);     % decision trust (pre-door-update)
    sc_all    = NaN(0,1);     % self-confidence (participant constant)
    y_follow  = NaN(0,1);     % 1=follow, 0=override
    y_outcome = NaN(0,1);     % 1=success, 0=failure (if present)

    nP = numel(participants);

    for i = 1:nP
        P   = participants(i);
        pid = pid_list(i);

        % Simulate model once (fast) to obtain grid, doorEvents, tau_hist
        sim = trust_simulate_or_predict_one_participant(mode, theta_star, P, dt);

        if ~isfield(sim, "t_grid") || ~isfield(sim, "tau_hist")
            error("[A7] Simulator missing t_grid/tau_hist for pid=%s.", pid);
        end
        if ~isfield(sim, "doorEvents") || isempty(sim.doorEvents)
            % No doors -> skip (unexpected given preprocessing, but safe)
            continue;
        end

        t_grid    = double(sim.t_grid(:));
        tau_hist  = double(sim.tau_hist(:));
        doorEvents= sim.doorEvents;

        % Self-confidence from sim
        sc = NaN;
        if isfield(sim, "sc") && ~isempty(sim.sc)
            sc = double(sim.sc);
        end

        if ~isfinite(sc)
            warning("[A7] pid=%s has NaN self-confidence (sc). Check upstream preprocessing.", pid);
        end

        nDoor = numel(doorEvents);

        % We assume 60 doors; keep robust checks but do not hard error
        if nDoor ~= 60
            warning("[A7] pid=%s has %d doorEvents (expected 60). Proceeding.", pid, nDoor);
        end

        for d = 1:nDoor
            ev = doorEvents(d);

            % door time
            td = NaN;
            if isfield(ev, "t") && ~isempty(ev.t), td = double(ev.t); end

            % grid index at door time (preferred if cached)
            k = NaN;
            if isfield(ev, "k_grid") && ~isempty(ev.k_grid)
                k = double(ev.k_grid);
            else
                % fallback: nearest t_grid index
                if isfinite(td) && ~isempty(t_grid)
                    [~, k] = min(abs(t_grid - td));
                else
                    k = 1;
                end
            end
            if ~isfinite(k) || k < 1, k = 1; end
            if k > numel(tau_hist), k = numel(tau_hist); end

            % decision trust is pre-update at door -> previous grid index
            k_pre = max(k - 1, 1);
            tau_d = tau_hist(k_pre);

            % risk
            r = NaN;
            if isfield(ev, "risk_value") && ~isempty(ev.risk_value) && isfinite(ev.risk_value)
                r = double(ev.risk_value);
            elseif isfield(sim, "risk_hist") && ~isempty(sim.risk_hist)
                rh = double(sim.risk_hist(:));
                if k >= 1 && k <= numel(rh), r = rh(k); end
            end

            % behavior labels
            yF = NaN;
            if isfield(ev, "followed") && ~isempty(ev.followed)
                yF = double(ev.followed);
            end
            yO = NaN;
            if isfield(ev, "outcome") && ~isempty(ev.outcome)
                yO = double(ev.outcome);
            end

            % trial indexing
            g  = d; % global door index in sequence
            b  = ceil(g / 20);
            tb = g - (b-1)*20;

            pid_all(end+1,1)    = pid; %#ok<AGROW>
            trial_g(end+1,1)    = g;   %#ok<AGROW>
            block(end+1,1)      = b;   %#ok<AGROW>
            trial_b(end+1,1)    = tb;  %#ok<AGROW>
            t_door(end+1,1)     = td;  %#ok<AGROW>
            risk(end+1,1)       = r;   %#ok<AGROW>
            tau_dec(end+1,1)    = tau_d; %#ok<AGROW>
            sc_all(end+1,1)     = sc;  %#ok<AGROW>
            y_follow(end+1,1)   = yF;  %#ok<AGROW>
            y_outcome(end+1,1)  = yO;  %#ok<AGROW>
        end
    end

    % -------------------------
    % Assemble table (+ derived fields)
    % -------------------------
    T = table();
    T.participant_id  = pid_all;
    T.door_index      = trial_g;
    T.block_index     = block;
    T.trial_in_block  = trial_b;
    T.t_door_s        = t_door;
    T.risk            = risk;
    T.tau_decision    = tau_dec;
    T.self_confidence = sc_all;

    % Derived fields for behavioral models
    T.sc_centered     = T.self_confidence - 0.5;
    T.margin_treshold = T.tau_decision - T.self_confidence; % (tau - sc), baseline threshold-model margin

    % Labels
    T.followed        = y_follow;
    T.outcome         = y_outcome;

    % Valid label flag (do not drop rows; A8/A9 can filter)
    T.is_valid_label  = isfinite(T.followed) & (T.followed==0 | T.followed==1);

    % -------------------------
    % Summaries
    % -------------------------
    meta = struct();
    meta.run_id      = char(run_id);
    meta.split       = char(splitName);
    meta.mode        = char(mode);
    meta.dt          = dt;
    meta.theta_star  = theta_star;
    meta.theta_dim   = numel(theta_star);
    meta.participants_file = char(partMat);
    meta.a1_manifest = char(manifestPath);
    meta.created     = char(datetime('now','Format','yyyy-MM-dd HH:mm:ss'));
    meta.n_participants = nP;
    meta.n_rows = height(T);
    meta.n_sc_nan = sum(~isfinite(T.self_confidence));

    % Overall summary
    sumT = table();
    sumT.split          = splitName;
    sumT.n_participants = nP;
    sumT.n_rows         = height(T);
    sumT.n_valid_labels = sum(T.is_valid_label);

    if sumT.n_valid_labels > 0
        sumT.follow_rate = mean(T.followed(T.is_valid_label));
    else
        sumT.follow_rate = NaN;
    end

    sumT.tau_mean   = mean(T.tau_decision, 'omitnan');
    sumT.tau_std    = std(T.tau_decision, 'omitnan');
    sumT.risk_mean  = mean(T.risk, 'omitnan');
    sumT.sc_mean    = mean(T.self_confidence, 'omitnan');
    sumT.sc_nan_rows= sum(~isfinite(T.self_confidence));
    sumT.margin_mean= mean(T.margin_treshold, 'omitnan');

    % Per-participant summary
    byp = table();
    byp.participant_id = unique(T.participant_id);
    byp.N = zeros(height(byp),1);
    byp.N_valid = zeros(height(byp),1);
    byp.follow_rate = NaN(height(byp),1);
    byp.tau_mean = NaN(height(byp),1);
    byp.risk_mean = NaN(height(byp),1);
    byp.sc = NaN(height(byp),1);
    byp.sc_centered = NaN(height(byp),1);
    byp.margin_mean = NaN(height(byp),1);

    for ii = 1:height(byp)
        pid = byp.participant_id(ii);
        mask = (T.participant_id == pid);

        byp.N(ii) = sum(mask);
        maskV = mask & T.is_valid_label;
        byp.N_valid(ii) = sum(maskV);

        if byp.N_valid(ii) > 0
            byp.follow_rate(ii) = mean(T.followed(maskV));
        end

        byp.tau_mean(ii)   = mean(T.tau_decision(mask), 'omitnan');
        byp.risk_mean(ii)  = mean(T.risk(mask), 'omitnan');
        byp.margin_mean(ii)= mean(T.margin_treshold(mask), 'omitnan');

        % self-confidence constant per participant (take first finite)
        scv = T.self_confidence(mask);
        scv = scv(isfinite(scv));
        if ~isempty(scv)
            byp.sc(ii) = scv(1);
            byp.sc_centered(ii) = scv(1) - 0.5;
        end
    end

    % -------------------------
    % Save outputs
    % -------------------------
    save(outMat, "T", "-v7.3");
    writetable(T, outCsv);
    writetable(sumT, sumCsv);
    writetable(byp, bypCsv);

    save(metaMat, "meta");
    save_json(metaJson, meta);

    fprintf("[Step A7] Behavior dataset built (%s).\n", splitName);
    fprintf("          Rows: %d (valid labels: %d)\n", height(T), sum(T.is_valid_label));
    fprintf("          Output dir: %s\n", outDir);
end
