function [t_grid, doors_at_k, doorEvents, meas_at_k, measurements] = ...
         build_time_grid_and_events(P, dt)
% build_time_grid_and_events  Build time grid and attach door/measurement events.
%
%   [t_grid, doors_at_k, doorEvents, meas_at_k, measurements] = ...
%        build_time_grid_and_events(P, dt)
%
% This function constructs a uniform time grid for a single participant and
% attaches:
%   - door events (for trust dynamics updates), and
%   - trust measurements (questionnaires and probes mapped to a 40-item scale).
%
% It does not simulate any trust dynamics; it only prepares the temporal
% structure and event bookkeeping used by the trust model.
%
% Inputs:
%   P   - Participant struct from participants_time_stepT1.mat, expected to
%         contain:
%           P.doorTrials(k).t_s      : door completion time (seconds)
%           P.trustProbes(j).t_s     : trust-probe time (seconds)
%           P.trustProbes(j).value_40: trust value in 0..100 mapped to
%                                      40-item scale
%           P.trustProbes(j).questionnaire_type (optional):
%                                      "t40_pre", "t40_post",
%                                      "t14_mid1", "t14_mid2", or ""
%           P.questionnaires.t40_post.t_s, t14_mid1.t_s, t14_mid2.t_s
%                                      (used only to ensure time coverage)
%   dt  - Time step size in seconds for the grid (e.g., 0.1). If omitted
%         or empty, defaults to 0.1.
%
% Outputs:
%   t_grid      - Row vector of time points [0, dt, 2*dt, ..., T_max].
%
%   doorEvents  - Struct array (nDoor x 1) with fields:
%                   .idx_door    - index into P.doorTrials (1..nDoor)
%                   .t           - door event time (seconds)
%                   .k_grid      - index into t_grid
%                   .outcome     - 1=success, 0=failure (from .correct)
%                   .followed    - 1=followed, 0=overrode
%                   .risk_value  - risk_value used for situational trust
%                   .block_index - block index (if available)
%                   .trial_index - trial index within block (if available)
%
%   doors_at_k  - Cell array (K x 1), where doors_at_k{k} is a vector of
%                 indices into doorEvents for doors that occur at t_grid(k).
%
%   measurements - Struct array (nMeas x 1) with fields:
%                    .t        - measurement time (seconds)
%                    .k_grid   - index into t_grid
%                    .y        - trust value in [0,1] (normalized 40-scale)
%                    .kind     - "t40_post", "t14_mid1", "t14_mid2", "probe"
%                    .label    - label string (e.g., "P001_t14_mid1_3")
%
%   meas_at_k   - Cell array (K x 1), where meas_at_k{k} is a vector of
%                 indices into measurements for measurements at t_grid(k).
%
% Assumptions:
%   - Time enrichment (Step T1) has already been applied so that:
%       * P.doorTrials(:).t_s and P.trustProbes(:).t_s exist and are
%         relative times (seconds).
%   - All trust measurements (probes and questionnaire-based) have already
%     been mapped to a 40-item equivalent percentage and stored in
%     P.trustProbes(j).value_40 (0..100). The questionnaire_type field
%     indicates whether a probe aligns with t40_post, t14_mid1, t14_mid2,
%     or is simply a mid-interaction probe.
%   - The pre-40 questionnaire (t40_pre) is used as an anchor during
%     preprocessing but is intentionally skipped as a measurement here.
%
% Important:
%   - This function only constructs the temporal grid and event lists. It
%     does not update or evolve trust states.

    if nargin < 2 || isempty(dt)
        dt = 0.1;  % default 100 ms
    end

    % ------------------------------------------------------------
    % 1) Collect all relevant times: doors + measurements
    % ------------------------------------------------------------
    door_t = [];
    if isfield(P, "doorTrials") && ~isempty(P.doorTrials)
        % Door event times (seconds)
        door_t = [P.doorTrials.t_s];   % row or column, doesn't matter
    end

    meas_t = [];

    % Questionnaires (40-post / 14-mid1 / 14-mid2)
    % These times ensure that the time grid covers all measurement events.
    if isfield(P, "questionnaires") && ~isempty(P.questionnaires)
        Q = P.questionnaires;

        if isfield(Q, "t40_post") && isfield(Q.t40_post, "t_s")
            meas_t(end+1) = Q.t40_post.t_s; 
        end
        if isfield(Q, "t14_mid1") && isfield(Q.t14_mid1, "t_s")
            meas_t(end+1) = Q.t14_mid1.t_s; 
        end
        if isfield(Q, "t14_mid2") && isfield(Q.t14_mid2, "t_s")
            meas_t(end+1) = Q.t14_mid2.t_s; 
        end
    end

    % Trust probes (all trust measurements are ultimately taken from here)
    probe_t = [];
    if isfield(P, "trustProbes") && ~isempty(P.trustProbes)
        probe_t = [P.trustProbes.t_s];
        meas_t  = [meas_t, probe_t];
    end

    % All times considered for grid construction
    all_times = [door_t, meas_t];

    if isempty(all_times)
        error("build_time_grid_and_events: no door or measurement times found for this participant.");
    end

    % Ensure times are non-negative overall (T_max must be >= 0). Negative
    % times may exist but only relative to the chosen origin; we require at
    % least one non-negative endpoint to define the grid extent.
    T_max = max(all_times);
    if T_max < 0
        error("build_time_grid_and_events: all times are negative? Check time enrichment.");
    end

    % ------------------------------------------------------------
    % 2) Build time grid: from 0 to T_max + one dt step
    % ------------------------------------------------------------
    t_grid = 0:dt:(T_max + dt);
    K      = numel(t_grid);

    % Helper to map a continuous time (seconds) to a grid index. Rounds to
    % nearest grid point and clamps to [1, K] to avoid indexing errors.
    time2idx = @(t) max(1, min(K, round(t ./ dt) + 1));  % clamp just in case

    % ------------------------------------------------------------
    % 3) Door events: map each doorTrial to grid index
    % ------------------------------------------------------------
    % Define a template struct for one door event.
    doorTemplate = struct( ...
        'idx_door',   NaN, ...
        't',          NaN, ...
        'k_grid',     NaN, ...
        'outcome',    NaN, ...
        'followed',   NaN, ...
        'risk_value', NaN, ...
        'block_index',NaN, ...
        'trial_index',NaN);

    doorEvents = doorTemplate([]);  % empty by default

    if isempty(door_t)
        doors_at_k = cell(K,1);
    else
        nDoor = numel(P.doorTrials);
        % Preallocate doorEvents to the number of door trials.
        doorEvents = repmat(doorTemplate, nDoor, 1);

        for d = 1:nDoor
            t_door = P.doorTrials(d).t_s;
            k_grid = time2idx(t_door);

            % Outcome from .correct: 1=success, 0=failure.
            outcome  = double(getfield_default(P.doorTrials(d), "correct", 0));
            % Followed: 1=followed recommendation, 0=override.
            followed = double(getfield_default(P.doorTrials(d), "followed", 1));
            % Risk value used for situational trust.
            risk_val = getfield_default(P.doorTrials(d), "risk_value", NaN);
            % Block and trial indices for reference.
            blk_idx  = getfield_default(P.doorTrials(d), "block_index", NaN);
            tri_idx  = getfield_default(P.doorTrials(d), "trial_index", NaN);

            doorEvents(d).idx_door    = d;
            doorEvents(d).t           = t_door;
            doorEvents(d).k_grid      = k_grid;
            doorEvents(d).outcome     = outcome;
            doorEvents(d).followed    = followed;
            doorEvents(d).risk_value  = risk_val;
            doorEvents(d).block_index = blk_idx;
            doorEvents(d).trial_index = tri_idx;
        end

        % Inverse map: for each grid index k, record which doors occur there.
        doors_at_k = cell(K,1);
        for d = 1:nDoor
            k = doorEvents(d).k_grid;
            doors_at_k{k}(end+1) = d; %#ok<AGROW>
        end
    end

    % ------------------------------------------------------------
    % 4) Measurements: used for fitting and diagnostics
    %
    % All trust measurements (questionnaire-based and probes) are taken
    % from P.trustProbes, where:
    %   - value_40 holds the 0..100 equivalent on the 40-item scale, and
    %   - questionnaire_type marks special questionnaire-linked entries.
    % The pre-40 (t40_pre) anchor is intentionally skipped.
    % ------------------------------------------------------------
    measurements = struct( ...
        't',      {}, ...   % time in seconds
        'k_grid', {}, ...   % nearest time-grid index
        'kind',   {}, ...   % "t40_post", "t14_mid1", "t14_mid2", "probe"
        'y',      {}, ...   % scalar in [0,1] (normalized from 0..100)
        'label',  {}  ...   % optional string ID
    );
    
        function add_meas(t_val, y_perc, kindStr, labelStr)
            % Local helper to append one measurement:
            %   - maps time to grid index,
            %   - normalizes percentage (0..100) to [0,1],
            %   - appends struct to the measurements array.
            k = time2idx(t_val);
            % Convert 0..100 → 0..1
            y_norm = y_perc / 100;
    
            m = struct();
            m.t      = t_val;
            m.k_grid = k;
            m.kind   = string(kindStr);
            m.y      = y_norm;
            m.label  = string(labelStr);
    
            measurements(end+1,1) = m; %#ok<AGROW>
        end
    
    % ---- 4a) Use P.trustProbes for ALL trust measurements ----
    if isfield(P, "trustProbes") && ~isempty(P.trustProbes)
        nTP = numel(P.trustProbes);
    
        for j = 1:nTP
            TP = P.trustProbes(j);
    
            % Require a valid time stamp and a valid 40-item mapped value.
            if ~isfield(TP, "t_s") || isempty(TP.t_s) || ~isfinite(TP.t_s)
                continue;
            end
            if ~isfield(TP, "value_40") || isempty(TP.value_40) ...
                    || ~isfinite(double(TP.value_40))
                continue;
            end
    
            t_val  = TP.t_s;
            y_perc = double(TP.value_40);  % 0..100
    
            % Determine questionnaire linkage, if any.
            qt = "";
            if isfield(TP, "questionnaire_type") && ~isempty(TP.questionnaire_type)
                qt = string(TP.questionnaire_type);
            end
    
            % Skip the pre-40 anchor (used elsewhere in preprocessing).
            if qt == "t40_pre"
                continue;
            end
    
            % Choose 'kind' for this measurement:
            %   - questionnaire-linked: "t14_mid1", "t14_mid2", "t40_post"
            %   - plain probe: "probe"
            if qt == ""
                kindStr = "probe";
            else
                kindStr = qt;   % "t14_mid1", "t14_mid2", "t40_post"
            end
    
            % Construct a label for diagnostics and plotting.
            labelStr = sprintf("%s_%s_%d", ...
                string(P.participant_id), kindStr, j);
    
            add_meas(t_val, y_perc, kindStr, labelStr);
        end
    end

    % ------------------------------------------------------------
    % 5) Build meas_at_k: indices of measurements at each grid time
    % ------------------------------------------------------------
    meas_at_k = cell(K,1);
    nMeas = numel(measurements);
    for m = 1:nMeas
        k = measurements(m).k_grid;
        meas_at_k{k}(end+1) = m; %#ok<AGROW>
    end 
end

% ------------------------------------------------------------
% Small helper: safe field access with default
% ------------------------------------------------------------
function v = getfield_default(S, fieldName, defaultVal)
% getfield_default  Return S.(fieldName) if it exists and is non-empty; otherwise defaultVal.
%
%   v = getfield_default(S, fieldName, defaultVal)
%
% This helper avoids repeated isfield/empty checks when extracting fields
% from doorTrials and similar structs.
    if isstruct(S) && isfield(S, fieldName)
        v = S.(fieldName);
        if isempty(v)
            v = defaultVal;
        end
    else
        v = defaultVal;
    end
end
