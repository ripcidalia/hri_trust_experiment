function sim = trust_simulate_or_predict_one_participant(mode, theta, P, dt, behavior_params)
% trust_simulate_or_predict_one_participant  Simulate or predict trust dynamics for one participant.
%
%   sim = trust_simulate_or_predict_one_participant(mode, theta, P, dt, behavior_params)
%
% This function runs a full forward simulation of the trust model for a
% single participant, given a parameter vector θ and a time step size.
% It builds or uses a cached version of the time grid, attaches all door
% events and trust measurements, steps the trust dynamics forward in time,
% and records:
%   - the trajectory of total trust and its components, and
%   - model predictions at the times where trust is measured.
%
% The function will either use the recorded door event trials outcomes to deterministically
% update trust dynamics (mode: "simple"), or couple the behavioral model to predict user
% choices at door trials (mode: "coupled"). In the case of coupling with the behavioral model,
% another parameter ("behavior_params") can be passed to control the behavior_params of the probabilistic
% transition between the two behavioral states (follow/not follow).
%
% Inputs:
%   mode
%       Simulation mode (string):
%         - "simple", or "coupled".
%
%   theta
%       Parameter vector (column or row) that is interpreted by
%       trust_theta_to_params. The exact layout of theta is handled
%       entirely inside trust_theta_to_params.
%
%   P
%       Participant struct, typically coming from:
%         - participants_probes_mapped_stepM4.mat
%       and enriched by:
%         - stepT1_add_times (P.doorTrials(k).t_s, P.trustProbes(j).t_s, etc.)
%       It is assumed that build_time_grid_and_events(P, dt) succeeds
%       and that trust_init_state can initialize a valid state.
%
%   dt
%       Time step in seconds for the simulation time grid.
%       If omitted or empty, dt defaults to 1 second.
%
%   behavior_params
%       Relevant parameters for the behavioral model (required for "coupled").
%
% Output:
%   sim        Struct with simulation outputs:
%                 .t_grid        [K x 1] vector of time points (seconds)
%                 .doorEvents    [nDoor x 1] struct array (from get_cached_time_grid_and_events)
%                 .sc            self-confidence value (scalar)
%                 .tau_hist      [K x 1] total trust trajectory τ(t)
%                 .tau_lat_hist  [K x 1] latent component trajectory τ_lat(t)
%                 .tau_rep_hist  [K x 1] reputation component trajectory τ_rep(t)
%                 .tau_sit_hist  [K x 1] situational component trajectory τ_sit(t)
%                 .risk_hist     [K x 1] risk value history r(t)
%                 .measurements  struct array (including measurement times and ground-truth y)
%                 .y_hat         [nMeas x 1] model predictions at those measurement times
%
%               Coupled-mode-only additional logging (NEW; non-breaking):
%                 .coupled                   struct (only present in "coupled" mode)
%                 .coupled.followed_sampled  [nDoor x 1] sampled followed actions used by simulator
%                 .coupled.p_follow          [nDoor x 1] p_follow used for sampling at each door
%                 .coupled.outcome_used      [nDoor x 1] outcome applied after counterfactual inversion
%                 .coupled.t_door            [nDoor x 1] door event times (seconds)
%                 .coupled.block_index       [nDoor x 1] block index if present in doorEvents else NaN
%                 .coupled.door_index        [nDoor x 1] door index if present in doorEvents else (1:nDoor)
%
% Notes:
%   - The evolution within each time step is performed by trust_step, which
%     updates all trust components and internal substates.
%   - Multiple door events are allowed at the same grid time t_k and are
%     applied sequentially.
%   - Measurements (questionnaires and probes) do not alter the trust state;
%     they only induce a read-out of the current τ at their respective times.

    if isempty(mode)
        warning("trust_simulate_or_predict_one_participant: no mode specified. Proceeding with simple mode (no behavioral prediction).");
        mode = "simple";
    end

    if mode == "coupled" && (nargin < 5 || isempty(behavior_params))
        error("trust_simulate_or_predict_one_participant: no behavioral parameters specified for coupled mode. Please provide as input.");
    end

    if nargin < 4 || isempty(dt)
        dt = 1;
    end

    mode = string(lower(mode));

    % ------------------------------------------------------------
    % 1) Map θ → params
    % ------------------------------------------------------------
    params = trust_theta_to_params(theta);

    % ------------------------------------------------------------
    % 2) Build time grid and attach door / measurement events
    %    (CACHED: depends only on participant data + dt, not on theta)
    % ------------------------------------------------------------
    [t_grid, doors_at_k, doorEvents, meas_at_k, measurements] = ...
        get_cached_time_grid_and_events(P, dt);

    K     = numel(t_grid);
    nMeas = numel(measurements);
    nDoor = numel(doorEvents);

    % ------------------------------------------------------------
    % 3) Initialize trust state at t ≈ 0 (pre-40 anchor)
    % ------------------------------------------------------------
    state = trust_init_state(params, P);

    % Ensure latent component, last_risk and self-confidence are present.
    if ~isfield(state, "tau_lat") || isempty(state.tau_lat)
        state.tau_lat = state.tau - state.tau_rep;
    end
    if ~isfield(state, "last_risk")
        state.last_risk = NaN;
    end
    if ~isfield(state, "sc")
        state.sc = 0.5;
    end

    % Latent-episode substate (used by trust_update_latent_sequence)
    state.lat = struct();
    state.lat.mode       = "none";
    state.lat.lambda_seq = NaN;
    state.lat.kappa_seq  = NaN;
    state.lat.sigma      = NaN;
    state.lat.tau0       = state.tau_lat;

    % ------------------------------------------------------------
    % 4) Preallocate component histories and prediction array
    % ------------------------------------------------------------
    tau_hist      = NaN(K,1);
    tau_lat_hist  = NaN(K,1);
    tau_rep_hist  = NaN(K,1);
    tau_sit_hist  = NaN(K,1);
    risk_hist     = NaN(K,1);
    y_hat         = NaN(nMeas,1);

    % ------------------------------------------------------------
    % NEW: Coupled-mode per-door logging (non-breaking)
    % ------------------------------------------------------------
    if mode == "coupled"
        coupled = struct();
        coupled.followed_sampled = NaN(nDoor,1);
        coupled.p_follow          = NaN(nDoor,1);
        coupled.outcome_used      = NaN(nDoor,1);
        coupled.t_door            = NaN(nDoor,1);

        % Optional indices for convenience in A9 (robust to missing fields)
        coupled.block_index = NaN(nDoor,1);
        coupled.door_index  = NaN(nDoor,1);

        for d = 1:nDoor
            if isfield(doorEvents(d), "block_index")
                coupled.block_index(d) = double(doorEvents(d).block_index);
            end
            if isfield(doorEvents(d), "door_index")
                coupled.door_index(d) = double(doorEvents(d).door_index);
            elseif isfield(doorEvents(d), "door_index_global")
                coupled.door_index(d) = double(doorEvents(d).door_index_global);
            else
                coupled.door_index(d) = d; % safe fallback
            end
        end
    end

    % ------------------------------------------------------------
    % 5) Main simulation loop over time grid
    % ------------------------------------------------------------
    for k = 1:K
        t_k = t_grid(k);

        % Door events scheduled at this grid index (possibly multiple)
        door_indices = doors_at_k{k};

        if ~isempty(door_indices)
            for j = 1:numel(door_indices)
                d_idx = door_indices(j);
                D     = doorEvents(d_idx);

                ev = struct();
                ev.t          = t_k;
                ev.type       = "door";
                ev.risk_value = D.risk_value;

                switch mode
                    case "simple"
                        ev.outcome  = D.outcome;
                        ev.followed = D.followed;

                    case "coupled"
                        [ev.followed, p_follow] = behavioral_model(state, behavior_params);

                        % Counterfactual inversion rule (as you already use)
                        if ev.followed ~= D.followed
                            ev.outcome = 1 - D.outcome;
                        else
                            ev.outcome = D.outcome;
                        end

                        % NEW: log sampled decision + p_follow + applied outcome
                        coupled.followed_sampled(d_idx) = ev.followed;
                        coupled.p_follow(d_idx)          = p_follow;
                        coupled.outcome_used(d_idx)      = ev.outcome;
                        coupled.t_door(d_idx)            = t_k;

                    otherwise
                        error('trust_simulate_or_predict_one_participant: unknown mode "%s". Use "simple" or "coupled".', mode);
                end

                state = trust_step(state, ev, params);
            end
        else
            ev = struct();
            ev.t    = t_k;
            ev.type = "none";
            state = trust_step(state, ev, params);
        end

        % Store model predictions at any measurement indices assigned to this grid time
        if ~isempty(meas_at_k{k})
            for idx = meas_at_k{k}
                y_hat(idx) = state.tau;
            end
        end

        % Situational trust for logging (using the updated state and last_risk)
        if isnan(state.last_risk)
            tau_sit = 0.0;
        else
            tau_sit = trust_compute_situational( ...
                state.last_risk, state.tau_disp, state.sc, params);
        end

        % Record component histories and risk
        tau_hist(k)     = state.tau;
        tau_lat_hist(k) = state.tau_lat;
        tau_rep_hist(k) = state.tau_rep;
        tau_sit_hist(k) = tau_sit;
        risk_hist(k)    = state.last_risk;
    end

    % ------------------------------------------------------------
    % 6) Package output struct
    % ------------------------------------------------------------
    sim = struct();
    sim.t_grid       = t_grid(:);
    sim.doorEvents   = doorEvents;
    sim.sc           = state.sc;
    sim.tau_hist     = tau_hist;
    sim.tau_lat_hist = tau_lat_hist;
    sim.tau_rep_hist = tau_rep_hist;
    sim.tau_sit_hist = tau_sit_hist;
    sim.risk_hist    = risk_hist;
    sim.measurements = measurements;
    sim.y_hat        = y_hat;

    % NEW: attach coupled logging only in coupled mode
    if mode == "coupled"
        sim.coupled = coupled;
    end
end
