function state_next = trust_step(state_cur, event, params)
% trust_step  Advance the trust state by a single time step on the grid.
%
%   state_next = trust_step(state_cur, event, params)
%
% This function updates the full trust state from time t_k to t_{k+1},
% where the next time point is given by event.t. It combines:
%   - reputation decay over the interval Δt (only after the first door
%     trial is observed),
%   - latent trust evolution between door events,
%   - discrete personal-experience updates at door trials,
%   - situational trust given the current / last risk level.
%
% The total trust is decomposed as:
%
%       tau[k] = tau_lat[k] + tau_rep[k] + alpha_sit*tau_sit(r[k]),
%
% and evolves according to two regimes:
%
%   Rule 1 (HRI event, i.e. door trial at this step):
%       tau_lat[k+1] = tau_lat[k] + tau_exp(n)
%                     (personal experience update only)
%       (then a new latent episode is prepared from tau_lat[k+1])
%       tau_rep[k+1] = decay(tau_rep[k], Δt_rep)
%                     with Δt_rep = 0 for the very first door trial
%                     and Δt_rep = Δt thereafter
%       tau_sit[k+1] = tau_sit(r_event)
%       tau[k+1]     = tau_lat[k+1] + tau_rep[k+1] + tau_sit[k+1]
%
%   Rule 2 (no HRI event in this step):
%       tau_lat[k+1] = tau_lat^lat_sequence(tau_lat[k], tau_disp; Δt)
%       tau_rep[k+1] = decay(tau_rep[k], Δt_rep)
%                     with Δt_rep = 0 before the first door trial
%                     and Δt_rep = Δt afterwards
%       tau_sit[k+1] = tau_sit(r_last)
%       tau[k+1]     = tau_lat[k+1] + tau_rep[k+1] + tau_sit[k+1]
%
% followed by clipping of tau[k+1] to [0,1].
%
% Inputs:
%   state_cur   Current trust state (see trust_init_state), with fields:
%                 .t           Current time (seconds)
%                 .tau         Current total trust in [0,1]
%                 .tau_disp    Dispositional trust in (0,1)
%                 .tau_rep     Current reputation component in [-1,1]
%                 .tau_lat     Current latent component in [0,1]
%                 .exp         Struct with personal-experience counters:
%                                 .n_succ  consecutive successes
%                                 .n_fail  consecutive failures
%                 .last_risk   Last seen risk_value (scalar in [0,1] or NaN)
%                 .sc          Self-confidence (scalar in [0,1], optional)
%                 .lat         Latent-episode state (mode, lambda_seq, etc.)
%                 .rep_active  Logical flag: true iff reputation decay
%                              is currently active (optional; defaults false)
%                 .participant_id (optional, for diagnostics)
%
%   event       Struct describing what happens at the next time point:
%                 .t          Absolute time of this step (seconds)
%                 .type       Event type, e.g.:
%                               "door", "t40_post", "t14_mid1",
%                               "t14_mid2", "probe", ...
%                 .risk_value Risk value for door events (scalar, optional)
%                 .outcome    Door outcome (1=success, 0=failure) for doors
%                 .followed   1=followed, 0=overrode (door events)
%
%   params      Parameter struct with subfields:
%                 .lat, .rep, .exp, .sit, ...
%               These substructs are passed to the respective update
%               functions (latent, reputation, personal experience,
%               situational trust).
%
% Output:
%   state_next  Updated state at time event.t, with all components
%               advanced by the time step and current event.

    % Start from the current state; update fields in state_next
    state_next = state_cur;

    % ------------------------------------------------------------
    % 1) Time step Δt between current time and event time
    % ------------------------------------------------------------
    dt = event.t - state_cur.t;
    if dt < 0
        warning("trust_step: negative dt (%.3f). Forcing dt=0.", dt);
        dt = 0;
    end

    % Determine whether this step corresponds to a door (HRI) event.
    isDoorEvent = isfield(event, "type") && (event.type == "door");

    % ------------------------------------------------------------
    % 2) Reputation decay with activation gating
    % ------------------------------------------------------------
    % Reputation is kept constant until the first door trial. After that,
    % it decays continuously over the exact time intervals between events.
    %
    %   - Before any door:       Δt_rep = 0  → no decay
    %   - At the first door:     Δt_rep = 0  → decay starts afterwards
    %   - After first door:      Δt_rep = Δt
    %
    rep_active_cur = getfield_with_default(state_cur, "rep_active", false);

    if ~rep_active_cur
        if isDoorEvent
            % First door trial: enable reputation dynamics but do not
            % retroactively apply decay over the pre-door interval.
            dt_rep = 0;
            rep_active_next = true;
        else
            % Still before any door trial: freeze reputation.
            dt_rep = 0;
            rep_active_next = false;
        end
    else
        % After the first door: decay over the full interval Δt.
        dt_rep = dt;
        rep_active_next = true;
    end

    tau_rep_next = trust_update_reputation(state_cur.tau_rep, dt_rep, params);
    state_next.tau_rep    = tau_rep_next;
    state_next.rep_active = rep_active_next;

    % ------------------------------------------------------------
    % 3) Self-confidence and risk (for situational trust)
    % ------------------------------------------------------------
    % Self-confidence: participant-specific level stored in state, with
    % a neutral default if absent.
    sc = getfield_with_default(state_cur, "sc", 0.5);

    % Retrieve last known risk (if any)
    if ~isfield(state_cur, "last_risk")
        last_risk = NaN;
    else
        last_risk = state_cur.last_risk;
    end

    % If this event carries a new risk_value (door trial), update it.
    % Otherwise use the last seen risk (for non-door events).
    if isfield(event, "risk_value") && ~isempty(event.risk_value) ...
            && isfinite(event.risk_value)
        r = event.risk_value;
    else
        r = last_risk;
    end
    state_next.last_risk = r;

    % Compute situational trust for current risk r.
    % If r is NaN (no risk seen yet) the situational term is suppressed.
    if isnan(r)
        tau_sit = 0.0;
    else
        tau_sit = trust_compute_situational(r, state_cur.tau_disp, sc, params);
    end
    
    % Get weight for situational trust component (alpha_sit)
    if isfield(params, "sit") && isfield(params.sit, "lambda_sit")
        alpha_sit = params.sit.alpha_sit;
    end

    % ------------------------------------------------------------
    % 4) Ensure latent-related fields exist (tau_lat and state.lat)
    % ------------------------------------------------------------
    % Latent component (fallback if absent: total minus reputation,
    % ignoring any situational effect).
    if ~isfield(state_cur, "tau_lat") || isempty(state_cur.tau_lat)
        tau_lat_cur = state_cur.tau - state_cur.tau_rep;
    else
        tau_lat_cur = state_cur.tau_lat;
    end

    % Latent-episode state (episode-specific parameters)
    if ~isfield(state_cur, "lat") || isempty(state_cur.lat)
        lat_state_cur = struct();
        lat_state_cur.mode       = "none";
        lat_state_cur.lambda_seq = NaN;
        lat_state_cur.kappa_seq  = NaN;
        lat_state_cur.sigma      = NaN;
        lat_state_cur.tau0       = tau_lat_cur;
    else
        lat_state_cur = state_cur.lat;
    end

    % Personal-experience substate (streak counters)
    exp_state_next = state_cur.exp;
    if ~isfield(exp_state_next, "n_succ"), exp_state_next.n_succ = 0; end
    if ~isfield(exp_state_next, "n_fail"), exp_state_next.n_fail = 0; end

    % ------------------------------------------------------------
    % 5) Latent evolution and personal experience
    % ------------------------------------------------------------
    tau_lat_next   = tau_lat_cur;
    delta_tau_exp  = 0.0;
    lat_state_next = lat_state_cur;

    if isDoorEvent
        % --------------------------------------------------------
        % Rule 1: HRI event (door trial)
        %
        %   - Personal-experience term updates latent trust based on
        %     success/failure streaks.
        %   - A new latent episode is prepared starting from the updated
        %     latent value.
        % --------------------------------------------------------
        outcome  = getfield_with_default(event, "outcome", 0);
        followed = getfield_with_default(event, "followed", 1);

        % Personal-experience update → Δτ_exp(n)
        [delta_tau_exp, exp_state_next] = trust_update_personal_experience( ...
            state_cur.exp, outcome, followed, params);

        % Update latent component by adding the event-based increment
        tau_lat_next = tau_lat_cur + delta_tau_exp;

        % Prepare a new latent episode starting at this updated level
        lat_state_next = trust_prepare_latent_sequence( ...
            tau_lat_next, state_cur.tau_disp, params);

    else
        % --------------------------------------------------------
        % Rule 2: No HRI event in this time step
        %
        %   - Latent trust evolves according to the latent sequence
        %     dynamics (above/below/none) over Δt.
        % --------------------------------------------------------
        [tau_lat_next, lat_state_next] = trust_update_latent_sequence( ...
            tau_lat_cur, state_cur.tau_disp, dt, params, lat_state_cur);
    end

    % ------------------------------------------------------------
    % 6) Combine components and clip total trust
    % ------------------------------------------------------------
    tau_combined = tau_lat_next + tau_rep_next + (alpha_sit*tau_sit);

    % Participant ID for diagnostic logging (optional)
    pid = "";
    if isfield(state_cur, "participant_id")
        pid = string(state_cur.participant_id);
    end

% % --- (OPTIONAL) Diagnostic logging for anomalous values (pre-clipping)  ---
% % --- Kept commented out by default, because it blows up simulation time ---
%     if ~isreal(tau_lat_next) || ~isfinite(tau_lat_next) || ...
%             tau_lat_next < -1e-3 || tau_lat_next > 1+1e-3
%         trust_debug_log("tau_lat_invalid", struct( ...
%             "participant_id", pid, ...
%             "t", state_cur.t, ...
%             "event_type", getfield_with_default(event, "type", ""), ...
%             "tau_lat_cur", tau_lat_cur, ...
%             "tau_lat_next_raw", tau_lat_next, ...
%             "tau_disp", state_cur.tau_disp, ...
%             "dt", dt));
%     end
% 
%     if ~isreal(tau_rep_next) || ~isfinite(tau_rep_next) || ...
%             tau_rep_next < -1-1e-3 || tau_rep_next > 1+1e-3
%         trust_debug_log("tau_rep_invalid", struct( ...
%             "participant_id", pid, ...
%             "t", state_cur.t, ...
%             "event_type", getfield_with_default(event, "type", ""), ...
%             "tau_rep_cur", state_cur.tau_rep, ...
%             "tau_rep_next_raw", tau_rep_next, ...
%             "dt_rep", dt_rep));
%     end
% 
%     if ~isreal(tau_sit) || ~isfinite(tau_sit) || ...
%             tau_sit < -1-1e-3 || tau_sit > 1+1e-3
%         trust_debug_log("tau_sit_invalid", struct( ...
%             "participant_id", pid, ...
%             "t", state_cur.t, ...
%             "event_type", getfield_with_default(event, "type", ""), ...
%             "tau_disp", state_cur.tau_disp, ...
%             "sc", sc, ...
%             "risk", r, ...
%             "tau_sit_raw", tau_sit));
%     end
% 
%     if ~isreal(tau_combined) || ~isfinite(tau_combined)
%         trust_debug_log("tau_combined_invalid", struct( ...
%             "participant_id", pid, ...
%             "t", state_cur.t, ...
%             "event_type", getfield_with_default(event, "type", ""), ...
%             "tau_lat_next_raw", tau_lat_next, ...
%             "tau_rep_next_raw", tau_rep_next, ...
%             "tau_sit", tau_sit, ...
%             "tau_combined_raw", tau_combined));
%     end
%     % --- End diagnostic logging ---

    % Apply final clipping to keep total trust in [0,1]
    tau_next = trust_clip(tau_combined);

    % ------------------------------------------------------------
    % 7) Populate updated state fields
    % ------------------------------------------------------------
    state_next.t       = event.t;
    state_next.tau     = tau_next;
    state_next.tau_lat = tau_lat_next;
    state_next.exp     = exp_state_next;   % updated streak counters
    state_next.lat     = lat_state_next;   % updated latent-episode state
end


% -----------------------------------------------------------------
% Local helper: safe field access with default
% -----------------------------------------------------------------
function v = getfield_with_default(S, fieldName, defaultVal)
% getfield_with_default  Safe struct field access with fallback.
%
%   v = getfield_with_default(S, fieldName, defaultVal)
%
% Returns S.(fieldName) if it exists, otherwise defaultVal.

    if isstruct(S) && isfield(S, fieldName)
        v = S.(fieldName);
    else
        v = defaultVal;
    end
end
