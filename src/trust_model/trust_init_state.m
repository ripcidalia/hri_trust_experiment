function state = trust_init_state(params, P)
% trust_init_state  Initialize the full trust state for one participant.
%
%   state = trust_init_state(params, P)
%
% This function constructs the initial trust state at the start of the
% simulation (typically aligned with the pre 40-item questionnaire time).
% It sets up:
%   - the dispositional anchor,
%   - the initial reputation component,
%   - the initial latent baseline (aligned with dispositional trust),
%   - the total trust (without situational contribution),
%   - the personal-experience substate (streak counters),
%   - the self-confidence scalar used in the situational component.
%
% Inputs:
%   params        Struct with model parameter substructs:
%                    .disp, .sit, .rep, .exp, .lat
%                 This function currently uses:
%                    params.rep   (for trust_initial_reputation)
%                 All other fields are passed through to downstream
%                 functions when the state is used later.
%
%   P             Participant struct, typically taken from:
%                    participants_time_stepT1.mat  or
%                    participants_probes_mapped_stepM4.mat
%                 Expected fields (as produced by preprocessing):
%                    .questionnaires.t40_pre.total_percent
%                    .reviews / .emergency (if present)
%                    .participant_id
%
% Output:
%   state         Struct representing the initial trust state:
%                    .t              Current time (seconds, usually 0 at 40-pre)
%                    .tau            Current total trust in [0,1]
%                    .tau_disp       Dispositional trust anchor in [0,1]
%                    .tau_rep        Reputation component in [-1,1]
%                    .tau_lat        Latent component (baseline, excl. rep/sit)
%                    .exp            Personal-experience substate:
%                                      .n_succ  consecutive successes
%                                      .n_fail  consecutive failures
%                    .last_risk      Last seen risk value (NaN if none)
%                    .sc             Self-confidence in [0,1]
%                    .participant_id Participant label (string)
%
% Decomposition at any time step:
%       tau = tau_lat + tau_rep + tau_sit(r),
% where tau_sit(r) is computed on-the-fly from the current risk level r.

    % ---------------------------------------------------------------------
    % 1) Dispositional trust (from pre 40-item, mapped to [0,1])
    % ---------------------------------------------------------------------
    tau_disp = trust_compute_dispositional(P);

    % ---------------------------------------------------------------------
    % 2) Initial reputation component in [-1,1]
    %    (derived from reputation / reviews phase)
    % ---------------------------------------------------------------------
    tau_rep0 = trust_initial_reputation(params, P);

    % ---------------------------------------------------------------------
    % 3) Initial latent trust: start at dispositional level
    %    At t = 0 we assume no deviation yet, so latent = dispositional.
    % ---------------------------------------------------------------------
    tau_lat0 = tau_disp;

    % ---------------------------------------------------------------------
    % 4) Initial total trust (no situational contribution at t=0)
    %    Situational trust tau_sit(r) only appears when risk is observed.
    % ---------------------------------------------------------------------
    tau0 = trust_clip(tau_lat0 + tau_rep0);

    % ---------------------------------------------------------------------
    % 5) Self-confidence (sc) derived from emergency trial
    %
    %    emergency.choice ∈ {"self","robot"} when has_response = true.
    %
    %    Mapping:
    %      - "robot": robot-trusting participant  → tau_disp > sc
    %           sc = tau_disp / 2
    %      - "self" : self-trusting participant   → tau_disp <= sc
    %           sc = (tau_disp + 1)/2
    %      - missing / unknown:
    %           sc = 0.5 (neutral baseline)
    %
    %    This mapping guarantees:
    %      - "robot" ⇒ sc < tau_disp  ⇒ tau_disp > sc  (robot-trusting branch)
    %      - "self"  ⇒ sc > tau_disp  ⇒ tau_disp <= sc (self-trusting branch)
    % ---------------------------------------------------------------------
    sc_default = 0.5;
    sc = sc_default;

    if isfield(P, "emergency") && ~isempty(P.emergency)
        E = P.emergency;
        hasResp = isfield(E, "has_response") && logical(E.has_response);
        if hasResp && isfield(E, "choice") && ~isempty(E.choice)
            ch = string(E.choice);
            if ch == "robot"
                % Robot-trusting: place self-confidence below dispositional trust
                sc = tau_disp / 2;
            elseif ch == "self"
                % Self-trusting: place self-confidence above dispositional trust
                sc = (tau_disp + 1) / 2;
            else
                % Any unexpected label falls back to neutral self-confidence
                sc = sc_default;
            end
        end
    end

    % Clamp self-confidence to [0,1] as a precaution
    sc = max(min(sc, 1.0), 0.0);

    % ---------------------------------------------------------------------
    % 6) Assemble initial state struct
    % ---------------------------------------------------------------------
    state = struct();
    state.t        = 0.0;          % by convention, t=0 at questionnaire40pre
    state.tau      = tau0;         % total initial trust
    state.tau_disp = tau_disp;     % dispositional anchor
    state.tau_rep  = tau_rep0;     % initial reputation contribution
    state.tau_lat  = tau_lat0;     % latent baseline at start

    % Personal experience substate (used by trust_update_personal_experience)
    state.exp = struct();
    state.exp.n_succ = 0;          % consecutive successes
    state.exp.n_fail = 0;          % consecutive failures

    % No door has been seen yet → no risk observed
    state.last_risk = NaN;

    % Self-confidence used by situational trust and behaviour model
    state.sc = sc;

    % Optional participant identifier (helpful for diagnostics/logging)
    if isfield(P, "participant_id") && ~isempty(P.participant_id)
        state.participant_id = string(P.participant_id);
    else
        state.participant_id = "";
    end
end
