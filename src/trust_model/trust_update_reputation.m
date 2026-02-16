function tau_rep_next = trust_update_reputation(tau_rep_cur, dt, params)
% trust_update_reputation  Exponential decay of the reputation trust term.
%
%   tau_rep_next = trust_update_reputation(tau_rep_cur, dt, params)
%
% This function updates the *reputation* component of trust, tau_rep, over
% a time step Δt, modelling an exponential decay of its magnitude towards
% zero while preserving its sign. The reputation term is typically
% initialised from pre-experiment reviews (see trust_initial_reputation)
% and then decays during the interaction.
%
% Inputs:
%   tau_rep_cur : current reputation component, scalar or array in [-1, 1].
%                 Can be positive (favourable reviews), negative
%                 (unfavourable reviews), or zero (neutral).
%
%   dt          : time step in seconds (scalar, dt ≥ 0). This is the
%                 continuous-time increment used in the trust simulation
%                 between door events.
%
%   params      : model parameter struct. Only the substruct params.rep is
%                 used here:
%                     params.rep.lambda_rep  ≥ 0
%                 where lambda_rep controls how quickly the reputation term
%                 decays towards zero (larger values → faster decay).
%
% Outputs:
%   tau_rep_next : updated reputation component after Δt, same size as
%                  tau_rep_cur, clipped to [-1, 1].
%
% Model:
%   - Given the current reputation value tau_rep_cur ∈ [-1, 1], we update
%     only its magnitude via an exponential decay:
%
%         mag_cur  = |tau_rep_cur|
%         mag_next = mag_cur * exp(-lambda_rep * dt)
%
%     while preserving the sign:
%
%         tau_rep_next = sign(tau_rep_cur) * mag_next
%
%   - This implements a simple first-order decay towards zero, with
%     lambda_rep acting as the decay rate to be identified in the
%     parameter estimation procedure.
%
% Notes:
%   - If params.rep.lambda_rep is not present, a default of 0.0 is used,
%     corresponding to "no decay" (tau_rep_next = tau_rep_cur).
%   - The function supports vector inputs for tau_rep_cur; dt and
%     lambda_rep are treated as scalars.

    % ----------------------------------------------------------------------
    % 1) Read reputation decay parameter (lambda_rep)
    % ----------------------------------------------------------------------
    % Default: no decay if lambda_rep is not provided.
    lambda_rep = 0.0;

    if isfield(params, "rep") && isfield(params.rep, "lambda_rep")
        lambda_rep = params.rep.lambda_rep;
    end

    % Ensure a non-negative time step (negative dt would be unphysical here).
    dt = max(dt, 0);

    % ----------------------------------------------------------------------
    % 2) Sign-preserving exponential decay of reputation magnitude
    % ----------------------------------------------------------------------
    % Current magnitude of the reputation term.
    mag_cur = abs(tau_rep_cur);

    % Exponentially decayed magnitude after time dt.
    mag_next = mag_cur .* exp(-lambda_rep * dt);

    % Reattach the original sign to obtain the updated reputation term.
    tau_rep_next = sign(tau_rep_cur) .* mag_next;

    % ----------------------------------------------------------------------
    % 3) Numerical safety: force real and clip to [-1, 1]
    % ----------------------------------------------------------------------
    tau_rep_next = real(tau_rep_next);
    tau_rep_next = max(min(tau_rep_next, 1), -1);
end
