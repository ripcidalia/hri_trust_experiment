function [delta_tau_exp, exp_state_next] = trust_update_personal_experience( ...
    exp_state_cur, outcome, followed, params)
% trust_update_personal_experience  Update trust from personal HRI outcomes.
%
%   [delta_tau_exp, exp_state_next] = trust_update_personal_experience( ...
%       exp_state_cur, outcome, followed, params)
%
% This function implements the *personal-experience* trust update. It
% modifies trust based on streaks of consecutive successes or failures on
% door trials, independently of the other trust components (dispositional,
% reputation, situational).
%
% The update is purely event-based: it is called at each door trial and
% depends only on the binary outcome (success/failure) and the length of
% the current success/failure streak for that participant.
%
% Inputs:
%   exp_state_cur : struct representing the current personal-experience
%                   state for a participant, with fields
%                       .n_succ  (non-negative integer)
%                           current streak length of consecutive successes
%                       .n_fail  (non-negative integer)
%                           current streak length of consecutive failures
%                   If any of these fields are missing, they are initialised
%                   to 0 inside this function.
%
%   outcome       : logical or 0/1
%                       1 → the chosen door led to a "success" outcome
%                       0 → the chosen door led to a "failure" outcome
%
%   followed      : logical or 0/1 (1 = followed the drone recommendation,
%                   0 = overrode). For this personal-experience component
%                   we only care about the outcome of the *chosen* door and
%                   do not distinguish between following/overriding. It is
%                   kept in the interface for completeness and potential
%                   future extensions.
%
%   params        : full parameter struct. Only params.exp is used here:
%                       params.exp.phi_fail   (ϕ_fail ∈ (0,1))
%                       params.exp.phi_succ   (ϕ_succ ∈ (0,1))
%                       params.exp.a_succ     (a_succ ∈ (0,1))
%
%                   These define the magnitude and shape of the trust
%                   increments for failures and successes:
%                       - ϕ controls the first failure drop in trust.
%                       - ψ controls the first success boost in trust.
%                       - a shapes the saturating success profile.
%
% Outputs:
%   delta_tau_exp  : scalar trust increment due to this event (can be
%                    positive or negative). It is meant to be added to the
%                    latent trust component at the door-trial time.
%
%   exp_state_next : updated experience-state struct with new values of
%                    .n_succ and .n_fail after processing this event.
%
% Failure model (consecutive failures, n = 1, 2, ...):
%   - Define λ_fail = -ln(1 - ϕ), with ϕ ∈ (0,1).
%   - The nth failure contribution is:
%
%         τ_exp^fail(n) = exp(-λ_fail * n) * (1 - exp(λ_fail)),
%
%     which satisfies τ_exp^fail(1) = -ϕ and yields a negative, decaying
%     contribution as the failure streak grows.
%
% Success model (consecutive successes, n = 1, 2, ...):
%   - A shaped, saturating profile based on a logistic construction:
%
%       τ_exp^suc(n) = (1 - a) * [ σ(κ_suc (n - x_1/2)) ...
%                                  - σ(κ_suc (n - 1 - x_1/2)) ],
%
%       σ(z)      = 1 / (1 + exp(-z))
%       κ_suc     = -ln((1 - ψ)/(ψ - a)) - ln(-a)
%       x_1/2     = ln(-a) / ( ln((1 - ψ)/(ψ - a)) + ln(-a) )
%
%     chosen so that τ_exp^suc(1) = ψ (first success increases trust by ψ)
%     and subsequent successes follow a saturating profile controlled by a.
%
% Notes:
%   - If the parameters are invalid (e.g., ϕ outside (0,1), ψ outside (0,1),
%     a ≥ 0, or formula degeneracies), the function safely returns
%     delta_tau_exp = 0 for that event, leaving trust unchanged.
%   - The function does not clip the output; clipping is handled at the
%     level of the overall trust state.

    % ----------------------------------------------------------------------
    % 1) Ensure streak counters exist
    % ----------------------------------------------------------------------
    if ~isfield(exp_state_cur, "n_succ"), exp_state_cur.n_succ = 0; end
    if ~isfield(exp_state_cur, "n_fail"), exp_state_cur.n_fail = 0; end

    % Keep 'followed' in the signature for future use; currently ignored.
    %#ok<NASGU>
    outcome = logical(outcome);

    % Extract experience-related parameter substruct (may be empty).
    if isfield(params, "exp")
        pexp = params.exp;
    else
        pexp = struct();
    end

    % ----------------------------------------------------------------------
    % 2) Update streak counters based on success or failure
    % ----------------------------------------------------------------------
    if outcome
        % Success: increment success streak, reset failure streak.
        exp_state_cur.n_succ = exp_state_cur.n_succ + 1;
        exp_state_cur.n_fail = 0;
        n = exp_state_cur.n_succ;

        % Compute Δτ for the nth consecutive success.
        delta_tau_exp = compute_success_delta(n, pexp);
    else
        % Failure: increment failure streak, reset success streak.
        exp_state_cur.n_fail = exp_state_cur.n_fail + 1;
        exp_state_cur.n_succ = 0;
        n = exp_state_cur.n_fail;

        % Compute Δτ for the nth consecutive failure.
        delta_tau_exp = compute_failure_delta(n, pexp);
    end

    % Return updated experience state.
    exp_state_next = exp_state_cur;
end

% ========================================================================
% Helper: failure contribution τ_exp^fail(n)
% ========================================================================
function delta = compute_failure_delta(n, pexp)
    % Retrieve φ (phi_fail). If not valid, return zero contribution.
    phi = getfield_with_default(pexp, "phi_fail", NaN);

    % Valid range for φ is (0,1); otherwise we skip the update.
    if ~isfinite(phi) || phi <= 0 || phi >= 1
        delta = 0.0;
        return;
    end

    % λ_fail = -ln(1 - φ)
    lambda_fail = -log(1 - phi);

    % τ_exp^fail(n) = exp(-λ_fail * n) * (1 - exp(λ_fail))
    % This is negative for φ > 0 and yields τ_exp^fail(1) = -φ.
    delta = exp(-lambda_fail * n) * (1 - exp(lambda_fail));
end

% ========================================================================
% Helper: success contribution τ_exp^suc(n)
% ========================================================================
function delta = compute_success_delta(n, pexp)
    % Retrieve ψ (phi_succ) and a (a_succ). If invalid, return zero.
    psi = getfield_with_default(pexp, "phi_succ", NaN);  % magnitude at first success
    a   = getfield_with_default(pexp, "a_succ",   NaN);  % shape parameter

    % ψ, a should be in (0,1).
    if ~isfinite(psi) || ~isfinite(a) || psi <= 0 || psi >= 1 || a <= 0
        delta = 0.0;
        return;
    end

    % Guard against degenerate combinations in the closed-form definitions.
    denom = (psi + a);
    if denom <= 0
        delta = 0.0;
        return;
    end

    term1 = (1 - psi) / denom;
    if term1 <= 0
        delta = 0.0;
        return;
    end

    if a <= 0
        delta = 0.0;
        return;
    end

    % κ_suc = -ln((1 - ψ)/(ψ + a)) - ln(a)
    kappa_suc = -log(term1) - log(a);

    % Denominator for x_1/2
    denom_x = log(term1) + log(a);
    if denom_x == 0
        delta = 0.0;
        return;
    end

    % x_1/2 = ln(a) / ( ln((1 - ψ)/(ψ + a)) + ln(a) )
    x_half = log(a) / denom_x;

    % Logistic function σ(z) = 1 / (1 + exp(-z))
    sigma = @(z) 1 ./ (1 + exp(-z));

    % τ_exp^suc(n) = (1 + a) [ σ(κ (n - x_1/2)) - σ(κ (n - 1 - x_1/2)) ]
    z1 = kappa_suc * (n - x_half);
    z0 = kappa_suc * (n - 1 - x_half);

    delta = (1 + a) * ( sigma(z1) - sigma(z0) );
end

% ========================================================================
% Helper: get field from struct with default value
% ========================================================================
function v = getfield_with_default(S, fieldName, defaultVal)
    if isstruct(S) && isfield(S, fieldName)
        v = S.(fieldName);
    else
        v = defaultVal;
    end
end
