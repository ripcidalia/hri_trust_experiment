function lat_state = trust_prepare_latent_sequence(tau_lat0, tau_disp, params)
% trust_prepare_latent_sequence  Initialise a latent trust episode.
%
%   lat_state = trust_prepare_latent_sequence(tau_lat0, tau_disp, params)
%
% This function prepares the latent-trust state immediately after a door
% trial. Given the current latent trust value and the participant's
% dispositional trust, it determines whether a new latent “episode” is
% started and, if so, pre-computes episode-specific parameters for the
% subsequent evolution:
%
%   - If tau_lat0 is sufficiently close to tau_disp, no latent drift is
%     activated and the mode is "none".
%   - If tau_lat0 is sufficiently above tau_disp, an "above" episode is
%     created, with an exponential decay rate λ_{τ0,τdisp}.
%   - If tau_lat0 is sufficiently below tau_disp, a "below" episode is
%     created, with a logistic growth rate κ_{τ0,τdisp} and an internal
%     logistic state σ[0].
%
% The heavy algebra (mapping from base rates to episode-specific λ/κ) is
% performed once per episode and cached in lat_state. The state is then
% used by trust_update_latent_sequence.m when advancing in time.
%
% Inputs:
%   tau_lat0  - latent trust component immediately AFTER the door event,
%               scalar in [0,1].
%   tau_disp  - dispositional trust for this participant, scalar in (0,1).
%   params    - global parameter struct. The latent-trust configuration is
%               expected under params.lat with fields (all optional):
%                   .eps_lat      - tolerance around tau_disp for “no drift”
%                   .lambda_lat   - base exponential rate
%                   .kappa_lat    - base logistic rate
%                   .gamma_above  - shape parameter for “above” mapping
%                   .epsilon_above- minimum gap parameter for “above”
%                   .gamma_below  - shape parameter for “below” mapping
%                   .epsilon_below- minimum gap parameter for “below”
%                   .tau_offset   - offset used in the logistic transform
%
% Output:
%   lat_state - struct describing the current latent episode:
%                   .mode        - "above", "below", or "none"
%                   .lambda_seq  - episode-specific decay rate for
%                                  “above” episodes (NaN otherwise)
%                   .kappa_seq   - episode-specific growth rate for
%                                  “below” episodes (NaN otherwise)
%                   .sigma       - internal logistic state for “below”
%                                  episodes (NaN otherwise)
%                   .tau0        - latent starting value for this episode
%
% Conceptual model:
%   - Around tau_disp, an indifference band of width eps_lat is defined.
%     If the latent value falls within this band, no additional drift is
%     applied (mode = "none").
%
%   - For tau_lat0 > tau_disp + eps_lat, a two-step exponential mapping is
%     used to derive λ_{τ0,τdisp} from the global base rate λ_{1,0}:
%
%       λ_{1,τdisp}   = λ_{1,0} * ln(γ / (1 - τ_disp)) / ln(τ_disp)
%       λ_{τ0,τdisp}  = λ_{1,τdisp} * ln(ε / (τ0 - τ_disp)) ...
%                                 / ln( (1 - τ0) / (1 - τ_disp) )
%
%     with γ = gamma_above and ε = epsilon_above.
%
%   - For tau_lat0 < tau_disp - eps_lat, a two-step logistic mapping is
%     used to derive κ_{τ0,τdisp} from κ_{0,1}, together with an initial
%     logistic state σ[0] = τ_offset / (τ_disp - τ0 + τ_offset).
%
% Numerical safety:
%   - If any intermediate expression becomes invalid (log of non-positive
%     arguments, divisions by zero, etc.), the function falls back to
%     simple base rates (lambda_lat / kappa_lat) and/or safe defaults.

    % ---------------------------------------------------------------------
    % Extract latent-parameter substruct and set defaults
    % ---------------------------------------------------------------------
    if isfield(params, "lat")
        plat = params.lat;
    else
        plat = struct();
    end

    % Numerical tolerance around tau_disp to decide whether to apply drift.
    eps_lat = getfield_default(plat, "eps_lat", 1e-3);

    % Global base rates λ_{1,0} and κ_{0,1} (optimisation variables).
    lambda_lat = getfield_default(plat, "lambda_lat", 1e-3);  % λ_{1,0}
    kappa_lat  = getfield_default(plat, "kappa_lat",  1e-3);  % κ_{0,1}

    % Hyperparameters for the above/below transformations.
    gamma_above    = getfield_default(plat, "gamma_above",    0.05);
    epsilon_above  = getfield_default(plat, "epsilon_above",  0.01);

    gamma_below    = getfield_default(plat, "gamma_below",    0.05);
    epsilon_below  = getfield_default(plat, "epsilon_below",  0.01);
    tau_offset     = getfield_default(plat, "tau_offset",     0.05);

    % ---------------------------------------------------------------------
    % Initialise output struct for this latent episode
    % ---------------------------------------------------------------------
    lat_state = struct();
    lat_state.mode       = "none";   % default: no active episode
    lat_state.lambda_seq = NaN;
    lat_state.kappa_seq  = NaN;
    lat_state.sigma      = NaN;
    lat_state.tau0       = tau_lat0;

    % Deviation from dispositional trust.
    d = tau_lat0 - tau_disp;

    % If very close to dispositional trust, do not create a drift episode.
    if abs(d) <= eps_lat
        lat_state.mode = "none";
        return;
    end

    % ---------------------------------------------------------------------
    % Case 1: "above" latent episode (tau_lat0 > tau_disp + eps_lat)
    % ---------------------------------------------------------------------
    if d > eps_lat
        % Two-step exponential transformation, as derived in the thesis:
        %   λ_{1,τdisp} = λ_{1,0} * ln(γ / (1 - τ_disp)) / ln(τ_disp)
        %   λ_{τ0,τdisp} = λ_{1,τdisp} * ln(ε / (τ0 - τ_disp)) ...
        %                                / ln( (1 - τ0) / (1 - τ_disp) )
        lambda_seq = compute_lambda_above(tau_lat0, tau_disp, ...
                                          lambda_lat, gamma_above, epsilon_above);

        lat_state.mode       = "above";
        lat_state.lambda_seq = lambda_seq;
        lat_state.kappa_seq  = NaN;
        lat_state.sigma      = NaN;
        return;
    end

    % ---------------------------------------------------------------------
    % Case 2: "below" latent episode (tau_lat0 < tau_disp - eps_lat)
    % ---------------------------------------------------------------------
    kappa_seq = compute_kappa_below(tau_lat0, tau_disp, ...
                                    kappa_lat, gamma_below, ...
                                    epsilon_below, tau_offset);

    % Initial σ[0] from the logistic derivation:
    %   σ[0] = τ_offset / ( τ_disp - τ0 + τ_offset )
    denom = tau_disp - tau_lat0 + tau_offset;
    if denom <= 0
        % Degenerate case; fall back to a small positive σ.
        sigma0 = 1e-3;
    else
        sigma0 = tau_offset / denom;
    end

    lat_state.mode       = "below";
    lat_state.lambda_seq = NaN;
    lat_state.kappa_seq  = kappa_seq;
    lat_state.sigma      = sigma0;
end


% =====================================================================
% Helper: compute λ_{τ0,τdisp} for the "above" case
% =====================================================================
function lambda_seq = compute_lambda_above(tau0, tau_disp, lambda_lat, gamma, epsilon)
    % Default to the base rate if anything becomes invalid.
    lambda_seq = max(lambda_lat, 0);

    % Guard against pathological values of tau_disp and tau0.
    if tau_disp <= 0 || tau_disp >= 1
        return;
    end
    if tau0 <= tau_disp || tau0 >= 1
        return;
    end

    % Step 1: λ_{1,τdisp}.
    num1 = gamma / (1 - tau_disp);
    den1 = tau_disp;
    if num1 <= 0 || den1 <= 0
        return;
    end
    lambda_1_disp = lambda_lat * ( log(num1) / log(den1) );

    % Step 2: λ_{τ0,τdisp}.
    num2 = epsilon / (tau0 - tau_disp);
    den2 = (1 - tau0) / (1 - tau_disp);
    if num2 <= 0 || den2 <= 0
        % If invalid, fall back to λ_{1,τdisp}.
        lambda_seq = max(lambda_1_disp, 0);
        return;
    end

    lambda_seq = lambda_1_disp * ( log(num2) / log(den2) );

    % Final safety: ensure non-negative and finite.
    if ~isfinite(lambda_seq) || lambda_seq < 0
        lambda_seq = max(lambda_lat, 0);
    end
end


% =====================================================================
% Helper: compute κ_{τ0,τdisp} for the "below" case
% =====================================================================
function kappa_seq = compute_kappa_below(tau0, tau_disp, ...
                                         kappa_lat, gamma, epsilon, tau_offset)
    % Default to the base rate if anything becomes invalid.
    kappa_seq = max(kappa_lat, 0);

    % Basic guards on tau_disp, tau0, and tau_offset.
    if tau_disp <= 0 || tau_disp >= 1
        return;
    end
    if tau0 < 0 || tau0 >= tau_disp
        % We expect tau0 < tau_disp, but also tau0 ∈ [0, tau_disp[.
        return;
    end
    if tau_offset <= 0
        return;
    end

    % -----------------------------------------------------------------
    % Step 1: κ_{0,τdisp}
    % -----------------------------------------------------------------
    %   κ_{0,τdisp} = κ_{0,1} *
    %       [ ln( γ / (τdisp + τoffset - γ) ) - ln( τdisp / τoffset ) ]
    %       ----------------------------------------------------------
    %       [ ln(τoffset) + ln( (1 - τdisp) / (τdisp + τoffset) ) ]
    num1_a = gamma;
    den1_a = tau_disp + tau_offset - gamma;
    num1_b = tau_disp;
    den1_b = tau_offset;
    den1_c = tau_offset;
    den1_d = (1 - tau_disp) / (tau_disp + tau_offset);

    if num1_a <= 0 || den1_a <= 0 || num1_b <= 0 || den1_b <= 0 ...
            || den1_c <= 0 || den1_d <= 0
        return;
    end

    top1 = log(num1_a / den1_a) - log(num1_b / den1_b);
    bot1 = log(den1_c) + log(den1_d);

    if bot1 == 0
        return;
    end

    kappa_0_disp = kappa_lat * (top1 / bot1);

    % -----------------------------------------------------------------
    % Step 2: κ_{τ0,τdisp}
    % -----------------------------------------------------------------
    %   κ_{τ0,τdisp} = κ_{0,τdisp} *
    %      [ ln( (τdisp - τ0)/τoffset ) - ln( ε / (τdisp - ε - τ0 + τoffset) ) ]
    %      -------------------------------------------------------------------
    %      [ ln( τdisp / τoffset ) - ln( τ0 / (τdisp - τ0 + τoffset) ) ]
    num2_a = (tau_disp - tau0);
    den2_a = tau_offset;
    num2_b = epsilon;
    den2_b = tau_disp - epsilon - tau0 + tau_offset;

    num2_c = tau_disp;
    den2_c = tau_offset;
    num2_d = tau0;
    den2_d = tau_disp - tau0 + tau_offset;

    if num2_a <= 0 || den2_a <= 0 || num2_b <= 0 || den2_b <= 0 ...
            || num2_c <= 0 || den2_c <= 0 || num2_d <= 0 || den2_d <= 0
        % Fall back to κ_{0,τdisp} if the second step breaks.
        if isfinite(kappa_0_disp) && kappa_0_disp > 0
            kappa_seq = kappa_0_disp;
        end
        return;
    end

    top2 = log(num2_a / den2_a) - log(num2_b / den2_b);
    bot2 = log(num2_c / den2_c) - log(num2_d / den2_d);

    if bot2 == 0
        if isfinite(kappa_0_disp) && kappa_0_disp > 0
            kappa_seq = kappa_0_disp;
        end
        return;
    end

    kappa_seq = kappa_0_disp * (top2 / bot2);

    % Final safety checks.
    if ~isfinite(kappa_seq) || kappa_seq < 0
        if isfinite(kappa_0_disp) && kappa_0_disp > 0
            kappa_seq = kappa_0_disp;
        else
            kappa_seq = max(kappa_lat, 0);
        end
    end
end


% =====================================================================
% Small helper: safe field access with default
% =====================================================================
function v = getfield_default(S, fieldName, defaultVal)
    if isstruct(S) && isfield(S, fieldName)
        v = S.(fieldName);
        if isempty(v)
            v = defaultVal;
        end
    else
        v = defaultVal;
    end
end
