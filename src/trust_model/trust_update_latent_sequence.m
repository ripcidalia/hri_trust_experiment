function [tau_lat_next, lat_state_next] = trust_update_latent_sequence( ...
    tau_lat_cur, tau_disp, dt, params, lat_state_cur)
% trust_update_latent_sequence  Evolve the latent trust component over time.
%
%   [tau_lat_next, lat_state_next] = trust_update_latent_sequence( ...
%       tau_lat_cur, tau_disp, dt, params, lat_state_cur)
%
% This function updates the latent trust component between HRI events over a
% single time step dt. It uses an episode-based representation of latent
% dynamics, where an “episode” is defined whenever the latent trust is
% sufficiently above or below the participant’s dispositional trust:
%
%   - If tau_lat_cur is within an epsilon band around tau_disp, no latent
%     drift is applied and the mode is set to "none".
%   - If tau_lat_cur > tau_disp + eps_lat, an "above" episode is active,
%     and latent trust decays exponentially towards tau_disp.
%   - If tau_lat_cur < tau_disp - eps_lat, a "below" episode is active,
%     and latent trust grows logistically towards tau_disp.
%
% Episode-specific parameters (lambda_seq / kappa_seq and, for the below
% regime, the internal logistic state sigma) are carried in lat_state_cur
% and updated as needed. The mapping from global parameters to episode-
% specific rates follows the analytical transformations used in the thesis.
%
% Inputs:
%   tau_lat_cur   - current latent trust component, scalar in [0,1].
%   tau_disp      - dispositional trust for this participant, scalar in (0,1).
%   dt            - time step in seconds (>= 0).
%   params.lat    - struct containing latent trust parameters:
%                       .eps_lat
%                       .lambda_lat
%                       .kappa_lat
%                       .gamma_above,  .epsilon_above
%                       .gamma_below,  .epsilon_below
%                       .tau_offset
%                   All fields are optional; defaults are used if missing.
%   lat_state_cur - struct describing the current latent episode, with fields:
%                       .mode       = "none" | "above" | "below"
%                       .lambda_seq = episode-specific decay rate (above)
%                       .kappa_seq  = episode-specific growth rate (below)
%                       .sigma      = internal logistic state (below)
%                       .tau0       = latent value at episode start
%
% Outputs:
%   tau_lat_next   - updated latent trust component for the next time step.
%   lat_state_next - updated latent episode state (mode and internal params).
%
% Behaviour:
%   - If dt = 0 or tau_lat_cur is within eps_lat of tau_disp, the latent
%     component is left unchanged and any episode is reset to "none".
%   - Otherwise, the function either continues the current episode
%     (above/below) or restarts a new one if the sign of the deviation
%     relative to tau_disp has changed.
%   - For "above" episodes, an exponential decay towards tau_disp is used.
%   - For "below" episodes, a logistic growth in an internal sigma variable
%     is used, which is then mapped back to tau_lat using the specified
%     offset and scaling.
%   - Numerical safeguards ensure non-negative rates and clipping of trust
%     values to [0,1].

    % ---------------------------------------------------------------------
    % 1) Default episode state if none was provided
    % ---------------------------------------------------------------------
    if nargin < 5 || isempty(lat_state_cur)
        lat_state_cur = struct();
    end

    if ~isfield(lat_state_cur, "mode"),       lat_state_cur.mode       = "none"; end
    if ~isfield(lat_state_cur, "lambda_seq"), lat_state_cur.lambda_seq = NaN;    end
    if ~isfield(lat_state_cur, "kappa_seq"),  lat_state_cur.kappa_seq  = NaN;    end
    if ~isfield(lat_state_cur, "sigma"),      lat_state_cur.sigma      = NaN;    end
    if ~isfield(lat_state_cur, "tau0"),       lat_state_cur.tau0       = tau_lat_cur; end

    lat_state_next = lat_state_cur;

    % ---------------------------------------------------------------------
    % 2) Read latent parameters (with defaults if missing)
    % ---------------------------------------------------------------------
    eps_lat   = getfield_with_default(params, "lat", struct(), "this is bullshit, because when we determine the ",   1e-3);
    lambda_lat  = getfield_with_default(params, "lat", struct(), "lambda_lat",  1e-4);
    kappa_lat   = getfield_with_default(params, "lat", struct(), "kappa_lat",   1e-4);

    gamma_ab  = getfield_with_default(params, "lat", struct(), "gamma_above",   0.05);
    eps_ab    = getfield_with_default(params, "lat", struct(), "epsilon_above", 0.01);

    gamma_bl  = getfield_with_default(params, "lat", struct(), "gamma_below",   0.05);
    eps_bl    = getfield_with_default(params, "lat", struct(), "epsilon_below", 0.01);
    tau_off   = getfield_with_default(params, "lat", struct(), "tau_offset",    0.05);

    % Basic input sanitisation
    dt = max(dt, 0);
    tau_lat_cur = trust_clip(tau_lat_cur);
    tau_disp    = trust_clip(tau_disp);

    % Deviation from dispositional trust
    delta = tau_lat_cur - tau_disp;

    % ---------------------------------------------------------------------
    % 3) Deadzone: if very close to tau_disp or dt=0, no latent drift
    % ---------------------------------------------------------------------
    if dt == 0 || abs(delta) <= eps_lat
        tau_lat_next = tau_lat_cur;

        % Reset episode state within the deadzone
        lat_state_next.mode       = "none";
        lat_state_next.lambda_seq = NaN;
        lat_state_next.kappa_seq  = NaN;
        lat_state_next.sigma      = NaN;
        lat_state_next.tau0       = tau_lat_cur;
        return;
    end

    % ---------------------------------------------------------------------
    % 4) Decide (or re-decide) which latent regime applies
    % ---------------------------------------------------------------------
    mode = lat_state_cur.mode;

    % Determine whether to start a new episode:
    %   - mode "none" always restarts,
    %   - "above" with delta <= 0 implies crossing tau_disp,
    %   - "below" with delta >= 0 implies crossing tau_disp.
    if mode == "none"
        need_restart = true;
    elseif mode == "above" && delta <= 0
        need_restart = true;
    elseif mode == "below" && delta >= 0
        need_restart = true;
    else
        need_restart = false;
    end

    if need_restart
        % New episode initialisation based on the sign of the deviation.
        if delta > eps_lat
            % ABOVE: tau_lat > tau_disp
            mode = "above";
            lambda_seq = compute_lambda_above(tau_disp, tau_lat_cur, ...
                                              lambda_lat, gamma_ab, eps_ab);
            kappa_seq  = NaN;
            sigma      = NaN;
            tau0_ep    = tau_lat_cur;

        elseif delta < -eps_lat
            % BELOW: tau_lat < tau_disp
            mode = "below";
            [kappa_seq, sigma0] = compute_kappa_below(tau_disp, tau_lat_cur, ...
                                                      kappa_lat, gamma_bl, eps_bl, tau_off);
            lambda_seq = NaN;
            sigma      = sigma0;
            tau0_ep    = tau_lat_cur;
        else
            % Should not occur due to deadzone check, but kept for safety
            mode       = "none";
            lambda_seq = NaN;
            kappa_seq  = NaN;
            sigma      = NaN;
            tau0_ep    = tau_lat_cur;
        end

        lat_state_next.mode       = mode;
        lat_state_next.lambda_seq = lambda_seq;
        lat_state_next.kappa_seq  = kappa_seq;
        lat_state_next.sigma      = sigma;
        lat_state_next.tau0       = tau0_ep;
    else
        % Continue within existing episode: reuse pre-computed parameters.
        lambda_seq = lat_state_cur.lambda_seq;
        kappa_seq  = lat_state_cur.kappa_seq;
        sigma      = lat_state_cur.sigma;
        tau0_ep    = lat_state_cur.tau0;
    end

    % ---------------------------------------------------------------------
    % 5) Apply the regime-specific latent update over dt
    % ---------------------------------------------------------------------
    switch mode
        case "above"
            % Exponential decay of latent trust towards tau_disp.
            %   tau_lat_next = tau_disp + (tau_lat_cur - tau_disp) * exp(-lambda_seq * dt)
            if ~isfinite(lambda_seq) || lambda_seq <= 0
                tau_lat_next = tau_lat_cur;
            else
                tau_lat_next = tau_disp + (tau_lat_cur - tau_disp) * exp(-lambda_seq * dt);
            end

            % Avoid overshoot below tau_disp due to numerical issues.
            if tau_lat_next < tau_disp
                tau_lat_next = tau_disp;
            end

        case "below"
            % Logistic increase of an internal sigma towards 1, mapped back
            % to tau_lat via an affine transform.
            if ~isfinite(kappa_seq) || kappa_seq <= 0 || ~isfinite(sigma)
                tau_lat_next = tau_lat_cur;
            else
                % Euler step on sigma:
                %   sigma' = sigma + κ * sigma * (1 - sigma) * dt
                sigma_new = sigma + kappa_seq * sigma * (1 - sigma) * dt;
                % Clamp σ to [0,1] to maintain consistent mapping.
                sigma_new = min(max(sigma_new, 0), 1);

                % tau_lat(t) = tau0 - tau_off + (tau_disp - tau0 + tau_off) * sigma
                tau_lat_next = tau0_ep - tau_off + (tau_disp - tau0_ep + tau_off) * sigma_new;

                % Avoid overshoot above tau_disp.
                if tau_lat_next > tau_disp
                    tau_lat_next = tau_disp;
                end

                % Store updated sigma in the episode state.
                lat_state_next.sigma = sigma_new;
            end

        otherwise
            % No active latent regime: keep latent trust unchanged.
            tau_lat_next = tau_lat_cur;
    end

    % Final clipping for numerical safety.
    tau_lat_next = trust_clip(tau_lat_next);

end

% =====================================================================
% Helper: compute lambda for ABOVE regime
% =====================================================================
function lambda_val = compute_lambda_above(tau_disp, tau0, lambda_lat, gamma, eps_val)
% Compute the episode-specific exponential rate λ_{τ0,τdisp} for the
% "above" regime given:
%   - tau_disp  : dispositional trust
%   - tau0      : latent trust at episode start (tau0 > tau_disp)
%   - lambda_lat  : base rate λ_{1,0}
%   - gamma     : shape parameter for the mapping
%   - eps_val   : minimum gap parameter
%
% The mapping follows a two-step transformation:
%   1) Derive λ_{1,τdisp} from λ_{1,0}.
%   2) Derive λ_{τ0,τdisp} from λ_{1,τdisp}.

    % Defensive defaults and basic guards
    if ~isfinite(lambda_lat) || lambda_lat <= 0
        lambda_val = 0;
        return;
    end
    if tau_disp <= 0 || tau_disp >= 1
        lambda_val = lambda_lat;
        return;
    end
    if gamma <= 0 || gamma >= 1
        gamma = 0.05;
    end
    if eps_val <= 0
        eps_val = 0.01;
    end

    % λ_{1,τdisp}
    lambda_1_disp = lambda_lat * ( ...
        log(gamma / (1 - tau_disp)) / log(tau_disp) );

    % λ_{τ0,τdisp}
    num   = log(eps_val / (tau0 - tau_disp));
    denom = log((1 - tau0) / (1 - tau_disp));

    if denom == 0
        lambda_val = lambda_1_disp;
    else
        lambda_val = lambda_1_disp * (num / denom);
    end

    % Final safety check.
    if ~isfinite(lambda_val) || lambda_val < 0
        lambda_val = max(lambda_lat, 0);
    end
end

% =====================================================================
% Helper: compute kappa + initial sigma for BELOW regime
% =====================================================================
function [kappa_val, sigma0] = compute_kappa_below(tau_disp, tau0, ...
                                                   kappa_lat, gamma, eps_val, tau_off)
% Compute the episode-specific logistic rate κ_{τ0,τdisp} and initial
% logistic state σ[0] for the "below" regime given:
%   - tau_disp  : dispositional trust
%   - tau0      : latent trust at episode start (tau0 < tau_disp)
%   - kappa_lat   : base rate κ_{0,1}
%   - gamma     : shape parameter
%   - eps_val   : minimum gap parameter
%   - tau_off   : offset used in the logistic mapping.
%
% The computation proceeds in two stages:
%   1) Derive κ_{0,τdisp} from κ_{0,1}.
%   2) Derive κ_{τ0,τdisp} from κ_{0,τdisp}.
% The initial σ[0] is obtained from tau_disp, tau0, and tau_off.

    % Defaults if base rate is invalid
    if ~isfinite(kappa_lat) || kappa_lat <= 0
        kappa_val = 0;
        sigma0    = 0.5;
        return;
    end
    if tau_disp <= 0 || tau_disp >= 1
        kappa_val = kappa_lat;
        sigma0    = 0.5;
        return;
    end
    if gamma <= 0 || gamma >= 1
        gamma = 0.05;
    end
    if eps_val <= 0
        eps_val = 0.01;
    end
    if tau_off <= 0
        tau_off = 0.05;
    end

    % Initial σ[0] from the logistic mapping:
    %   σ(0) = tau_offset / (tau_disp - tau0 + tau_offset)
    sigma0 = tau_off / (tau_disp - tau0 + tau_off);

    % κ_{0,τdisp}
    num1 = log(gamma / (tau_disp + tau_off - gamma)) - log(tau_disp / tau_off);
    den1 = log(tau_off) + log((1 - tau_disp) / (tau_disp + tau_off));
    kappa_0_disp = kappa_lat * (num1 / den1);

    % κ_{τ0,τdisp}
    num2 = log( (tau_disp - tau0) / tau_off ) ...
           - log( eps_val / (tau_disp - eps_val - tau0 + tau_off) );
    den2 = log( tau_disp / tau_off ) ...
           - log( tau0 / (tau_disp - tau0 + tau_off) );

    if den2 == 0
        kappa_val = kappa_0_disp;
    else
        kappa_val = kappa_0_disp * (num2 / den2);
    end

    if ~isfinite(kappa_val) || kappa_val < 0
        kappa_val = max(kappa_lat, 0);
    end
end

% =====================================================================
% Helper: safe getfield on nested structs with default
% =====================================================================
function v = getfield_with_default(S, subfield, defaultStruct, fieldName, defaultVal)
% Safely access S.(subfield).(fieldName) with a default value.
%
% Inputs:
%   S            - outer struct (e.g., params).
%   subfield     - name of sub-struct field (e.g., 'lat').
%   defaultStruct- unused placeholder (kept for signature compatibility).
%   fieldName    - field name inside S.(subfield).
%   defaultVal   - value returned if field is missing or empty.
%
% Output:
%   v            - value of S.(subfield).(fieldName), or defaultVal.

    if ~isstruct(S) || ~isfield(S, subfield) || ~isstruct(S.(subfield))
        v = defaultVal;
        return;
    end
    sub = S.(subfield);
    if isfield(sub, fieldName) && ~isempty(sub.(fieldName))
        v = sub.(fieldName);
    else
        v = defaultVal;
    end
end
