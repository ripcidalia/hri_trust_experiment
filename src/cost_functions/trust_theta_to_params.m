function params = trust_theta_to_params(theta)
% trust_theta_to_params  Map parameter vector to structured model parameters.
%
%   params = trust_theta_to_params(theta)
%
% This helper converts the global parameter vector theta (as used by
% fmincon / trust_cost_all) into a structured parameter representation
% used by the trust model components. No transformation or bounding of
% theta is performed here; this function is a pure reshaping/mapping
% utility.
%
% Parameter layout:
%   theta(1) = lambda_rep   (reputation decay rate)                 >= 0
%   theta(2) = alpha_sit    (situational trust component weight) in [0,1]
%   theta(3) = lambda_sit   (situational risk sensitivity)           > 0
%   theta(4) = phi_fail     (first failure magnitude)            in [0,1]
%   theta(5) = phi_succ     (first success magnitude)            in [0,1]
%   theta(6) = a_succ       (success-shape parameter)                < 0
%   theta(7) = lambda_lat   (base latent "above" rate)               > 0
%   theta(8) = kappa_lat    (base latent "below" rate)               > 0
%
% Fixed design constants (not estimated from data):
%   params.lat.eps_lat       - deadzone around tau_disp for no drift
%   params.lat.gamma_above   - shaping parameter for "above" episodes
%   params.lat.epsilon_above - epsilon for "above" rate scaling
%   params.lat.gamma_below   - shaping parameter for "below" episodes
%   params.lat.epsilon_below - epsilon for "below" rate scaling
%   params.lat.tau_offset    - logistic offset for "below" dynamics
%
% These parameters are consumed by:
%   - trust_update_personal_experience (params.exp.*)
%   - trust_prepare_latent_sequence / trust_update_latent_sequence (params.lat.*)
%   - trust_update_reputation (params.rep.*)
%   - trust_compute_situational (params.sit.*)

    % Require at least the expected number of free parameters.
    if numel(theta) < 8
        error("trust_theta_to_params: expected at least 8 parameters, got %d.", numel(theta));
    end

    params = struct();

    % ------------------------------------------------------------
    % Reputation dynamics (global decay of initial reputation)
    % ------------------------------------------------------------
    params.rep = struct();
    params.rep.lambda_rep = theta(1);   % decay rate for reputation

    % ------------------------------------------------------------
    % Situational trust (instantaneous risk-based component)
    % ------------------------------------------------------------
    params.sit = struct();
    params.sit.alpha_sit  = theta(2);   % situational trust weight factor
    params.sit.lambda_sit = theta(3);   % decay rate for situational trust

    % ------------------------------------------------------------
    % Personal experience (event-based updates at door trials)
    % ------------------------------------------------------------
    params.exp = struct();
    params.exp.phi_fail   = theta(4);   % first failure magnitude
    params.exp.phi_succ   = theta(5);   % first success magnitude
    params.exp.a_succ     = theta(6);   % success streak shape parameter

    % ------------------------------------------------------------
    % Latent trust dynamics (between door events)
    % ------------------------------------------------------------
    params.lat = struct();

    % Episode-specific base rates (estimated from data)
    params.lat.lambda_lat = theta(7);   % base decay rate for "above" latent episodes
    params.lat.kappa_lat  = theta(8);   % base growth rate for "below" latent episodes

    % Small deadzone around tau_disp where no latent drift is applied.
    % This is kept fixed rather than estimated.
    params.lat.eps_lat = 1e-2;

    % Fixed hyperparameters for the lambda / kappa transformations.
    % These determine how the base rates are transformed to
    % episode-specific rates as a function of tau_disp and tau_lat0.
    % Defined by graphical inspection.
    params.lat.gamma_above   = 0.25;
    params.lat.epsilon_above = 0.02;

    params.lat.gamma_below   = 0.10;
    params.lat.epsilon_below = 0.01;
    params.lat.tau_offset    = 0.1;
    
    % ------------------------------------------------------------
    % Dispositional trust
    % ------------------------------------------------------------
    % Dispositional trust is derived directly from questionnaires and
    % currently has no free parameters.
    params.disp = struct();
end
