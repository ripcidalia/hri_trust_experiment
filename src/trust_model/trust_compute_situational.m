function tau_sit = trust_compute_situational(risk_value, tau_disp, sc, params)
% trust_compute_situational  Compute situational trust contribution.
%
%   tau_sit = trust_compute_situational(risk_value, tau_disp, sc, params)
%
% This function computes the *situational* component of trust, i.e. the
% instantaneous change in trust driven by the current door risk and the
% participant’s self-confidence. It does **not** integrate over time; it is
% designed to be evaluated at (or around) door-trial events and then added
% to the latent and reputation components:
%
%       tau(t) = tau_lat(t) + tau_rep(t) + alpha_sit*tau_sit(risk(t))
%
% The behaviour depends on whether the participant is more inclined to rely
% on the robot or on themselves:
%
%   - Robot-trusting (tau_disp > sc):
%       situational trust increases with risk:
%
%           tau_sit(r) = 1 - 2*exp(-lambda_sit * r)
%
%   - Self-trusting (tau_disp <= sc):
%       situational trust decreases with risk:
%
%           tau_sit(r) = 2*exp(-lambda_sit * r) - 1
%
% where r = risk_value ∈ [0,1] and lambda_sit > 0 is a sensitivity
% parameter.
%
% Inputs:
%   risk_value  - scalar in [0,1]; perceived risk of the current door
%                 (typically P.doorTrials(k).risk_value).
%   tau_disp    - dispositional trust (scalar in [0,1]) for this participant.
%   sc          - self-confidence level (scalar in [0,1]).
%   params      - struct of model parameters. Only params.sit.lambda_sit is used:
%                 params.sit.lambda_sit : positive scalar risk sensitivity.
%
% Output:
%   tau_sit     - scalar situational trust term in [-1,1].
%
% Notes:
%   - If risk_value is NaN or non-finite, it is treated as 0.
%   - This function does not store state; it is purely instantaneous.

    % ----------------------------------------------------------------------
    % 1) Retrieve situational sensitivity parameter (lambda_sit)
    % ----------------------------------------------------------------------
    % Default value in case params.sit.lambda_sit is missing. The actual
    % value used in the model is set during parameter identification.
    lambda_sit = 1.0;

    if isfield(params, "sit") && isfield(params.sit, "lambda_sit")
        lambda_sit = params.sit.lambda_sit;
    end

    % ----------------------------------------------------------------------
    % 2) Sanitize and clamp risk input to [0,1]
    % ----------------------------------------------------------------------
    if isnan(risk_value) || ~isfinite(risk_value)
        risk_value = 0;
    end
    risk_value = max(min(risk_value, 1), 0);

    % ----------------------------------------------------------------------
    % 3) Dual-exponential situational trust contribution
    % ----------------------------------------------------------------------
    % Robot-trusting vs. self-trusting split based on dispositional trust
    % relative to self-confidence.
    if tau_disp > sc
        % Robot-trusting user: higher risk leads to higher reliance on the robot
        tau_sit = 1 - 2*exp(-lambda_sit * risk_value);
    else
        % Self-trusting user: higher risk leads to lower reliance on the robot
        tau_sit = 2*exp(-lambda_sit * risk_value) - 1;
    end
end
