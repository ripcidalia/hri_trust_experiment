function cost = trust_cost_one_participant(theta, P, dt, weights, mode, behavior_params)
% trust_cost_one_participant  WLS cost for one participant.
%
%   cost = trust_cost_one_participant(theta, P, dt, weights)
%
% This function computes the **weighted least-squares (WLS)** cost for a
% single participant, given:
%   - a global parameter vector θ,
%   - a preprocessed participant struct P, and
%   - per-instrument measurement weights.
%
% The cost is based on the discrepancy between:
%   - the model-predicted trust values at all measurement times, and
%   - the observed trust values (questionnaires and probes),
% scaled by instrument-specific weights:
%   - 40-item (post)
%   - 14-item (two mid-block questionnaires mapped to 40-scale)
%   - single trust probes (mapped to 40-scale).
%
% Inputs:
%   theta      
%       Parameter vector for the global trust model (all participants).
%       The mapping θ → params is handled by trust_theta_to_params,
%       which is called inside trust_simulate_or_predict_one_participant.
%
%   P          
%       Participant struct (single element), typically from:
%         - participants_probes_mapped_stepM4.mat,
%       enriched with:
%         - time information (stepT1_add_times.m) and
%         - calibrated probe values (stepM2/stepM4).
%
%   dt         
%       Time step (in seconds) for the trust simulation grid.
%       If omitted or empty, dt defaults to 1.
%
%   weights
%       Struct specifying measurement weights with fields:
%         .w40     weight for 40-item questionnaires
%         .w14     weight for 14-item questionnaires (mapped to 40-scale)
%         .w_probe weight for single trust probes (mapped to 40-scale)
%       If omitted or empty, this function attempts to load:
%          measurement_weights.mat  (created in stepM5).
%
%   mode
%       Simulation mode (string):
%               - "simple", or "coupled"
%
%   behavior_params 
%       Relevant parameters for the behavioral model.
%
% Output:
%   cost     Scalar WLS cost for this participant:
%              cost = Σ_j w_j * (y_obs_j - y_hat_j)^2,
%            where j ranges over all measurements for this participant.
%
% Dependencies:
%   - trust_simulate_or_predict_one_participant (forward simulation)
%   - measurement_weights.mat (when weights are not provided)

    % -----------------------------
    % 1) Handle defaults
    % -----------------------------

    if nargin < 3 || isempty(dt)
        dt = 1;
    end

    % If weights are not provided, load from default MAT file
    if nargin < 4 || isempty(weights)
        if ~isfile("measurement_weights.mat")
            error("trust_cost_one_participant: measurement_weights.mat not found.");
        end
        S = load("measurement_weights.mat");
        if ~isfield(S, "weights")
            error("trust_cost_one_participant: 'weights' struct not found in measurement_weights.mat");
        end
        weights = S.weights;
    end

    % Basic sanity on weights: require all three instrument fields
    if ~isfield(weights, "w40") || ~isfield(weights, "w14") || ~isfield(weights, "w_probe")
        error("trust_cost_one_participant: weights must have fields w40, w14, w_probe.");
    end

    if nargin < 5 || isempty(mode)
        mode = "simple";
        behavior_params = struct();
    end

    if mode == "coupled" && (nargin < 6 || isempty(behavior_params))
        error("trust_cost_one_participant: no behavioral model parameters specified for coupled mode.");
    end

    % -----------------------------
    % 2) Run trust simulation
    % -----------------------------
    try
        % Forward simulation of the participant trajectory + predictions
        sim = trust_simulate_or_predict_one_participant(mode, theta, P, dt, behavior_params);
    catch ME
        % If the simulation fails for this participant, assign a large penalty
        warning("trust_cost_one_participant: simulation failed for participant %s: %s", ...
                string(P.participant_id), ME.message);
        cost = 1e6;
        return;
    end

    meas  = sim.measurements;
    y_hat = sim.y_hat;

    % No measurements → no contribution to the global cost
    if isempty(meas)
        cost = 0.0;
        return;
    end

    % -----------------------------
    % 3) Build WLS cost
    % -----------------------------
    nMeas = numel(meas);

    % Observed values and weights (aligned with measurement order)
    y_obs = zeros(nMeas, 1);
    w_vec = zeros(nMeas, 1);

    for j = 1:nMeas
        % Observed trust value (already in [0,1] 40-scale)
        y_obs(j) = meas(j).y;

        % Measurement kind: "t40_post", "t14_mid1", "t14_mid2", "probe", ...
        kind = string(meas(j).kind);

        % Map measurement kind → appropriate weight
        if kind == "t40_post"
            w_vec(j) = weights.w40;
        elseif kind == "t14_mid1" || kind == "t14_mid2"
            w_vec(j) = weights.w14;
        elseif kind == "probe"
            w_vec(j) = weights.w_probe;
        else
            % Unknown type: default to probe weight for robustness
            w_vec(j) = weights.w_probe;
        end
    end

    % Sanity check: predictions must match number of measurements
    if numel(y_hat) ~= nMeas
        error("trust_cost_one_participant: y_hat length (%d) != nMeas (%d).", ...
              numel(y_hat), nMeas);
    end

    % Residuals and WLS cost
    residuals = y_obs - y_hat(:);
    cost = sum( w_vec .* (residuals.^2) );

end
