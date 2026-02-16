function total_cost = trust_cost_all(theta, participants, dt, weights, mode, behavior_params)
% trust_cost_all  Weighted least-squares cost over all participants.
%
%   total_cost = trust_cost_all(theta, participants, dt, weights)
%
% This function aggregates the **per-participant** weighted least-squares
% (WLS) costs into a single scalar objective to be minimized by the
% optimizer (e.g., fmincon). It simply calls
% `trust_cost_one_participant` for each participant and sums the resulting
% costs.
%
% Inputs:
%   theta
%       Parameter vector (global model parameters shared across all
%       participants). The mapping θ → model parameters is handled inside
%       trust_cost_one_participant via trust_theta_to_params.
%
%   participants
%       Array of participant structs, typically loaded from:
%           - participants_probes_mapped_stepM4.mat
%       Each struct should already contain:
%           - time-enriched events (stepT1_add_times.m)
%           - calibrated questionnaire and probe information (steps M1–M4).
%
%   dt
%       Time step (in seconds) for the trust simulation grid used for all
%       participants (e.g., dt = 0.1). If omitted or empty, dt defaults to
%       1.
%
%   weights
%       Struct with measurement weights, expected fields:
%           .w40     weight for 40-item questionnaires
%           .w14     weight for 14-item questionnaires (mapped to 40-scale)
%           .w_probe weight for single trust probes (mapped to 40-scale)
%       If omitted or left empty, the weights will be internally loaded by
%       trust_cost_one_participant from:
%           measurement_weights.mat
%       (produced by stepM5_save_measurement_weights.m).
%
%   mode
%       Simulation mode (string):
%           "simple", or "coupled".
%
%   behavior_params 
%       Relevant parameters for the behavioral model.
%
% Output:
%   total_cost
%       Scalar WLS cost obtained by summing the per-participant costs:
%
%           total_cost = Σ_i cost_i,
%
%       where cost_i is the WLS cost for participant i.
%       A large penalty (1e6) is returned if the aggregated cost becomes
%       non-finite or non-scalar due to numerical issues.
%
% Notes:
%   - This function does not perform any normalization by number of
%     measurements; the weighting is handled at the per-participant level
%     through the provided measurement weights.
%   - This is the objective that should be passed to the global optimizer.

    % ------------------------------------------------------------
    % 1) Handle optional inputs
    % ------------------------------------------------------------
    if nargin < 3 || isempty(dt)
        dt = 1;
    end

    % If weights are not provided, let trust_cost_one_participant handle
    % loading of measurement_weights.mat
    if nargin < 4
        weights = [];
    end

    if nargin < 5 || isempty(mode)
        mode = "simple";
        behavior_params = struct();
    end

    if mode == "coupled" && (nargin < 6 || isempty(behavior_params))
        error("trust_cost_all: no behavioral model parameters specified for coupled mode.");
    end

    % ------------------------------------------------------------
    % 2) Accumulate per-participant costs
    % ------------------------------------------------------------
    N = numel(participants);
    total_cost = 0.0;

    for i = 1:N
        P   = participants(i);
        c_i = trust_cost_one_participant(theta, P, dt, weights, mode, behavior_params);
        % Simple sum over participants; measurement-type weights are
        % already applied inside trust_cost_one_participant.
        total_cost = total_cost + c_i;
    end

    % ------------------------------------------------------------
    % 3) Final safety checks on total_cost
    % ------------------------------------------------------------
    % Ensure real scalar output
    if ~isreal(total_cost)
        total_cost = real(total_cost);
    end

    if ~isscalar(total_cost) || ~isfinite(total_cost)
        % If something still went wrong, fall back to a large penalty
        total_cost = 1e6;
    end
end
