function debug_cost_gradient()
% debug_cost_gradient  Probe sensitivity of trust_cost_all to parameter changes.
%
%   debug_cost_gradient()
%
% This diagnostic function evaluates how the global WLS cost function
% trust_cost_all responds to small perturbations of each parameter in a
% baseline parameter vector. It is intended for:
%   - Checking numerical stability of the cost function with respect to
%     each parameter.
%   - Identifying parameters that cause errors, NaNs, or Infs when
%     perturbed slightly.
%   - Gaining a qualitative sense of parameter sensitivity (via changes in
%     the scalar cost value).
%
% The function:
%   1) Loads participant data and measurement weights.
%   2) Defines a baseline parameter vector theta0.
%   3) Evaluates trust_cost_all at theta0.
%   4) Perturbs each parameter by a small epsilon and re-evaluates the cost,
%      reporting whether the perturbed cost is valid and its value.
%
% Inputs:
%   None. All required data files and parameters are hard-coded.
%
% Assumptions:
%   - File derived/participants_probes_mapped_stepM4.mat exists and
%     contains the variable participants_probes_mapped.
%   - File derived/measurement_weights.mat exists and contains the variable
%     weights (a struct or compatible object passed to trust_cost_all).
%   - trust_cost_all.m is available on the MATLAB path and expects
%     arguments (theta, participants, dt, weights).
%
% This function does not perform gradient-based optimization; it only
% performs finite-difference style perturbations for debugging.

    % ---------------------------------------------------------------------
    % 1) Load participants and measurement weights
    % ---------------------------------------------------------------------
    load derived/participants_probes_mapped_stepM4.mat participants_probes_mapped
    participants = participants_probes_mapped;
    load derived/measurement_weights.mat weights

    % Simulation time step (must be consistent with trust_cost_all usage).
    dt = 0.1;

    % ---------------------------------------------------------------------
    % 2) Baseline parameter vector theta0
    % ---------------------------------------------------------------------
    % Parameter ordering and values must match trust_cost_all expectations.
    theta0 = [0.001;
              0.15;
              0.10;
              0.20;
              3.0;
              1e-4;
              1e-4];

    fprintf("\n=== DEBUGGING PARAMETER SENSITIVITY ===\n");

    % Compute baseline cost for reference.
    base_cost = trust_cost_all(theta0, participants, dt, weights);
    fprintf("Baseline cost = %.4f\n", base_cost);

    % ---------------------------------------------------------------------
    % 3) Perturb each parameter and re-evaluate cost
    % ---------------------------------------------------------------------
    % Small perturbation magnitude used for sensitivity probing.
    eps = 1e-4;

    for i = 1:length(theta0)
        theta_test = theta0;
        theta_test(i) = theta_test(i) + eps;

        fprintf("\n-- Testing parameter %d perturbation --\n", i);

        try
            c = trust_cost_all(theta_test, participants, dt, weights);

            % Check that the returned cost is a valid, finite scalar.
            if ~isscalar(c) || ~isreal(c) || ~isfinite(c)
                fprintf("Param %d produced INVALID cost: ", i);
                disp(c);
            else
                fprintf("Param %d OK, cost = %.4f\n", i, c);
            end
        catch ME
            % Report any runtime errors encountered for this perturbation.
            fprintf("Param %d ERROR: %s\n", i, ME.message);
        end
    end

    fprintf("\n=== END DEBUG ===\n");
end
