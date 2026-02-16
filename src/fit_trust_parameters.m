function [theta_hat, fval, exitflag, output] = ...
    fit_trust_parameters(method, dt, preset, theta0)
% fit_trust_parameters  Fit global trust-dynamics parameters via WLS.
%
%   [theta_hat, fval, exitflag, output] = ...
%       fit_trust_parameters(method, dt)
%
%   [theta_hat, fval, exitflag, output] = ...
%       fit_trust_parameters(method, dt, preset)
%
%   [theta_hat, fval, exitflag, output] = ...
%       fit_trust_parameters(method, dt, preset, theta0)
%
% This function is the conceptual entry point for parameter fitting. It
% dispatches to a solver-specific implementation:
%   - "fmincon"      : gradient-based constrained optimisation
%   - "ga"           : genetic algorithm (derivative-free, global)
%   - "patternsearch": derivative-free local search (polling-based)
%
% Inputs:
%   method - optimisation method (string):
%              "fmincon", "ga", or "patternsearch"
%   dt     - time step for simulation (seconds)
%   preset - solver preset name (optional, method-dependent)
%            For GA:          "moderate" | "heavy" | "overnight"
%            For patternsearch: "moderate" | "heavy" | "overnight"
%   theta0 - optional initial parameter guess (Nparams x 1 vector). If provided,
%            it is used to initialise the solver.
%
% Outputs:
%   theta_hat - estimated parameter vector:
%       [ lambda_rep
%         alpha_sit
%         lambda_sit
%         phi_fail
%         phi_succ
%         a_succ
%         lambda_lat
%         kappa_lat ]
%
%   fval      - final WLS cost
%   exitflag  - solver exit flag
%   output    - solver output struct

    if nargin < 2
        error('fit_trust_parameters requires at least (method, dt).');
    end
    if nargin < 3
        preset = [];
    end
    if nargin < 4
        theta0 = [];
    end

    method = string(lower(method));

    switch method
        case "fmincon"
            [theta_hat, fval, exitflag, output] = ...
                fit_trust_parameters_fmincon(dt, theta0);

        case "ga"
            if isempty(preset)
                preset = "moderate";
            end
            [theta_hat, fval, exitflag, output] = ...
                fit_trust_parameters_ga(dt, preset, theta0);

        case "patternsearch"
            if isempty(preset)
                preset = "moderate";
            end
            [theta_hat, fval, exitflag, output] = ...
                fit_trust_parameters_patternsearch(dt, preset, theta0);

        otherwise
            error('fit_trust_parameters: unknown method "%s". Use "fmincon", "ga", or "patternsearch".', method);
    end
end
