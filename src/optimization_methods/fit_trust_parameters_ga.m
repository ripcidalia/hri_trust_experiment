function [theta_hat, fval, exitflag, output] = fit_trust_parameters_ga(dt, preset, theta0)
% fit_trust_parameters_ga  Fit global trust-dynamics parameters via WLS + GA.
%
%   [theta_hat, fval, exitflag, output] = ...
%       fit_trust_parameters_ga(dt, preset)
%
%   [theta_hat, fval, exitflag, output] = ...
%       fit_trust_parameters_ga(dt, preset, theta0)
%
% Inputs:
%   dt      - time step for simulation (seconds)
%   preset  - GA compute-budget preset:
%               "moderate" | "heavy" | "overnight"
%   theta0  - optional initial parameter guess (8 x 1 vector). If provided,
%             it is injected into the initial GA population.
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
%   exitflag  - ga exit flag
%   output    - ga output struct
%
% Notes:
%   - The inequality constraint phi_succ <= phi_fail is enforced by
%     construction using a re-parameterisation:
%         phi_succ = rho * phi_fail,  with rho ∈ [0,1].

    if nargin < 2
        error('fit_trust_parameters_ga requires (dt, preset).');
    end
    if nargin < 3
        theta0 = [];
    end

    %% ------------------------------------------------------------
    % 1) Load participants and measurement weights
    % -------------------------------------------------------------
    fprintf('[fit_trust_parameters_ga] Loading data...\n');

    matP = "derived/participants_probes_mapped_stepM4.mat";
    if ~isfile(matP)
        error("File not found: %s. Run preprocessing (M1–M4) first.", matP);
    end
    S = load(matP, "participants_probes_mapped");
    if ~isfield(S, "participants_probes_mapped")
        error("Variable participants_probes_mapped not found in %s.", matP);
    end
    participants = S.participants_probes_mapped;
    N = numel(participants);
    fprintf('[fit_trust_parameters_ga] Loaded %d participants.\n', N);

    % Measurement weights (already computed via LOPO CV)
    matW = "derived/measurement_weights.mat";
    if ~isfile(matW)
        error("File not found: %s. You said this should contain 'weights'.", matW);
    end
    W = load(matW, "weights");
    if ~isfield(W, "weights")
        error("measurement_weights.mat does not contain 'weights'.");
    end
    meas_weights = W.weights;

    fprintf('[fit_trust_parameters_ga] Loaded measurement weights:\n');
    disp(meas_weights);

    %% ------------------------------------------------------------
    % 2) Bounds in theta space (same physical bounds as fmincon)
    % -------------------------------------------------------------
    %   lambda_rep ∈ [1e-4, 0.10], reputation should affect only first door trials
    %   alpha_sit  ∈ [0.00, 1.00], weight factor
    %   lambda_sit ∈ [0.10, 2.00], from graphical inspection
    %   phi_fail   ∈ [0.00, 1.00]
    %   phi_succ   ∈ [0.00, 1.00], with constraint phi_succ <= phi_fail
    %   a_succ     ∈ [0.01, 0.65], from graphical inspection
    %   lambda_lat ∈ [1e-6, 1e-2], not intended for fast dynamics
    %   kappa_lat  ∈ [1e-6, 1e-2], not intended for fast dynamics


    lb_theta = [
        1e-4;    % lambda_rep
        0;       % alpha_sit
        0.10;    % lambda_sit
        0;       % phi_fail
        0;       % phi_succ
        0.01;    % a_succ
        1e-6;    % lambda_lat
        1e-6;    % kappa_lat
    ];

    ub_theta = [
        0.10;    % lambda_rep
        1;       % alpha_sit
        2;       % lambda_sit
        1;       % phi_fail
        1;       % phi_succ
        0.65;    % a_succ
        1e-2;    % lambda_lat
        1e-2;    % kappa_lat
    ];

    %% ------------------------------------------------------------
    % 3) Re-parameterisation for phi_succ <= phi_fail
    % -------------------------------------------------------------
    % Optimisation variable:
    %   x = [ lambda_rep
    %         alpha_sit
    %         lambda_sit
    %         phi_fail
    %         rho
    %         a_succ
    %         lambda_lat
    %         kappa_lat ]
    %
    % with:
    %   phi_succ = rho * phi_fail,  rho ∈ [0,1].
    %
    lb_x = lb_theta;
    ub_x = ub_theta;

    % Replace bounds for the third variable (rho)
    lb_x(5) = 0.0;
    ub_x(5) = 1.0;

    %% ------------------------------------------------------------
    % 4) Define SAFE objective wrapper for GA
    % -------------------------------------------------------------
    % Raw cost evaluated in theta space:
    raw_fun_theta = @(theta) trust_cost_all(theta, participants, dt, meas_weights);

    % Robust wrapper: catches errors AND non-real/non-scalar/NaN/Inf and
    % returns a large penalty so GA never sees an undefined objective.
    function J = safe_obj_theta(theta_vec)
        persistent n_error_calls
        if isempty(n_error_calls)
            n_error_calls = 0;
        end

        try
            J_raw = raw_fun_theta(theta_vec);

            % ---- Hard checks: J must be a real, finite scalar ----
            if ~isscalar(J_raw) || ~isreal(J_raw) || ~isfinite(J_raw)
                n_error_calls = n_error_calls + 1;
                if n_error_calls <= 10
                    warning(['safe_obj_theta: invalid cost value (non-scalar, ' ...
                             'non-real, or non-finite). Using large penalty.']);
                end
                J = 1e6;
                return;
            end

            % All good
            J = J_raw;

        catch ME
            n_error_calls = n_error_calls + 1;
            if n_error_calls <= 10
                warning('safe_obj_theta: error in trust_cost_all at some theta: %s', ME.message);
            end
            J = 1e6;
        end
    end

    % Map x -> theta (enforces phi_succ <= phi_fail by construction)
    function theta = x_to_theta(x_vec)
        x_vec = x_vec(:);
        theta = x_vec;

        phi_fail = theta(2);
        rho      = theta(3);

        % Ensure rho stays within [0,1] even if GA produces slight numerical drift
        rho = min(max(rho, 0.0), 1.0);

        phi_succ = rho * phi_fail;
        theta(5) = phi_succ;

        % Safety: keep theta within physical bounds
        theta = min(max(theta, lb_theta), ub_theta);
    end

    obj = @(x) safe_obj_theta(x_to_theta(x));

    %% ------------------------------------------------------------
    % 5) Select GA preset and build options
    % -------------------------------------------------------------
    [popSize, maxGens, stallGens] = ga_preset(preset);

    options = optimoptions('ga', ...
        'Display', 'iter', ...
        'PopulationSize', popSize, ...
        'MaxGenerations', maxGens, ...
        'MaxStallGenerations', stallGens, ...
        'FunctionTolerance', 1e-6, ...
        'PlotFcn', [], ...
        'UseVectorized', false, ...
        'UseParallel', true);

    fprintf('[fit_trust_parameters_ga] Using preset "%s": pop=%d, gens=%d, stall=%d\n', ...
        preset, popSize, maxGens, stallGens);

    %% ------------------------------------------------------------
    % 6) Optional initial population seeding
    % -------------------------------------------------------------
    if ~isempty(theta0)
        fprintf('[fit_trust_parameters_ga] Seeding GA with provided initial guess.\n');

        theta0 = theta0(:);
        phi0   = theta0(2);
        psi0   = theta0(3);

        % Recover rho from phi_succ = rho * phi_fail
        if phi0 > 0
            rho0 = psi0 / phi0;
        else
            rho0 = 0.0;
        end
        rho0 = min(max(rho0, 0.0), 1.0);

        x0 = theta0;
        x0(3) = rho0;

        % Inject x0 as part of the initial population
        options = optimoptions(options, ...
            'InitialPopulationMatrix', x0.');
    end

    %% ------------------------------------------------------------
    % 7) Call GA
    % -------------------------------------------------------------
    fprintf('[fit_trust_parameters_ga] Starting GA...\n');

    nvars = 8;
    A = [];
    b = [];
    Aeq = [];
    beq = [];
    nonlcon = [];

    [x_hat, fval, exitflag, output] = ga( ...
        obj, nvars, A, b, Aeq, beq, lb_x, ub_x, nonlcon, options);

    theta_hat = x_to_theta(x_hat);

    fprintf('[fit_trust_parameters_ga] Done.\n');
    fprintf('  exitflag = %d\n', exitflag);
    fprintf('  final cost fval = %.6f\n', fval);
    fprintf('  theta_hat =\n');
    disp(theta_hat);

end


% ------------------------------------------------------------
% Local helper: GA preset definitions
% ------------------------------------------------------------
function [popSize, maxGens, stallGens] = ga_preset(preset)
    preset = string(lower(preset));

    switch preset
        case "moderate"
            popSize   = 60;
            maxGens   = 40;
            stallGens = 20;

        case "heavy"
            popSize   = 80;
            maxGens   = 60;
            stallGens = 30;

        case "overnight"
            popSize   = 120;
            maxGens   = 80;
            stallGens = 40;

        otherwise
            error('fit_trust_parameters_ga: unknown preset "%s". Use "moderate", "heavy", or "overnight".', preset);
    end
end
