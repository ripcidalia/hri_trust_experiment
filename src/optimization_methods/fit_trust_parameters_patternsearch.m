function [theta_hat, fval, exitflag, output] = ...
    fit_trust_parameters_patternsearch(dt, preset, theta0)
% fit_trust_parameters_patternsearch  Fit global trust-dynamics parameters via WLS + patternsearch.
%
%   [theta_hat, fval, exitflag, output] = ...
%       fit_trust_parameters_patternsearch(dt, preset)
%
%   [theta_hat, fval, exitflag, output] = ...
%       fit_trust_parameters_patternsearch(dt, preset, theta0)
%
% Inputs:
%   dt      - time step for simulation (seconds)
%   preset  - patternsearch compute-budget preset:
%               "moderate" | "heavy" | "overnight"
%   theta0  - optional initial parameter guess (8 x 1 vector). If not provided,
%             a default heuristic initialisation is used.
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
%   exitflag  - patternsearch exit flag
%   output    - patternsearch output struct
%
% Notes:
%   - The inequality constraint phi_succ <= phi_fail is enforced by
%     construction using a re-parameterisation:
%         phi_succ = rho * phi_fail,  with rho ∈ [0,1].
%     patternsearch optimises x where x(3)=rho, and theta is recovered.

    if nargin < 2
        error('fit_trust_parameters_patternsearch requires (dt, preset).');
    end
    if nargin < 3
        theta0 = [];
    end
    preset = string(lower(preset));

    %% ------------------------------------------------------------
    % 1) Load participants and measurement weights
    % -------------------------------------------------------------
    fprintf('[fit_trust_parameters_patternsearch] Loading data...\n');

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
    fprintf('[fit_trust_parameters_patternsearch] Loaded %d participants.\n', N);

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

    fprintf('[fit_trust_parameters_patternsearch] Loaded measurement weights:\n');
    disp(meas_weights);

    %% ------------------------------------------------------------
    % 2) Default initial parameter guess theta0
    % -------------------------------------------------------------
    % Order:
    %   1: lambda_rep
    %   2: alpha_sit
    %   3: lambda_sit
    %   4: phi_fail
    %   5: phi_succ
    %   6: a_succ
    %   7: lambda_lat
    %   8: kappa_lat

    if isempty(theta0)
        theta0 = [
            1e-3;   % lambda_rep
            0.5;    % alpha_sit
            1.0;    % lambda_sit
            0.15;   % phi_fail
            0.10;   % phi_succ
            0.20;   % a_succ
            1e-4;   % lambda_lat
            1e-4;   % kappa_lat
        ];
        fprintf('[fit_trust_parameters_patternsearch] Using default initial guess.\n');
    else
        fprintf('[fit_trust_parameters_patternsearch] Using user-provided initial guess.\n');
        theta0 = theta0(:);
    end

    %% ------------------------------------------------------------
    % 3) Bounds on parameters
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
    % 4) Re-parameterisation for phi_succ <= phi_fail
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
    lb_x = lb_theta;
    ub_x = ub_theta;
    lb_x(5) = 0.0;
    ub_x(5) = 1.0;

    % Convert theta0 -> x0
    phi0 = theta0(4);
    psi0 = theta0(5);
    if phi0 > 0
        rho0 = psi0 / phi0;
    else
        rho0 = 0.0;
    end
    rho0 = min(max(rho0, 0.0), 1.0);

    x0 = theta0(:);
    x0(5) = rho0;

    %% ------------------------------------------------------------
    % 5) Define SAFE objective wrapper for patternsearch
    % -------------------------------------------------------------
    raw_fun_theta = @(theta) trust_cost_all(theta, participants, dt, meas_weights);

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

            J = J_raw;

        catch ME
            n_error_calls = n_error_calls + 1;
            if n_error_calls <= 10
                warning('safe_obj_theta: error in trust_cost_all at some theta: %s', ME.message);
            end
            J = 1e6;
        end
    end

    function theta = x_to_theta(x_vec)
        x_vec = x_vec(:);
        theta = x_vec;

        phi_fail = theta(4);
        rho      = theta(5);
        rho = min(max(rho, 0.0), 1.0);

        phi_succ = rho * phi_fail;
        theta(5) = phi_succ;

        % Safety: keep theta within physical bounds
        theta = min(max(theta, lb_theta), ub_theta);
    end

    obj = @(x) safe_obj_theta(x_to_theta(x));

    %% ------------------------------------------------------------
    % 6) Select preset and build patternsearch options
    % -------------------------------------------------------------
    [maxEvals, maxIters, initMesh, tolMesh] = patternsearch_preset(preset);

    options = optimoptions('patternsearch', ...
        'Display', 'iter', ...
        'UseParallel', true, ...
        'MaxFunctionEvaluations', maxEvals, ...
        'MaxIterations', maxIters, ...
        'InitialMeshSize', initMesh, ...
        'MeshTolerance', tolMesh);

    fprintf('[fit_trust_parameters_patternsearch] Using preset "%s": evals=%d, iters=%d\n', ...
        preset, maxEvals, maxIters);

    %% ------------------------------------------------------------
    % 7) Call patternsearch
    % -------------------------------------------------------------
    fprintf('[fit_trust_parameters_patternsearch] Starting patternsearch...\n');

    A = [];
    b = [];
    Aeq = [];
    beq = [];
    nonlcon = [];

    [x_hat, fval, exitflag, output] = patternsearch( ...
        obj, x0, A, b, Aeq, beq, lb_x, ub_x, nonlcon, options);

    theta_hat = x_to_theta(x_hat);

    fprintf('[fit_trust_parameters_patternsearch] Done.\n');
    fprintf('  exitflag = %d\n', exitflag);
    fprintf('  final cost fval = %.6f\n', fval);
    fprintf('  theta_hat =\n');
    disp(theta_hat);

end


% ------------------------------------------------------------
% Local helper: patternsearch preset definitions
% ------------------------------------------------------------
function [maxEvals, maxIters, initMesh, tolMesh] = patternsearch_preset(preset)
    preset = string(lower(preset));

    switch preset
        case "moderate"
            maxEvals = 2000;
            maxIters = 200;
            initMesh = 0.10;
            tolMesh  = 1e-3;

        case "heavy"
            maxEvals = 6000;
            maxIters = 400;
            initMesh = 0.20;
            tolMesh  = 5e-4;

        case "overnight"
            maxEvals = 15000;
            maxIters = 800;
            initMesh = 0.30;
            tolMesh  = 1e-4;

        otherwise
            error('fit_trust_parameters_patternsearch: unknown preset "%s". Use "moderate", "heavy", or "overnight".', preset);
    end
end
