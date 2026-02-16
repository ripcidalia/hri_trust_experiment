function measure_trust_cost_runtime(dt)
% measure_trust_cost_runtime  Benchmark the runtime of trust_cost_all.
%
%   measure_trust_cost_runtime()
%   measure_trust_cost_runtime(dt)
%
% This utility function benchmarks the runtime of a single evaluation of
% the global trust model cost function trust_cost_all, and also reports
% the runtime of simulating a single participant. It is intended for
% profiling and sanity-checking the computational load of the parameter
% estimation pipeline.
%
% The function:
%   1) Loads time-annotated participants with mapped trust probes.
%   2) Loads measurement weights used in the WLS cost.
%   3) Evaluates trust_cost_all once as a warm-up (JIT, caching, etc.).
%   4) Times a single trust_cost_all call across all participants.
%   5) Repeats the timing multiple times and reports mean and std.
%   6) Times trust_simulate_or_predict_one_participant for a single participant and
%      extrapolates an approximate total cost if all participants were
%      simulated individually.
%
% Inputs (optional):
%   dt  - Time step for simulation (in seconds), passed to
%         trust_cost_all and trust_simulate_or_predict_one_participant.
%         If omitted or empty, dt defaults to 0.1, consistent with the
%         choice used in fit_trust_parameters for runtime assessment.
%
% Assumptions:
%   - File derived/participants_probes_mapped_stepM4.mat exists and
%     contains a variable participants_probes_mapped with the participant
%     structs used during model fitting.
%   - File measurement_weights.mat exists and contains a struct 'weights'
%     with measurement weights for questionnaires and probes.
%   - trust_cost_all.m and trust_simulate_or_predict_one_participant.m are available
%     on the MATLAB path and consistent with the trust model and data.
%
% This function does not modify any data or parameters; it only measures
% runtime characteristics for the existing implementation.

    if nargin < 1 || isempty(dt)
        dt = 0.25;  % Default time step, consistent with fit_trust_parameters
    end

    fprintf('=== Measuring trust_cost_all runtime (dt = %.3f) ===\n', dt);

    % ------------------------------------------------------------
    % 1) Load participants
    % ------------------------------------------------------------
    % Load the participant data with trust probes mapped to the
    % 40-item-equivalent scale. This is the same structure used in
    % parameter estimation.
    S = load("derived/participants_probes_mapped_stepM4.mat", "participants_probes_mapped");
    participants = S.participants_probes_mapped;
    N = numel(participants);
    fprintf('Loaded %d participants.\n', N);

    % ------------------------------------------------------------
    % 2) Load measurement weights
    % ------------------------------------------------------------
    % Load the measurement weights that determine the relative influence
    % of different measurement types in the WLS cost function.
    W = load("derived/measurement_weights.mat", "weights");
    weights = W.weights;
    disp('Loaded measurement weights:');
    disp(weights);

    % ------------------------------------------------------------
    % 3) Define the same theta0 as in fit_trust_parameters
    % ------------------------------------------------------------
    % Use the same initial parameter vector as in fit_trust_parameters for
    % consistency when timing trust_cost_all.
    theta0 = [ ...
        3e-3;   % 1) lambda_rep
        0.50;   % 2) alpha_sit
        1.0;    % 3) lambda_sit
        0.15;   % 4) phi_fail
        0.10;   % 5) phi_succ
        0.20;   % 6) a_succ
        1e-4;   % 7) lambda_lat
        1e-4;   % 8) kappa_lat
    ];

    % Warm-up call to account for JIT compilation and internal caching
    % effects, so that subsequent timings are more representative.
    c0 = trust_cost_all(theta0, participants, dt, weights); %#ok<ASGLU>
    fprintf('Warm-up cost = %.4f\n', c0);

    % ------------------------------------------------------------
    % 4) Time a single call
    % ------------------------------------------------------------
    % Measure the runtime of a single evaluation of trust_cost_all over
    % all participants.
    tic;
    c1 = trust_cost_all(theta0, participants, dt, weights); %#ok<ASGLU>
    t1 = toc;

    fprintf('Single trust_cost_all call took %.3f seconds. Cost = %.4f\n', t1, c1);

    % ------------------------------------------------------------
    % 5) Optional: repeat a few times and average
    % ------------------------------------------------------------
    % Repeat the timing several times to estimate the average and
    % variability of the runtime.
    nRepeat = 3;
    times = zeros(nRepeat,1);
    for r = 1:nRepeat
        tic;
        trust_cost_all(theta0, participants, dt, weights);
        times(r) = toc;
    end
    fprintf('Repeated %d times: mean = %.3f s, std = %.3f s\n', ...
        nRepeat, mean(times), std(times));

    % ------------------------------------------------------------
    % 6) Optional: time a single participant simulation
    % ------------------------------------------------------------
    % Time the simulation for one participant using trust_simulate_or_predict_one_participant
    % and extrapolate the approximate total cost if all participants had a
    % similar simulation runtime.
    idx = 1;  % pick a participant to inspect
    P = participants(idx);

    fprintf('\n--- Timing trust_simulate_or_predict_one_participant for participant %d (ID=%s) ---\n', ...
        idx, string(P.participant_id));

    tic;
    sim = trust_simulate_or_predict_one_participant("simple", theta0, P, dt); 
    tP = toc;

    fprintf('trust_simulate_or_predict_one_participant took %.3f seconds (dt = %.3f)\n', tP, dt);
    fprintf('If all participants cost similarly, full cost is approx %.3f seconds.\n', tP * N);

    fprintf('=== Done measuring runtime ===\n');
end
