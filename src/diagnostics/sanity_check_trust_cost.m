function sanity_check_trust_cost(theta)
% sanity_check_trust_cost  Diagnostic wrapper for trust_cost_one_participant/all.
%
%   sanity_check_trust_cost()
%   sanity_check_trust_cost(theta_hat)
%
% This diagnostic function evaluates the per-participant and total cost of
% the trust model for a given parameter vector, using the same cost
% definition as the fitting pipeline. It is intended as a quick sanity
% check to:
%   - Confirm that trust_cost_one_participant runs without errors for all
%     participants.
%   - Inspect the magnitude and variability of per-participant costs.
%   - Compare different parameter sets (e.g., baseline vs. fitted).
%
% The function:
%   1) Loads derived participant data and measurement weights.
%   2) For each participant, calls trust_cost_one_participant.
%   3) Prints the per-participant costs and the total sum.
%   4) Produces a stem plot of per-participant costs and a histogram of
%      their distribution.
%
% Inputs (optional):
%   theta  - Parameter vector used by trust_cost_one_participant (7x1 or
%            compatible). If omitted or empty, a default baseline vector is
%            used. The semantics and ordering of theta must be consistent
%            with trust_cost_one_participant / trust_cost_all.
%
% Assumptions:
%   - File derived/participants_probes_mapped_stepM4.mat exists and
%     contains a variable participants_probes_mapped with participant
%     structs.
%   - File derived/measurement_weights.mat exists and contains a struct
%     'weights' with measurement weights (w40, w14, w_probe, etc.).
%   - trust_cost_one_participant.m is available on the MATLAB path and
%     consistent with the parameter ordering used here.
%
% This function does not modify any data; it only evaluates and visualizes
% costs for diagnostic purposes.

    % ---------------------------------------------------------------------
    % 0) Handle optional input
    % ---------------------------------------------------------------------
    if nargin < 1 || isempty(theta)
        % Default theta vector used as a baseline sanity-check point.
        % The interpretation and ordering of elements must match
        % trust_cost_one_participant. Numerical values are chosen to be
        % consistent with other scripts in this project.
        theta = [ ...            
            3e-3;   % 1) lambda_rep
            0.5;    % 2) aplha_sit
            1;      % 3) lambda_sit
            0.15;   % 4) phi_fail
            0.10;   % 5) phi_succ
            0.20;   % 6) a_succ
            1e-4;   % 7) lambda_lat
            1e-4;   % 8) kappa_lat
        ];
        fprintf("Using DEFAULT theta0:\n");
    else
        fprintf("Using PROVIDED theta:\n");
    end

    disp(theta);

    % ---------------------------------------------------------------------
    % 1) Load participants
    % ---------------------------------------------------------------------
    % Load the derived participant structures with probes mapped to the
    % 40-item-equivalent scale.
    matFile = "derived/participants_probes_mapped_stepM4.mat";
    if ~isfile(matFile)
        error("File not found: %s", matFile);
    end

    S = load(matFile, "participants_probes_mapped");
    if ~isfield(S, "participants_probes_mapped")
        error("Variable 'participants_probes_mapped' not found in %s.", matFile);
    end
    participants = S.participants_probes_mapped;
    N = numel(participants);
    fprintf("\n=== Sanity check: trust_cost_all / trust_cost_one_participant ===\n");
    fprintf("Loaded %d participants.\n", N);

    % ---------------------------------------------------------------------
    % 2) Load measurement weights
    % ---------------------------------------------------------------------
    % Load the measurement weights that define the WLS scaling of different
    % trust measurements (questionnaires, probes).
    Wfile = "derived/measurement_weights.mat";
    if ~isfile(Wfile)
        error("Measurement weights file not found: %s", Wfile);
    end

    W = load(Wfile);
    if ~isfield(W, "weights")
        error("File %s does not contain 'weights'.", Wfile);
    end
    weights = W.weights;

    fprintf("Loaded measurement weights:\n");
    disp(weights);

    % ---------------------------------------------------------------------
    % 3) Simulation settings
    % ---------------------------------------------------------------------
    % Time step used during simulation. This is independent of the dt used
    % in the fitting script and is chosen for diagnostic purposes.
    dt = 1;

    % ---------------------------------------------------------------------
    % 4) Compute total cost and per-participant contributions
    % ---------------------------------------------------------------------
    % For each participant, compute the individual cost using the current
    % parameter vector theta and accumulate the total cost.
    per_cost = zeros(N,1);

    for i = 1:N
        P = participants(i);
        Ji = trust_cost_one_participant(theta, P, dt, weights);
        per_cost(i) = Ji;

        fprintf("Participant %2d (ID=%s): cost = %.4f\n", ...
            i, string(P.participant_id), Ji);
    end

    J_total = sum(per_cost);
    fprintf("\nTOTAL COST = %.4f\n", J_total);

    % ---------------------------------------------------------------------
    % 5) Plot per-participant costs
    % ---------------------------------------------------------------------
    % Clean and prepare cost values for plotting, ignoring any non-finite
    % values in the visualizations while still reporting their presence.
    per_cost = double(per_cost(:));       % numeric column vector
    per_cost_real = real(per_cost);       % ensure real
    finiteMask = isfinite(per_cost_real);

    if ~all(finiteMask)
        warning("Some per-participant costs are non-finite; ignoring in plots.");
    end

    idx  = find(finiteMask);
    vals = per_cost_real(finiteMask);

    figure('Name','Cost diagnostics','NumberTitle','off');
    set(gcf, 'Color', 'w');

    % Bar-like stem plot of all finite per-participant costs
    subplot(1,2,1);
    stem(idx, vals, 'filled');
    xlabel('Participant index');
    ylabel('Cost J_i');
    title('Per-participant costs');
    grid on;

    % Histogram of finite per-participant costs
    subplot(1,2,2);
    histogram(vals, 20);
    xlabel('Cost J_i');
    ylabel('Count');
    title('Distribution of per-participant costs');
    grid on;

end
