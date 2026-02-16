function run_trust_diagnostics(participant_index, dt, theta, mode, behavior_params)
% run_trust_diagnostics  Simulate and visualize trust dynamics for one participant.
%
%   run_trust_diagnostics()
%   run_trust_diagnostics(participant_index)
%   run_trust_diagnostics(participant_index, dt)
%   run_trust_diagnostics(participant_index, dt, theta)
%
% This diagnostic function simulates the full trust-dynamics model for a
% single participant and visualizes the time evolution of:
%   - Total trust τ(t)
%   - Latent trust τ_lat(t)
%   - Reputation component τ_rep(t)
%   - Situational component τ_sit(t)
%   - Risk signal r(t)
%
% Door trial times and trust measurement times are marked on the plots.
% Observed trust measurements are overlaid on the total-trust plot,
% grouped by type (40-item, 14-item, probe) when labels are available.

    if nargin < 1 || isempty(participant_index)
        participant_index = 1;
    end
    if nargin < 2 || isempty(dt)
        dt = 0.1;
    end

    if nargin < 4 || isempty(mode)
        mode = "simple";
        behavior_params = struct();
    end

    if mode == "coupled" && (nargin < 5 || isempty(behavior_params))
        error("run_trust_diagnostics: no behavioral model parameters specified for coupled mode.");
    end

    % ------------------------------------------------------------
    % 0) Default theta if not provided
    % ------------------------------------------------------------
    if nargin < 3 || isempty(theta)
        theta = zeros(8,1);
        theta(1) = 3e-3;  % lambda_rep
        theta(2) = 0.5;   % alpha_sit
        theta(3) = 1;     % lambda_sit
        theta(4) = 0.15;  % phi_fail
        theta(5) = 0.10;  % phi_succ
        theta(6) = 0.20;  % a_succ
        theta(7) = 1e-4;  % lambda_lat
        theta(8) = 1e-4;  % kappa_lat
    else
        theta = theta(:);
        if numel(theta) ~= 8
            error("run_trust_diagnostics: theta must be 8x1 (got %d elements).", numel(theta));
        end
    end

    % ------------------------------------------------------------
    % 1) Load participant data
    % ------------------------------------------------------------
    matFile = "derived/participants_probes_mapped_stepM4.mat";
    if ~isfile(matFile)
        error("run_trust_diagnostics: file %s not found on path.", matFile);
    end

    S = load(matFile);
    if ~isfield(S, "participants_probes_mapped")
        error("run_trust_diagnostics: variable 'participants_probes_mapped' not found in %s.", matFile);
    end

    participants = S.participants_probes_mapped;
    N = numel(participants);
    if participant_index < 1 || participant_index > N
        error("run_trust_diagnostics: participant_index=%d out of range [1,%d].", ...
              participant_index, N);
    end

    P = participants(participant_index);
    fprintf("Diagnostics for participant %d (ID=%s)\n", ...
        participant_index, string(P.participant_id));

    % ------------------------------------------------------------
    % 2) Simulate or predict participant's trust dynamics
    % ------------------------------------------------------------
    
    sim = trust_simulate_or_predict_one_participant(mode, theta, P, dt, behavior_params);

    % ------------------------------------------------------------
    % 3) Basic console summary
    % ------------------------------------------------------------
    fprintf("Final trust τ(T)       = %.3f\n", sim.tau_hist(end));
    fprintf("Final latent τ_lat(T)  = %.3f\n", sim.tau_lat_hist(end));
    fprintf("Final rep τ_rep(T)     = %.3f\n", sim.tau_rep_hist(end));
    fprintf("Final risk r(T)        = %.3f\n", sim.risk_hist(end));

    % ------------------------------------------------------------
    % 4) Prepare measurement overlays (using trustProbes structure)
    % ------------------------------------------------------------
    [t_meas, y_meas, meas_types] = extract_measurements_for_plot(sim.measurements);

    hasTypes = ~isempty(meas_types);

    is40    = false(size(t_meas));
    is14    = false(size(t_meas));
    isProbe = false(size(t_meas));

    if hasTypes
        T = string(meas_types);

        % 40-item questionnaires (pre/post)
        is40 = (T == "t40_pre")  | (T == "t40_post") | ...
               (T == "q40_pre")  | (T == "q40_post");

        % 14-item mid questionnaires
        is14 = (T == "t14_mid1") | (T == "t14_mid2") | ...
               (T == "q14_mid1") | (T == "q14_mid2");

        % Single trust probes
        isProbe = (T == "probe") | (T == "trust_probe");
    end

    % Use the same times for vertical measurement lines.
    meas_times = t_meas;

    % ------------------------------------------------------------
    % 5) Plot diagnostics
    % ------------------------------------------------------------
    figure('Name','Trust Diagnostics','NumberTitle','off');
    set(gcf, 'Color','w');

    door_times = [sim.doorEvents.t];

    % --- Panel 1: Total trust with measurement overlays ---
    subplot(4,1,1);

    % Main simulated trajectory
    h_sim = plot(sim.t_grid, sim.tau_hist, 'LineWidth', 1.5, 'DisplayName','simulated');
    hold on;
    % Door markers (vertical dashed lines, hidden from legend)
    for t_d = door_times
        h_door = plot([t_d t_d], [0 1], '--', 'LineWidth', 0.5, 'Color',[0.7 0.7 0.7]);
        set(h_door, 'HandleVisibility','off');
    end

    % Measurement time markers (vertical dotted lines, hidden from legend)
    for t_m = meas_times
        h_mline = plot([t_m t_m], [0 1], ':', 'LineWidth', 0.5, 'Color',[0.85 0.85 0.85]);
        set(h_mline, 'HandleVisibility','off');
    end

    % Overlay observed trust measurements
    h40    = gobjects(0);
    h14    = gobjects(0);
    hProbe = gobjects(0);
    hOther = gobjects(0);
    hAll   = gobjects(0);

    if ~isempty(t_meas)
        if hasTypes
            if any(is40)
                h40 = plot(t_meas(is40), y_meas(is40), 'o', ...
                    'MarkerSize', 6, 'MarkerFaceColor','b', 'Color','b', ...
                    'DisplayName','40-item');
            end
            if any(is14)
                h14 = plot(t_meas(is14), y_meas(is14), 's', ...
                    'MarkerSize', 6, 'MarkerFaceColor','g', 'Color','g', ...
                    'DisplayName','14-item');
            end
            if any(isProbe)
                hProbe = plot(t_meas(isProbe), y_meas(isProbe), '^', ...
                    'MarkerSize', 6, 'MarkerFaceColor','r', 'Color','r', ...
                    'DisplayName','probe');
            end

            otherMask = ~(is40 | is14 | isProbe);
            if any(otherMask)
                hOther = plot(t_meas(otherMask), y_meas(otherMask), 'x', ...
                    'MarkerSize', 6, 'Color','k', ...
                    'DisplayName','other');
            end
        else
            % No type information: one marker group
            hAll = plot(t_meas, y_meas, 'o', ...
                'MarkerSize', 6, 'MarkerFaceColor','k', 'Color','k', ...
                'DisplayName','measurements');
        end
    end

    ylim([0 1]);
    xlabel('Time [s]');
    ylabel('\tau');
    title('Total trust and measurements');
    grid on;

    % Build a clean list of handles for the legend
    legHandles = [h_sim; h40(:); h14(:); hProbe(:); hOther(:); hAll(:)];
    legHandles = legHandles(isgraphics(legHandles));
    legend(legHandles, 'Location','best');


    % --- Panel 2: Latent component ---
    subplot(4,1,2);
    plot(sim.t_grid, sim.tau_lat_hist, 'LineWidth', 1.5);
    hold on;
    for t_d = door_times
        plot([t_d t_d], [0 1], '--', 'LineWidth', 0.5, 'Color',[0.7 0.7 0.7]);
    end
    ylim([0 1]);
    xlabel('Time [s]');
    ylabel('\tau^{lat}');
    title('Latent trust');
    grid on;

    % --- Panel 3: Reputation component ---
    subplot(4,1,3);
    plot(sim.t_grid, sim.tau_rep_hist, 'LineWidth', 1.5);
    hold on;
    yL = ylim;
    for t_d = door_times
        plot([t_d t_d], yL, '--', 'LineWidth', 0.5, 'Color',[0.7 0.7 0.7]);
    end
    ylim(yL);
    xlabel('Time [s]');
    ylabel('\tau^{rep}');
    title('Reputation');
    grid on;

    % --- Panel 4: Situational + risk ---
    subplot(4,1,4);
    yyaxis left;
    plot(sim.t_grid, sim.tau_sit_hist, 'LineWidth', 1.5);
    ylabel('\tau^{sit}');
    ylim([-1 1]);

    yyaxis right;
    plot(sim.t_grid, sim.risk_hist, '--', 'LineWidth', 1.0);
    ylabel('risk');
    ylim([-1 1]);

    xlabel('Time [s]');
    title('Situational trust and risk');
    grid on;

    sgtitle(sprintf('Participant %d (ID=%s)', ...
        participant_index, string(P.participant_id)));

end

% -------------------------------------------------------------------------
% Helper: extract measurement times, values, and types for plotting
% -------------------------------------------------------------------------
function [t_meas, y_meas, meas_types] = extract_measurements_for_plot(measurements)
% extract_measurements_for_plot  Prepare measurement overlays for plotting.
%
%   [t_meas, y_meas, meas_types] = extract_measurements_for_plot(measurements)
%
% Inputs:
%   measurements  - struct array returned by build_time_grid_and_events,
%                   typically derived from P.trustProbes. Expected to
%                   contain at least measurement times and values, with
%                   type labels when available.
%
% Outputs:
%   t_meas     - row vector of measurement times (1 x M)
%   y_meas     - row vector of measurement values (1 x M)
%   meas_types - cell array of type labels (1 x M), or {} if unavailable.

    if isempty(measurements)
        t_meas     = [];
        y_meas     = [];
        meas_types = {};
        return;
    end

    % --- Times ---
    if isfield(measurements, 't')
        t_meas = [measurements.t];
    elseif isfield(measurements, 't_s')
        % Directly from trustProbes
        t_meas = [measurements.t_s];
    else
        warning('extract_measurements_for_plot: no time field found; using indices as proxy.');
        t_meas = 1:numel(measurements);
    end
    t_meas = t_meas(:).';  % row vector

    % --- Values ---
    % Try several possible value fields in order of preference.
    candidateFields = {'y','trust','score'};
    y_meas = [];
    for k = 1:numel(candidateFields)
        f = candidateFields{k};
        if isfield(measurements, f)
            y_meas = [measurements.(f)];
            break;
        end
    end

    if isempty(y_meas)
        warning('extract_measurements_for_plot: could not find a value field; using zeros.');
        y_meas = zeros(size(t_meas));
    end
    y_meas = y_meas(:).';

    % --- Types ---
    if isfield(measurements, 'questionnaire_type')
        meas_types = {measurements.questionnaire_type};
    elseif isfield(measurements, 'type')
        meas_types = {measurements.type};
    elseif isfield(measurements, 'kind')
        meas_types = {measurements.kind};
    else
        meas_types = {};
    end
end
