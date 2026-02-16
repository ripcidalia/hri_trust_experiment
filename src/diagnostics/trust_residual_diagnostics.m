function [residuals_tbl, summary_tbl] = trust_residual_diagnostics(run_id, varargin)
% trust_residual_diagnostics  Global residual analysis over all participants.
%
%   [residuals_tbl, summary_tbl] = trust_residual_diagnostics(theta)
%   [residuals_tbl, summary_tbl] = trust_residual_diagnostics(theta, dt)
%
% This function performs a global residual diagnostic of the trust model
% across all participants used in the fitting pipeline. For each
% participant, it:
%
%   1) Runs the forward simulation via trust_simulate_or_predict_one_participant.
%   2) Extracts observed trust measurements (y_obs) and model predictions
%      (y_hat) at all measurement times.
%   3) Computes residuals: r = y_obs - y_hat.
%
% All residuals are aggregated into a single table with one row per
% measurement, and summary statistics are computed per measurement type
% (40-item, 14-item, probe).
%
% Inputs:
%   theta      
%       8x1 parameter vector (as used in trust_cost_all /
%       trust_cost_one_participant), with layout:
%         1: lambda_rep
%         2: alpha_sit
%         3: lambda_sit
%         4: phi_fail
%         5: phi_succ
%         6: a_succ
%         7: lambda_lat
%         8: kappa_lat
%
%   dt         
%       Time step (seconds) for the simulation grid. If omitted or
%       empty, defaults to 1s (consistent with the fitting runs).
%
%   mode       
%       Simulation mode (string):
%         - "simple", or "coupled"
%
%   behavior_params 
%       Relevant parameters for the behavioral model.
%
% Outputs:
%   residuals_tbl  - MATLAB table with one row per measurement, columns:
%                      participant_index
%                      participant_id
%                      meas_index
%                      t              (time in seconds)
%                      kind           ("t40_post", "t14_mid1", "t14_mid2", "probe")
%                      y_obs          (observed trust, 0..1)
%                      y_hat          (predicted trust, 0..1)
%                      residual       (y_obs - y_hat)
%                      weight         (instrument weight used in WLS)
%
%   summary_tbl    - Table of summary statistics per measurement type:
%                      kind
%                      N
%                      mean_residual
%                      std_residual
%                      rmse
%                      mean_abs_residual
%
% Side effects:
%   - Produces a diagnostic figure with:
%       * Histogram of all residuals.
%       * Boxplot of residuals grouped by measurement type.
%       * Scatter of residual vs. predicted.
%       * Residual vs. time.
%
% Requirements:
%   - derived/participants_probes_mapped_stepM4.mat
%   - derived/measurement_weights.mat  or  measurement_weights.mat
%   - trust_simulate_or_predict_one_participant on the MATLAB path.


    theta_default = [ 0.0524
                      0.0290
                      0.8307
                      0.0417
                      0.0124
                      0.3787
                      0.0058
                      0.0043];
   
    % Parse inputs
    p = inputParser;
    p.addParameter("Split", "valid", @(s) isstring(s) || ischar(s));  % "train" or "valid"
    p.addParameter("Theta", theta_default, @(x) isnumeric(x) && isscalar(x) && x>0);  % allow user defined theta
    p.addParameter("Dt", 1, @(x) isnumeric(x) && isscalar(x) && x>=0);  % allow user defined dt    
    p.parse(varargin{:});
    args = p.Results;

    dt = args.Dt;
    
    % ------------------------------------------------------------
    % 1) Load participants,  measurement weights and theta
    % ------------------------------------------------------------

    if nargin < 1 || isempty(run_id)
        % No run_id specified, proceed with general participant struct
        matP = "derived/participants_probes_mapped_stepM4.mat";
        weightsP = "derived/measurement_weights.mat";
        theta = args.Theta;
        thetaP = "user_defined";
    else  
        splitName = lower(string(args.Split));
        if ~(splitName == "train" || splitName == "valid")
            error("Split must be 'train' or 'valid'. Got: %s", splitName);
        end
  
        % Resolve participant file for requested split
        if splitName == "train"
            matP = fullfile("derived", "analysis_runs", run_id, "stepA1_prepare_analysis", "participants_probes_mapped_stepM4.mat");
        else
            matP = fullfile("derived", "analysis_runs", run_id, "stepA1_prepare_analysis", "participants_valid_probes_mapped_stepM4.mat");
        end

        % Weights and theta are independent of the split, not of the run_id
        weightsP = fullfile("derived", "analysis_runs", run_id, "stepA1_prepare_analysis", "measurement_weights.mat");
        thetaP = fullfile("derived", "analysis_runs", run_id, "stepA3_model_selection", "theta_star.mat");
    end
    
    if ~isfile(matP)
        error("trust_residual_diagnostics: file %s not found.", matP);
    end

    S = load(matP);
    if ~isfield(S, "participants_probes_mapped")
        error("trust_residual_diagnostics: variable 'participants_probes_mapped' not found in %s.", matP);
    end
    participants = S.participants_probes_mapped;
    N = numel(participants);
    fprintf("[trust_residual_diagnostics] Loaded %d participants.\n", N);

    weights = [];
    W = load(weightsP);
    if isfield(W, "weights")
            weights = W.weights;
    end

    if isempty(weights)
        error("trust_residual_diagnostics: could not find measurement_weights.mat (derived/ or current folder).");
    end

    if ~isfield(weights, "w40") || ~isfield(weights, "w14") || ~isfield(weights, "w_probe")
        error("trust_residual_diagnostics: weights struct must contain fields w40, w14, w_probe.");
    end

    fprintf("[trust_residual_diagnostics] Loaded measurement weights:\n");
    disp(weights);

    if thetaP ~= "user_defined"
        T = load(thetaP);
        if isfield(T, "theta_star")
            theta = T.theta_star;
        end
    end
  
    if numel(theta) ~= 8
        error("trust_residual_diagnostics: theta must be 8x1 (got %d elements).", numel(theta));
    end

    % ------------------------------------------------------------
    % 3) Accumulate residuals across all participants
    % ------------------------------------------------------------
    participant_index_all = [];
    participant_id_all    = strings(0,1);
    meas_index_all        = [];
    t_all                 = [];
    kind_all              = strings(0,1);
    y_obs_all             = [];
    y_hat_all             = [];
    residual_all          = [];
    weight_all            = [];

    total_meas = 0;

    for i = 1:N
        P = participants(i);
        pid = string(P.participant_id);

        try
            sim = trust_simulate_or_predict_one_participant("simple", theta, P, dt);
        catch ME
            warning("trust_residual_diagnostics: simulation failed for participant %s: %s", ...
                    pid, ME.message);
            continue;
        end

        meas  = sim.measurements;
        y_hat = sim.y_hat;

        if isempty(meas)
            continue;
        end

        nMeas = numel(meas);
        if numel(y_hat) ~= nMeas
            warning("trust_residual_diagnostics: y_hat length (%d) != #meas (%d) for participant %s. Skipping.", ...
                    numel(y_hat), nMeas, pid);
            continue;
        end

        % Preallocate per-participant arrays
        t_i        = zeros(nMeas,1);
        kind_i     = strings(nMeas,1);
        y_obs_i    = zeros(nMeas,1);
        y_hat_i    = zeros(nMeas,1);
        residual_i = zeros(nMeas,1);
        w_i        = zeros(nMeas,1);

        for j = 1:nMeas
            t_i(j)     = meas(j).t;
            kind_i(j)  = string(meas(j).kind);
            y_obs_i(j) = meas(j).y;
            y_hat_i(j) = y_hat(j);
            residual_i(j) = y_obs_i(j) - y_hat_i(j);

            % Map measurement type to its WLS weight
            kj = kind_i(j);
            if kj == "t40_post"
                w_i(j) = weights.w40;
            elseif kj == "t14_mid1" || kj == "t14_mid2"
                w_i(j) = weights.w14;
            elseif kj == "probe"
                w_i(j) = weights.w_probe;
            else
                % Unknown type: default to probe weight for robustness
                w_i(j) = weights.w_probe;
            end
        end

        participant_index_all = [participant_index_all; i * ones(nMeas,1)];
        participant_id_all    = [participant_id_all; repmat(pid, nMeas, 1)];
        meas_index_all        = [meas_index_all; (1:nMeas).'];
        t_all                 = [t_all; t_i];
        kind_all              = [kind_all; kind_i];
        y_obs_all             = [y_obs_all; y_obs_i];
        y_hat_all             = [y_hat_all; y_hat_i];
        residual_all          = [residual_all; residual_i];
        weight_all            = [weight_all; w_i];

        total_meas = total_meas + nMeas;
    end

    fprintf("[trust_residual_diagnostics] Aggregated %d measurements across all participants.\n", total_meas);

    % ------------------------------------------------------------
    % 4) Build output table
    % ------------------------------------------------------------
    residuals_tbl = table( ...
        participant_index_all, ...
        participant_id_all, ...
        meas_index_all, ...
        t_all, ...
        kind_all, ...
        y_obs_all, ...
        y_hat_all, ...
        residual_all, ...
        weight_all, ...
        'VariableNames', { ...
            'participant_index', ...
            'participant_id', ...
            'meas_index', ...
            't', ...
            'kind', ...
            'y_obs', ...
            'y_hat', ...
            'residual', ...
            'weight'});

    % ------------------------------------------------------------
    % 5) Summary statistics per measurement type
    % ------------------------------------------------------------
    if isempty(residual_all)
        warning("trust_residual_diagnostics: no residuals collected; summary table will be empty.");
        summary_tbl = table();
        return;
    end

    kinds_unique = unique(kind_all);
    nKinds = numel(kinds_unique);

    kind_col     = strings(nKinds,1);
    N_col        = zeros(nKinds,1);
    mean_res_col = zeros(nKinds,1);
    std_res_col  = zeros(nKinds,1);
    rmse_col     = zeros(nKinds,1);
    mean_abs_col = zeros(nKinds,1);

    for k = 1:nKinds
        kk = kinds_unique(k);
        mask = (kind_all == kk);

        r_k   = residual_all(mask);
        w_k   = weight_all(mask);

        kind_col(k) = kk;
        N_col(k)    = numel(r_k);

        if ~isempty(r_k)
            mean_res_col(k) = mean(r_k);
            std_res_col(k)  = std(r_k);
            % Weighted RMSE (using same weights as in cost)
            rmse_col(k)     = sqrt( sum(w_k .* (r_k.^2)) / max(sum(w_k), eps) );
            mean_abs_col(k) = mean(abs(r_k));
        else
            mean_res_col(k) = NaN;
            std_res_col(k)  = NaN;
            rmse_col(k)     = NaN;
            mean_abs_col(k) = NaN;
        end
    end

    summary_tbl = table( ...
        kind_col, ...
        N_col, ...
        mean_res_col, ...
        std_res_col, ...
        rmse_col, ...
        mean_abs_col, ...
        'VariableNames', { ...
            'kind', ...
            'N', ...
            'mean_residual', ...
            'std_residual', ...
            'rmse', ...
            'mean_abs_residual'});

    fprintf("[trust_residual_diagnostics] Summary statistics by measurement type:\n");
    disp(summary_tbl);

    % ------------------------------------------------------------
    % 6) Diagnostic plots
    % ------------------------------------------------------------
    figure('Name','Global trust residual diagnostics','NumberTitle','off');
    set(gcf, 'Color','w');

    % Panel 1: Histogram of all residuals
    subplot(2,2,1);
    histogram(residual_all, 30);
    xlabel('residual = y_{obs} - y_{hat}');
    ylabel('count');
    title('Residual histogram');
    grid on;

    % Panel 2: Boxplot by measurement type
    subplot(2,2,2);
    boxplot(residual_all, categorical(kind_all));
    xlabel('measurement type');
    ylabel('residual');
    title('Residuals by measurement type');
    grid on;

    % Panel 3: Residual vs. predicted
    subplot(2,2,3);
    scatter(y_hat_all, residual_all, 10, 'filled');
    xlabel('y_{hat}');
    ylabel('residual');
    title('Residual vs. predicted');
    grid on;

    % Panel 4: Residual vs. time
    subplot(2,2,4);
    scatter(t_all, residual_all, 10, 'filled');
    xlabel('time [s]');
    ylabel('residual');
    title('Residual vs. time');
    grid on;

end
