% main.m
%
% MAIN Entry point for the SAR trust + behavior modelling pipeline.
%
% This script runs the end-to-end workflow for the human–robot trust study,
% from raw data preprocessing and measurement mapping through trust model
% fitting, analysis reporting, behavioral modelling, robustness checks, and
% coupled rollout validation.
%
% -------------------------------------------------------------------------
% Pipeline overview
%
% A) Data processing (project-wide; produces cleaned participant structs)
%   1) preprocessing.m
%      - Load raw experimental logs/events
%      - Clean and standardize event streams and timestamps
%      - Produce per-participant, time-aligned data for modelling
%
%   2) postprocessing.m
%      - Map questionnaire/probe responses into model-ready measurements
%      - Apply measurement weights/calibration and generate diagnostics
%
% B) Trust dynamics model fitting (produces artifacts in derived/fit_runs)
%   3) run_trust_optimisation_pipeline(cfg)
%      - Fit trust-dynamics parameters (theta) in "simple" mode on a fixed dt
%      - Save run results and checkpoints for reproducibility
%
% C) Analysis run creation and reporting
%    (run-local outputs under derived/analysis_runs/<run_id>/)
%
%   4)  stepA0_import_fit_results(cfg)
%       - Register optimisation outputs and create an analysis run ID
%
%   5)  stepA1_prepare_analysis_run(run_id)
%       - Archive inputs for this run (participant splits, mappings, metadata)
%
%   6)  stepA2_simple_mode_fit_report(run_id, resultsMatPath)
%       - Trust model fit quality summaries and figures
%
%   7)  stepA3_select_theta_for_coupled(run_id, resultsMatPath)
%       - Select theta_star for downstream prediction and coupled analyses
%
%   8)  stepA4_sensitivity_simple_mode(run_id, resultsMatPath)
%       - Sensitivity analysis around fitted trust parameters
%
%   9)  stepA5_compare_baselines_simple_mode(run_id, resultsMatPath)
%       - Compare simple-mode baselines using held-out evaluation metrics
%
%  10)  stepA6_report_baseline_comparison_simple_mode(run_id)
%       - Compile baseline-comparison reporting artifacts
%
%  11)  stepA7_build_behavior_dataset(run_id)
%       - Build per-door behavioral dataset (follow/override decisions)
%
%  12)  stepA8_behavior_fit_eval(run_id)
%       - Fit behavioral models on TRAIN and evaluate on VALID
%
%  13)  stepA9_behavior_rollouts(run_id)
%       - Coupled generative rollout analysis using global behavior models
%
%  14)  stepA10_behavior_fit_by_participant(run_id)
%       - Fit behavioral models per participant (door-resampling bootstrap)
%       - Select best model via BIC
%
%  15)  stepA11_behavior_param_robustness(run_id)
%       - Robustness analysis of behavioral parameters:
%         * random door splits (quantitative)
%         * blockwise stability (qualitative: stable / drifting / under-identified)
%
%  16)  stepA12_behavior_rollouts_personalized(run_id)
%       - Personalized coupled rollouts using per-participant behavior params
%       - Guardrail fallback for under-identified or invalid fits
%
%  17)  stepA13_trust_divergence_sanity_check(run_id)
%       - Trust realism diagnostics:
%         * divergence between simple replay and coupled trust trajectories
%         * effect of personalization vs global behavior parameters
%         * full-time-grid divergence metrics and visualizations
%
% -------------------------------------------------------------------------
% Assumptions
%   - The repository root is the working directory (so "src" and "derived"
%     paths resolve correctly).
%   - All required dependencies are available on the MATLAB path via src/.
% -------------------------------------------------------------------------

clear; clc;

% Add project source tree (utilities, models, steps, plotting helpers, etc.).
addpath(genpath("src"));

% -------------------------------------------------------------------------
% A) Data processing
% -------------------------------------------------------------------------
preprocessing
postprocessing

% Optional dataset overview diagnostics (overall, train split, valid split).
analyze_participants_overview();
analyze_participants_overview("Split", "train");
analyze_participants_overview("Split", "valid");

% -------------------------------------------------------------------------
% Reputation diagnostics and figure generation
% -------------------------------------------------------------------------
% Evaluate reputation-related effects and generate diagnostic figures/PDFs.
reputation_bias("derived/participants_train_stepV.mat", "particpant_set_analysis/global/figs");

% -------------------------------------------------------------------------
% Preparation for trust model fitting
% -------------------------------------------------------------------------
cfg = struct();
cfg.dt = 1.0;  % Simulation time step [s]

% Initial parameter vector (theta0) for optimization.
cfg.theta0 = [
    3e-3       % lambda_rep
    1          % alpha_sit
    1          % lambda_sit
    0.5        % phi_fail
    0.45       % phi_succ
    0.10       % a_succ
    1e-4       % lambda_lat
    1e-4       % kappa_lat
];

% Run naming and output locations.
cfg.run_tag        = "thesis_fit_v2";
cfg.results_dir    = "derived/fit_runs";
cfg.checkpoint_dir = "derived/checkpoints";

% -------------------------------------------------------------------------
% B) Trust model fitting
% -------------------------------------------------------------------------
results = run_trust_optimisation_pipeline(cfg); %#ok<NASGU>

% -------------------------------------------------------------------------
% C) Analysis steps (A0-A13)
% -------------------------------------------------------------------------
overwrite_flag = true;

[run_id, resultsMatPath] = stepA0_import_fit_results(cfg, "Overwrite", overwrite_flag);

stepA1_prepare_analysis_run(run_id, "Overwrite", overwrite_flag);
stepA2_simple_mode_fit_report(run_id, resultsMatPath);
stepA3_select_theta_for_coupled(run_id, resultsMatPath);
stepA4_sensitivity_simple_mode(run_id, resultsMatPath);
stepA5_compare_baselines_simple_mode(run_id, resultsMatPath);
stepA6_report_baseline_comparison_simple_mode(run_id);

% Build behavior datasets for both splits (train/valid).
stepA7_build_behavior_dataset(run_id, "Split", "train", "Overwrite", overwrite_flag);
stepA7_build_behavior_dataset(run_id, "Split", "valid", "Overwrite", overwrite_flag);

% Behavioral fitting and coupled rollout analyses.
stepA8_behavior_fit_eval(run_id, "Overwrite", overwrite_flag);
stepA9_behavior_rollouts(run_id, "Overwrite", overwrite_flag);
stepA10_behavior_fit_by_participant(run_id, "Overwrite", overwrite_flag);
stepA11_behavior_param_robustness(run_id, "Overwrite", overwrite_flag);
stepA12_behavior_rollouts_personalized(run_id, "Overwrite", overwrite_flag);
stepA13_trust_divergence_sanity_check(run_id, "Overwrite", overwrite_flag);

% -------------------------------------------------------------------------
% Further analysis steps
% -------------------------------------------------------------------------
trust_residual_diagnostics(run_id);
