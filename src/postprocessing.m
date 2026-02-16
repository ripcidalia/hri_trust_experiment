% ExperimentalDataProcessing/postprocessing.m
%
% POSTPROCESSING  Run the measurement calibration and mapping pipeline (M1–M5).
%
% This script orchestrates the postprocessing measurement steps (M1–M5).
% It operates on participant data produced by the preprocessing pipeline and
% generates:
%   - Global linear mappings that express heterogeneous trust instruments on
%     a unified 40-item questionnaire scale.
%   - Augmented participant datasets containing mapped mid-block questionnaire
%     totals and per-probe 40-scale equivalents.
%   - Residual-variance estimates used to derive relative measurement weights
%     for downstream Weighted Least Squares (WLS) fitting.
%
% The steps are:
%   M1: Fit global OLS mappings to the 40-item scale using questionnaire anchors
%       and probe-anchor pairs (saves mapping coefficients).
%   M2: Apply the 14→40 mapping to mid-block 14-item questionnaire totals and
%       store mapped values per participant.
%   M3: Estimate residual variances for probe→40 fits via LOPO in two contexts:
%       (i) pre/post anchors and (ii) mid-block (mapped) anchors.
%   M4: Populate 40-scale equivalents for all trust probes; for probes tied to
%       questionnaire time points, enforce consistency by assigning the
%       questionnaire-derived 40-scale totals.
%   M5: Consolidate measurement weights from residual variance estimates and
%       save a unified weights struct for downstream WLS objectives.
%
% This script is intended to be run from the project root and assumes:
%   - Source functions are available under src/ (added to the MATLAB path).
%   - Input participant MAT files exist under derived/ as referenced below.
%   - Outputs are written under derived/.
%
% No computational logic is implemented here; this file sequences calls to the
% dedicated step functions and defines the file paths passed between them.

%% Step M1: Global OLS calibration mappings (questionnaire anchors and probes)

% Computes global linear mappings to the 40-item questionnaire scale and
% writes derived/measurement_stepM1_calibration.mat.
stepM1_ols_calibration("derived/participants_train_stepV.mat");

%% Step M2: Apply 14→40 mapping to mid-block 14-item questionnaire totals

% Uses the M1 14→40 mapping to compute:
%   t14_mid1.total_percent_40 and t14_mid2.total_percent_40
% for each participant and writes derived/participants_mapped14_stepM2.mat.
stepM2_apply_14mapping("derived/participants_train_stepV.mat");

%% Step M3: LOPO residual variance estimation for probe→40 fits

% Estimates pooled residual variances for probe→40 mappings using:
%   - pre/post 40-item anchor context, and
%   - mid-block (mapped) anchor context,
% and writes derived/measurement_step3_residual_variances.mat.
stepM3_residuals_variances("derived/participants_mapped14_stepM2.mat");

%% Step M4: Populate 40-scale equivalents for all trust probes

% Adds value_40 to every probe and copies questionnaire timestamps for
% anchor-typed probes; writes derived/participants_probes_mapped_stepM4.mat.
stepM4_apply_probe_mapping("derived/participants_mapped14_stepM2.mat", ...
                           "derived/measurement_stepM1_calibration.mat");

%% Step M5: Compute and save unified measurement weights

% Builds derived/measurement_weights.mat containing relative weights for
% the 40-item questionnaire, the 14-item questionnaire, and probe measures.
stepM5_save_measurement_weights( ...
    "derived/measurement_step3_residual_variances.mat");
