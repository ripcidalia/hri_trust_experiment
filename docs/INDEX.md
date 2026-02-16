# MATLAB Codebase Index

This document provides a structured index of the MATLAB scripts and functions in the trust- and behavior-modelling framework, with concise descriptions for each component and their role in the end-to-end pipeline.

It complements the technical documentation in `docs/01–08` and is intended as a practical navigation reference for the codebase.

---

## Top-Level Orchestration Scripts (repository root)

### `main.m`
Primary entry point for the full pipeline:
- preprocessing,
- measurement mapping,
- trust-model fitting,
- Step A analysis,
- behavioral modelling and coupled rollouts.

---

# `src/` — MATLAB Source Modules

### `preprocessing.m`
Runs preprocessing steps 1–4 and time annotation step T1.

### `postprocessing.m`
Runs measurement mapping and weighting steps (M1–M6).

### `fit_trust_parameters.m`
Global parameter estimation using WLS with a selectable solver backend (`fmincon`, `ga`, `patternsearch`).

### `run_trust_optimisation_pipeline.m`
Orchestrates the multi-stage optimisation sequence (GA → GA → fmincon → patternsearch), with checkpointing and run logging to `derived/fit_runs`.

Below, subsystems are grouped according to the processing and modelling pipeline.

---

# 1) Preprocessing & Data Structuring

### Data ingestion and utilities

### `read_events_table.m`
Reads CSV event logs into MATLAB tables.

### `normalize_headers.m`
Standardises column names across event logs.

### `sort_by_ts.m`
Sorts events chronologically.

### `decode_json_columns.m`
Decodes JSON-encoded fields in the logs.

### `safejsondecode.m`
Failsafe wrapper around `jsondecode`.

### `ensure_vars_exist.m`
Ensures required variables exist in a table.

---

### Participant construction

### `step1_loader.m`
Loads and prepares raw event tables.

### `build_participant_struct.m`
Constructs unified participant structs.

### `extract_door_trials.m`
Extracts door-trial interactions.

### `extract_door_trials_with_order.m`
Variant preserving event ordering.

### `extract_trust_probes.m`
Extracts trust-probe slider measurements.

### `extract_trust_probes_linked.m`
Links probes to surrounding events.

### `extract_trust_probes_with_mapping.m`
Maps raw probes through calibration functions.

### `extract_questionnaires.m`
Extracts 40-item and 14-item questionnaires.

### `extract_reviews.m`
Extracts review-based reputation data.

### `extract_demographics.m`
Extracts demographic metadata.

### `harmonize_struct_fields.m`
Ensures field consistency across participant structs.

### `project_common_fields.m`
Projects participants onto a common schema.

### `compute_block_counts.m`
Computes interaction counts per phase.

### `get_numeric_field.m`
Utility: safely retrieves numeric fields.

### `get_string_field.m`
Utility: safely retrieves string fields.

### `print_step1_diagnostics.m`
Reports diagnostics after Step 1 import.

---

### Validation and filtering

### `step2_build_participants.m`
Builds full participant struct array.

### `validate_participant.m`
Runs per-participant consistency checks.

### `check_timeline.m`
Validates chronological consistency.

### `check_door_trials.m`
Checks completeness of door-trial data.

### `check_trust_probes.m`
Validates trust-probe timelines.

### `check_questionnaires.m`
Ensures questionnaire structure validity.

### `check_demographics.m`
Validates demographic data.

### `probe_alignment_report.m`
Diagnostics for probe–event alignment.

### `step3_validate.m`
Runs validation across all participants.

### `step4_filter_participants.m`
Removes invalid or incomplete participants.

---

### Time enrichment and dataset preparation

### `stepT1_add_times.m`
Adds timestamps and inter-event timing fields.

### `build_time_grid_and_events.m`
Constructs uniform simulation time grid and event mapping.

### `get_cached_time_grid_and_events.m`
Memoized wrapper for time-grid construction.

### `split_participants_by_index.m`
Splits participants into training/validation sets by index.

### `stepV_split_train_valid.m`
Driver for generating TRAIN/VALID splits and saving separate MAT files.

---

# 2) Measurement Mapping & Weighting (M1–M6)

### `stepM1_ols_calibration.m`
Global calibration of 14-item → 40-item questionnaire mapping.

### `stepM2_apply_14mapping.m`
Applies the 14→40 mapping to all participants.

### `stepM3_residuals_variances.m`
Estimates residual variances used for probe weighting.

### `stepM4_apply_probe_mapping.m`
Applies the probe→40 mapping.

### `stepM5_save_measurement_weights.m`
Constructs the measurement-weights struct for WLS.

### `reputation_bias.m`
Computes negativity-bias metrics and diagnostics from review data.

---

# 3) Trust Model Core (Dynamics)

### State utilities

### `trust_clip.m`
Clips trust values to the interval [0,1].

### `trust_init_state.m`
Creates the initial trust state for a participant.

### `trust_debug_log.m`
Debug logging for numerical anomalies.

---

### Trust components

### `trust_compute_dispositional.m`
Maps pre-interaction questionnaire score to dispositional trust.

### `trust_compute_situational.m`
Computes situational trust from risk and self-confidence.

### `trust_initial_reputation.m`
Extracts initial reputation from review data.

### `trust_update_reputation.m`
Applies exponential decay to reputation component.

### `trust_update_personal_experience.m`
Updates latent trust using success/failure streaks.

### `trust_prepare_latent_sequence.m`
Initialises latent drift episodes.

### `trust_update_latent_sequence.m`
Propagates latent trust between events.

---

### Simulation

### `trust_step.m`
Applies one trust update step.

### `trust_simulate_or_predict_one_participant.m`
Runs full continuous-time trust simulation and prediction.

---

# 4) Behavioral Model

### `behavioral_model.m`
Probabilistic decision model predicting follow vs override based on current trust and self-confidence.

---

# 5) Cost Functions & Parameter Transformation

### `trust_cost_one_participant.m`
Weighted least-squares cost for a single participant.

### `trust_cost_all.m`
Aggregates WLS cost across all participants.

### `trust_theta_to_params.m`
Maps optimisation vector θ to a structured parameter representation.

### `measure_trust_cost_runtime.m`
Benchmarks cost-function and simulation runtime.

### `debug_cost_gradient.m`
Diagnostic script for probing parameter sensitivity.

---

# 6) Optimisation Methods (Solver Backends)

### `fit_trust_parameters_fmincon.m`
Local constrained optimisation using `fmincon`.

### `fit_trust_parameters_ga.m`
Global optimisation using a Genetic Algorithm.

### `fit_trust_parameters_patternsearch.m`
Derivative-free optimisation using `patternsearch`.

---

# 7) Diagnostics & Evaluation

### Core diagnostics

### `sanity_check_trust_cost.m`
Computes per-participant costs and produces diagnostic plots.

### `run_trust_diagnostics.m`
Simulates and visualises trust components for one participant.

### `trust_residual_diagnostics.m`
Global residual analysis with summary tables and figures.

### `test_trust_modules.m`
Smoke tests for trust-model components.

---

### Dataset and participant diagnostics

### `analyze_participants_overview.m`
Generates descriptive statistics and reporting plots for participant datasets.

### `diag_behavior_thresholds.m`
Diagnostics for decision-threshold behaviour at door events.

### `run_ga_sweep_pipeline.m`
Executes multiple GA optimisation runs with checkpointing for robustness analysis.

---

# 8) Step A Analysis Pipeline

These scripts implement the run-local analysis workflow used after trust fitting.

### A-step orchestration

### `stepA0_import_fit_results.m`
Imports optimisation outputs and prepares analysis run inputs.

### `stepA1_prepare_analysis_run.m`
Archives inputs and creates a run-local analysis directory.

### `stepA2_simple_mode_fit_report.m`
Evaluates SIMPLE-mode fits on TRAIN and VALID sets.

### `stepA3_select_theta_for_coupled.m`
Selects the best fitted parameter vector for coupled analysis.

### `stepA4_sensitivity_simple_mode.m`
Performs sensitivity and stability analysis around fitted parameters.

### `stepA5_compare_baselines_simple_mode.m`
Compares SIMPLE-mode predictions with baseline models.

### `stepA6_report_baseline_comparison_simple_mode.m`
Generates reporting artifacts for baseline comparisons.

---

### Behavioral modelling and evaluation

### `stepA7_build_behavior_dataset.m`
Builds door-level behavioural datasets.

### `stepA8_behavior_fit_eval.m`
Fits behavioral models and evaluates performance.

### `stepA9_behavior_rollouts.m`
Coupled generative rollouts using global behavioral parameters.

### `stepA10_behavior_fit_by_participant.m`
Fits behavioral models per participant.

### `stepA11_behavior_param_robustness.m`
Assesses robustness and identifiability of behavioral parameters.

### `stepA12_behavior_rollouts_personalized.m`
Personalized coupled rollouts using participant-specific parameters.

### `stepA13_trust_divergence_sanity_check.m`
Quantifies divergence between SIMPLE and COUPLED trust trajectories.

---

# 9) Analysis Utilities (`src/analysis/utils`)

Supporting functions used throughout the Step A pipeline.

### Data and file utilities
- `ensure_dir.m`
- `must_exist_file.m`
- `copy_into_dir.m`
- `safe_copy_into_dir.m`
- `resolve_runlocal_or_source.m`
- `file_info_struct.m`
- `save_json.m`

### Participant and measurement helpers
- `load_participants_struct.m`
- `read_participant_ids.m`
- `compare_id_sets.m`
- `extract_measurements.m`
- `get_dispositional_only.m`

### Parameter and optimisation utilities
- `discover_theta_hats.m`
- `find_theta_in_struct.m`
- `optimizerDisplayName.m`
- `paramDisplayName.m`

### Metrics and summaries
- `compute_weighted_metrics.m`
- `summarize_residuals.m`
- `weight_for_kind.m`

### Plotting and reporting
- `thesisStyle.m`
- `thesisFinalizeFigure.m`
- `thesisExport.m`
- `save_residual_diagnostic_figure.m`
- `save_participant_metric_figure.m`
- `behavioralDisplayName.m`
- `measurementDisplayName.m`
- `methodDisplayName.m`

### Simulation helpers
- `trust_simulate_baseline_one_participant.m`

---

## Repository Navigation Summary

```

/ (repo root)
│
├── main.m
├── README.md
│
├── src/
│   ├── preprocessing.m
│   ├── postprocessing.m
│   ├── fit_trust_parameters.m
│   ├── run_trust_optimisation_pipeline.m
│   ├── preprocessing/
│   ├── measurement_mapping/
│   ├── trust_model/
│   ├── behavioral_model/
│   ├── cost_functions/
│   ├── optimization_methods/
│   ├── diagnostics/
│   └── analysis/
│
└── docs/
├── INDEX.md
├── 01_preprocessing.md
├── 02_measurement_mapping_and_weights.md
├── 03_trust_model.md
├── 04_cost_optimisation.md
├── 05_optimization_methods.md
├── 06_diagnostics.md
├── 07_analysis_pipeline.md
└── 08_behavioral_model.md

```

This document serves as the canonical index for navigating the MATLAB codebase and understanding the role of each script within the full modelling and analysis pipeline.