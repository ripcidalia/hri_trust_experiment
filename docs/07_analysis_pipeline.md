# 07 — Analysis Pipeline (Step A0–A13)
*(MATLAB Trust Modelling Framework – Full Documentation)*

---

## Overview

This document describes the **Step A analysis pipeline** implemented under `src/analysis/`. The A-steps provide a **run-local, manifest-driven evaluation and reporting workflow** that sits on top of the optimisation and simulation code.

At a high level, the pipeline:

- Imports optimisation outputs into an analysis-friendly format (A0),
- Prepares a **run-scoped** analysis directory with archived inputs and provenance manifests (A1),
- Evaluates fitted trust-model parameters on **TRAIN** and **VALID** splits in **SIMPLE** mode (A2),
- Selects a single best SIMPLE-mode parameter vector for downstream coupled analyses (A3),
- Runs sensitivity and stability analyses without refitting (A4),
- Compares the selected SIMPLE trust model against baseline predictors (A5),
- Produces thesis-ready figures and tables for baseline comparison (A6),
- Builds a door-level behavior dataset from simulation-aligned decision variables (A7),
- Fits and evaluates behavior models on TRAIN→VALID (A8),
- Performs coupled (closed-loop) rollouts on VALID using global behavior fits (A9),
- Fits behavior models per participant on VALID (mechanistic within-subject analysis) (A10),
- Assesses robustness/identifiability of per-participant behavior parameters (A11),
- Performs personalized coupled rollouts using per-participant behavior parameters (A12),
- Checks realism/sanity of continuous-time trust divergence across modes/behavior parameterizations (A13).

A core design principle is **separation of concerns**:
- Optimisation produces fit runs under `derived/fit_runs/`.
- Analysis reads those results and writes **run-local** artifacts under `derived/analysis_runs/<run_id>/`.
- Shared canonical `derived/` paths are not overwritten unless explicitly enabled.

---

## Directory conventions

### Fit runs (upstream)
Fit outputs are expected under:
- `derived/fit_runs/<fit_run_dir>/`
  - `manifest.mat`
  - `final/run_summary.mat`

These are produced by the trust optimisation pipeline (e.g., `run_trust_optimisation_pipeline`).

### Analysis runs (this module)
Analysis outputs are written under:
- `derived/analysis_runs/<run_id>/`

Each step writes into its own subdirectory:
- `stepA1_prepare_analysis/`
- `stepA2_simple_mode/`
- `stepA3_model_selection/`
- `stepA4_sensitivity/`
- `stepA5_baseline_comparison/`
- `stepA6_report/`
- `stepA7_behavior_dataset/`
- `stepA8_behavior_fit_eval/`
- `stepA9_behavior_rollouts/`
- `stepA10_behavior_fit_by_participant/`
- `stepA11_behavior_param_robustness/`
- `stepA12_behavior_rollouts_personalized/`
- `stepA13_trust_divergence_sanity/`

---

## The A-step workflow at a glance

A typical end-to-end analysis sequence is:

```matlab
% A0: import fit results into A-step-compatible format
[run_id, resultsMatPath] = stepA0_import_fit_results(cfg);

% A1: prepare run-local analysis inputs (train/valid + calib + weights)
runInfo = stepA1_prepare_analysis_run(run_id);

% A2: evaluate SIMPLE-mode fits on TRAIN and VALID
stepA2_simple_mode_fit_report(run_id, resultsMatPath);

% A3: select theta_star (best SIMPLE-mode candidate by VALID metrics)
stepA3_select_theta_for_coupled(run_id, resultsMatPath);

% A4: SIMPLE-mode sensitivity analysis around theta_star and across candidates
stepA4_sensitivity_simple_mode(run_id, resultsMatPath);

% A5: compare theta_star against baselines (TRAIN + VALID)
stepA5_compare_baselines_simple_mode(run_id, resultsMatPath);

% A6: generate thesis-ready report bundle from A5 outputs
stepA6_report_baseline_comparison_simple_mode(run_id);

% A7–A9: global behavior modeling and coupled rollouts (TRAIN→VALID)
stepA7_build_behavior_dataset(run_id);
stepA8_behavior_fit_eval(run_id);
stepA9_behavior_rollouts(run_id);

% A10–A13: per-participant behavior modeling + personalized rollouts + sanity checks
stepA10_behavior_fit_by_participant(run_id);
stepA11_behavior_param_robustness(run_id);
stepA12_behavior_rollouts_personalized(run_id);
stepA13_trust_divergence_sanity_check(run_id);
````

Not all steps are mandatory for every experiment. A0–A6 cover trust-model evaluation and baseline comparisons; A7–A13 extend the pipeline to behavioral modeling and coupled generative analyses.

---

## Step A0 — Import fit results (`stepA0_import_fit_results.m`)

**Purpose:** Provide a minimal interface layer between the **fit pipeline** and the **A-step analysis** pipeline.

This step:

1. Infers a fit-run directory from `cfg.run_tag` and `cfg.dt` (or uses an override),
2. Loads final optimisation summary + manifest produced by the fit pipeline,
3. Repackages fitted parameter vectors into an A2-compatible MAT file exposing `theta_hat_*` fields and `cfg.dt`,
4. Writes run-local outputs and a manifest for provenance/auditability.

No refitting or recomputation is performed.

**Inputs:**

* `cfg` (struct), required:

  * `cfg.dt` (numeric scalar)
  * `cfg.run_tag` (string/char)

**Name-value options (all optional):**

* `"FitBaseDir"` default `"derived/fit_runs"`
* `"FitRunDir"` explicit fit directory override (default `""`)
* `"OutDir"` default `"derived/analysis_runs"`
* `"ResultsMatName"` default `"fit_results_stepA0.mat"`
* `"Overwrite"` default `false`

**Outputs:**

* `run_id` (string): derived from `cfg.run_tag` and `cfg.dt` (intended for Step A1+)
* `resultsMatPath` (string): path to the A2-compatible results MAT file
* `runInfo` (struct): manifest summarizing inputs/outputs (also saved to `<stepDir>/manifest.mat`)

**Assumptions:**

* Fit pipeline produced:

  * `<FitRunDir>/manifest.mat`
  * `<FitRunDir>/final/run_summary.mat`

---

## Step A1 — Prepare run-local analysis inputs (`stepA1_prepare_analysis_run.m`)

**Purpose:** Create a run-scoped analysis directory and deterministic, archived inputs for downstream A-steps.

This step:

1. Validates required inputs (TRAIN/VALID splits, calibration, weights),
2. Applies TRAIN-derived calibrations to VALID (14-item and probe mappings),
3. Archives key artifacts into a run-specific folder,
4. Writes a manifest (MAT + JSON) capturing provenance + metadata.

**Inputs:**

* `run_id` (string/char): analysis run identifier

**Key name-value options (optional):**

* `"TrainSplitPath"` default `"derived/participants_train_stepV.mat"`
* `"ValidSplitPath"` default `"derived/participants_valid_stepV.mat"`
* `"TrainParticipantsProbesPath"` default `"derived/participants_probes_mapped_stepM4.mat"` (archived if present)
* `"CalibPath"` default `"derived/measurement_stepM1_calibration.mat"` (expects `calib`)
* `"WeightsPath"` default `"derived/measurement_weights.mat"` (expects `weights`)
* `"OutDir"` default `"derived/analysis_runs"`
* `"Overwrite"` default `false`
* `"WriteCanonicalLatest"` default `false` (optionally overwrites convenience copies in `derived/`)

**Outputs:**

* `runInfo` manifest saved to:

  * `<stepDir>/manifest.mat`
  * `<stepDir>/manifest.json`

**Dependencies / mapping steps used:**

* `stepM2_apply_14mapping` (applied to VALID)
* `stepM4_apply_probe_mapping` (applied to VALID)

---

## Step A2 — SIMPLE-mode fit evaluation (`stepA2_simple_mode_fit_report.m`)

**Purpose:** Evaluate one or more fitted parameter vectors for the **SIMPLE trust model** on TRAIN and VALID, using run-local artifacts prepared by A1.

This step:

* Loads `theta_hat_*` vectors and `cfg.dt`,
* Loads TRAIN/VALID participant sets and weights via the Step A1 manifest,
* Simulates all participants in SIMPLE mode,
* Computes weighted residual tables and summary metrics,
* Writes per-optimizer artifacts and diagnostic figures under:
  `derived/analysis_runs/<run_id>/stepA2_simple_mode/`.

**Inputs:**

* `run_id`
* `resultsMatPath`: MAT containing:

  * one or more `theta_hat_*` fields
  * `cfg.dt`

**Outputs (written files):**

* `meta.mat`
* `<optimizer>_<split>.mat`
* `<optimizer>_<split>_residuals.csv`
* `<optimizer>_<split>_summary.csv`
* `<optimizer>_<split>_participants.csv`
* `optimizer_comparison.mat`
* `optimizer_comparison.csv`
* Diagnostic figures (PDF + FIG) exported via `thesisExport`

**Key dependencies:**

* Simulation: `trust_simulate_or_predict_one_participant`
* Utilities:

  * `discover_theta_hats`, `weight_for_kind`, `compute_weighted_metrics`,
    `summarize_residuals`, `resolve_runlocal_or_source`, `load_participants_struct`
* Plot helpers:

  * `save_residual_diagnostic_figure`, `save_participant_metric_figure`,
    `thesisStyle`, `thesisFinalizeFigure`, `thesisExport`

---

## Step A3 — Select theta for coupled-mode (`stepA3_select_theta_for_coupled.m`)

**Purpose:** Choose a single parameter vector `theta_star` from SIMPLE-mode candidates evaluated in A2 for downstream coupled-mode analyses.

If `resultsMatPath` is omitted/empty, it is resolved from:

* `derived/analysis_runs/<run_id>/stepA2_simple_mode/meta.mat` (`meta.results_file`)

**Selection rule (lexicographic):**

1. Minimize `valid_wRMSE`
2. Minimize `valid_wMAE`
3. Minimize `abs(valid_wBias)` if available, else `abs(valid_bias)`
4. Minimize `train_wRMSE`

**Optional name-value:**

* `"TieTol"` default `0.0` (round ranking keys before sorting to reduce numerical flip-flops)

**Outputs (written files):**

* `derived/analysis_runs/<run_id>/stepA3_model_selection/`

  * `selection.mat` / `selection.json`
  * `theta_star.mat`
  * `optimizer_comparison_with_selection.csv`

---

## Step A4 — SIMPLE-mode sensitivity analysis (`stepA4_sensitivity_simple_mode.m`)

**Purpose:** Run simulation-only sensitivity and stability analyses in SIMPLE mode around `theta_star` and across all `theta_hat_*` candidates, using:

* A4.1 Optimizer-to-optimizer stability (TRAIN + VALID; overall + per-kind metrics)
* A4.2 One-at-a-time (OAT) local sensitivity around `theta_star` (± directions)
* A4.3 Morris-like local screening in a bounded neighborhood (no refits)
* A4.4 Pairwise 2D grid map for top-two parameters (by Morris mu_star)

**Inputs:**

* `run_id`
* `resultsMatPath` (optional; inferred from A3 when omitted)
* `selectedThetaMatPath` (optional; inferred from A3, else best VALID wRMSE)

**Key name-value options:**

* `"EpsList"` default `[0.01 0.05 0.10]` (relative perturbations)
* `"MorrisCfg"` struct overrides (e.g., `delta_rel`, `box_rel`, `r_paths`, `seed`, `zero_scale`)
* `"GridFactors"` default `linspace(0.9, 1.1, 8)`

**Outputs:** Tables (CSV/MAT) and figures (PDF+FIG) under `stepA4_sensitivity/`.

---

## Step A5 — Baseline comparison (`stepA5_compare_baselines_simple_mode.m`)

**Purpose:** Compare the selected SIMPLE trust model (`theta_star`) against baseline predictors on TRAIN and VALID, using the **same simulator measurement alignment** for direct comparability.

Baselines:

* **B1.1** `const_dispositional`
* **B1.2** `const_global_train_mean`
* **B1.3** `const_oracle_participant_mean`
* **B2** `bump_asymmetric` (fit `delta_plus`, `delta_minus` on TRAIN)
* **B3** `bump_symmetric` (fit `Delta` on TRAIN)
* **B4** OPTIMo-lite:

  * `optimo_lite` (filtered; probes enabled; scored pre-update)
  * `optimo_lite_outcome_only` (ablation; probes disabled; scored pre-update)

**Methodological consistency:**

* Same simulator-provided time grid and measurement sampling.
* Same weighted metrics computed from aligned residuals.

**Inputs:**

* `run_id`
* `resultsMatPath` (optional; inferred from A3 if empty)
* `selectedThetaMatPath` (optional; inferred from A3 if empty)

**Optional:**

* `"Clip01"` default `true`

**Outputs (written files):**

* `derived/analysis_runs/<run_id>/stepA5_baseline_comparison/`

  * residual tables, summaries, participant tables
  * fitted baseline parameters
  * improvements tables
  * `A5_plot_data.mat` (consumed by A6)

**Dependencies:**

* Simulation:

  * `trust_simulate_or_predict_one_participant`
  * `trust_simulate_baseline_one_participant`

---

## Step A6 — Report bundle (`stepA6_report_baseline_comparison_simple_mode.m`)

**Purpose:** Create a thesis-ready reporting bundle from the consolidated MAT output produced by A5 (`A5_plot_data.mat`). This step avoids recomputing metrics from raw CSVs to ensure consistent reporting.

It also generates per-participant validation trajectory plots by re-simulating:

* the fitted SIMPLE model, and
* each baseline method.

**Inputs:**

* `run_id`
* `inA5Dir` (optional; defaults to `.../stepA5_baseline_comparison`)

**Outputs:**

* `derived/analysis_runs/<run_id>/stepA6_report/`:

  * CSV tables, text summaries, figures
  * Per-participant trajectory plots under:
    `participant_trajectories/<participant_id>/`

---

## Step A7 — Build behavior dataset (`stepA7_build_behavior_dataset.m`)

**Purpose:** Build a run-local, door-trial-level dataset for follow/override prediction, supporting behavior modeling (A8) and coupled evaluation (A9).

**Causality design choice:**

* Decision trust must be evaluated **before** door-event update:

  * `tau_decision = tau_hist(k_grid-1)`

**Outputs:**

* `derived/analysis_runs/<run_id>/stepA7_behavior_dataset/`

  * `behavior_dataset_<split>.csv` / `.mat`
  * summary tables and meta manifests (MAT + JSON)

**Notes / included fields:**

* Components (`tau_lat`, `tau_rep`, `tau_sit`) are **not** stored.
* `sc` stored as participant-constant; derived fields include:

  * `sc_centered = sc - 0.5`
  * `margin_treshold = tau_decision - sc` (spelling preserved)

---

## Step A8 — Fit/evaluate behavior models (`stepA8_behavior_fit_eval.m`)

**Purpose:** Fit simple behavioral models on TRAIN and evaluate on VALID using the door-level datasets from A7. Metrics are reported with **override as the positive class**.

Behavior models (follow probability):

* **Model 0:** `p_follow = clamp(tau_decision, 0, 1)` (no fitting)
* **Model 1:** `p_follow = sigmoid( k * (tau_decision - self_confidence) )`
* **Model 2:** offset + lapse:

  * `z = k*tau_decision + beta*(sc-0.5)`
  * `p_follow = (1-eps)*sigmoid(z) + eps*0.5`

Override probability:

* `p_override = 1 - p_follow` (override is positive class for override-focused metrics)

**Outputs:**

* `fit_params.mat` (parameters + profile grids + bootstrap samples)
* VALID metric tables (CSV/MAT), deltas vs baselines, calibration bins
* `A8_plot_data.mat`
* figures exported via `thesisExport`

---

## Step A9 — Coupled rollouts (global behavior) (`stepA9_behavior_rollouts.m`)

**Purpose:** Evaluate the coupled trust+behavior system on VALID by running **closed-loop** simulations in `"coupled"` mode. Decisions are sampled from a behavior model and fed back into trust dynamics using the simulator’s counterfactual outcome inversion rule:

* If `sampled_follow ~= recorded_follow` then `outcome := 1 - recorded_outcome`
* Else `outcome := recorded_outcome`

**Targets:** Emergent interaction signatures (override timing, switching, streak/gap structure), not pointwise prediction.

**Outputs:**

* `A9_rollout_stats.mat` and `.csv`
* pooled diagnostic figures, meta manifests

**Signature stats (override-centric):**

* follow/override rates (overall + optional per block)
* switch probabilities
* inter-override gaps (mean/median/p90)
* override streak lengths (mean/p90)
* follow streak lengths (reported but interpret cautiously)

---

## Step A10 — Fit behavior per participant (`stepA10_behavior_fit_by_participant.m`)

**Purpose:** Mechanistic within-participant analysis on VALID: fit models 0/1/2 per participant and report likelihood-based fit metrics and uncertainty via door bootstrap.

Metrics per participant and model:

* NLL, Brier score
* AIC, BIC

Model selection:

* Best by BIC with a parsimony rule (`DeltaBIC_Parsimony`, default 2).

Outputs include:

* `A10_params_by_participant.csv` and `.mat`
* meta manifests and figures

---

## Step A11 — Robustness of per-participant behavior params (`stepA11_behavior_param_robustness.m`)

**Purpose:** Assess identifiability/robustness of A10 per-participant parameters with:

* **A11.1 Random split stability:** repeated random door splits into fit/held-out, recording parameter samples (when applicable) and held-out metrics (NLL, Brier, corr).
* **A11.2 Blockwise stability:** fit the A10-selected best model separately on blocks 1/2/3; label drift as stable/drifting/under_identified and save per-participant plots.

Outputs include:

* split stability tables
* blockwise parameter tables
* meta manifests and figures

---

## Step A12 — Personalized coupled rollouts (`stepA12_behavior_rollouts_personalized.m`)

**Purpose:** Evaluate generative realism of coupled simulations on VALID using per-participant behavior parameters from A10 (optionally guarded by A11).

Guardrail fallback (default enabled):

* If under-identified or invalid params, degrade model:

  * Model 2 → Model 1 → Model 0
* Fallback decisions are recorded.

Outputs include:

* per-participant and pooled observed-vs-simulated signature comparisons
* figures (pooled and by participant)
* MAT/CSV bundles plus meta manifests

---

## Step A13 — Trust divergence sanity check (`stepA13_trust_divergence_sanity_check.m`)

**Purpose:** Continuous-time trust realism check on the full simulation grid. Quantifies divergence between:

A) SIMPLE-mode trust replay vs COUPLED-mode trust

* COUPLED with GLOBAL behavior parameters (A8 fit)
* COUPLED with PERSONALIZED behavior parameters (A10 + A11 guards)

B) COUPLED with PERSONALIZED vs GLOBAL behavior parameters

* Matched RNG seeds so differences reflect parameter changes rather than sampling noise

Error curve:

* Option A: `e(t_k) = tau_cpl(t_k) - tau_simple(t_k)`
* Option B: `e(t_k) = tau_cpl_personal(t_k) - tau_cpl_global(t_k)`

Per-rollout divergence metrics:

* `IAD = sum |e(t_k)| * dt`
* `MAE_t`, `RMSE_t`
* `MaxAbs = max |e(t_k)|`

Outputs:

* participant-level summaries (CSV/MAT)
* pooled normalized-time summaries
* optional rollout-level MAT (can be large)
* figures (pooled and by participant), meta manifests

---

## Utility layer (`src/analysis/utils/`)

The analysis pipeline relies on a set of utilities for:

* run-local provenance and manifests,
* locating and loading participants and weights,
* discovering candidate parameter vectors,
* computing weighted residual metrics and summaries,
* exporting thesis-style figures.

Key utilities include:

* File/provenance: `must_exist_file`, `ensure_dir`, `file_info_struct`, `save_json`,
  `copy_into_dir`, `safe_copy_into_dir`, `resolve_runlocal_or_source`
* Participant handling: `load_participants_struct`, `read_participant_ids`, `compare_id_sets`
* Theta handling: `discover_theta_hats`, `find_theta_in_struct`
* Metrics: `weight_for_kind`, `extract_measurements`, `compute_weighted_metrics`, `summarize_residuals`
* Display names: `methodDisplayName`, `optimizerDisplayName`, `measurementDisplayName`,
  `behavioralDisplayName`, `paramDisplayName`
* Plot/export: `thesisStyle`, `thesisFinalizeFigure`, `thesisExport`,
  `save_residual_diagnostic_figure`, `save_participant_metric_figure`
* Baseline simulation: `trust_simulate_baseline_one_participant`

---

## Relevant files

### Step functions

* `src/analysis/stepA0_import_fit_results.m`
* `src/analysis/stepA1_prepare_analysis_run.m`
* `src/analysis/stepA2_simple_mode_fit_report.m`
* `src/analysis/stepA3_select_theta_for_coupled.m`
* `src/analysis/stepA4_sensitivity_simple_mode.m`
* `src/analysis/stepA5_compare_baselines_simple_mode.m`
* `src/analysis/stepA6_report_baseline_comparison_simple_mode.m`
* `src/analysis/stepA7_build_behavior_dataset.m`
* `src/analysis/stepA8_behavior_fit_eval.m`
* `src/analysis/stepA9_behavior_rollouts.m`
* `src/analysis/stepA10_behavior_fit_by_participant.m`
* `src/analysis/stepA11_behavior_param_robustness.m`
* `src/analysis/stepA12_behavior_rollouts_personalized.m`
* `src/analysis/stepA13_trust_divergence_sanity_check.m`

### Utilities

* `src/analysis/utils/behavioralDisplayName.m`
* `src/analysis/utils/compare_id_sets.m`
* `src/analysis/utils/compute_weighted_metrics.m`
* `src/analysis/utils/copy_into_dir.m`
* `src/analysis/utils/discover_theta_hats.m`
* `src/analysis/utils/ensure_dir.m`
* `src/analysis/utils/extract_measurements.m`
* `src/analysis/utils/file_info_struct.m`
* `src/analysis/utils/find_theta_in_struct.m`
* `src/analysis/utils/get_dispositional_only.m`
* `src/analysis/utils/load_participants_struct.m`
* `src/analysis/utils/measurementDisplayName.m`
* `src/analysis/utils/methodDisplayName.m`
* `src/analysis/utils/must_exist_file.m`
* `src/analysis/utils/optimizerDisplayName.m`
* `src/analysis/utils/paramDisplayName.m`
* `src/analysis/utils/read_participant_ids.m`
* `src/analysis/utils/resolve_runlocal_or_source.m`
* `src/analysis/utils/safe_copy_into_dir.m`
* `src/analysis/utils/save_json.m`
* `src/analysis/utils/save_participant_metric_figure.m`
* `src/analysis/utils/save_residual_diagnostic_figure.m`
* `src/analysis/utils/summarize_residuals.m`
* `src/analysis/utils/thesisExport.m`
* `src/analysis/utils/thesisFinalizeFigure.m`
* `src/analysis/utils/thesisStyle.m`
* `src/analysis/utils/trust_simulate_baseline_one_participant.m`
* `src/analysis/utils/weight_for_kind.m`
