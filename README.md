````md
# Human–Robot Trust + Behavior Modelling Framework (MATLAB)

This repository contains an end-to-end MATLAB framework for modelling **human trust in an autonomous Search & Rescue (SAR) robot** and its coupling to **human reliance behavior** (follow vs override). It was developed for a TU Delft MSc thesis (Aerospace Engineering, Control & Simulation).

The framework covers the full workflow:

- **Raw data → cleaned, time-aligned participants** (preprocessing)
- **Trust measurement calibration** across questionnaires and probes (mapping + weights)
- **Continuous-time trust dynamics simulation** (interpretable component model)
- **Global parameter fitting** via Weighted Least Squares (WLS) with multiple solvers
- **Diagnostics** (residuals, trajectory inspection, runtime checks)
- **Run-local analysis pipeline** (train/valid evaluation, baselines, sensitivity, behavior fitting, coupled rollouts)

---

## Quickstart

From the repository root in MATLAB:

```matlab
% 1) Run the full pipeline (recommended)
main
````

If you prefer to run parts manually:

```matlab
% A) Data processing (project-wide artifacts in derived/)
preprocessing
postprocessing

% B) Fit trust parameters (writes to derived/fit_runs/<run_id>/)
cfg.dt      = 1.0;
cfg.run_tag = "example_run";
results = run_trust_optimisation_pipeline(cfg);

% C) Create an analysis run (writes to derived/analysis_runs/<analysis_run_id>/)
[run_id, resultsMatPath] = stepA0_import_fit_results(cfg);
stepA1_prepare_analysis_run(run_id);
stepA2_simple_mode_fit_report(run_id, resultsMatPath);
stepA3_select_theta_for_coupled(run_id, resultsMatPath);
```

---

## Pipeline overview

`main.m` orchestrates the full workflow:

### A) Data processing (project-wide; produces cleaned participant structs)

1. `preprocessing.m`

   * Loads raw experimental logs/events
   * Cleans and standardizes event streams and timestamps
   * Produces time-aligned participant structs for modelling

2. `postprocessing.m`

   * Maps questionnaire/probe responses into model-ready measurements
   * Applies calibration and computes measurement weights

### B) Trust dynamics model fitting (run-local artifacts in `derived/fit_runs/`)

3. `run_trust_optimisation_pipeline(cfg)`

   * Fits the global trust-dynamics parameter vector **θ** in **"simple"** mode
   * Creates a run folder with manifests, checkpoints, logs, and final outputs

### C) Step A analysis pipeline (run-local artifacts in `derived/analysis_runs/<run_id>/`)

4. `stepA0_import_fit_results(cfg)`
5. `stepA1_prepare_analysis_run(run_id)`
6. `stepA2_simple_mode_fit_report(run_id, resultsMatPath)`
7. `stepA3_select_theta_for_coupled(run_id, resultsMatPath)`
8. `stepA4_sensitivity_simple_mode(run_id, resultsMatPath)`
9. `stepA5_compare_baselines_simple_mode(run_id, resultsMatPath)`
10. `stepA6_report_baseline_comparison_simple_mode(run_id)`
11. `stepA7_build_behavior_dataset(run_id)`
12. `stepA8_behavior_fit_eval(run_id)`
13. `stepA9_behavior_rollouts(run_id)`
14. `stepA10_behavior_fit_by_participant(run_id)`
15. `stepA11_behavior_param_robustness(run_id)`
16. `stepA12_behavior_rollouts_personalized(run_id)`
17. `stepA13_trust_divergence_sanity_check(run_id)`

---

## Repository structure

```
root/
│
├─ main.m                                – End-to-end entry point (trust + behavior pipeline)
├─ README.md
│
├─ src/
│   ├─ preprocessing.m                   – Preprocessing orchestrator (raw → participants)
│   ├─ postprocessing.m                  – Measurement mapping + weights orchestrator
│   ├─ fit_trust_parameters.m            – Solver dispatcher (WLS fitting)
│   ├─ run_trust_optimisation_pipeline.m – 4-stage fitting pipeline with checkpointing + archiving
│   │
│   ├─ preprocessing/                    – Raw extraction + validation modules
│   ├─ measurement_mapping/              – Calibration + mapping steps (M1–M6)
│   ├─ trust_model/                      – Trust dynamics + simulator (SIMPLE/COUPLED)
│   ├─ behavioral_model/                 – Probabilistic follow/override model
│   ├─ cost_functions/                   – WLS costs + parameter mapping utilities
│   ├─ optimization_methods/             – fmincon / GA / patternsearch implementations
│   ├─ diagnostics/                      – Residuals, plots, sanity checks, reporting helpers
│   └─ analysis/                         – Step A analysis pipeline (A0–A13) + run-local utilities
│
├─ derived/                              – Generated artifacts (inputs/outputs, cached runs)
└─ docs/                                 – Full technical documentation (01–08)
```

---

## Key artifacts and where they live

### Data processing (project-wide)

Typical inputs/outputs under `derived/`:

* `derived/participants_time_stepT1.mat`
  Time-aligned participants produced by preprocessing.

Optional split files (when using train/valid workflow):

* `derived/participants_train_stepV.mat`
* `derived/participants_valid_stepV.mat`

### Measurement calibration and weights

Produced by postprocessing / mapping steps:

* `derived/measurement_stepM1_calibration.mat`
  Global OLS mappings (14-equivalent→40, probe→40).

* `derived/participants_mapped14_stepM2.mat`
  Participants with mapped 14-item mid-block questionnaire totals.

* `derived/measurement_step3_residual_variances.mat`
  LOPO residual variance estimates for probes (anchor + mid-block contexts).

* `derived/participants_probes_mapped_stepM4.mat`
  Participants with per-probe `value_40` (plus anchor consistency at questionnaire probes).

* `derived/measurement_weights.mat`
  Unified weights struct (`w40`, `w14`, `w_probe`) for WLS objectives.

### Trust fitting runs

Created by `run_trust_optimisation_pipeline(cfg)`:

* `derived/fit_runs/<run_id>/manifest.mat`
* `derived/fit_runs/<run_id>/final/run_summary.mat`
* `derived/fit_runs/<run_id>/logs/`
* `derived/fit_runs/<run_id>/checkpoints/` (run-local, crash-safe)

### Step A analysis runs

Created by Step A0/A1 and used by downstream A-steps:

* `derived/analysis_runs/<analysis_run_id>/stepA1_prepare_analysis/manifest.(mat|json)`
* Run-local copies of TRAIN/VALID inputs, calibration, weights, and mapped VALID sets
* Reporting tables + figures per step under:
  `derived/analysis_runs/<analysis_run_id>/stepA2_...`, `stepA5_...`, etc.

---

## What to read next (docs index)

The technical docs live in `docs/`:

* `docs/01_preprocessing.md` — Raw logs → validated, time-aligned participants
* `docs/02_measurement_mapping_and_weights.md` — Instrument mapping + weights (M1–M6)
* `docs/03_trust_model.md` — Trust dynamics core (state, step update, simulation)
* `docs/04_cost_optimisation.md` — WLS cost functions + parameter transformation
* `docs/05_optimization_methods.md` — Solver backends and presets
* `docs/06_diagnostics.md` — Residual analysis + plotting utilities
* `docs/07_analysis_pipeline.md` — Step A run-local analysis pipeline (A0–A13)
* `docs/08_behavioral_model.md` — Probabilistic follow/override model

---

## Reproducibility notes

* All major pipeline stages write **manifest** files (MAT and, where applicable, JSON) capturing provenance.
* Fit runs are run-local under `derived/fit_runs/<run_id>/`.
* Analysis outputs are run-local under `derived/analysis_runs/<analysis_run_id>/`.
* The simulator uses a cached time grid/event alignment to reduce overhead across repeated evaluations for the same participant and `dt`.

---

## MATLAB requirements

Recommended: MATLAB R2022b or newer.

Toolboxes typically required (depending on which parts you run):

* Optimization Toolbox (fmincon)
* Global Optimization Toolbox (GA, patternsearch)
* Statistics and Machine Learning Toolbox
* Parallel Computing Toolbox (optional; parallel objective evaluations)

---

## Data location

Place raw data under:

```
data/raw_events/*.csv
```

Intermediate and output artifacts are written under `derived/`.

---

## Author

**Pedro Rodrigues Correia da Silva**  
MSc Aerospace Engineering — Control & Simulation Track  
Delft University of Technology (TU Delft)  
E-mail: [P.RodriguesCorreiaDaSilva@student.tudelft.nl](mailto:P.RodriguesCorreiaDaSilva@student.tudelft.nl)
