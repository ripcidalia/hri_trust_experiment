# 06 — Diagnostics & Analysis
*(MATLAB Trust Modelling Framework – Full Documentation)*

---

## Overview

This document describes the diagnostic and analysis tools included in the trust–dynamics codebase. These tools help evaluate model fit quality, inspect residual patterns, validate parameter behaviour, and generate plots used during development and thesis reporting.

Diagnostics are **read-only** with respect to participant data and fitted parameters: they do not modify the trust model, do not re-fit parameters, and do not overwrite the derived datasets (except for saving diagnostic artefacts such as figures, tables, and MAT summaries).

---

## 1. What diagnostics are for

After fitting the trust model (WLS optimisation), diagnostics help you to:

- Verify that residuals behave sensibly (distribution, bias, variance by instrument).
- Detect systematic prediction errors by measurement type (40-post, 14-mid, probes).
- Identify participants that disproportionately contribute to the final cost.
- Visualise the decomposition of trust into components (latent, reputation, situational).
- Inspect stability / sensitivity of the optimisation pipeline.
- Generate reporting figures and descriptive dataset statistics.

---

## 2. Diagnostic scripts and their roles

| File | Purpose |
|------|---------|
| `trust_residual_diagnostics.m` | Global residual analysis across all participants: tables + summary stats + core diagnostic figure. |
| `run_trust_diagnostics.m` | Single-participant simulation + plots of trust components, risk, and measurement overlays. |
| `sanity_check_trust_cost.m` | Per-participant cost evaluation and basic plots to detect outliers / dominance. |
| `test_trust_modules.m` | Smoke-test for core trust modules using a synthetic participant and hand-crafted event sequence. |
| `run_ga_sweep_pipeline.m` | Automation utility: run multiple GA fits (overnight preset) with checkpointing and restartability. |
| `analyze_participants_overview.m` | Dataset overview reporting: demographics, set membership, review distributions, questionnaire distributions, balance matrices. |
| `stepM6_reputation_bias.m` | Diagnostic for review-prime effects and negativity bias; writes metrics + optional figures. |
| `diag_behavior_thresholds.m` | Print/return per-participant decision-threshold diagnostics computed from the A7 behaviour dataset. |
| `reputation_bias.m` | (Legacy/alternate) reputation-bias analysis; conceptually overlaps with `stepM6_reputation_bias`. |

---

## 3. Global residual analysis

### `trust_residual_diagnostics.m`

**Purpose:** Perform a global residual diagnostic of the trust model across all participants used in the fitting pipeline.

**Signatures:**
```matlab
[residuals_tbl, summary_tbl] = trust_residual_diagnostics(theta)
[residuals_tbl, summary_tbl] = trust_residual_diagnostics(theta, dt)
````

**High-level procedure (per participant):**

1. Run forward simulation using `trust_simulate_or_predict_one_participant`.
2. Extract observed trust measurements `y_obs` and model predictions `y_hat` at measurement times.
3. Compute residuals `r = y_obs - y_hat`.
4. Append residuals into a long-format table (one row per measurement).

**Inputs:**

* `theta` — 8×1 parameter vector used by the fitting pipeline (same layout as cost functions):

  1. `lambda_rep`
  2. `alpha_sit`
  3. `lambda_sit`
  4. `phi_fail`
  5. `phi_succ`
  6. `a_succ`
  7. `lambda_lat`
  8. `kappa_lat`
* `dt` — simulation time step (seconds). If omitted/empty, defaults to 1s (consistent with many fitting runs).

**Outputs:**

* `residuals_tbl` — table with one row per measurement, including:

  * `participant_index`, `participant_id`
  * `meas_index`
  * `t` (seconds)
  * `kind` (`"t40_post" | "t14_mid1" | "t14_mid2" | "probe"`)
  * `y_obs`, `y_hat` (0..1)
  * `residual` (`y_obs - y_hat`)
  * `weight` (instrument weight used in WLS)
* `summary_tbl` — per-kind summary statistics:

  * `N`
  * `mean_residual`
  * `std_residual`
  * `rmse`
  * `mean_abs_residual`

**Side effects:**

* Produces a diagnostic figure with:

  * Histogram of residuals
  * Boxplot grouped by measurement type
  * Residual vs predicted scatter
  * Residual vs time

**Requirements / expected files:**

* `derived/participants_probes_mapped_stepM4.mat`
* `derived/measurement_weights.mat` (or `measurement_weights.mat`)
* Trust simulation code on path: `trust_simulate_or_predict_one_participant`

---

## 4. Single-participant simulation visualisation

### `run_trust_diagnostics.m`

**Purpose:** Simulate and visualise trust dynamics for one participant.

**Signatures:**

```matlab
run_trust_diagnostics()
run_trust_diagnostics(participant_index)
run_trust_diagnostics(participant_index, dt)
run_trust_diagnostics(participant_index, dt, theta)
```

**What it plots:**

* Total trust (\tau(t))
* Latent component (\tau_{lat}(t))
* Reputation component (\tau_{rep}(t))
* Situational component (\tau_{sit}(t))
* Risk signal (r(t))

**Annotations:**

* Door trial times
* Trust measurement times
* Observed measurements overlaid on total-trust plot (grouped by type when labels are available)

**When to use:**

* Qualitative inspection of fitted dynamics.
* Debugging participant-specific failure modes.
* Producing interpretability figures (component decomposition).

---

## 5. Cost-level sanity checks

### `sanity_check_trust_cost.m`

**Purpose:** Evaluate per-participant and total costs using the same WLS cost definition as fitting, and visualise the distribution of cost contributions.

**Signatures:**

```matlab
sanity_check_trust_cost()
sanity_check_trust_cost(theta_hat)
```

**Procedure:**

1. Load participant data and measurement weights.
2. For each participant, call `trust_cost_one_participant`.
3. Print per-participant costs and total sum.
4. Produce:

   * Stem plot of per-participant costs
   * Histogram of the cost distribution

**Use cases:**

* Ensure all participants simulate successfully for a given `theta`.
* Check that a small number of participants do not dominate the objective.
* Compare baseline vs fitted parameter vectors.

---

## 6. Model smoke tests

### `test_trust_modules.m`

**Purpose:** Basic end-to-end smoke test for core trust model components using a synthetic participant and a small, hand-crafted event sequence.

**What it verifies:**

* `trust_init_state` can initialise a state with minimal inputs.
* `trust_step` can process representative event types without errors.
* The resulting trust trajectory is sensible enough for quick visual inspection.

**Notes:**

* Numeric values are not intended to be realistic or calibrated.
* Intended for developer regression testing during refactors and extensions.

---

## 7. Automated multi-run fitting sweeps

### `run_ga_sweep_pipeline.m`

**Purpose:** Run multiple independent GA optimisations using:

* `method = "ga"`
* `preset = "overnight"`

**Features:**

* Crash-safe, restartable checkpointing:

  * Completed runs saved as `checkpoints/ga_run_XXX.mat`
  * Re-running resumes incomplete runs

**Inputs (conceptual):**

* `cfg.dt` — simulation timestep (seconds)
* `N` — number of GA runs (positive integer)

**Optional:**

* `cfg.run_tag` — descriptive tag (default: `"trust_fit_ga_sweep"`)
* `cfg.base_dir` — results directory (default: `"derived/fit_runs"`)

**Use cases:**

* Assess variability of GA solutions.
* Explore multi-modality / robustness of fitted parameters.
* Build a library of candidate fits for later comparison.

---

## 8. Participant dataset overview reporting

### `analyze_participants_overview.m`

**Purpose:** Generate descriptive statistics, reporting tables, and figures used for dataset characterization.

**Typical outputs:**

* Demographics distributions (age range, gender)
* Door-sequence set membership counts (Set A ... Set H)
* Review expectation distribution and expectation-vs-response scatter
* Questionnaire score distributions (histograms + by-set boxplots)
* Reporting tables for device type, browser, emergency choice
* Balance matrices for multiple categorical variables

**Inputs (name-value):**

* `"Split"` (optional): `"train"` or `"valid"`.

  * If omitted, analyzes the full set in `derived/participants_time_stepT1.mat`.

**File outputs (typical):**

* Figures: `derived/participant_set_analysis/<scope>/figs/*.png` and `*.fig`
* Tables: `derived/participant_set_analysis/<scope>/tables/*.csv`
* MAT summary: `derived/participant_set_analysis/<scope>_analysis.mat`

**Dependencies (report styling):**

* `thesisStyle`, `thesisFinalizeFigure`, `thesisExport`
* (optional) `measurementDisplayName` for legend labels

---

## 9. Reputation / negativity bias diagnostics

### `stepM6_reputation_bias.m`

**Purpose:** Quantify how strongly participants update their opinion about the robot’s reputation as a function of review valence (expected reputation) versus their own post-review rating (response). This is diagnostic-only and does not affect measurement mapping or fitting.

**Signatures:**

```matlab
stepM6_reputation_bias(timeMatPath)
stepM6_reputation_bias(timeMatPath, saveFigDir)
```

**Per-participant extraction:**

* Scalar expected value (E) (review valence)
* Scalar response value (R)

**Computed metrics:**

* Raw negativity bias: compare mean (|R|) for negative vs positive (E)
* Normalized bias: influence index (I_i = |R_i|/|E_i|) for non-zero (E)
* Neutral-condition summaries for (E = 0)
* Reliability flag based on minimum group sizes

**Output (file):**

* Writes `derived/measurement_reputation.mat` containing `repMetrics` with:

  * vectors of `expected_all`, `response_all`, `participant_id`
  * group counts and means
  * `B_raw`, `B_norm`
  * reliability fields and timestamp

**Assumptions:**

* `.reviews` populated during preprocessing (Step 2 / `extract_reviews`)
* First review item contains `.response_struct{1}.expected` and `.response_struct{1}.response`

---

## 10. Behavioural threshold diagnostics (A7-derived)

### `diag_behavior_thresholds.m`

**Purpose:** Print (and optionally return) per-participant decision-threshold diagnostics from the A7 behaviour dataset.

**Key idea:**
Uses **pre-door trust** (`tau_decision = tau_hist(k_grid-1)`) rather than post-update trust at door time.

**Computed per participant and scope (global + block1..3):**

1. `min tau_decision` where `followed == 1`
2. `max tau_decision` where `followed == 0`
3. `self_confidence`
4. `gap = min_follow - max_override`

**Usage examples:**

```matlab
diag_behavior_thresholds("RUN_001", "valid");
T = diag_behavior_thresholds("RUN_001", "train", "Print", false);
```

---

## 11. Summary

The diagnostics and analysis module provides:

* Global residual evaluation (`trust_residual_diagnostics`)
* Deep, interpretable single-participant visualisation (`run_trust_diagnostics`)
* Sanity checks at the cost-function level (`sanity_check_trust_cost`)
* Regression/smoke testing for core modules (`test_trust_modules`)
* Automated multi-run fit sweeps with checkpointing (`run_ga_sweep_pipeline`)
* Dataset characterisation reports (`analyze_participants_overview`)
* Review-prime / negativity bias metrics (`stepM6_reputation_bias`)
* Behavioural threshold reporting from the behaviour dataset (`diag_behavior_thresholds`)

Together, these tools help ensure the model remains interpretable, stable, and scientifically defensible during development and reporting.

---

## Relevant files

* `src/diagnostics/analyze_participants_overview.m`
* `src/diagnostics/diag_behavior_thresholds.m`
* `src/diagnostics/reputation_bias.m`
* `src/diagnostics/run_ga_sweep_pipeline.m`
* `src/diagnostics/run_trust_diagnostics.m`
* `src/diagnostics/sanity_check_trust_cost.m`
* `src/diagnostics/test_trust_modules.m`
* `src/diagnostics/trust_residual_diagnostics.m`
