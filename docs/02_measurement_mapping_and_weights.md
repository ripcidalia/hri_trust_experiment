# 02 — Measurement Mapping & Weights
*(MATLAB Trust Modelling Framework – Full Documentation)*

---

## Overview

This document describes how the project calibrates and combines heterogeneous trust measurement instruments onto a **common 40-item questionnaire scale**, and how it derives **relative measurement weights** for downstream **Weighted Least Squares (WLS)** model fitting.

The measurement layer covers:

- **40-item questionnaire** totals (pre and post) — treated as the **reference instrument**.
- **14-item questionnaire** totals (mid-block measurements) — mapped to the 40-item scale.
- **Single trust probes** (slider ratings throughout the task) — mapped to the 40-item scale, with special handling at questionnaire anchors.

The pipeline produces:

1. **Global linear mappings**:
   - 14-equivalent → 40-item scale
   - probe value → 40-item scale
2. **Mapped participant datasets**:
   - mid-block 14-item totals expressed on the 40-item scale
   - per-probe 40-scale equivalents
3. **Residual-variance estimates** (via LOPO) and a consolidated **weights struct** for WLS objectives.

All steps operate on participant data produced by preprocessing (Step T1) and write outputs under `derived/`.

---

## Inputs and Outputs

### Inputs

- `derived/participants_time_stepT1.mat`  
  Expected to contain `participants_clean` (time-enriched participant structs).

Depending on step, a mapped participants file is also used:

- `derived/participants_mapped14_stepM2.mat`  
  Expected to contain `participants_mapped` (participants with mapped mid-block 14-item totals).

### Main outputs

| Step | Output file | Contents |
|------|-------------|----------|
| M1 | `derived/measurement_stepM1_calibration.mat` | `calib`: global OLS mappings (14→40 and probe→40) + participant IDs |
| M2 | `derived/participants_mapped14_stepM2.mat` | `participants_mapped`: mid-block 14-item totals mapped to 40-scale + `info` |
| M3 | `derived/measurement_step3_residual_variances.mat` | `residualVars`: LOPO residual variances (anchor + mid-block contexts) |
| M4 | `derived/participants_probes_mapped_stepM4.mat` | `participants_probes_mapped`: all probes augmented with `value_40` (+ anchor timestamps) |
| M5 | `derived/measurement_weights.mat` | `weights`: unified weights + residual variances + metadata |

---

## High-level pipeline (M1–M5)

The measurement mapping and weighting pipeline is implemented under:

- `src/postprocessing.m` (orchestrator)
- `src/measurement_mapping/` (step functions)

Execution order:

```matlab
stepM1_ols_calibration("derived/participants_time_stepT1.mat");

stepM2_apply_14mapping("derived/participants_time_stepT1.mat", ...
                       "derived/measurement_stepM1_calibration.mat", ...
                       "derived/participants_mapped14_stepM2.mat");

stepM3_residuals_variances("derived/participants_mapped14_stepM2.mat");

stepM4_apply_probe_mapping("derived/participants_mapped14_stepM2.mat", ...
                           "derived/measurement_stepM1_calibration.mat", ...
                           "derived/participants_probes_mapped_stepM4.mat");

stepM5_save_measurement_weights("derived/measurement_step3_residual_variances.mat");
````

---

## Design principles

### Unified scale

All heterogeneous instruments are expressed on the **40-item percentage scale** (0..100). Downstream, the trust simulation and cost functions typically normalize to **[0,1]** when constructing measurement vectors.

### Linear mappings

This module uses simple global linear mappings:

* 14-equivalent to 40-item:
  $$Q_{40} \approx a_{14}\, Q_{14}^{eq} + b_{14}$$


* probe value to 40-item:
  $$Q_{40} \approx a_{1},P + b_{1}$$
  where (P) is the raw probe value in 0..100.

### Reliability via residual variance

Residual variances are estimated via **leave-one-participant-out (LOPO)** in two contexts:

* **Anchor context**: probes after `t40_pre` and `t40_post` predict the corresponding 40-item totals.
* **Mid-block context**: probes after `t14_mid1` and `t14_mid2` predict the corresponding *mapped* mid-block totals on the 40-scale.

These variances are then packaged into `measurement_weights.mat`.

---

## Orchestrator script

### `src/postprocessing.m`

**Purpose:** Run the measurement calibration and mapping pipeline (M1–M5).

**Behavior (no computation inside):**

* Sequences the step calls and defines file paths.
* Assumes source functions are on the MATLAB path (`src/`).
* Assumes inputs exist under `derived/`.
* Writes outputs under `derived/`.

---

## Step M1 — Global OLS calibration mappings

### `stepM1_ols_calibration.m`

**Purpose:** Fit global (pooled) OLS mappings for:

1. **14-equivalent → 40-item scale** using questionnaire anchors:

   * `t40_pre.total_percent` and `t40_post.total_percent` as targets
   * `t40_pre.trust14_equiv_total_percent` and `t40_post.trust14_equiv_total_percent` as predictors

2. **probe → 40-item scale** using probe anchors:

   * `t40_pre.total_percent` and `t40_post.total_percent` as targets
   * probe values recorded after `t40_pre` and `t40_post` as predictors

**Inputs:**

* `cleanMatPath` (optional): MAT with `participants_clean`
  Default: `derived/participants_time_stepT1.mat`

**Outputs (file):**

* `derived/measurement_stepM1_calibration.mat` containing `calib` with:

  * `calib.a14`, `calib.b14` — 14-equivalent → 40 mapping
  * `calib.a1`,  `calib.b1`  — probe → 40 mapping
  * `calib.ids` — participant IDs

**Assumptions:**

* Each participant has `.questionnaires` with:

  * `t40_pre.total_percent`, `t40_post.total_percent`
  * `t40_pre.trust14_equiv_total_percent`, `t40_post.trust14_equiv_total_percent`
* Each participant has `.trustProbes` entries with:

  * `origin`, `questionnaire_type`, `value`
  * relevant anchor probes satisfy:

    * `origin == "after_questionnaire"`
    * `questionnaire_type ∈ {"t40_pre","t40_post"}`
* At least ~3 participants for a minimally robust pooled fit.

---

## Step M2 — Apply 14→40 mapping to mid-block questionnaires

### `stepM2_apply_14mapping.m`

**Purpose:** Map mid-block 14-item questionnaire totals onto the 40-item percentage scale using the M1 calibration coefficients.

For each participant:

* If `t14_mid1.total_percent` exists:
  $$\mathrm{t14\_mid1.total\_percent_40} = a_{14},\mathrm{t14\_mid1.total\_percent} + b_{14}$$
* If `t14_mid2.total_percent` exists:
  $$\mathrm{t14\_mid2.total\_percent_40} = a_{14},\mathrm{t14\_mid2.total\_percent} + b_{14}$$

Missing or non-coercible values are stored as `NaN`, but output fields are created for consistency.

**Inputs:**

* `cleanMatPath` (optional): MAT with `participants_clean`
  Default: `derived/participants_time_stepT1.mat`
* `calibPath` (optional): MAT with `calib.a14`, `calib.b14`
  Default: `derived/measurement_stepM1_calibration.mat`
* `outPath` (optional): output MAT
  Default: `derived/participants_mapped14_stepM2.mat`

**Output (file):**

* MAT at `outPath` containing:

  * `participants_mapped` — participants with:

    * `P(i).questionnaires.t14_mid1.total_percent_40`
    * `P(i).questionnaires.t14_mid2.total_percent_40`
  * `info` — metadata:

    * `source_clean_file`, `calib_file`, `created`, `n_participants`

---

## Step M3 — LOPO residual variances for probes

### `stepM3_residuals_variances.m`

**Purpose:** Estimate pooled residual variances for probe→40 predictions via LOPO in two contexts:

**A) Anchor context (pre/post):**

* Targets:

  * `t40_pre.total_percent`, `t40_post.total_percent`
* Predictors:

  * probe value after `t40_pre`, probe value after `t40_post`

**B) Mid-block context (mid1/mid2):**

* Targets:

  * `t14_mid1.total_percent_40`, `t14_mid2.total_percent_40` (from Step M2)
* Predictors:

  * probe value after `t14_mid1`, probe value after `t14_mid2`

**LOPO mechanism (per context):**
For held-out participant (i), fit on all others:
$$
\widehat{Q_{40}} = a^{(i)}_1,Probe + b^{(i)}_1
$$
Compute residuals on held-out pairs and pool across participants to estimate sample variances.

**Input:**

* `cleanMatPath` (optional): MAT with `participants_mapped` (output of M2)
  Default: `derived/participants_mapped14_stepM2.mat`

**Output (file):**

* `derived/measurement_step3_residual_variances.mat` containing `residualVars`:

  * `var_anchor` — pooled variance of anchor-context LOPO residuals
  * `var_mid` — pooled variance of mid-block-context LOPO residuals
  * `n_anchor` — number of non-NaN residuals used for `var_anchor`
  * `n_mid` — number of non-NaN residuals used for `var_mid`

**Assumptions:**

* Participants contain questionnaire totals and mapped mid-block totals:

  * `t40_pre.total_percent`, `t40_post.total_percent`
  * `t14_mid1.total_percent_40`, `t14_mid2.total_percent_40`
* Relevant probes satisfy:

  * `origin == "after_questionnaire"`
  * `questionnaire_type ∈ {"t40_pre","t40_post","t14_mid1","t14_mid2"}`
* Sufficient non-NaN probe/target pairs exist in each context.

---

## Step M4 — Apply probe mapping and enforce anchor consistency

### `stepM4_apply_probe_mapping.m`

**Purpose:** Augment each probe with a `value_40` field (40-item scale equivalent).

Special handling ensures **consistency at questionnaire anchors**:

* If a probe is tied to a questionnaire time point (`questionnaire_type` in `{t40_pre, t40_post, t14_mid1, t14_mid2}`),

  * `value_40` is taken **directly** from the corresponding questionnaire total on the 40-scale (or mapped 40-scale total for mid-block),
  * and `t_s` is copied from the questionnaire entry.

* Otherwise:

  * apply the global probe→40 mapping:
    $$
    value_{40} = a_1,value + b_1
    $$
  * If the raw probe value is `NaN`, set `value_40 = NaN`.

**Inputs:**

* `cleanMatPath` (optional): MAT with `participants_mapped` (typically Step M2 output)
  Default: `derived/participants_mapped14_stepM2.mat`
* `calibPath` (optional): MAT with `calib.a1`, `calib.b1`
  Default: `derived/measurement_stepM1_calibration.mat`
* `outPath` (optional): output MAT
  Default: `derived/participants_probes_mapped_stepM4.mat`

**Output (file):**

* MAT at `outPath` containing:

  * `participants_probes_mapped` — participants with each probe extended by:

    * `.value_40` (probe expressed on 40-item scale)
    * `.t_s` for anchor-typed probes (timestamp copied from questionnaire)
  * `info` — metadata (source paths, mapping coeffs, timestamps)

**Assumptions:**

* Probe structs have fields: `value`, `questionnaire_type`
* Questionnaire structs contain totals and timestamps:

  * `t40_pre.total_percent`, `t40_pre.t_s`
  * `t40_post.total_percent`, `t40_post.t_s`
  * `t14_mid1.total_percent_40`, `t14_mid1.t_s`
  * `t14_mid2.total_percent_40`, `t14_mid2.t_s`

---

## Step M5 — Save unified measurement weights

### `stepM5_save_measurement_weights.m`

**Purpose:** Consolidate measurement weighting parameters used downstream in WLS trust fitting.

It loads LOPO residual variance estimates and packages a unified `weights` struct for cost functions.

**Inputs:**

* `residualVarsPath` (optional): MAT with `residualVars` containing:

  * `var_anchor`, `var_mid`
    Default: `derived/measurement_step3_residual_variances.mat`

**Output (file):**

* `derived/measurement_weights.mat` containing `weights` with fields:

  * `w40` — reference weight for 40-item questionnaire (fixed to `1`)
  * `w14` — weight for 14-item questionnaire (**fixed proportion rule**)
  * `w_probe` — weight for probe measurements (computed from residual variances)
  * `var_anchor` — anchor-context residual variance (from Step M3)
  * `var_mid` — mid-block-context residual variance (from Step M3)
  * `info` — metadata (source file, timestamp, description)

**Notes on weight construction (as implemented):**

* `w40 = 1` defines the reference.
* `w14` is set via a fixed ratio (14/40), rather than from LOPO variance.
* `w_probe` is computed from `var_anchor` and `var_mid` using the expression in the code, producing a weight intended to be relative to the 40-item reference instrument.

---

## How downstream modules use these outputs

### Participant mappings used for simulation and fitting

* Step M2 provides mid-block questionnaire totals on the 40-scale:

  * `questionnaires.t14_mid1.total_percent_40`
  * `questionnaires.t14_mid2.total_percent_40`

* Step M4 provides per-probe 40-scale equivalents:

  * `trustProbes(k).value_40`

These fields are used when assembling measurement events for trust simulation and for building cost residuals.

### Weights used for WLS objectives

`derived/measurement_weights.mat` is loaded by cost functions (directly or indirectly) to weight measurement residuals by instrument reliability. Typical measurement “kinds” are:

* 40-item totals (reference): weight `w40`
* 14-item mid-block totals: weight `w14`
* probes: weight `w_probe`

The intent is that WLS fitting balances instruments not by count alone, but by their estimated (or prescribed) reliability.

---

## File inventory

Relevant files for this document:

* `src/postprocessing.m`
* `src/measurement_mapping/stepM1_ols_calibration.m`
* `src/measurement_mapping/stepM2_apply_14mapping.m`
* `src/measurement_mapping/stepM3_residuals_variances.m`
* `src/measurement_mapping/stepM4_apply_probe_mapping.m`
* `src/measurement_mapping/stepM5_save_measurement_weights.m`

