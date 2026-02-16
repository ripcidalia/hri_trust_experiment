# 04 — Cost Functions & Parameter Transformation
*(MATLAB Trust Modelling Framework – Full Documentation)*

---

## Overview

This document describes the **objective (cost) layer** used to fit the trust model parameters. It includes:

- Per-participant **weighted least-squares (WLS)** cost construction.
- Aggregation across participants into a single scalar objective.
- The parameter mapping utility that converts the optimisation vector \(\theta\) into structured model parameters used by the trust dynamics.
- Runtime profiling and sensitivity diagnostics.

This layer sits between:
- **Simulation** (`src/trust_model/`), and
- **Optimisation methods** (`src/optimization_methods/`, documented separately in `05_optimization_methods.md`).

---

## 1. Cost architecture

The cost function follows a standard “simulate → compare → sum” pattern:

1. **Simulate** the trust trajectory for each participant using the trust model and \(\theta\).
2. **Read out** model predictions \(\hat{y}_j\) at each measurement timestamp \(t_j\).
3. **Compare** with observed trust measurements \(y_j\) (questionnaires and probes), expressed on a common scale.
4. Compute a **weighted quadratic error**:
   \[
   J_P(\theta) = \sum_{j\in\mathcal{M}_P} w_j\,(y_j - \hat{y}_j)^2
   \]
   where weights \(w_j\) depend on measurement type.
5. **Sum across participants**:
   \[
   J(\theta) = \sum_{i=1}^{N} J_{P_i}(\theta).
   \]

### Measurement weights

The per-measurement weights are instrument-specific and expected in a struct:

- `weights.w40` — 40-item questionnaire weight (reference)
- `weights.w14` — 14-item mid-block questionnaire weight (mapped to 40-scale)
- `weights.w_probe` — trust probe weight (mapped to 40-scale)

If weights are not passed explicitly, the cost function will attempt to load `measurement_weights.mat` (produced by Step M5 in the measurement-mapping pipeline).

---

## 2. File inventory (cost & mapping)

All files in this module live under:

- `src/cost_functions/`

Included files:

- `trust_cost_one_participant.m`
- `trust_cost_all.m`
- `trust_theta_to_params.m`
- `measure_trust_cost_runtime.m`
- `debug_cost_gradient.m`

---

## 3. Per-participant WLS cost

### `trust_cost_one_participant.m`

**Purpose:** Compute the weighted least-squares cost for a single participant.

**Signature:**
```matlab
cost = trust_cost_one_participant(theta, P, dt, weights, mode, behavior_params)
````

**Inputs:**

* `theta` — global parameter vector (\theta) shared across all participants.

  * Mapping (\theta \rightarrow params) is handled via `trust_theta_to_params` (invoked inside the simulator).
* `P` — participant struct (one element), typically from:

  * `derived/participants_probes_mapped_stepM4.mat`
    and enriched with:
  * time information (Step T1) and calibrated probe values (Steps M2/M4).
* `dt` — simulation time step (seconds). Defaults to 1 if omitted/empty.
* `weights` — measurement weights struct with fields `w40`, `w14`, `w_probe`.

  * If omitted/empty, the function attempts to load `measurement_weights.mat`.
* `mode` — simulation mode string:

  * `"simple"` or `"coupled"`.
* `behavior_params` — behavioral model parameters (used for `"coupled"` mode).

**Output:**

* `cost` — scalar WLS cost:
  \[
  cost = \sum_j w_j,(y_{\text{obs},j} - y_{\text{hat},j})^2.
  \]

**Core steps:**

1. Run a forward simulation (and measurement readout) via:

   * `trust_simulate_or_predict_one_participant(mode, theta, P, dt, behavior_params)`
2. Build observation and prediction vectors:

   * observed `y_obs` from the participant’s measurement structs,
   * predicted `y_hat` from the simulator readout.
3. Assign weights (w_j) per measurement type:

   * post 40-item (`w40`)
   * mid-block 14-item mapped to 40-scale (`w14`)
   * probes mapped to 40-scale (`w_probe`)
4. Return the weighted sum of squared residuals.

**Dependencies:**

* `trust_simulate_or_predict_one_participant` (trust model forward simulation)
* `measurement_weights.mat` (when weights not provided)

---

## 4. Global cost over participants

### `trust_cost_all.m`

**Purpose:** Compute the total WLS objective by summing per-participant costs.

**Signature:**

```matlab
total_cost = trust_cost_all(theta, participants, dt, weights, mode, behavior_params)
```

**Inputs:**

* `theta` — global parameter vector.
* `participants` — array of participant structs, typically from:

  * `derived/participants_probes_mapped_stepM4.mat`
    containing time-enriched events and mapped measurements.
* `dt` — simulation time step (seconds). Defaults to 1 if omitted/empty.
* `weights` — measurement weights struct (`w40`, `w14`, `w_probe`).

  * If omitted/empty, weights are loaded internally (via the per-participant cost).
* `mode` — `"simple"` or `"coupled"`.
* `behavior_params` — behavioral model parameters (for coupled mode).

**Output:**

* `total_cost` — scalar:
  \[
  total_cost = \sum_i cost_i.
  \]

**Robustness behavior:**

* Aggregates costs across participants.
* Returns a large penalty (e.g. `1e6`) if the aggregated cost becomes non-finite or non-scalar due to numerical issues.

**Notes:**

* No normalization by number of measurements is performed here; weighting is handled via measurement weights in the per-participant computation.

---

## 5. Parameter transformation

### `trust_theta_to_params.m`

**Purpose:** Convert the optimisation vector (\theta) into a structured parameter representation `params` used by the trust model components.

**Signature:**

```matlab
params = trust_theta_to_params(theta)
```

**Behavior:**

* Pure mapping / reshaping utility.
* Does not apply bounding, constraints, or transforms; it simply interprets entries of `theta` as model parameters.

**Parameter layout (length 8):**

* `theta(1) = lambda_rep` — reputation decay rate ((\ge 0))
* `theta(2) = alpha_sit` — situational component weight (in ([0,1]))
* `theta(3) = lambda_sit` — situational risk sensitivity ((> 0))
* `theta(4) = phi_fail` — first failure magnitude (in ([0,1]))
* `theta(5) = phi_succ` — first success magnitude (in ([0,1]))
* `theta(6) = a_succ` — success-shape parameter ((< 0))
* `theta(7) = lambda_lat` — base latent “above” rate ((> 0))
* `theta(8) = kappa_lat` — base latent “below” rate ((> 0))

**Fixed design constants (not estimated):**

* `params.lat.eps_lat` — deadzone around (\tau_{\text{disp}})
* `params.lat.gamma_above`, `params.lat.epsilon_above` — shaping/scaling for “above” episodes
* `params.lat.gamma_below`, `params.lat.epsilon_below` — shaping/scaling for “below” episodes
* `params.lat.tau_offset` — logistic offset for “below” dynamics

**Consumed by:**

* `trust_update_personal_experience` via `params.exp.*`
* `trust_prepare_latent_sequence` / `trust_update_latent_sequence` via `params.lat.*`
* `trust_update_reputation` via `params.rep.*`
* `trust_compute_situational` via `params.sit.*`

---

## 6. Runtime profiling

### `measure_trust_cost_runtime.m`

**Purpose:** Benchmark the runtime of:

* one call to `trust_cost_all` across all participants, and
* one call to `trust_simulate_or_predict_one_participant` for a single participant.

**Signature:**

```matlab
measure_trust_cost_runtime()
measure_trust_cost_runtime(dt)
```

**Behavior:**

1. Load participants with mapped probes (for fitting).
2. Load measurement weights.
3. Warm up once (JIT compilation and internal caching).
4. Time repeated evaluations of:

   * `trust_cost_all(...)`
5. Time a single participant simulation and provide a rough extrapolation.

**Inputs:**

* `dt` — simulation time step (seconds). Defaults to 0.1 if omitted/empty (for runtime assessment).

**Assumptions:**

* `derived/participants_probes_mapped_stepM4.mat` exists and contains `participants_probes_mapped`.
* `derived/measurement_weights.mat` exists and contains `weights`.

---

## 7. Cost sensitivity diagnostics

### `debug_cost_gradient.m`

**Purpose:** Probe sensitivity of the global cost to small perturbations of each parameter in a baseline vector.

**Signature:**

```matlab
debug_cost_gradient()
```

**Behavior:**

1. Load participants and weights.
2. Define a baseline `theta0`.
3. Evaluate `trust_cost_all(theta0, ...)`.
4. For each parameter index (k):

   * perturb (\theta_k) by a small epsilon,
   * re-evaluate the cost,
   * report validity (finite / non-finite) and magnitude changes.

**Use cases:**

* Identify parameters that cause NaNs/Infs under small perturbations.
* Detect extreme scaling sensitivity before running an optimizer.

**Assumptions:**

* Data files are present under `derived/`.
* `trust_cost_all` expects `(theta, participants, dt, weights)` in the configured pipeline.

---

## 8. Summary

This module defines the objective that drives parameter estimation:

* `trust_cost_one_participant` computes per-participant WLS error using measurement-type weights.
* `trust_cost_all` sums costs over participants and applies robust penalty behavior when needed.
* `trust_theta_to_params` provides the bridge from (\theta) to the structured parameters used by the trust model.
* `measure_trust_cost_runtime` and `debug_cost_gradient` support profiling and stability diagnostics.

Optimisation solvers and their usage are documented separately in `05_optimization_methods.md`.
