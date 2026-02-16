# 05 — Optimisation Methods for Trust Model Calibration
*(MATLAB Trust Modelling Framework – Full Documentation)*

---

## Overview

This document describes the optimisation methods implemented for calibrating the global parameter vector
\(\theta \in \mathbb{R}^8\) of the trust model. All methods minimise the same **weighted least-squares (WLS)**
objective (see `04_cost_optimisation.md`), but differ in numerical assumptions, robustness to non-smoothness,
and computational cost.

Implemented strategies:

- **fmincon** — gradient-based constrained local optimisation
- **patternsearch** — derivative-free deterministic optimisation
- **genetic algorithm (GA)** — stochastic population-based global optimisation

Implementation location:

- Dispatcher: `src/fit_trust_parameters.m`
- Solver backends: `src/optimization_methods/`

---

## 1. Common optimisation interface

All solver backends target the same problem:

\[
\min_{\theta}\; J(\theta) = \sum_{i=1}^{N} J_{P_i}(\theta),
\]

where \(J_{P_i}\) is the per-participant WLS cost and the global objective is computed by `trust_cost_all`.

### Shared inputs and outputs

All methods are invoked via the dispatcher:

```matlab
[theta_hat, fval, exitflag, output] = fit_trust_parameters(method, dt)
[theta_hat, fval, exitflag, output] = fit_trust_parameters(method, dt, preset)
[theta_hat, fval, exitflag, output] = fit_trust_parameters(method, dt, preset, theta0)
````

**Inputs:**

* `method` — `"fmincon"`, `"ga"`, or `"patternsearch"`
* `dt` — simulation time step in seconds
* `preset` — compute-budget preset (method-dependent; optional)
* `theta0` — optional initial guess (8×1)

**Outputs:**

* `theta_hat` — estimated parameter vector (8×1)
* `fval` — final WLS objective value
* `exitflag` — solver exit flag
* `output` — solver output struct

### Parameter vector layout

All methods estimate the same 8 parameters:

[
\theta =
\begin{bmatrix}
\lambda_{\text{rep}} \
\alpha_{\text{sit}} \
\lambda_{\text{sit}} \
\phi_{\text{fail}} \
\phi_{\text{succ}} \
a_{\text{succ}} \
\lambda_{\text{lat}} \
\kappa_{\text{lat}}
\end{bmatrix}
]

These are interpreted downstream by `trust_theta_to_params` (documented in `04_cost_optimisation.md`).

### Constraint handling (shared concept)

The inequality constraint:

[
\phi_{\text{succ}} \le \phi_{\text{fail}}
]

is enforced solver-specifically:

* **GA / patternsearch**: enforced **by construction** via re-parameterisation:
  [
  \phi_{\text{succ}} = \rho ,\phi_{\text{fail}},\quad \rho \in [0,1].
  ]
* **fmincon**: typically enforced through bounds/constraints (exact mechanism is solver-specific).

---

## 2. Dispatcher entry point

### `fit_trust_parameters.m`

**Purpose:** Conceptual entry point that selects an optimisation backend and standardises I/O.

**Signature:**

```matlab
[theta_hat, fval, exitflag, output] = fit_trust_parameters(method, dt, preset, theta0)
```

**Behavior:**

* Validates the requested `method`.
* Dispatches to one of:

  * `fit_trust_parameters_fmincon(dt, theta0)`
  * `fit_trust_parameters_ga(dt, preset, theta0)`
  * `fit_trust_parameters_patternsearch(dt, preset, theta0)`
* Returns the best parameter estimate and solver diagnostics.

---

## 3. fmincon (gradient-based constrained optimisation)

### `fit_trust_parameters_fmincon.m`

**Purpose:** Fit (\theta) using MATLAB `fmincon` for constrained local optimisation.

**Signatures:**

```matlab
[theta_hat, fval, exitflag, output] = fit_trust_parameters_fmincon(dt)
[theta_hat, fval, exitflag, output] = fit_trust_parameters_fmincon(dt, theta0)
```

**Key characteristics:**

* Uses `fmincon` (typically an interior-point style algorithm).
* Uses **finite-difference gradients** of the scalar objective (`trust_cost_all`).
* Deterministic given fixed options, initial guess, and dataset.

**When it works well:**

* Refinement from a good initial guess.
* Faster convergence near a smooth local basin.

**Common limitations in this project context:**

* Sensitivity to non-smoothness from:

  * event-driven updates (door events),
  * clipping and piecewise regime switching,
  * discontinuities from internal guards / fallbacks.

**Output:**

* Returns (\theta_{\hat{}}), final cost, and standard `fmincon` diagnostics.

---

## 4. patternsearch (derivative-free deterministic optimisation)

### `fit_trust_parameters_patternsearch.m`

**Purpose:** Fit (\theta) using MATLAB `patternsearch`, robust to non-differentiable objectives.

**Signatures:**

```matlab
[theta_hat, fval, exitflag, output] = fit_trust_parameters_patternsearch(dt, preset)
[theta_hat, fval, exitflag, output] = fit_trust_parameters_patternsearch(dt, preset, theta0)
```

**Inputs:**

* `dt` — simulation time step (seconds)
* `preset` — compute-budget preset:

  * `"moderate" | "heavy" | "overnight"`
* `theta0` — optional initial guess

**Key characteristics:**

* Derivative-free polling on an adaptive mesh.
* Naturally respects bound constraints.
* Deterministic for a fixed configuration.
* Can exploit parallel objective evaluations (MATLAB option-dependent).

**Constraint enforcement:**

* Uses re-parameterisation:
  [
  \phi_{\text{succ}} = \rho ,\phi_{\text{fail}},\quad \rho \in [0,1],
  ]
  and optimises an internal vector (x) where one component represents (\rho).
  The solver then recovers (\theta) from (x).

**When to use:**

* Default robust choice when objective smoothness is uncertain.
* Medium-duration runs where reliability is preferred over raw speed.

---

## 5. Genetic algorithm (global stochastic optimisation)

### `fit_trust_parameters_ga.m`

**Purpose:** Fit (\theta) using MATLAB `ga` for global search over a bounded domain.

**Signatures:**

```matlab
[theta_hat, fval, exitflag, output] = fit_trust_parameters_ga(dt, preset)
[theta_hat, fval, exitflag, output] = fit_trust_parameters_ga(dt, preset, theta0)
```

**Inputs:**

* `dt` — simulation time step (seconds)
* `preset` — compute-budget preset:

  * `"moderate" | "heavy" | "overnight"`
* `theta0` — optional initial guess injected into the initial population

**Key characteristics:**

* Population-based stochastic exploration using selection/crossover/mutation.
* Strong ability to escape local minima and explore multi-modal landscapes.
* High compute cost compared to local methods.
* Can be parallelised across population evaluations (MATLAB option-dependent).

**Constraint enforcement:**

* Enforces (\phi_{\text{succ}} \le \phi_{\text{fail}}) by construction using:
  [
  \phi_{\text{succ}} = \rho ,\phi_{\text{fail}},\quad \rho \in [0,1].
  ]

**Typical workflow role:**

* Global exploration to find good basins.
* Provide an initial guess for local refinement (GA → fmincon or GA → patternsearch).

**Reproducibility note:**

* GA is stochastic; runs can be made reproducible by fixing MATLAB’s RNG seed before calling the solver.

---

## 6. Presets and practical guidance

### Presets

Both GA and patternsearch expose compute-budget presets:

* `"moderate"` — faster iteration, suitable for development/debugging
* `"heavy"` — more thorough search, suitable for serious calibration runs
* `"overnight"` — maximum budget, intended for long runs

Preset meaning is solver-specific and typically controls:

* maximum iterations / generations,
* maximum objective evaluations,
* stall criteria,
* mesh refinement (patternsearch) or population settings (GA).

### Recommended workflow

1. **Development / debugging**

   * Use `"moderate"` preset with patternsearch or fmincon.
   * Use smaller participant subsets and/or coarser `dt`.

2. **Exploration**

   * Use GA or patternsearch with `"heavy"` preset.
   * Identify robust parameter regions.

3. **Final calibration**

   * Use GA/patternsearch output as `theta0` for `fmincon` refinement (optional).
   * Use the full participant set and production `dt`.

---

## 7. Summary

This module provides three complementary calibration methods:

* `fit_trust_parameters_fmincon` — efficient local refinement when smoothness allows.
* `fit_trust_parameters_patternsearch` — robust deterministic search for non-smooth objectives.
* `fit_trust_parameters_ga` — global exploration of complex cost landscapes.

All methods share a consistent interface through `fit_trust_parameters.m`, enabling systematic comparison while keeping the simulation and cost definitions fixed.

---

## Relevant files

* `src/fit_trust_parameters.m`
* `src/optimization_methods/fit_trust_parameters_fmincon.m`
* `src/optimization_methods/fit_trust_parameters_ga.m`
* `src/optimization_methods/fit_trust_parameters_patternsearch.m`

