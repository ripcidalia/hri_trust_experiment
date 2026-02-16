# 03 — Trust Model Core
*(MATLAB Trust Modelling Framework – Full Documentation)*

---

## Overview

This document describes the **core trust dynamics model** used to represent and predict participant trust in the robot over the course of a session. It covers:

- the decomposition of trust into interpretable components,
- the continuous-time / event-driven update logic,
- simulation over a time grid with door events and measurement readouts,
- the MATLAB implementation under `src/trust_model/`.

This document focuses on the **trust model itself** (dynamics + simulation). Parameter-vector mapping is handled elsewhere (e.g., in `src/cost_functions/`) and is therefore not covered here.

---

## 1. Trust decomposition

At each time point, total trust is computed as an additive decomposition and then clipped:

\[
\tau(t) = \tau_{\text{lat}}(t) + \tau_{\text{rep}}(t) + \alpha_{\text{sit}}\;\tau_{\text{sit}}(r(t)),
\quad \tau(t)\in[0,1].
\]

Where:

- **Dispositional trust** \(\tau_{\text{disp}}\) is a participant-specific constant derived from the pre 40-item questionnaire.
- **Latent trust** \(\tau_{\text{lat}}\) is a slowly varying baseline that drifts toward \(\tau_{\text{disp}}\) between events, and is updated discretely by personal experience at door trials.
- **Reputation trust** \(\tau_{\text{rep}}\) captures the influence of the review prime and decays toward zero over time.
- **Situational trust** \(\tau_{\text{sit}}(r)\) is an instantaneous term driven by door risk \(r\in[0,1]\) and modulated by self-confidence.

> Note: The situational term is in \([-1,1]\) (before the global clip), and is added to the other components through a gain \(\alpha_{\text{sit}}\) (stored in parameters).

---

## 2. File inventory (trust model)

All trust-model source files live under:

- `src/trust_model/`

Included files:

- `trust_clip.m`
- `trust_compute_dispositional.m`
- `trust_compute_situational.m`
- `trust_debug_log.m`
- `trust_init_state.m`
- `trust_initial_reputation.m`
- `trust_prepare_latent_sequence.m`
- `trust_simulate_or_predict_one_participant.m`
- `trust_step.m`
- `trust_update_latent_sequence.m`
- `trust_update_personal_experience.m`
- `trust_update_reputation.m`

---

## 3. Core utilities

### `trust_clip.m`

**Purpose:** Saturate trust values to the interval \([0,1]\).

**Key behavior:**
- Discards tiny imaginary parts via `real(x)` (numerical safety).
- Clips values below 0 to 0 and above 1 to 1, element-wise.

Used throughout the model after combining components and after updates that may slightly overshoot due to numerical effects.

---

### `trust_compute_dispositional.m`

**Purpose:** Compute participant-specific dispositional trust \(\tau_{\text{disp}}\).

**Definition:**
- Uses the pre 40-item questionnaire total in percent:
  \[
  \tau_{\text{disp}} = \frac{Q40_{\text{pre}}}{100}.
  \]

**Notes:**
- \(\tau_{\text{disp}}\) is constant for a participant during simulation.
- It serves as the anchor toward which latent trust drifts.

---

## 4. Situational component

### `trust_compute_situational.m`

**Purpose:** Compute the instantaneous situational trust contribution given risk and self-confidence.

This function is evaluated at (or around) door events and combined as:

\[
\tau(t) = \tau_{\text{lat}}(t) + \tau_{\text{rep}}(t) + \alpha_{\text{sit}}\;\tau_{\text{sit}}(r(t)).
\]

**Inputs:**
- `risk_value` — scalar in \([0,1]\) (non-finite treated as 0).
- `tau_disp` — dispositional trust in \([0,1]\).
- `sc` — self-confidence in \([0,1]\).
- `params.sit.lambda_sit` — positive risk sensitivity parameter.

**Behavior (two cases):**
- **Robot-trusting** (`tau_disp > sc`): situational trust **increases** with risk:
  \[
  \tau_{\text{sit}}(r) = 1 - 2e^{-\lambda_{\text{sit}} r}.
  \]
- **Self-trusting** (`tau_disp \le sc`): situational trust **decreases** with risk:
  \[
  \tau_{\text{sit}}(r) = 2e^{-\lambda_{\text{sit}} r} - 1.
  \]

**Output:**
- `tau_sit` — scalar in \([-1,1]\).

**Notes:**
- This is purely instantaneous; no time integration or stored state is involved.

---

## 5. Reputation component

### `trust_initial_reputation.m`

**Purpose:** Compute the initial reputation component \(\tau_{\text{rep},0}\in[-1,1]\) from review data.

**Current extraction logic (in priority order):**
1. From the first review item’s decoded response struct:
   - `P.reviews.items(1).response_struct{1}.expected`
2. Fallback:
   - `P.reviews.review_expected`
3. Fallback parameter default:
   - `params.rep.tau0`
4. Final fallback:
   - `0.0`

Result is clipped to \([-1,1]\).

**Notes:**
- The review **response** is currently extracted but not used in the final value (kept for possible later extensions such as negativity bias).

---

### `trust_update_reputation.m`

**Purpose:** Update reputation trust by exponential decay toward zero while preserving sign.

Given \(\tau_{\text{rep}}(t)\in[-1,1]\):

\[
|\tau_{\text{rep}}(t+\Delta t)| = |\tau_{\text{rep}}(t)|\;e^{-\lambda_{\text{rep}}\Delta t},
\quad
\tau_{\text{rep}}(t+\Delta t) = \mathrm{sign}(\tau_{\text{rep}}(t))\;|\tau_{\text{rep}}(t+\Delta t)|.
\]

**Inputs:**
- `tau_rep_cur` — scalar or array in \([-1,1]\).
- `dt` — time step (seconds, \(\ge 0\)).
- `params.rep.lambda_rep` — decay rate (default 0.0 if missing).

**Output:**
- `tau_rep_next` — same size as input, clipped to \([-1,1]\).

---

## 6. Personal experience updates at door events

### `trust_update_personal_experience.m`

**Purpose:** Compute the discrete trust increment \(\Delta\tau_{\text{exp}}\) at each door trial, based on success/failure streaks.

This component is **event-based** and only depends on:
- `outcome` (success/failure of the *chosen* door),
- current streak counters (consecutive successes / failures),
- parameters in `params.exp`.

**Inputs:**
- `exp_state_cur.n_succ`, `exp_state_cur.n_fail` — streak lengths (default 0 if missing).
- `outcome` — 1 for success, 0 for failure.
- `followed` — provided but not used for the update (kept for interface completeness).
- `params.exp.phi_fail` (\(\phi\)), `params.exp.phi_succ` (\(\psi\)), `params.exp.a_succ` (\(a\)).

**Outputs:**
- `delta_tau_exp` — scalar increment (positive or negative).
- `exp_state_next` — updated streak counters.

**Failure update (streak length \(n\)):**
- Uses a decaying negative contribution where the first failure drop equals \(-\phi\).
- Implemented via a transformation \(\lambda_{\text{fail}} = -\ln(1-\phi)\) and an exponential form that yields \(\tau^{{fail}}_{\text{exp}}(1)=-\phi\).

**Success update (streak length \(n\)):**
- Uses a shaped, saturating profile constructed from logistic differences.
- First success increase equals \(\psi\), and subsequent contributions saturate according to \(a\).

**Robustness notes:**
- If parameters are invalid or formulas degenerate, the function returns `delta_tau_exp = 0` safely.
- Clipping is not done here; it is handled when combining components in the overall state update.

---

## 7. Latent component: episode-based continuous drift

Latent trust \(\tau_{\text{lat}}(t)\) evolves **between door events**, drifting toward \(\tau_{\text{disp}}\). The implementation uses an **episode-based** representation:

- A dead-zone around \(\tau_{\text{disp}}\) where no drift is applied.
- Two regimes depending on whether latent trust is above or below the dispositional anchor:
  - **Above**: exponential decay down toward \(\tau_{\text{disp}}\).
  - **Below**: logistic growth up toward \(\tau_{\text{disp}}\).

### `trust_update_latent_sequence.m`

**Purpose:** Update \(\tau_{\text{lat}}\) over a time increment `dt`, maintaining an internal latent episode state.

**Inputs:**
- `tau_lat_cur` — scalar in \([0,1]\).
- `tau_disp` — scalar in \((0,1)\).
- `dt` — seconds, \(\ge 0\).
- `params.lat` — latent parameters (optional fields, defaults used if missing):
  - `.eps_lat`
  - `.lambda_lat`
  - `.kappa_lat`
  - `.gamma_above`, `.epsilon_above`
  - `.gamma_below`, `.epsilon_below`
  - `.tau_offset`
- `lat_state_cur` — struct describing the current latent episode:
  - `.mode` (`"none" | "above" | "below"`)
  - `.lambda_seq` (episode-specific decay rate)
  - `.kappa_seq` (episode-specific growth rate)
  - `.sigma` (internal logistic state for below-regime)
  - `.tau0` (latent value at episode start)

**Outputs:**
- `tau_lat_next`
- `lat_state_next`

**Key behavior:**
- If `dt == 0` or within `eps_lat` of `tau_disp`, no drift is applied and mode resets to `"none"`.
- If above/below regime changes sign relative to `tau_disp`, a new episode is started.
- Numerical safeguards ensure non-negative effective rates and clip outputs to \([0,1]\).

### `trust_prepare_latent_sequence.m`

**Purpose:** Prepare / reset the latent episode after a discrete door-event update.

This function is called after applying \(\Delta\tau_{\text{exp}}\) at a door trial, to ensure the next between-event drift episode starts from the updated latent value with consistent episode bookkeeping.

---

## 8. Full trust state and per-step update

### `trust_init_state.m`

**Purpose:** Initialize the full trust state for a participant at the start of simulation.

**Inputs:**
- `params` — model parameter struct with substructs:
  - `.rep` is used to initialize reputation via `trust_initial_reputation`
  - other fields are passed downstream through the state
- `P` — participant struct (from preprocessing / mapping), expected to include:
  - `P.questionnaires.t40_pre.total_percent`
  - optional `P.reviews` and `P.emergency`
  - `P.participant_id`

**Outputs:**
A state struct containing:
- `.t` — current time (seconds; typically aligned so pre-40 is at 0)
- `.tau` — total trust in \([0,1]\)
- `.tau_disp` — dispositional trust in \([0,1]\)
- `.tau_rep` — reputation component in \([-1,1]\)
- `.tau_lat` — latent component baseline in \([0,1]\)
- `.exp` — experience substate:
  - `.n_succ`, `.n_fail`
- `.last_risk` — last seen risk value (NaN if none)
- `.sc` — self-confidence in \([0,1]\)
- `.participant_id` — label for diagnostics

**Decomposition at runtime:**
\[
\tau = \tau_{\text{lat}} + \tau_{\text{rep}} + \alpha_{\text{sit}}\;\tau_{\text{sit}}(r).
\]

---

### `trust_step.m`

**Purpose:** Advance the trust state from time \(t_k\) to the next event time, applying:
- reputation decay over the elapsed \(\Delta t\),
- latent drift between events,
- discrete personal-experience updates at door trials,
- situational trust from the current / last risk.

**Inputs:**
- `state_cur` — current trust state from `trust_init_state`.
- `event` — struct describing the next event:
  - `.t` — absolute time (seconds) of the step
  - `.type` — `"door"`, `"probe"`, `"t14_mid1"`, `"t14_mid2"`, `"t40_post"`, etc.
  - door-only fields:
    - `.risk_value`, `.outcome`, `.followed`
- `params` — parameter struct with substructs `.lat`, `.rep`, `.exp`, `.sit`, etc.

**Output:**
- `state_next` — updated state at `event.t`.

**Rule summary:**

**Door event:**
- Apply personal experience update to latent:
  \[
  \tau_{\text{lat}} \leftarrow \tau_{\text{lat}} + \Delta\tau_{\text{exp}}.
  \]
- Prepare a new latent episode.
- Update reputation via decay (with a special rule to avoid decay before the first door is observed).
- Update situational trust from the event risk value.
- Combine components and clip total trust.

**Non-door event:**
- Drift latent via `trust_update_latent_sequence`.
- Decay reputation (active only after first door).
- Situational trust from last seen risk value.
- Combine components and clip.

**Notes:**
- Multiple door events can occur at the same grid time and are applied sequentially.
- Measurements do not alter the state; they induce a read-out of the current trust.

---

## 9. Simulation for one participant

### `trust_simulate_or_predict_one_participant.m`

**Purpose:** Run a full forward simulation of the trust model for a single participant and produce predictions at measurement times.

This function:
- builds (or loads from cache) the time grid and associated event lists,
- initializes state via `trust_init_state`,
- iterates over time, applying `trust_step`,
- records trajectories of total trust and its components,
- returns predicted trust values at measurement times.

**Modes:**
- `"simple"`: use recorded door outcomes deterministically.
- `"coupled"`: couple with the behavioral model to sample/produce followed actions and apply outcomes after counterfactual inversion; requires `behavior_params`.

**Inputs:**
- `mode` — `"simple"` or `"coupled"`.
- `theta` — parameter vector interpreted by the parameter-mapping layer (handled outside this document).
- `P` — participant struct, typically from mapped probes dataset and time-enriched:
  - door times `P.doorTrials(k).t_s`, probe times `P.trustProbes(j).t_s`, etc.
- `dt` — time step seconds (defaults to 1 if omitted/empty).
- `behavior_params` — required for `"coupled"` mode.

**Output struct `sim`:**
- `.t_grid` — time grid (seconds)
- `.doorEvents` — door event list (from cached grid/event builder)
- `.sc` — self-confidence scalar used in situational term
- trajectories:
  - `.tau_hist`, `.tau_lat_hist`, `.tau_rep_hist`, `.tau_sit_hist`
  - `.risk_hist`
- `.measurements` — measurement list (times + ground-truth y)
- `.y_hat` — model predictions at measurement times

**Coupled-mode-only additional logging (non-breaking):**
- `.coupled.followed_sampled`
- `.coupled.p_follow`
- `.coupled.outcome_used`
- `.coupled.t_door`
- `.coupled.block_index`
- `.coupled.door_index`

**Notes:**
- Evolution within each time step is handled by `trust_step`.
- Measurements do not influence the state; they only provide readout points.

---

## 10. Debug logging

### `trust_debug_log.m`

**Purpose:** Append timestamped diagnostics to a text file under `derived/`, intended for use during optimisation runs.

Typical uses include logging:
- non-finite trust values,
- unexpected transitions or invalid internal states,
- anomalous numerical behavior.

Designed to be lightweight and safe to call frequently.

---

## 11. Conceptual flow

```

participant P (mapped + time-enriched)
|
v
trust_simulate_or_predict_one_participant
|
+--> get_cached_time_grid_and_events (preprocessing cache; built elsewhere)
|
+--> trust_init_state
|
+--> loop over event times:
trust_step
- latent drift (between doors)
- personal experience update (at doors)
- reputation decay
- situational term from risk
- clip total trust
|
+--> read out predictions y_hat at measurement times

```

---

## 12. Summary

The trust-model core provides:

- a **modular decomposition** (latent + reputation + situational, anchored by dispositional trust),
- a **hybrid continuous/event-driven** update structure (drift between doors, discrete experience updates at doors),
- a **deterministic simulation engine** with optional coupling to a behavioral model,
- clear state bookkeeping and numerical safeguards (clipping, safe decay, episode management).

Together with preprocessing (Step 01) and measurement mapping (Step 02), this module forms the forward model used for fitting and evaluation.
```
