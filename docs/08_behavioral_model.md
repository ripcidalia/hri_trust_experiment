# 08 — Behavioral Model (Follow/Override Decision Model)
*(MATLAB Trust Modelling Framework – Full Documentation)*

---

## Overview

This document describes the **probabilistic behavioral model** used to predict whether a participant **follows** the robot’s recommendation (follow) or **overrides** it (not follow) at a door decision.

The behavioral model is intentionally lightweight and is designed to couple cleanly to the trust dynamics simulator:

- It consumes the **current trust state** (total trust `tau`) and a participant-specific **self-confidence** scalar `sc`.
- It produces a **Bernoulli** action (follow / not follow) and the corresponding **follow probability** `p_follow`.
- The decision boundary is modeled as a smooth logistic function of the **decision margin** `(tau - sc)`.

This module is implemented in:

- `src/behavioral_model/behavioral_model.m`

---

## Behavioral decision rule

At a decision time (typically a door trial), the model computes the probability of following the robot recommendation as:

\[
p_{\text{follow}} \;=\; \frac{1}{1 + \exp\bigl(-k\,(\tau - sc)\bigr)}.
\]

Where:

- \(\tau \in [0,1]\) is the **current total trust** (from the trust model state),
- \(sc \in [0,1]\) is the participant’s **self-confidence** (stored in the trust state),
- \(k > 0\) is a **steepness** (inverse-temperature) parameter controlling how sharply the probability transitions around the threshold \( \tau = sc \).

The model then samples the binary action:

- `behavior = 1` (follow) with probability \(p_{\text{follow}}\),
- `behavior = 0` (not follow / override) with probability \(1 - p_{\text{follow}}\).

This is equivalent to treating follow as a **Bernoulli random variable** with parameter \(p_{\text{follow}}\).

---

## Interpretation

### Decision threshold (self-confidence)
The term `sc` acts as a **decision threshold**:

- If \(\tau > sc\), the model tends to follow more often (\(p_{\text{follow}} > 0.5\)).
- If \(\tau < sc\), the model tends to override more often (\(p_{\text{follow}} < 0.5\)).
- If \(\tau = sc\), then \(p_{\text{follow}} = 0.5\).

### Steepness parameter \(k\)
The parameter \(k\) controls determinism:

- Small \(k\): smooth, noisy decisions (probabilities closer to 0.5 over a wide range of margins).
- Large \(k\): near-deterministic threshold behavior.

---

## Function: `behavioral_model.m`

### Location
- `src/behavioral_model/behavioral_model.m`

### Signature
```matlab
[behavior, p] = behavioral_model(state)
````

### Inputs

* `state` — trust state struct with relevant fields:

  * `state.tau` : current total trust, scalar in `[0,1]`
  * `state.sc`  : self-confidence, scalar in `[0,1]`

* `params` — parameters for models (as implemented in code)

  * The behavioral model uses at least the steepness parameter `k` (naming and storage are code-defined).
  * In coupled simulations, `params` is typically passed via the simulator as a `behavior_params` struct.

> Note: The header describes `behavioral_model(state)` but also mentions `params`. The implemented function may accept `params` explicitly or read it from `state`/persistent configuration. The conceptual interface is: **state + behavior parameters → (action, probability)**.

### Outputs

* `behavior` — sampled binary action:

  * `1` = followed recommendation
  * `0` = not followed (override)
* `p` — probability of following recommendation ((p_{\text{follow}}))

---

## How it is used in coupled simulation

In `"coupled"` mode, the trust simulator uses the behavioral model at door events to **sample** whether the participant follows or overrides. The sampled action is then fed back into the trust-update logic at the door.

This enables **closed-loop rollouts** where trust and behavior interact:

* trust influences follow probability,
* sampled follow/override decisions affect experienced outcomes,
* outcomes update trust via the personal-experience component,
* and the cycle continues across door trials.

The behavioral model itself does not update trust; it only produces an action given the current trust state and parameters.

---

## Assumptions and data conventions

* `tau` is expected to be a valid scalar in `[0,1]`.
* `sc` is expected to be a valid scalar in `[0,1]`.
* The model uses a logistic probability, so extreme margins and large (k) can produce probabilities extremely close to 0 or 1 (numerically stable under `exp` for typical parameter ranges, but saturation is expected).
* Sampling is stochastic; reproducibility depends on MATLAB RNG state when the model is called.

---

## Relevant files

* `src/behavioral_model/behavioral_model.m`