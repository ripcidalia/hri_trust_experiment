# 01 — Preprocessing & Data Structuring
*(MATLAB Trust Modelling Framework – Full Documentation)*

---

## Overview

This document describes the full preprocessing pipeline that converts the raw experimental event log (`rawData/events.csv`) into validated, time-enriched MATLAB participant structs used throughout the project (measurement mapping, trust simulation, optimisation, and diagnostics).

The preprocessing stage transforms a flat event table into per-participant structures with:

- **Normalized schema** (consistent column names and types).
- **Decoded JSON payloads** (e.g., `extra_json`, `response`) preserved both as raw text and decoded structs.
- **Structured extractions**: door trials, trust probes, questionnaires, review prime, demographics, and emergency choice.
- **Validation**: structural consistency checks + probe/event alignment checks.
- **Time enrichment (T1)**: continuous time stamps in seconds, aligned per participant.
- **Optional split (V)**: explicit training / validation participant sets.

---

## Inputs and Outputs

### Raw input

- `rawData/events.csv` — one row per event, including identifiers, timestamps, event type, and JSON-encoded payloads.

### Main outputs

| Step | Output file | Contents |
|------|-------------|----------|
| Step 1 | `derived/normalized_events_step1.mat` | Normalized event table (`T`) with decoded JSON and projected common fields. |
| Step 2 | `derived/participants_step2.mat` | `participants` array (one struct per participant), including emergency info. |
| Step 3 | `derived/validation_report_step3.mat` + `.csv` | Validation structs `V` summarizing data quality and issues. |
| Step 4 | `derived/participants_clean_step4.mat` | `participants_clean` filtered by validation + optional criteria; includes masks and `info`. |
| Step T1 | `derived/participants_time_stepT1.mat` | Time-enriched `participants_clean` plus metadata (`info_time`). |
| Step V (optional) | `derived/participants_train_stepV.mat` / `derived/participants_valid_stepV.mat` | Explicit train/validation participant sets + `info` metadata. |

> **Note on directories:** All outputs are written under `derived/`. The pipeline assumes it is run from the repository root.

---

## Pipeline overview (Steps 1–4, T1, and optional V)

At a high level:

1. **Step 1 — Load & normalize**
   - Read CSV into a MATLAB table.
   - Normalize column headers.
   - Decode JSON columns into struct variables.
   - Project frequently used fields into top-level table variables.
   - Sort events chronologically and print diagnostics.
   - Save `derived/normalized_events_step1.mat` (and a best-effort CSV export).

2. **Step 2 — Build participant structs**
   - Group events by participant identifier (fallback to `session_id` if needed).
   - Build one participant struct per group (`build_participant_struct`).
   - Extract emergency choice into `P.emergency`.
   - Harmonize fields and save `derived/participants_step2.mat`.

3. **Step 3 — Validate**
   - Run `validate_participant` for each participant.
   - Save `derived/validation_report_step3.mat` and a flat CSV summary.

4. **Step 4 — Filter**
   - Keep participants that pass required validation checks.
   - Optionally apply demographic / experiment filters (e.g., age range, gender, review condition, device type, emergency choice).
   - Save `derived/participants_clean_step4.mat`.

5. **Step T1 — Time enrichment**
   - Reload raw events table.
   - Compute participant-relative time `t_s` (seconds), using `ts_client` when possible, with fallback to `ts_seq`.
   - Shift time so the first `door_trial` occurs at **t = 10 s** (if present).
   - Inject time stamps into `timeline`, door trials, trust probes, and questionnaires.
   - Save `derived/participants_time_stepT1.mat`.

6. **Optional Step V — Train/validation split**
   - Split the time-enriched participant array by indices.
   - Save `derived/participants_train_stepV.mat` and `derived/participants_valid_stepV.mat`.

7. **Simulation support — time grid & event mapping**
   - For simulation and optimisation, each participant is mapped onto a uniform time grid (`build_time_grid_and_events`).
   - This mapping is memoized per `(participant, dt)` using `get_cached_time_grid_and_events` to avoid recomputation during optimisation.

---

## How to run preprocessing

### Recommended: run the orchestration script

The preprocessing pipeline is orchestrated by:

- `src/preprocessing.m`

It sequences the calls to:

- `step1_loader.m`
- `step2_build_participants.m`
- `step3_validate.m`
- `step4_filter_participants.m`
- `stepT1_add_times.m`

and assumes:

- raw data is at `rawData/events.csv`
- outputs go under `derived/`

### Optional: perform an explicit train/validation split

After Step T1, call:

- `stepV_split_train_valid(timeMatPath, train_idx, outDir)`

to create explicit training and validation `.mat` files.

---

## Module map (files under `src/preprocessing/`)

The preprocessing subsystem contains:

- **Orchestration / pipeline drivers**
  - `step1_loader.m`
  - `step2_build_participants.m`
  - `step3_validate.m`
  - `step4_filter_participants.m`
  - `stepT1_add_times.m`
  - `stepV_split_train_valid.m`

- **Table import, normalization, and projection**
  - `read_events_table.m`
  - `normalize_headers.m`
  - `sort_by_ts.m`
  - `decode_json_columns.m`
  - `safejsondecode.m`
  - `ensure_vars_exist.m`
  - `project_common_fields.m`
  - `print_step1_diagnostics.m`

- **Participant construction & extraction**
  - `build_participant_struct.m`
  - `extract_door_trials.m`
  - `extract_door_trials_with_order.m`
  - `extract_trust_probes.m`
  - `extract_trust_probes_linked.m`
  - `extract_trust_probes_with_mapping.m`
  - `extract_questionnaires.m`
  - `extract_reviews.m`
  - `extract_demographics.m`
  - `harmonize_struct_fields.m`
  - `get_numeric_field.m`
  - `get_string_field.m`

- **Validation**
  - `validate_participant.m`
  - `check_timeline.m`
  - `check_door_trials.m`
  - `check_trust_probes.m`
  - `check_questionnaires.m`
  - `check_demographics.m`
  - `compute_block_counts.m`
  - `probe_alignment_report.m`

- **Time grid + caching**
  - `build_time_grid_and_events.m`
  - `get_cached_time_grid_and_events.m`

- **Train/validation split utilities**
  - `split_participants_by_index.m`

---

## Step-by-step details

### Step 1 — Load & normalize (`step1_loader.m`)

**Purpose:** Load the raw CSV and produce a normalized event table suitable for downstream processing.

**Processing sequence (as implemented):**
1. Read the raw CSV into a table.
2. Normalize column headers (lowercase, underscores, etc.).
3. Decode JSON fields (e.g., `extra_json`, `response`) into struct columns.
4. Project commonly used fields from nested structs into top-level table variables via `project_common_fields`.
5. Sort chronologically (by `ts_seq` when available).
6. Print basic diagnostics.
7. Save `derived/normalized_events_step1.mat` and a best-effort CSV export.

**Primary output:**
- `derived/normalized_events_step1.mat` (contains table `T`)

---

### Step 2 — Build participant structs (`step2_build_participants.m`)

**Purpose:** Group the normalized event table by participant and build per-participant structs.

**Key behavior:**
- Groups by `participant_id` with fallback to `session_id` if needed.
- Builds each participant using `build_participant_struct(Tk)`.
- Extracts the emergency-trial choice (if present) and stores it under:
  - `P.emergency.has_response`
  - `P.emergency.choice` (normalized to lowercase; `"self"` / `"robot"` or empty)
- Harmonizes fields across participants for consistency.
- Saves `derived/participants_step2.mat`.

---

### Step 3 — Validate (`step3_validate.m`)

**Purpose:** Run consistency checks over all participants and export a compact validation report.

**Validation covers (via `validate_participant`):**
- timeline consistency
- door trial integrity
- trust probe integrity
- questionnaire presence/validity
- demographics validity (soft)
- emergency trial interpretation (soft)
- block-wise door trial counts
- basic probe ↔ door alignment checks

**Outputs:**
- `derived/validation_report_step3.mat` (struct array `V`)
- `derived/validation_report_step3.csv` (flat summary)

---

### Step 4 — Filter participants (`step4_filter_participants.m`)

**Purpose:** Select participants that pass required validation checks, with optional additional filters.

**Filtering logic:**
- Always applies **validation-based** filtering (required checks).
- Optionally applies user-defined filters on:
  - `gender`, `age_range`
  - `set_id`
  - `review_condition`
  - `device_type`
  - `emergency_choice`

Each filter supports a `*_mode` of `"include"` (default) or `"exclude"`.

**Outputs:**
- `derived/participants_clean_step4.mat` containing:
  - `participants_clean`
  - `info`
  - `allOk` (validation + filters mask)
  - `allOkValidationOnly` (validation-only mask)
  - `V` (validation structs)

---

### Step T1 — Time enrichment (`stepT1_add_times.m`)

**Purpose:** Add continuous time stamps (seconds) to participant structs by reloading the raw event table.

**Key behavior:**
- Builds relative time `t_s` per event using:
  - `ts_client` where available, else
  - `ts_seq` as monotone fallback
- Shifts time so the first `door_trial` occurs at **t = 10 s** (if present).
- Injects times into:
  - `P.timeline_t_s` (aligned with `P.timeline`)
  - `P.doorTrials(k).t_s`
  - `P.trustProbes(j).t_s`
  - `P.questionnaires.*.t_s`
  - forces `P.questionnaires.t40_post.t_s` to **(last door trial + 10 s)**

**Output:**
- `derived/participants_time_stepT1.mat` containing time-enriched `participants_clean` and `info_time`.

---

### Optional Step V — Train/validation split (`stepV_split_train_valid.m`)

**Purpose:** Create explicit training and validation participant sets from the time-enriched data.

**Inputs:**
- `timeMatPath` — typically `derived/participants_time_stepT1.mat`
- `train_idx` — indices into the participant array defining training set
- `outDir` — output directory (default `derived/`)

**Outputs:**
- `derived/participants_train_stepV.mat`
- `derived/participants_valid_stepV.mat`

Each includes:
- `participants_train` / `participants_valid`
- `info` with `.source_file`, `.train_indices`, `.valid_indices`, `.n_total`, `.n_train`, `.n_valid`, `.created`

---

## Participant struct: key fields

Participants are created by `build_participant_struct` and then enriched by later steps. At a minimum, downstream modules expect fields like:

- Identifiers and metadata:
  - `participant_id`, `session_id`, `set_id`, `seed`, `device_type`, (browser/client fields when available)
- Timeline and trials:
  - `timeline` (string array of event labels, including door identifiers)
  - `doorTrials` (struct array; contains outcomes, followed flag, risk, block/trial indexing)
  - `trustProbes` (struct array; contains probe values and contextual metadata)
- Questionnaires, reviews, demographics:
  - `questionnaires`
  - `reviews`
  - `demographics`
- Emergency choice:
  - `emergency.has_response`, `emergency.choice`
- Time enrichment (after T1):
  - `timeline_t_s`
  - `doorTrials(:).t_s`
  - `trustProbes(:).t_s`
  - `questionnaires.*.t_s`

---

## Time grid and event mapping (simulation support)

### `build_time_grid_and_events.m`

**Purpose:** Construct a uniform time grid and attach door events and trust measurements to grid indices.

**Important assumptions:**
- Step T1 has been run (so `t_s` exists on door trials and probes).
- Trust probe measurements have already been mapped to a 40-item equivalent percentage in:
  - `P.trustProbes(j).value_40` (0..100)
- Probes may carry a `questionnaire_type` tag to align them with questionnaire moments.

**Outputs:**
- `t_grid` time vector
- `doorEvents` + `doors_at_k` mapping
- `measurements` + `meas_at_k` mapping

This output is purely bookkeeping — it does **not** simulate trust dynamics.

### `get_cached_time_grid_and_events.m`

Memoizes the grid and event mapping per `(participant, dt)` so optimisation can reuse the same structures across many objective evaluations (safe because they do not depend on model parameters).

---

## Summary

The preprocessing layer converts the raw SAR trust study event log into a clean, validated, and time-enriched participant representation:

1. **Step 1:** CSV → normalized event table (`normalized_events_step1.mat`)
2. **Step 2:** event table → participant structs (`participants_step2.mat`)
3. **Step 3:** validation report (`validation_report_step3.mat/.csv`)
4. **Step 4:** filtered participants (`participants_clean_step4.mat`)
5. **Step T1:** time-enriched participants (`participants_time_stepT1.mat`)
6. **Optional Step V:** explicit train/validation split

These outputs form the required foundation for measurement mapping, trust simulation, cost evaluation, optimisation, and analysis.
