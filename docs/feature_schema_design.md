# Feature Schema Design — VitalDB 5-Minute Window Monitoring

**Status:** proposal for review · **Schema version:** `fs-1.0` · **Date:** 2026-08-16

---

## 0. Scope and ground rules

This document specifies **only** the shape of the feature table that a future monitoring model would
consume. It does not specify the model, does not define anomalies, and makes no clinical claims.
Every threshold that appears here is a *statistical* or *data-quality* constant, never a
physiological one.

Inherited preprocessing contract (unchanged by this design):

| Item | Value |
|---|---|
| Window size | 5 min (300 s), non-overlapping |
| Minimum usable coverage | 70% |
| Core signals | `Solar8000/HR`, `Solar8000/PLETH_SPO2`, `Solar8000/RR_CO2` |
| Optional signal | `Solar8000/ART_MBP` |
| Interpolation | none — missing stays missing |
| Low-coverage windows | preserved, marked unusable |
| Raw data | untouched |

### 0.1 Grain and keys

- **One row = one (case, window).** Primary key `(case_id, window_index)`, sorted ascending.
- **Every window gets a row**, including fully unusable ones. Filtering is a *training-time*
  decision, not a build-time one. This keeps the table an honest record of the case timeline and
  makes gap-aware features computable.
- **Nothing crosses a case boundary.** All deltas, trends and run-counters reset at
  `window_index = 0`.
- **Strictly causal.** A row uses only the current window and earlier windows. No future
  information, no centered windows, no whole-case normalization. This is a hard rule so the table
  remains valid for an online monitoring setting.

### 0.2 Naming convention

`{signal}_{feature}` with fixed short signal prefixes:

| Track | Prefix | Role |
|---|---|---|
| `Solar8000/HR` | `hr` | core |
| `Solar8000/PLETH_SPO2` | `spo2` | core |
| `Solar8000/RR_CO2` | `rr` | core |
| `Solar8000/ART_MBP` | `mbp` | optional |

Window-scope features use `window_`, `core_`, `n_`, `run_`, or `cross_` prefixes.

### 0.3 Missing-value policy (applies to the whole table)

- Missing means **NULL**, always. No `0`, no `-999`, no forward-fill, no imputation, no
  "assume no change."
- Use nullable dtypes throughout (`Float64`, `Int64`, `boolean`, `category`) — not numpy floats
  where NaN would be ambiguous with a real value.
- **NULL is a distinguishable state and the model is expected to see it as such.** A separate
  `_usable` / `_present` boolean always accompanies nullable blocks so a model that cannot ingest
  NULL directly can be given `(flag, value-or-null)` pairs without inventing a value.
- **Descriptive stats vs. derived comparisons — the key line:**
  - *Descriptive* stats (mean/std/min/max/range/n_obs) are **computed and stored even when the
    signal is unusable**, because they are facts about observations that actually exist. They are
    not fabrications, and discarding them destroys information.
  - *Derived* comparisons (delta, pct change, trend, cross-signal, run counters) **never consume an
    unusable window.** A quality gate applies before any comparison.
  - The `_usable` flags are what let a consumer choose. Rejected alternative: nulling descriptive
    stats on unusable windows — it loses real data and makes "no observations at all"
    indistinguishable from "41% coverage."

### 0.4 The "previous usable window" rule (important)

Change features compare against **the most recent earlier window in which *that specific signal* was
usable** — not simply `window_index - 1`, and not the previous *fully usable* window.

Consequence: different signals in the same row may reference different back-windows. To keep this
honest and visible, every signal carries `{s}_ref_window_index` and `{s}_ref_gap_windows`. A
`gap = 1` delta spans 5 minutes; a `gap = 4` delta spans 20 minutes. **Those are not the same
quantity, and the model must be able to tell them apart.** Exposing the gap rather than capping it
is the proposal; see Open Question 6.

---

## 1. Block A — Identity and window index

| Column | R/O | Meaning | How computed | Prev? | Missing/unusable |
|---|---|---|---|---|---|
| `case_id` | R | VitalDB case identifier | carried from preprocessing | no | never null |
| `window_index` | R | 0-based window number within the case | sequential counter | no | never null |
| `window_start_s` | R | Window start, seconds from case start | `window_index x 300` | no | never null |
| `window_end_s` | R | Window end, seconds from case start | `window_start_s + window_dur_s` | no | never null |
| `window_dur_s` | R | Nominal window length in seconds | 300.0, or shorter for a trailing partial window | no | never null |
| `is_trailing_partial` | R | Last window of the case is shorter than 300 s | `window_dur_s < 300` | no | never null |
| `schema_version` | R | Feature-schema version string | constant `"fs-1.0"` | no | never null |

*Note:* a trailing partial window changes the coverage denominator. It is flagged rather than
dropped so the decision stays with the consumer.

---

## 2. Block B — Current-state features (per signal)

For each `s` in `{hr, spo2, rr, mbp}` — 5 columns each, **20 total**.

| Column | R/O | Meaning | How computed | Prev? | Missing/unusable |
|---|---|---|---|---|---|
| `{s}_mean` | R (core) / O (mbp) | Average value of observations in the window | arithmetic mean of observed samples | no | NULL if `n_obs = 0`; **computed even if unusable** |
| `{s}_std` | R / O | Spread of values inside the window | sample std, `ddof=1` | no | NULL if `n_obs < 2` |
| `{s}_min` | R / O | Lowest observed value | min of observed samples | no | NULL if `n_obs = 0` |
| `{s}_max` | R / O | Highest observed value | max of observed samples | no | NULL if `n_obs = 0` |
| `{s}_range` | R / O | Peak-to-trough spread within the window | `{s}_max - {s}_min` | no | NULL if `n_obs = 0` |

`{s}_range` is arithmetically derivable from min/max but is kept explicitly: axis-aligned models
(trees, and most simple detectors) cannot synthesize a difference between two columns, so leaving it
out silently removes the feature.

Units are native track units (HR: bpm, SpO2: %, RR: breaths/min, MBP: mmHg). No unit conversion, no
scaling, no clipping at this stage.

---

## 3. Block C — Change features vs. previous usable window

Per signal. `hr`, `rr`, `mbp` get 7 columns; `spo2` gets 6 (no pct change — see §3.1).
**27 total.**

| Column | R/O | Meaning | How computed | Prev? | Missing/unusable |
|---|---|---|---|---|---|
| `{s}_delta` | R / O | Change in the window average since this signal was last usable | `{s}_mean(cur) - {s}_mean(ref)` | **yes** | NULL if current window unusable for `s`, or no earlier usable window exists |
| `{s}_pct_change` | R (hr, rr) / O (mbp) | Same change expressed as a percentage of the reference value | `100 x delta / {s}_mean(ref)` | **yes** | NULL as above; also NULL if `{s}_mean(ref) = 0` |
| `{s}_delta_std` | O | Whether the signal became more or less variable | `{s}_std(cur) - {s}_std(ref)` | **yes** | NULL if either std is NULL |
| `{s}_trend_score` | R / O | Size of the change relative to the signal's own noise (unitless) | `delta / pooled_std`, where `pooled_std = sqrt((std_cur^2 + std_ref^2) / 2)` | **yes** | NULL if delta NULL or `pooled_std` NULL or `pooled_std = 0` |
| `{s}_trend` | R / O | `increasing` / `decreasing` / `stable` | `increasing` if `trend_score > TREND_K`, `decreasing` if `< -TREND_K`, else `stable`. `TREND_K = 1.0` (configurable) | **yes** | NULL if `trend_score` NULL. If `pooled_std = 0`: `stable` iff `delta = 0`, else sign of delta |
| `{s}_ref_window_index` | R / O | Which window this row was compared against | index of last usable window for `s` | **yes** | NULL if none exists |
| `{s}_ref_gap_windows` | R / O | How many windows back the comparison reaches (1 = immediately previous) | `window_index - ref_window_index` | **yes** | NULL if none exists |

### 3.1 Why SpO2 gets no percent change

SpO2 is already a percentage, bounded and compressed near its ceiling. A drop from 99 to 94 is a
5-point change but only a -5.1% "percent change," which is both numerically tiny and semantically
confusing (a percent of a percent). `spo2_delta` in **percentage points** is the only well-defined
change measure for this signal. Documented as a deliberate omission rather than emitted as a
permanently-null column.

### 3.2 Why `trend_score` exists alongside `{s}_trend`

`TREND_K` is a knob, and any fixed cut discards information. The continuous `trend_score` is the
model-facing feature; the categorical `{s}_trend` exists for explainability, for the run-length
counters, and for the cross-signal block. Both are kept so the discretization is never the only
representation. `pooled_std` is **self-referential** — the signal's own within-window variability —
so "stable" means "moved less than this signal normally wobbles," not any physiological statement.

---

## 4. Block D — Data quality (per signal)

5 columns per signal, **20 total**. Applies equally to core and optional signals.

| Column | R/O | Meaning | How computed | Prev? | Missing/unusable |
|---|---|---|---|---|---|
| `{s}_n_obs` | R / O | Number of observations present in the window | count of non-missing samples | no | never null; `0` is valid and meaningful |
| `{s}_coverage_pct` | R / O | Percentage of the window actually observed | `100 x n_obs / expected_n` (see Open Question 1) | no | never null; `0.0` when no data |
| `{s}_usable` | R / O | Whether this signal met the 70% coverage bar | `coverage_pct >= 70` | no | never null |
| `{s}_max_gap_s` | R / O | Longest continuous stretch inside the window with no observation | max inter-observation interval, including from window start to first sample and last sample to window end | no | never null; `= window_dur_s` when `n_obs = 0` |
| `{s}_missing_time_s` | R / O | Total unobserved time in the window | `window_dur_s - observed_time` | no | never null; `= window_dur_s` when `n_obs = 0` |

`max_gap_s` and `missing_time_s` together separate two very different failure modes that
`coverage_pct` alone conflates: **one long dropout** (low coverage, huge max gap) vs. **dense
intermittent loss** (same coverage, small max gap). That distinction matters for a monitoring model
and costs two columns.

---

## 5. Block E — Window-level quality

**6 columns.**

| Column | R/O | Meaning | How computed | Prev? | Missing/unusable |
|---|---|---|---|---|---|
| `n_core_usable` | R | How many of the 3 core signals are usable (0–3) | `hr_usable + spo2_usable + rr_usable` | no | never null |
| `all_core_usable` | R | All three core signals met the coverage bar | `n_core_usable == 3` | no | never null |
| `window_usable` | R | Row-level usability verdict — the intended training filter | `= all_core_usable` (proposal; see Open Question 5) | no | never null |
| `core_coverage_min_pct` | R | Worst core-signal coverage in this window | `min` over the 3 core `coverage_pct` | no | never null |
| `core_coverage_mean_pct` | R | Average core-signal coverage | mean over the 3 core `coverage_pct` | no | never null |
| `mbp_present` | R | Arterial pressure data exists at all in this window | `mbp_n_obs > 0` | no | never null |

`window_usable` is stored as its own column rather than left implicit, so a downstream policy change
is a one-line rebuild, not a re-derivation scattered across consumers. `n_core_usable` is retained so
a looser policy (e.g. "at least 2 core signals") can be applied without recomputing anything.

---

## 6. Block F — Cross-signal co-movement

**6 columns.** These describe *whether multiple signals moved at the same time and in which relative
direction*. They encode no medical relationship and no diagnosis.

| Column | R/O | Meaning | How computed | Prev? | Missing/unusable |
|---|---|---|---|---|---|
| `cross_basis_n` | R | How many core signals actually had a computable trend this row (0–3) | count of non-null `{s}_trend` over core signals | **yes** | never null |
| `cross_n_moving` | R | How many core signals are not `stable` | count of core `{s}_trend` in `{increasing, decreasing}` | **yes** | NULL if `cross_basis_n < 2` |
| `cross_hr_spo2_codir` | R | Do HR and SpO2 move the same way, opposite ways, or is one flat? | `+1` both same direction, `-1` opposite, `0` either is `stable` | **yes** | NULL if either trend NULL |
| `cross_hr_rr_codir` | R | Same, for HR and RR | as above | **yes** | NULL if either trend NULL |
| `cross_spo2_rr_codir` | R | Same, for SpO2 and RR | as above | **yes** | NULL if either trend NULL |
| `cross_change_magnitude` | R | Overall size of simultaneous movement across core signals | RMS of non-null core `trend_score`: `sqrt(mean(score^2))` | **yes** | NULL if `cross_basis_n = 0` |

Design notes:

- **Co-direction is computed from `{s}_trend` labels, not raw `sign(delta)`.** Raw sign flips on
  noise; a two-window pair that both barely wobbled would otherwise register as a strong "opposite
  movement" signal. Using labels means the noise band gates the comparison.
- `cross_basis_n` is published so the model can tell "no signals moved" from "we couldn't tell." The
  RMS in `cross_change_magnitude` averages only over available signals, which would otherwise
  silently change meaning between rows.
- ART_MBP is **excluded** from all cross-signal features. Including it would make the cross block's
  availability depend on the presence of an arterial line, which changes its meaning between cases
  (see Open Question 7).
- No pair is privileged and no pair is annotated with an expected relationship. All three core pairs
  are emitted symmetrically.

---

## 7. Block G — Temporal persistence (run-length counters)

**9 columns.** All are counts of *consecutive* windows ending at (and including) the current one.

| Column | R/O | Meaning | How computed | Prev? | Missing/unusable |
|---|---|---|---|---|---|
| `run_usable` | R | Consecutive usable windows up to now | `+1` if `window_usable` else reset to `0` | **yes** | never null; `0` on an unusable window |
| `run_unusable` | R | Consecutive unusable windows up to now | `+1` if not `window_usable` else `0` | **yes** | never null |
| `run_hr_up` | R | Consecutive windows where HR was classified `increasing` | `+1` if `hr_trend = increasing` else `0` | **yes** | never null; **reset to 0 if HR unusable** |
| `run_hr_down` | R | Same, `decreasing` | symmetric | **yes** | as above |
| `run_spo2_up` | R | Consecutive windows where SpO2 was `increasing` | symmetric | **yes** | as above |
| `run_spo2_down` | R | Same, `decreasing` | symmetric | **yes** | as above |
| `run_rr_up` | R | Consecutive windows where RR was `increasing` | symmetric | **yes** | as above |
| `run_rr_down` | R | Same, `decreasing` | symmetric | **yes** | as above |
| `run_multi_change` | O | Consecutive windows with 2+ core signals moving at once | `+1` if `cross_n_moving >= 2` else `0` | **yes** | never null; `0` when `cross_n_moving` NULL |

Design notes:

- **Directional runs are per-signal and break on *that signal's* unusability**, not on window-level
  unusability. If RR drops out but HR is clean, the HR run should survive — the HR evidence is
  intact.
- **A run requires contiguous usable windows.** If a signal is unusable in window *k*, its run
  resets, even though `{s}_delta` at window *k+1* still compares back to *k-1*. Rationale: we cannot
  assert a direction persisted through a window we did not observe. `{s}_ref_gap_windows` carries the
  "we skipped one" information separately, so nothing is lost — it is just not laundered into a
  persistence count.
- **Up and down are both emitted for every core signal.** Emitting only `run_hr_up` and
  `run_spo2_down` would bake a directional prior into the schema. Symmetry keeps the table neutral.
- `stable` resets both directional counters. Runs count *sustained movement*, not *absence of
  reversal*.

**Total: ~95 columns** (7 identity + 20 state + 27 change + 20 quality + 6 window quality + 6 cross
+ 9 temporal). Most of that is mechanical expansion across 4 signals rather than conceptual
complexity — there are 7 distinct feature ideas here.

### 7.1 Minimum viable subset

If a leaner first pass is wanted, this ~28-column subset preserves the design's structure: identity
(5), `{hr,spo2,rr}_mean/_std` (6), `{hr,spo2,rr}_delta` (3), `{hr,spo2,rr}_trend_score` (3),
`{hr,spo2,rr}_coverage_pct/_usable` (6), `window_usable`, `n_core_usable`, `run_usable`,
`cross_change_magnitude`, `cross_n_moving`. Everything else can be added later without changing the
grain, keys, or null semantics.

---

## 8. Deliberately NOT included, and why

| Excluded | Reason |
|---|---|
| Any imputation, interpolation, forward-fill, or "carry last value" | Violates the preprocessing contract. Missing must stay missing. |
| Sentinel values (`0`, `-1`, `-999`) for missing | Indistinguishable from real measurements; poisons every downstream statistic. NULL only. |
| Clinical thresholds, cutoffs, or alarm criteria | Out of scope by instruction and not a claim this project is positioned to make. `TREND_K` is a noise-relative statistical band, not a physiological one. |
| Named clinical scores (NEWS, MEWS, SOFA, qSOFA, shock index) | Medical constructs. Shock index in particular (`HR/SBP`) is a composite with a clinical meaning we are not asserting. |
| `spo2_pct_change` | Percent-of-a-percent on a ceiling-bounded signal — numerically compressed and semantically confusing. Absolute percentage-point delta only. §3.1 |
| HRV / frequency-domain features (FFT, LF/HF, spectral entropy), wavelets, sample/approximate entropy, detrended fluctuation | Complex time-series transforms, explicitly deferred. Also unreliable on 70%-coverage windows with unmodeled gaps. |
| EWMA, rolling z-scores over long horizons, autocorrelation, changepoint statistics | Same deferral. `trend_score` covers the "is this movement large relative to normal" question with one interpretable ratio. |
| Least-squares slope over the last N windows (`{s}_slope3`) | Borderline — genuinely simple and a natural v1.1 candidate, but it needs a gap-handling policy for unusable windows and would be the first feature spanning more than two windows. Held back to keep v1 to pairwise comparisons. |
| Per-patient baseline normalization (deviation from the case's own first N windows) | Requires defining a baseline period, and a baseline drawn from early windows leaks whole-case information into early rows. Needs its own design pass. |
| Cross-case / global z-scoring | Not causal in an online setting, and couples the feature table to a particular training population. |
| Whole-case or future-looking aggregates, centered windows, overlapping windows | Break causality and/or the 5-minute non-overlapping contract. |
| Ratios between different signals (`hr/rr`, `hr/mbp`) | Unit-incoherent quotients whose interpretability claims rest on physiology we are not asserting. Co-direction and co-magnitude capture "moving together" without them. |
| ART_MBP in cross-signal features | Its availability varies by case, which would make cross-feature semantics case-dependent. §6 |
| Outlier removal, clipping, or physiologic-plausibility filtering | This is a *preprocessing contract change*, not a feature decision. Raised as a future contract question, not smuggled in here. |
| Static patient metadata (age, sex, ASA, surgery type, department) | Merges cleanly on `case_id` later. Keeping the window table purely signal-derived keeps the two concerns separable and avoids premature cohort coupling. |
| Any anomaly score, label, flag, or detector output | Out of scope by instruction. This table is model *input* only. |

---

## 9. Example rows

Synthetic values for one case, windows 8–10. Assumes Solar8000 numerics at 0.5 Hz →
`expected_n = 150` per 300 s window; `missing_time_s = (150 - n_obs) x 2`.

The three rows are constructed to exercise the tricky paths: **w8** everything clean, **w9** RR drops
below the bar (window unusable, but HR/SpO2 comparisons continue), **w10** everything back except
ART_MBP, and RR's delta reaches back two windows.

### 9.1 Compact view

| window_index | window_start_s | hr_mean | hr_delta | hr_trend | spo2_mean | spo2_delta | spo2_trend | rr_mean | rr_delta | rr_ref_gap | mbp_mean | n_core_usable | window_usable | run_hr_up | run_spo2_down | cross_n_moving |
|---:|---:|---:|---:|---|---:|---:|---|---:|---:|---:|---:|---:|---|---:|---:|---:|
| 8 | 2400 | 72.4 | +1.2 | stable | 98.2 | -0.1 | stable | 12.6 | +0.3 | 1 | 78.5 | 3 | true | 0 | 0 | 0 |
| 9 | 2700 | 79.8 | +7.4 | increasing | 97.1 | -1.1 | decreasing | 15.8 | *null* | *null* | 74.2 | 2 | false | 1 | 1 | 2 |
| 10 | 3000 | 88.1 | +8.3 | increasing | 95.4 | -1.7 | decreasing | 16.9 | +4.3 | 2 | 71.0 | 3 | true | 2 | 2 | 3 |

Note row 9: `rr_mean = 15.8` is **present** (it is a real average of 62 real observations) while
`rr_delta` is **null** (41% coverage fails the quality gate for comparisons). That is the §0.3 rule in
action. Row 10: `mbp_mean = 71.0` is present but `mbp_usable = false` — ART_MBP had only 18
observations (12% coverage), so its descriptive stats are kept while every MBP comparison feature is
null. Present-but-unusable and absent are different states.

### 9.2 Full record — `window_index = 10`

```
case_id                  = 1234          window_index           = 10
window_start_s           = 3000.0        window_end_s           = 3300.0
window_dur_s             = 300.0         is_trailing_partial    = false
schema_version           = "fs-1.0"

-- HR (usable) ------------------------------------------------------
hr_mean = 88.1   hr_std = 5.2   hr_min = 78.0   hr_max = 101.0   hr_range = 23.0
hr_n_obs = 148   hr_coverage_pct = 98.7   hr_usable = true
hr_max_gap_s = 4.0   hr_missing_time_s = 4.0
hr_delta = +8.3   hr_pct_change = +10.40   hr_delta_std = +1.2
hr_trend_score = +1.79   hr_trend = increasing
hr_ref_window_index = 9   hr_ref_gap_windows = 1

-- SpO2 (usable) ----------------------------------------------------
spo2_mean = 95.4   spo2_std = 0.9   spo2_min = 93.0   spo2_max = 97.0   spo2_range = 4.0
spo2_n_obs = 141   spo2_coverage_pct = 94.0   spo2_usable = true
spo2_max_gap_s = 8.0   spo2_missing_time_s = 18.0
spo2_delta = -1.7  (percentage points)     spo2_delta_std = +0.3
spo2_trend_score = -2.22   spo2_trend = decreasing
spo2_ref_window_index = 9   spo2_ref_gap_windows = 1
   [spo2_pct_change intentionally absent from schema -- §3.1]

-- RR (usable; reference reaches back past unusable w9) -------------
rr_mean = 16.9   rr_std = 2.1   rr_min = 12.0   rr_max = 22.0   rr_range = 10.0
rr_n_obs = 133   rr_coverage_pct = 88.7   rr_usable = true
rr_max_gap_s = 10.0   rr_missing_time_s = 34.0
rr_delta = +4.3   rr_pct_change = +34.13   rr_delta_std = +0.7
rr_trend_score = +2.41   rr_trend = increasing
rr_ref_window_index = 8   rr_ref_gap_windows = 2      <-- 10-minute span, not 5

-- ART_MBP (optional; present but below coverage bar) ---------------
mbp_mean = 71.0   mbp_std = 6.1   mbp_min = 60.0   mbp_max = 82.0   mbp_range = 22.0
mbp_n_obs = 18   mbp_coverage_pct = 12.0   mbp_usable = false
mbp_max_gap_s = 214.0   mbp_missing_time_s = 264.0
mbp_delta = null   mbp_pct_change = null   mbp_delta_std = null
mbp_trend_score = null   mbp_trend = null
mbp_ref_window_index = null   mbp_ref_gap_windows = null
   [descriptive stats kept -- they are real observations; only the
    derived comparisons are gated off by mbp_usable = false. §0.3]

-- Window quality ---------------------------------------------------
n_core_usable = 3   all_core_usable = true   window_usable = true
core_coverage_min_pct = 88.7   core_coverage_mean_pct = 93.8
mbp_present = true

-- Cross-signal -----------------------------------------------------
cross_basis_n = 3   cross_n_moving = 3
cross_hr_spo2_codir = -1   cross_hr_rr_codir = +1   cross_spo2_rr_codir = -1
cross_change_magnitude = 2.16

-- Temporal persistence ---------------------------------------------
run_usable = 1   run_unusable = 0
run_hr_up = 2    run_hr_down = 0
run_spo2_up = 0  run_spo2_down = 2
run_rr_up = 1    run_rr_down = 0     <-- reset by w9's unusable RR, not 3
run_multi_change = 2
```

### 9.3 Supporting values for windows 8 and 9

Window 8 — all four signals usable, everything `stable`:

```
hr:   n_obs=142  cov=94.7  mean=72.4  std=3.1  delta=+1.2  score=+0.40  trend=stable
spo2: n_obs=147  cov=98.0  mean=98.2  std=0.5  delta=-0.1  score=-0.22  trend=stable
rr:   n_obs=138  cov=92.0  mean=12.6  std=1.4  delta=+0.3  score=+0.22  trend=stable
mbp:  n_obs=140  cov=93.3  mean=78.5  std=5.0  delta=-1.0  score=-0.21  trend=stable
n_core_usable=3  window_usable=true  cross_basis_n=3  cross_n_moving=0
cross_change_magnitude=0.29   all run_* directional counters = 0   run_usable=9
```

Window 9 — RR at 41% coverage, so the window is unusable but HR/SpO2 comparisons continue:

```
hr:   n_obs=145  cov=96.7  mean=79.8  std=4.0  delta=+7.4  score=+2.07  trend=increasing
spo2: n_obs=144  cov=96.0  mean=97.1  std=0.6  delta=-1.1  score=-1.99  trend=decreasing
rr:   n_obs=62   cov=41.3  mean=15.8  std=2.2  usable=false  max_gap_s=112.0
      -> rr_delta, rr_pct_change, rr_trend_score, rr_trend, rr_ref_* all NULL
mbp:  n_obs=139  cov=92.7  mean=74.2  std=5.4  delta=-4.3  score=-0.83  trend=stable
n_core_usable=2  all_core_usable=false  window_usable=false
core_coverage_min_pct=41.3  core_coverage_mean_pct=78.0  mbp_present=true
cross_basis_n=2  cross_n_moving=2  cross_hr_spo2_codir=-1
cross_hr_rr_codir=null  cross_spo2_rr_codir=null  cross_change_magnitude=2.03
run_usable=0  run_unusable=1  run_hr_up=1  run_spo2_down=1  run_rr_up=0
run_multi_change=1
```

### 9.4 What these rows demonstrate

1. **Per-signal reference windows diverge.** At w10, HR and SpO2 compare to w9; RR compares to w8
   with `rr_ref_gap_windows = 2`. The gap column is the only thing preventing a 10-minute change from
   being read as a 5-minute one.
2. **Descriptive is not derived.** w9 keeps `rr_mean` (real data, 62 samples) but nulls `rr_delta`
   (fails the quality gate).
3. **Signal-level unusability does not cascade.** w9 is `window_usable = false` because of RR, yet
   `run_hr_up` still advances — HR's evidence was intact.
4. **Runs reset on gaps.** `run_rr_up = 1` at w10, not 3, because RR was unobservable at w9.
5. **Optional signal degrades cleanly.** MBP goes from usable (w8, w9) to present-but-unusable (w10)
   with no fabricated values and no effect on the core row.
6. **Nothing is dropped.** w9 is unusable and still emitted, so the timeline stays continuous and
   gap-aware features remain computable.

---

## 10. Storage and typing

- **Format:** Parquet, partitioned by `case_id`, sorted by `(case_id, window_index)`.
- **Dtypes:** `Int64` for counts/indices, `Float64` for continuous, `boolean` for flags, `category`
  (ordered: `decreasing < stable < increasing`) for `{s}_trend`, `string` for `schema_version`. All
  nullable — no numpy float NaN standing in for a nullable int.
- **Rebuild is deterministic** from preprocessing output plus `TREND_K` and the 70% coverage bar.
  Both constants should live in one config location and be recorded alongside the built table.
- **`schema_version` is per-row**, so a mixed-vintage table is detectable rather than silently
  inconsistent.

---

## 11. Open questions for this review

1. **Is `coverage_pct` count-based (`n_obs / expected_n`) or time-based
   (`observed_time / window_dur_s`)?** The examples assume count-based at 0.5 Hz. If preprocessing
   already emits coverage, this schema carries it through — but `missing_time_s` must be defined
   consistently or the two columns will disagree. *(This is the one item to resolve before anything
   is built.)*
2. **Which columns are already produced by preprocessing vs. new here?** `coverage_pct`, `n_obs`, and
   `{s}_usable` are probably carried; `max_gap_s` and `missing_time_s` are likely new. Worth marking
   each column `carried` / `derived` in the final version.
3. **Are `expected_n` values the same across all four tracks?** If Solar8000 tracks differ in
   sampling rate, `n_obs` is not comparable across signals and only `coverage_pct` is.
4. **Trailing partial windows — keep, or drop?** Proposal is keep-and-flag. Their coverage
   denominator differs.
5. **Should `window_usable` require all 3 core signals, or at least 2?** Proposal is all 3, with
   `n_core_usable` retained so the looser policy stays available without a rebuild.
6. **Should `ref_gap_windows` be capped?** Proposal is uncapped, with the gap exposed as a feature.
   The alternative — nulling deltas beyond, say, 3 windows — is cleaner for the model but discards
   real comparisons. Leaning uncapped.
7. **Is `mbp_present` safe as a model input?** It is largely a proxy for whether an arterial line was
   placed, which correlates with case type and monitoring intensity. As a *data-quality* column it is
   clearly correct; as a *predictive* feature it risks encoding case selection rather than signal
   behavior. Recommend keeping it in the table and excluding it from the model's feature list by
   default.
8. **`TREND_K = 1.0` — right default?** It sets how much movement counts as movement. It is a tunable
   statistical parameter with no clinical meaning, but it does propagate into `{s}_trend`, all
   `run_*` counters, and the entire cross-signal block. Worth a sensitivity check once real windows
   exist.

---

*Design document only. No implementation, no model code, no anomaly detection, no changes to the
preprocessing contract.*
