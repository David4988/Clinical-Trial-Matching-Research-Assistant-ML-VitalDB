# Model B Benchmark Report

Isolation Forest, 15 physiological features, contamination 0.1, 200 trees, canonical seed 20260816, 135 analyzed windows, 14 flagged.

> **There is no ground-truth anomaly label in this data.** Nothing in this report is a measure of correctness. No accuracy, precision, recall, F1, AUROC or AUPRC is computed, because none of them are defined here. Agreement between two unsupervised methods is agreement, not validation: two methods wrong in the same way agree perfectly.

Each section separates **OBSERVED RESULTS** — numbers produced by the runs — from **INTERPRETATION** — what they do and do not license us to say.

---

## Benchmark 1 — Seed stability

Model B refit under 5 alternate seeds. Same rows, same 15 features, same contamination, same tree count; only `random_state` differs.

### OBSERVED RESULTS

| seed | windows_flagged | top10_overlap_with_canonical | spearman_rank_correlation | pct_labels_unchanged |
|---|---|---|---|---|
| 20260816 | 14 | 10 | 1.0000 | 100.0000 |
| 1 | 14 | 9 | 0.9948 | 100.0000 |
| 7 | 14 | 10 | 0.9948 | 98.5200 |
| 42 | 14 | 10 | 0.9943 | 100.0000 |
| 1234 | 14 | 10 | 0.9926 | 100.0000 |
| 99991 | 14 | 9 | 0.9913 | 98.5200 |

- Flagged count across seeds: 14–14 (canonical 14).
- Top-10 overlap with canonical: 9–10 of 10 (mean 9.6).
- Spearman rank correlation: 0.9913–0.9948 (mean 0.9936).
- Labels unchanged: 98.52%–100.0% (mean 99.41%).

### INTERPRETATION

The ranking is highly reproducible across seeds (Spearman 0.9913–0.9948), and the flag count is fixed at 14 by the contamination parameter rather than by anything the model discovered, so it cannot vary. 99.41% of labels are unchanged on average, meaning the disagreement is confined to a small number of windows near the decision boundary — exactly where a fixed budget forces an arbitrary cut. This says the PROCEDURE is stable. It says nothing about whether the flagged windows are meaningful, and a seed sweep cannot address that.

---

## Benchmark 2 — Simple statistical baseline

`robust z-score: z = (x - median) / scale, scale = 1.4826*MAD, falling back to IQR/1.349 where MAD is zero; window score = max |z| over features`

Deterministic, no fitting, no randomness. 12 of 15 features contribute; 3 are excluded as degenerate (`spo2_max`, `spo2_delta`, `rr_delta`) because both MAD and IQR are zero — they are effectively constant across the analyzed windows and would otherwise divide by zero.

### OBSERVED RESULTS

- Isolation Forest flagged **14**, baseline flagged **14** (budget matched).
- Flagged by both: **12**. Isolation Forest only: 2. Baseline only: 2.
- Top-10 overlap: **8 of 10**.
- Spearman rank correlation: **0.8802**.

Top 10 by each method:

*Isolation Forest*

| caseid | window_index | anomaly_score | baseline_rank |
|---|---|---|---|
| 4.0000 | 62.0000 | 0.2389 | 1.0000 |
| 4.0000 | 64.0000 | 0.1044 | 3.0000 |
| 4.0000 | 63.0000 | 0.1037 | 11.0000 |
| 8.0000 | 6.0000 | 0.0955 | 2.0000 |
| 4.0000 | 61.0000 | 0.0902 | 4.0000 |
| 4.0000 | 66.0000 | 0.0848 | 10.0000 |
| 4.0000 | 65.0000 | 0.0810 | 5.0000 |
| 4.0000 | 60.0000 | 0.0707 | 7.0000 |
| 8.0000 | 11.0000 | 0.0565 | 20.0000 |
| 8.0000 | 2.0000 | 0.0474 | 9.0000 |

*Robust z-score baseline*

| caseid | window_index | baseline_score | driving_feature | anomaly_rank |
|---|---|---|---|---|
| 4 | 62 | 28.5822 | rr_std | 1 |
| 8 | 6 | 20.2116 | spo2_mean | 4 |
| 4 | 64 | 18.6817 | rr_std | 2 |
| 4 | 61 | 16.2081 | rr_std | 5 |
| 4 | 65 | 15.0890 | rr_std | 7 |
| 8 | 5 | 15.0044 | spo2_mean | 11 |
| 4 | 60 | 13.5411 | rr_std | 8 |
| 4 | 67 | 13.0575 | rr_std | 12 |
| 8 | 2 | 12.3064 | spo2_mean | 10 |
| 4 | 66 | 10.9611 | rr_std | 6 |

Case 4 episode (285–340 min):

- Isolation Forest: 8 of 11 windows → `[60, 61, 62, 63, 64, 65, 66, 67]`
- Baseline: 9 of 11 windows → `[59, 60, 61, 62, 63, 64, 65, 66, 67]`

### INTERPRETATION

A deterministic robust z-score with no model recovers 12 of the 14 Isolation Forest flags and 8 of its top 10, with Spearman 0.8802 across all 135 windows. Both methods identify the case 4 episode. Two unsupervised methods agreeing is not evidence that either is correct — they read the same 15 features, so shared blind spots are expected rather than surprising. What it does establish is that the flags are not an artifact of tree ensembling: a transparent rule reaches largely the same conclusion. Where they differ is informative, because the baseline scores on a single most-extreme feature while the Isolation Forest can combine several moderately unusual ones. Neither is declared superior here; the evidence does not support that claim in either direction.

---

## Benchmark 3 — Temporal coherence

Adjacency: consecutive window_index within the same case; a window that was not analysed breaks a run rather than being assumed continuous.

### OBSERVED RESULTS

- Flagged windows: **14**
- With an adjacent flagged neighbour: **12** (**85.71%** of flags)
- Runs: **4**, lengths `[8, 4, 1, 1]`
- Longest run: **8** windows (40 minutes)
- Isolated single-window flags: **2**

| caseid | windows_analyzed | flagged | with_adjacent_flagged_neighbour | pct_in_contiguous_runs | n_runs | longest_run |
|---|---|---|---|---|---|---|
| 2 | 46 | 0 | 0 | 0.0000 | 0 | 0 |
| 4 | 65 | 8 | 8 | 100.0000 | 1 | 8 |
| 8 | 15 | 6 | 4 | 66.6700 | 3 | 4 |
| 9 | 9 | 0 | 0 | 0.0000 | 0 | 0 |

### INTERPRETATION

85.71% of flags sit adjacent to another flag, in 4 runs, the longest spanning 8 consecutive windows (40 minutes). Independent noise over 14 flags in 135 windows would rarely produce a run that long, so the flags are picking up something with temporal extent rather than firing at random. Two cautions: the model scores each window independently with no temporal features, so clustering is a property of the underlying signal and not of the detector; and a sustained physiological state and a sustained artifact both produce runs, so coherence alone does not distinguish them.

---

## Benchmark 4 — Evidence review

Each flagged window classified by what its flag rests on, using the existing evidence objects. Rules:

- **physiologically_supported** — every core signal at 100% coverage AND at least one evidence flag (mean change or unusual dispersion)
- **mainly_data_quality** — at least one core signal below 100% coverage AND no evidence flag
- **ambiguous** — anything else: imperfect coverage alongside an evidence flag, or full coverage with no evidence flag at all

### OBSERVED RESULTS

- physiologically supported: **13**
- mainly data-quality related: **0**
- ambiguous: **1**

| caseid | window_index | anomaly_rank | min_coverage_pct | evidence_drivers | review |
|---|---|---|---|---|---|
| 4 | 62 | 1 | 100.0000 | hr_change; spo2_dispersion; rr_dispersion | physiologically_supported |
| 4 | 64 | 2 | 100.0000 | rr_dispersion | physiologically_supported |
| 4 | 63 | 3 | 100.0000 | hr_change; rr_dispersion | physiologically_supported |
| 8 | 6 | 4 | 100.0000 | spo2_change | physiologically_supported |
| 4 | 61 | 5 | 100.0000 | rr_dispersion | physiologically_supported |
| 4 | 66 | 6 | 100.0000 | rr_dispersion | physiologically_supported |
| 4 | 65 | 7 | 100.0000 | hr_dispersion; rr_dispersion | physiologically_supported |
| 4 | 60 | 8 | 100.0000 | hr_change; rr_dispersion | physiologically_supported |
| 8 | 11 | 9 | 100.0000 | hr_dispersion | physiologically_supported |
| 8 | 2 | 10 | 100.0000 | spo2_change; rr_change | physiologically_supported |
| 8 | 5 | 11 | 100.0000 | spo2_change | physiologically_supported |
| 4 | 67 | 12 | 95.3300 | rr_dispersion | ambiguous |
| 8 | 7 | 13 | 100.0000 | spo2_change | physiologically_supported |
| 8 | 4 | 14 | 100.0000 | hr_dispersion | physiologically_supported |

### INTERPRETATION

13 of 14 flags rest on signal behaviour at full coverage, 0 on data quality alone, and 1 are ambiguous. This is partly by construction: the coverage ablation removed the features that previously drove data-quality flags, so a low count here reflects that earlier decision working as intended rather than an independent discovery. The category names describe what the evidence points at, not whether the physiology is abnormal.

> 'Physiologically supported' means the flag rests on signal behaviour rather than on data quality. It is NOT a claim that the behaviour is abnormal, clinically meaningful, or medically accurate. No clinical review has taken place.

---

## What this benchmark does not establish

- No ground truth exists, so none of these benchmarks measures correctness. They measure reproducibility, agreement, structure and composition.
- Seed stability and baseline agreement both rest on the same 15 features over the same 135 windows. A blind spot shared by both methods is invisible to this benchmark by construction.
- The flagged count is fixed by contamination = 0.10 in every run, so no benchmark here can tell whether 14 is the right number.
- Four cases, and case 4 supplies nearly half the analyzed windows, so 'unusual' is defined largely by one case.
- Temporal coherence does not distinguish a sustained physiological state from a sustained artifact.
- The evidence review applies a mechanical rule to existing evidence objects. It is not clinical review and involved no clinician.
