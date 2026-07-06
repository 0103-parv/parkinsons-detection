# ParkiGait — clinical evaluation report

Real **CARE-PD** data: 2953 walking trials from 110 patients with
clinician-rated MDS-UPDRS gait scores. **Subject-level cross-validation** (no patient
in both train and test). 95% confidence intervals by **patient-level bootstrap**
(2000 resamples). **RESEARCH / SCREENING PROTOTYPE — NOT A MEDICAL DEVICE.**

## Primary result — abnormal-gait detection

| Metric | Value (95% CI) |
|---|---|
| **AUC** | **0.863 (0.819–0.902)** |
| Gait-speed-only AUC (baseline) | 0.843 (0.789–0.889) |
| Permutation test (200 label shuffles) | **p < 0.005** (0/200 shuffles ≥ observed; max shuffled AUC 0.53) |

**Gait speed is the dominant biomarker:** it alone reaches AUC 0.84, and
the full 25-feature model adds only **+0.020** (the CIs overlap, so
this increment is modest, not a large independent gain). That the model leans on gait
speed is clinically sensible.

## Screening operating point (threshold 0.30, tuned for ~90% sensitivity)

| Metric | Value (95% CI) |
|---|---|
| Sensitivity | 0.90 (0.85–0.94) |
| Specificity | 0.59 (0.47–0.69) |
| PPV | 0.75 (0.67–0.82) |
| NPV | 0.81 (0.73–0.89) |
| LR+ / LR− | 2.2 / 0.17 |

A screening tool is tuned for high sensitivity (catch most impaired gait, accept more
false positives, which a clinician then rules out). The default 0.5 threshold gives
sensitivity 0.79 / specificity 0.79.

## Sensitivity by true severity (does it catch MILD cases?)

| True severity | % flagged |
|---|---|
| UPDRS-gait 1 | 87% |
| UPDRS-gait 2 | 96% |
| UPDRS-gait 3 | 100% |

Catching mild (UPDRS-gait 1) cases is the hard, valuable part; severe cases are easy.

## Severity regression (secondary)

Pearson r **0.698 (0.612–0.771)** between predicted and clinician
UPDRS-gait (the harder continuous task).

## What drives the model (top features)

| Feature | AUC drop when shuffled |
|---|---|
| gait_speed | +0.175 |
| stride_len | +0.017 |
| elbow_rom | +0.007 |
| double_support | +0.004 |
| hip_rom | +0.002 |
| hip_rom_asym | +0.002 |
| stride_time_cv | +0.001 |
| arm_swing | +0.001 |

These are the biomarkers clinicians already use (gait speed, arm swing, stride,
posture), which is a good sign: the model's reasoning matches clinical knowledge.

## Honest limits

Joints are a canonical-skeleton approximation (not the licensed SMPL body model);
validation is on one public multi-site dataset, not a prospective clinical study; the
cohorts are Parkinson's cohorts, so "abnormal gait" here means UPDRS-gait > 0, a
screening signal, not a standalone diagnosis. See CLINICAL_SAFETY.md and LIMITATIONS.md.

Figure: `figures/clinical_eval.png` (ROC, calibration, feature importance, sensitivity
by severity). Reproduce: `python -m parkigait.clinical_eval --permute`.
