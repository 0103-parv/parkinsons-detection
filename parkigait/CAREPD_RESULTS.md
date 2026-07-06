# Real CARE-PD results — trained on real Parkinson's UPDRS labels

This is the result the whole project was built toward: the pipeline trained and
evaluated on the **real CARE-PD dataset** with **real clinician-rated UPDRS-gait
labels** and **subject-level cross-validation** (no subject in both train and test).

Reproduce:
```bash
python -m parkigait carepd-train --root data/CARE-PD --joint-source canonical_fk
```

## The numbers (measured, subject-level CV, held-out)

| Cohort | Subjects | Labelled walks | Held-out Pearson r | CV MAE | baseline MAE* |
|---|---|---|---|---|---|
| **PD-GaM** | 30 | 1701 | **0.61** | 0.50 | 0.67 |
| BMCLab | 23 | 781 | 0.38 | 0.55 | 0.68 |
| 3DGait | 43 | 90 | 0.31 | 0.70 | 0.70 |
| T-SDU-PD | 14 | 381 | 0.15 | 0.68 | 0.71 |
| **Pooled** | 110 | 2953 | **0.53** | 0.57 | 0.68 |

\* baseline = MAE of always predicting the mean UPDRS-gait label. Beating it means
the gait features carry real signal about the score.

## Honest reading

- **This is a real, positive result.** On the largest cohort (PD-GaM) the held-out
  correlation with UPDRS-gait is **r ≈ 0.61**, and pooled across 110 subjects it is
  **r ≈ 0.53**, beating the predict-the-mean baseline on 3 of 4 cohorts and pooled.
- **It is far below the synthetic demo (r ≈ 0.99) and the poster's claimed >0.90** —
  which were never real. This ~0.5 is what an honestly-validated gait-only model
  gets on real Parkinson's data. That is the number to report.
- **Two caveats keep it honest:**
  1. **Approximate joints.** We use canonical-skeleton forward kinematics on the
     real SMPL pose rotations (`smpl_fk.py`), NOT the licensed SMPL body model.
     This is stamped `joint_source="canonical_fk"` everywhere. Exact SMPL joints
     (register at smpl.is.tue.mpg.de) would likely improve the number.
  2. **A deliberately simple model.** Seven hand-built gait features + ridge
     regression. Richer temporal features or a learned model would likely do better
     — but the simple model is honest and interpretable.
- **Subject-level splitting is doing real work.** These numbers use no
  subject in both train and test; a naive walk-level split would inflate them
  (some cohorts have 50+ walks per subject), exactly the leakage trap the voice
  track's `detect.py` demonstrates.

## Why the cohorts differ

PD-GaM (r 0.61) is the largest and has a full UPDRS 0–3 range; T-SDU-PD (r 0.15)
has only 14 subjects and a compressed 0–2 range, so there is little variance to
predict and the estimate is noisy. This is expected — small, low-variance cohorts
give unstable correlations.

## What this unlocks for the project

The project can now say, truthfully: *"We trained on real CARE-PD UPDRS-gait labels
with subject-level validation and measured a held-out correlation of ~0.5–0.6.
Here is exactly how (approximate joints, simple features), and here is what would
raise it."* That is a defensible, honest scientific result — and a far stronger
position than an un-measured ">0.90" that collapses under one question.

See [CLINICAL_SAFETY.md](CLINICAL_SAFETY.md): a real correlation is still not a
clinical device. This is a research result, not a diagnostic.
