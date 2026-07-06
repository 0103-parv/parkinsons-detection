# Real CARE-PD results — trained on real Parkinson's UPDRS labels

This is the result the whole project was built toward: the pipeline trained and
evaluated on the **real CARE-PD dataset** with **real clinician-rated UPDRS-gait
labels** and **subject-level cross-validation** (no subject in both train and test).

Reproduce:
```bash
python -m parkigait carepd-rich    # rich 25-feature clinical model (best)
python -m parkigait carepd-train   # simple 7-feature baseline
```

## The numbers (measured, subject-level CV, held-out)

Two feature sets, both evaluated identically (subject-level GroupKFold, per-fold
standardization). The **rich clinical features** (25 biomarkers from the 3D SMPL
pose + real gait speed from the pelvis translation) substantially beat the simple
7-feature 2D baseline:

| Cohort | Walks | Simple (7 feats) r | **Rich (25 feats) r** | rich r² |
|---|---|---|---|---|
| **PD-GaM** | 1701 | 0.61 | **0.75** | 0.57 |
| 3DGait | 90 | 0.31 | **0.69** | 0.48 |
| BMCLab | 781 | 0.38 | **0.61** | 0.37 |
| T-SDU-PD | 381 | 0.15 | **0.36** | 0.13 |
| **Pooled** | 2953 | 0.53 | **0.70** | **0.49** |

**Pooled metrics (random forest, subject-level CV):** held-out **r = 0.698, r² = 0.49,
R² = 0.49**, MAE 0.434 vs. predict-mean baseline 0.682. Ridge 0.674, grad-boost 0.695.

**Subject-aggregated (rank patients by severity):** averaging a subject's held-out
predictions gives **r = 0.76–0.78** (r² ≈ 0.58–0.61; grad-boost 0.778) — a legitimate,
higher subject-level framing that denoises walk-to-walk variation. Still not leakage:
each subject's predictions come only from folds where that subject was held out.

### The signal ceiling (why it doesn't go to 0.90)

We tried to push it higher and it **plateaus at ~0.70**: adding 12 more biomechanical
features (37 total) did **not** beat the base 25 (best 37-feature r = 0.692). That is
the honest ceiling for gait-only UPDRS prediction with these (approximate) joints — in
line with published gait-vs-UPDRS studies. Getting meaningfully past it needs exact
SMPL joints (licensed) or clinical inputs beyond gait, **not** feature-tweaking.

### The leakage check (this is what makes it trustworthy)

**Permutation control: shuffle the labels, re-run the whole subject-level CV → held-
out r = +0.05** (≈ 0). If the pipeline were leaking (e.g. a subject in both train and
test, or fitting the scaler on test data), shuffled labels would still score high.
They don't — so the r ≈ 0.70 is real generalization signal, not an artifact.

**Robustness cross-check.** The *dishonest* walk-level split (subjects leaking across
folds) scores only r = 0.73 — barely above the honest subject-level 0.70. A model that
was memorizing subjects would show a large gap; a ~0.03 gap means the signal genuinely
generalizes to new people. (An independent adversarial audit reproduced every number,
ran the permutation control over 10 seeds, and empirically ruled out a cohort-mean
confound — a cohort-only predictor scores just 0.075 under subject-level CV.)

## Honest reading

- **This is a strong, real result:** held-out **r ≈ 0.70 pooled, 0.75 on the largest
  cohort (PD-GaM)**, correlating a gait-only model with clinician-rated UPDRS-gait,
  with subject-level validation and a passing permutation control. That is in the
  range published gait-vs-UPDRS studies report.
- **It is still NOT >0.90.** r ≈ 0.70 is the honest ceiling of this approach; the
  poster's >0.90 and the synthetic demo's 0.99 are not real numbers. Report 0.70.
- The features are clinically grounded and directional per the PD literature
  (gait speed 0.91→0.08 across UPDRS 0→3; trunk flexion 0.18→0.47; foot clearance
  and knee ROM collapse at severe; freezing index rises).
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
