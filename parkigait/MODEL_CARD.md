# Model Card — ParkiGait abnormal-gait detector

Reported in the spirit of Model Cards (Mitchell et al. 2019) and TRIPOD-AI, so a
clinician or reviewer can assess it quickly. **RESEARCH / SCREENING PROTOTYPE —
NOT A MEDICAL DEVICE.**

## Model details
- **What:** a classifier that takes gait features from a short walking video and
  outputs the probability that gait is impaired (MDS-UPDRS gait item > 0), plus an
  exploratory 0–3 severity estimate.
- **Type:** random forest (400 trees, depth 8) over 25 hand-engineered gait
  features; standardized per training fold. Also a random-forest regressor for severity.
- **Inputs:** 3D joint trajectories (from SMPL pose via canonical-skeleton forward
  kinematics) + pelvis translation → 25 clinical gait features (gait speed, cadence,
  stride length, joint ROM, arm swing, trunk flexion, variability, asymmetry, freeze
  index, double support, regularity).
- **Version / date:** v0.1, July 2026. Code: github.com/0103-parv/parkinsons-detection.

## Intended use
- **Intended:** research; a **screening / triage aid** that flags possible
  parkinsonian gait for a clinician to review; longitudinal gait tracking in studies.
- **Out of scope (do NOT use for):** diagnosis; autonomous or unsupervised care
  decisions; treatment choices; any clinical use without prospective validation,
  ethics approval, and regulatory clearance. Not validated outside CARE-PD.

## Training & evaluation data
- **Dataset:** CARE-PD (public, multi-site 3D-mesh Parkinson's gait archive), 4 cohorts
  with clinician MDS-UPDRS gait ratings: 3DGait, BMCLab, PD-GaM, T-SDU-PD.
- **Size:** 110 patients, ~2953 walking trials with UPDRS-gait labels (0–3).
- **Labels:** clinician-rated MDS-UPDRS gait item, per walk.

## Evaluation protocol (this is the part that matters)
- **Subject-level 5-fold cross-validation** — no patient appears in both train and
  test. Features standardized on the training fold only.
- **95% confidence intervals** by patient-level bootstrap (2000 resamples; the patient
  is the resampling unit).
- **Permutation test:** labels shuffled and the full CV re-run; the observed AUC is
  compared to the shuffled distribution.
- **Leakage independently audited** (a separate adversarial review reproduced the
  numbers, checked subject splitting / preprocessing / feature independence, and ruled
  out a cohort-mean confound).

## Quantitative performance (subject-level CV; see CLINICAL_EVAL.md)
- **Abnormal-gait detection AUC 0.86 (95% CI 0.82–0.90).**
- Gait-speed-only baseline AUC 0.84 → the full model adds ~0.02 (gait speed is the
  dominant biomarker; other features add a small increment).
- **Screening operating point (~90% sensitivity):** sensitivity 0.90 (0.85–0.94),
  specificity 0.59 (0.47–0.69), PPV 0.75, NPV 0.81, LR+ 2.2, LR− 0.17.
- **Sensitivity by severity:** UPDRS-gait 1 = 87%, 2 = 96%, 3 = 100% (it catches mild
  cases, not only obvious ones).
- **Severity regression:** Pearson r 0.70 (95% CI 0.61–0.77).
- **Permutation test:** p < 0.005 (200 label shuffles; 0/200 reached the observed
  AUC; max shuffled AUC 0.53) — the detection result is highly unlikely by chance.

## Interpretability
Permutation feature importance is led by **gait speed**, then stride length, elbow/hip
ROM, and double support — the biomarkers clinicians already use for parkinsonian gait.
The model's reasoning matches clinical knowledge rather than opaque artifacts.

## Subgroups / fairness
Per-cohort AUC ranges 0.79–0.87. A stratified fairness analysis across skin tone, age,
sex, and walking aids is **not yet done** (needs those labels) and is planned before
any clinical use.

## Ethical considerations & caveats
- Joints are a **canonical-skeleton approximation**, not the licensed SMPL body model;
  exact joints are expected to improve accuracy.
- Real gait is **confounded** (terrain, speed, footwear, camera angle can mimic PD);
  the output is a flag for review, not a verdict.
- Cohorts are Parkinson's cohorts, so "abnormal gait" = UPDRS-gait > 0, a screening
  signal, not a standalone PD diagnosis.
- Validated on one public dataset only; no prospective clinical study yet.

See CLINICAL_SAFETY.md (what real deployment requires) and LIMITATIONS.md (findings).
