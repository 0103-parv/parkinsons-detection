# CARE-PD training runbook — ParkiGait severity model

**RESEARCH / EDUCATIONAL PROTOTYPE. NOT A MEDICAL DEVICE. NOT FOR CLINICAL USE.**
Training on real CARE-PD labels does **not** make this model clinical. Read
[`../parkigait/CLINICAL_SAFETY.md`](../parkigait/CLINICAL_SAFETY.md) and
[`../parkigait/HONEST_STATUS.md`](../parkigait/HONEST_STATUS.md) before and after you
run anything here.

This is the concrete, command-by-command path from "access granted" to an honest
held-out number. Everything below assumes:

- working dir `/Users/parvmehndiratta/parkinsons-detection`
- venv python at `.venv/bin/python`
- the loader `parkigait/carepd.py` and the trainer `parkigait/severity.py` unchanged

Anything I could not verify against the live dataset card on 2026-07-05 is tagged
**VERIFY against the dataset card** — gating and licensing policies change, so
re-check before you rely on it.

---

## 0. Provenance (what CARE-PD is, from the code + the live sources)

- **Dataset card:** https://huggingface.co/datasets/vida-adl/CARE-PD (`repo_type=dataset`)
- **Code / getting-started:** https://github.com/TaatiTeam/CARE-PD
- **Terms of use:** https://neurips2025.care-pd.ca/terms-of-use
- **Paper (cite this):** CARE-PD: A Multi-Site Anonymized Clinical Dataset for
  Parkinson's Disease Gait Assessment, NeurIPS 2025 D&B, arXiv:2510.04312
- **License (VERIFY):** the *current dataset card* states **CC BY-NC-ND 4.0**
  (Attribution-NonCommercial-**NoDerivatives**). Note: `carepd.py` hard-codes
  `CC BY-NC 4.0` in `CAREPD_LICENSE` — treat the **card** as authoritative and
  respect the ND (no-derivatives) clause if that is what it says today.
- Multi-site: 9 harmonized cohorts, 8 clinical centres, ~362 participants. Each
  walk is an anonymized **SMPL** body-mesh sequence; ~1/3 of walks carry a
  clinician-rated `UPDRS_GAIT` score (the gait item of the MDS-UPDRS motor exam).

You can print the loader's own account of all this at any time (works with **no**
data present — it never fabricates):

```bash
cd /Users/parvmehndiratta/parkinsons-detection
.venv/bin/python -c "from parkigait.carepd import CAREPDDataset; print(CAREPDDataset('/nonexistent').describe())"
```

---

## 1. Request access (the dataset is gated on Hugging Face)

CARE-PD is distributed through a **gated** Hugging Face dataset repo. You cannot
download the pickles until you are logged in and have accepted the terms on the
dataset card. (**VERIFY against the dataset card** — the automated page fetch on
2026-07-05 could not confirm the gate button behind the login wall; both the paper
and `carepd.py` describe it as gated, so assume a gate and follow the steps.)

1. Create / log in to a Hugging Face account: https://huggingface.co/join
2. Open the dataset card while logged in: https://huggingface.co/datasets/vida-adl/CARE-PD
3. **Read the terms of use in full** before agreeing:
   https://neurips2025.care-pd.ca/terms-of-use — these are a data-use agreement.
   Core obligations (from the card / terms, VERIFY):
   - Attribute and cite **both** the CARE-PD paper **and** the original cohort
     papers for every cohort you touch.
   - **Never** attempt to re-identify, contact, or de-anonymize any participant.
   - Comply with GDPR / HIPAA / PIPEDA and keep the data secure; do not re-host.
   - **Non-commercial** use only (and honor the **ND** clause if present today).
4. On the card, click the access / "Agree and access repository" control and
   accept. Some gated repos grant instantly; others require author approval and
   an email confirmation. **VERIFY the exact flow on the card** — if it is
   approval-gated, wait for the grant before step 2 of the download below.
5. Create a HF access token with **read** scope:
   https://huggingface.co/settings/tokens

Authenticate the CLI once the token exists:

```bash
cd /Users/parvmehndiratta/parkinsons-detection
.venv/bin/pip install -U "huggingface_hub[cli]"
.venv/bin/huggingface-cli login    # paste the read token when prompted
```

---

## 2. Download the data and point `CAREPDDataset` at it

The verified command from the CARE-PD GitHub getting-started downloads the whole
dataset repo (the 9 per-cohort `*.pkl` files plus `folds/` and
`Canonicalized_SMPL_pickles/`). Pick a local root and stick with it:

```bash
export CAREPD_ROOT="$HOME/data/care-pd"          # any writable path you like
mkdir -p "$CAREPD_ROOT"

.venv/bin/huggingface-cli download vida-adl/CARE-PD \
  --repo-type dataset \
  --local-dir "$CAREPD_ROOT"
```

What you should get at `$CAREPD_ROOT` (VERIFY against the card — sizes as seen on
the tree on 2026-07-05):

```
3DGait.pkl (8 MB)  BMCLab.pkl (127 MB)  DNE.pkl (24 MB)  E-LC.pkl (438 MB)
KUL-DT-T.pkl (117 MB)  PD-GaM.pkl (85 MB)  T-LTC.pkl (107 MB)
T-SDU-PD.pkl (27 MB)  T-SDU.pkl (186 MB)
Canonicalized_SMPL_pickles/    folds/    README.md
```

The loader auto-discovers `*.pkl` at the root **or** under an `assets/datasets/`
subtree (`CAREPDDataset._probe`), so either layout works. Confirm the loader now
sees the data (this reads no poses — pure availability probe):

```bash
.venv/bin/python - <<'PY'
import os
from parkigait.carepd import CAREPDDataset
ds = CAREPDDataset(os.environ["CAREPD_ROOT"])
print("available:", ds.is_available)
print(ds.describe())
PY
```

You want `available: True` and a "AVAILABLE at ... (N pickle file(s) found)"
status. If it says NOT available, the message tells you exactly why (wrong path /
no pickles) — fix that before going on. **It will never quietly fall back to
synthetic data.**

### 2a. SMPL body model — the real gate on producing joints

`carepd.py` stores SMPL params, not keypoints. To turn `(pose, trans, beta)` into
3D joints it needs the **SMPL body model** (separately licensed) and `smplx`:

```bash
.venv/bin/pip install smplx torch
# then obtain SMPL model weights from https://smpl.is.tue.mpg.de (separate license)
```

**Known limitation (do not skip):** even with `smplx` installed,
`_smpl_pose_to_joints()` in `carepd.py` **deliberately raises**
`CAREPDNotAvailable` — the smplx forward pass is left un-wired on purpose so no
run emits joints from an unverified SMPL configuration. Before any real training
you must:

1. **VERIFY the SMPL variant per cohort** against the dataset card (SMPL vs
   SMPL+H vs SMPL-X — the joint count shifts indices), and
2. wire the smplx forward pass into `_smpl_pose_to_joints` so it returns
   `(T, 24, 3)` joints in the documented SMPL joint order.

That wiring is a code change and is **out of scope for this runbook** (the task
here is docs-only), but nothing downstream can run until it is done. Track it as
the first real-data engineering task.

---

## 3. Extract features + labels and train with SUBJECT-LEVEL groups

There are two honest ways to get a held-out number. Use the first for a quick,
self-contained regression baseline; use the second when you want the full
two-head (`clf` + `reg`) model that the rest of ParkiGait consumes.

### 3a. The built-in one-shot path (`train_severity_from_carepd`)

This is the batteries-included entry point. It loads only `UPDRS_GAIT`-labelled
walks (`require_updrs=True`), groups walks **by subject**, builds subject-level
K-folds itself, fits a `Ridge` regressor per fold, and returns cross-validated
MAE on the real gait scores:

```bash
.venv/bin/python - <<'PY'
import os, json
from parkigait.carepd import train_severity_from_carepd
res = train_severity_from_carepd(os.environ["CAREPD_ROOT"], n_splits=5, seed=0)
print(json.dumps({k: v for k, v in res.items() if k != "model"}, indent=2))
PY
```

You get back `n_subjects`, `n_labelled_walks`, `cv_mae_updrs_gait`,
`cv_mae_per_fold`, and `split: "subject-level (no subject in both train and
test)"`.

> **Bug to fix first (real API mismatch):** `train_severity_from_carepd` calls
> `from parkigait.gaitfeat import extract_gait_features`, but the actual function
> in `gaitfeat.py` is **`extract_features`** — there is no `extract_gait_features`.
> As written this import fails and the function raises `CAREPDNotAvailable`
> ("the gait-feature extractor could not be imported"). Fix the import name in
> `carepd.py` (or add an alias) before this path can run. It is a one-line wiring
> error, not fabricated data.

### 3b. The full two-head model (`severity.train_from_features`)

Use this to fit the shipped `SeverityModel` (classifier + Ridge regressor) on
real labels. You build the arrays yourself so you control the labels and the
`groups`. Signature (from `severity.py`):

```python
train_from_features(X, y_label, y_sev, calibrated_on, groups=None, seed=0)
#   X:              (n, 7)  features in GAIT_FEATURE_ORDER  (feats.as_vector())
#   y_label:        (n,)    0/1 PD-sign label
#   y_sev:          (n,)    continuous severity (UPDRS_GAIT, 0..4-ish)
#   calibrated_on:  str     -> stamp the real dataset name, e.g. "CARE-PD (UPDRS_GAIT)"
#   groups:         (n,)    subject ids  <-- PASS THIS for subject-level CV
```

Collect features/labels/groups by iterating the loader, then call it. (This uses
`extract_features` — the correct name — so it side-steps the 3a bug.)

```bash
.venv/bin/python - <<'PY'
import os, numpy as np
from parkigait.carepd import CAREPDDataset
from parkigait.gaitfeat import extract_features
from parkigait.severity import train_from_features

ds = CAREPDDataset(os.environ["CAREPD_ROOT"], require_updrs=True)

X, y_sev, groups = [], [], []
for pose_seq, updrs, meta in ds.iter_sequences():   # yields only labelled walks
    if updrs is None:
        continue
    X.append(extract_features(pose_seq).as_vector())
    y_sev.append(float(updrs))
    groups.append(meta["subject_id"])               # <-- subject id per walk

X = np.asarray(X, dtype=np.float64)
y_sev = np.asarray(y_sev, dtype=np.float64)
groups = np.asarray(groups)

# Define the binary PD-sign label from the gait score however your protocol
# specifies (VERIFY against the dataset card / your study design). A common,
# defensible choice: UPDRS_GAIT >= 1 counts as a motor sign. Document your cut.
y_label = (y_sev >= 1.0).astype(int)

model, cv = train_from_features(
    X, y_label, y_sev,
    calibrated_on="CARE-PD (real UPDRS_GAIT, subject-level CV)",
    groups=groups,                                  # <-- the leakage firewall
    seed=0,
)
print("CV kind:            ", cv["cv_kind"])         # want "subject-level"
print("held-out AUC:       ", cv["auc_mean"])
print("held-out accuracy:  ", cv["acc_mean"])
print("severity Pearson r: ", cv["severity_pearson_mean"])
print("n:                  ", cv["n"])
print(cv["note"])
# model.save()   # persist only if you intend to reuse this REAL-calibrated model
PY
```

### WHY subject-level splits are non-negotiable

Each CARE-PD subject contributes **many** walks. If you split at the *walk* level,
the same person's walks land in both train and test. The model then "recognizes
the person," not the disease sign, and the held-out score is inflated and
meaningless — you are grading the model on data it effectively already saw.

This is the exact leakage lesson called out across the repo. `CLINICAL_SAFETY.md`
(step 2 of "what real deployment requires") points at the **voice track's**
`detect.py` for the same rule: *"strict subject-level splits (no patient in both
train and test) ... naive splits inflate scores."* Both real-data paths above
enforce it:

- `train_severity_from_carepd` partitions **subjects** (never walks) across folds.
- `train_from_features` uses `GroupKFold(groups=subject_id)` and reports
  `cv_kind: "subject-level"` **only** when you pass `groups` with >= 3 unique
  subjects; without groups it silently degrades to `record-level` KFold — so
  **always pass `groups`**, and confirm the printed `cv_kind` is `subject-level`.

If `cv_kind` comes back `record-level`, stop — your number is not trustworthy.

---

## 4. Read the honest held-out metrics

Report the **held-out cross-validated** numbers only — never a train score:

- **3a path:** `cv_mae_updrs_gait` (mean absolute error, in UPDRS-gait points).
  Lower is better; on a 0..3/0..4 scale a real MAE will be well above 0.
- **3b path:** `auc_mean`, `acc_mean` (classifier), and `severity_pearson_mean`
  (predicted vs real UPDRS_GAIT). Confirm `cv_kind == "subject-level"`.

### What a believable real number looks like

The shipped model is trained on the **synthetic** cohort, where severity is a
control we dialed in, so it reports a near-perfect fit (Pearson ~0.99). **That is
a method demo, not clinical performance** (`HONEST_STATUS.md`: the ">90% Pearson
with UPDRS" row is "REAL (synthetic data ONLY)").

On **real** CARE-PD, expect the number to be **substantially lower** — real gait
is noisy, multi-site (different cameras/rooms), and the SMPL->BlazePose remap
plus derived heel/foot joints add slack. A held-out Pearson in a moderate range
(well below 0.99) and a nonzero MAE is the **honest, expected** result, not a
failure. A real run that *still* reports ~0.99 is a red flag for leakage — check
that the split was subject-level and that subjects did not bleed across folds.

Frame it exactly as `HONEST_STATUS.md` prescribes: quote the measured real number,
say it is exploratory, and do **not** dress a synthetic result up as a clinical
correlation.

---

## 5. Hard limits before any clinical use

Training on real UPDRS labels produces an **exploratory research result** and
nothing more. Before this could touch a real patient, per
[`../parkigait/CLINICAL_SAFETY.md`](../parkigait/CLINICAL_SAFETY.md):

1. **Data-use agreement** honored (you accepted CARE-PD's terms in step 1);
   ideally a partnership to collect data from the actual clinics/cameras/patients
   that would use it.
2. **Training + internal validation** on real UPDRS with strict **subject-level**
   splits (done above — that is necessary, not sufficient).
3. **External validation** on an independent site the model never trained on
   (CARE-PD's multi-site structure makes leave-one-cohort-out the natural test).
4. **Prospective clinical study** with a pre-registered protocol, powered for
   sensitivity/specificity vs a gold standard, plus **fairness audits** across
   skin tone (Fitzpatrick), age, sex, and mobility aids.
5. **IRB / ethics approval and informed consent** for every study.
6. **Regulatory clearance** as Software as a Medical Device — CDSCO (India) /
   FDA (US) / MDR (EU) as applicable — with a quality system, ISO 14971 risk
   management, and a clinical evaluation report.
7. **Human-in-the-loop** framing even then: a decision-support aid for a
   clinician, never an autonomous diagnostician.

If anyone asks "can a hospital use this once it is trained on CARE-PD?" the
answer is still **no**, and the checklist above is what it would take.

---

## Quick reference — commands

```bash
cd /Users/parvmehndiratta/parkinsons-detection
export CAREPD_ROOT="$HOME/data/care-pd"

# access (after accepting terms on the gated HF card)
.venv/bin/pip install -U "huggingface_hub[cli]"
.venv/bin/huggingface-cli login

# download
.venv/bin/huggingface-cli download vida-adl/CARE-PD \
  --repo-type dataset --local-dir "$CAREPD_ROOT"

# confirm the loader sees it (reads no poses)
.venv/bin/python -c "import os; from parkigait.carepd import CAREPDDataset; d=CAREPDDataset(os.environ['CAREPD_ROOT']); print('available:', d.is_available); print(d.describe())"

# SMPL model (required to produce joints; forward pass still needs wiring in code)
.venv/bin/pip install smplx torch
```

**Pre-flight code fixes this runbook depends on (both in `parkigait/carepd.py`):**
1. Wire the smplx forward pass in `_smpl_pose_to_joints` for the correct
   per-cohort SMPL variant (currently raises by design).
2. Fix the feature-extractor import in `train_severity_from_carepd`:
   `extract_gait_features` -> `extract_features` (the real name in `gaitfeat.py`).
