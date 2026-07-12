# Parkinson's detection — honest ML on real clinical data

A complete, **genuinely working** Parkinson's detector built on real, public clinical data, with
the methodological rigor that most published pipelines skip. It runs on a laptop — no GPU, no
gated datasets.

> **Scope, stated plainly.** This is an **educational / methods reference on public data** — it is
> **not** a medical device and **not** anyone's competition submission. The point is the
> *methodology*: how to validate a clinical detector honestly so it doesn't fool itself.

## The headline result: validate on *people*, not recordings

The classic UCI Parkinson's voice dataset has ~6 recordings per patient. A naive cross-validation
split puts some of a patient's recordings in **train** and others in **test**, so the model learns
to recognize the *person*, not the *disease* — and reports a beautiful, **dishonest** score. The
honest question is "does it work on a **new** patient?", which requires that no subject's recordings
cross the train/test boundary.

| Evaluation | AUC | What it means |
|---|---|---|
| Record-level CV (naive) | **~0.95** | leaks the patient across train/test — inflated |
| **Subject-level CV (honest)** | **~0.78** | generalizes to a **new** person |
| Inflation from leakage | **+0.17 AUC** | pure false confidence the naive number buys |

Then a **verification-gated search** finds that a tiny 3-feature voice panel (`spread1`,
`MDVP:Fhi`, `D2`) reaches **subject-level AUC ~0.91** — beating all 22 features, which overfit on
only 32 people. And it **replicates** on a second, independent dataset (Sakar et al. 2018, **252
people, 752 features**) at **~0.91**.

## ParkiGait — the video gait pipeline (the ISEF edge-VLM project, made real)

**[`parkigait/`](parkigait/)** is a genuinely working, on-device pipeline: a walking
video → MediaPipe skeleton → STTP topology-preserving token pruning → clinical gait
features → an *exploratory* motor-sign score, plus LieQ mixed-precision quantization,
an out-of-distribution guard, a web app, an ablation/robustness study, and honest
docs. It runs on a laptop CPU (~27 to 90 ms/frame depending on clip, ~364 MB). It is a **research prototype,
not a medical device** — see [`parkigait/HONEST_STATUS.md`](parkigait/HONEST_STATUS.md),
[`parkigait/CLINICAL_SAFETY.md`](parkigait/CLINICAL_SAFETY.md), and
[`parkigait/LIMITATIONS.md`](parkigait/LIMITATIONS.md). Start at
[`parkigait/README.md`](parkigait/README.md).

```bash
python -m venv .venv && source .venv/bin/activate
pip install -r parkigait/requirements.txt
python -m parkigait demo --severity 0.6     # or: scan a real video, serve, eval, ablation
```

## What's here (voice track)

```
parkinsons/      the real-data detector (UCI voice; this is the main project)
  detect.py        subject-level vs record-level CV across 4 models — the leakage gap
  panel_search.py  a gated search for the minimal voice-feature panel
  final_model.py   trains + saves the deployable model; predict(recording) -> P(PD)
  detect_sakar.py  independent replication on the larger Sakar 2018 dataset
  download.py      fetches both datasets
  test_detect.py   smoke tests
  data/parkinsons.data   the UCI dataset (committed, 40 KB, runs offline)
gait/            synthetic-gait illustrations (no real data; methods demos)
  gait_detect.py        a gait-feature detector + minimal-panel search
  gait_quant_policy.py  mixed-precision (LieQ-style) quantization as a gated search
mentat/          vendored verification-gated kernel (Problem/Verdict/Memory/solve) — zero-dep
```

The **`mentat/`** folder is a vendored copy of the kernel from
[github.com/0103-parv/mentat](https://github.com/0103-parv/mentat): nothing becomes a believed
result until a verifier passes it. Here it gates the feature-panel search so the panel is only kept
if it generalizes to held-out **people**.

## Run it

```bash
pip install -r requirements.txt          # numpy, scikit-learn, pandas
python -m parkinsons.download            # fetch the datasets (UCI committed; Sakar fetched)
python -m parkinsons.detect              # the honest report (subject-level vs leaky)
python -m parkinsons.panel_search        # the gated minimal-panel search
python -m parkinsons.final_model         # train + save + predict on a recording
python -m parkinsons.detect_sakar        # independent replication (252 people)
python -m parkinsons.test_detect         # smoke tests
python gait/gait_detect.py               # synthetic-gait detector + panel search
```

## Data

- **UCI Parkinsons** (Little et al., 2007) — 195 sustained-phonation voice recordings, 32 people,
  22 dysphonia features. Committed under `parkinsons/data/` (40 KB), runs offline.
- **UCI Parkinson's Disease Classification** (Sakar et al., 2018) — 756 recordings, 252 people, 752
  features. Fetched by `download.py` (≈5 MB).

Both are public UCI datasets. No private or patient-identifying data is included.

## Honest limits

- The detector is a strong, *honestly-validated* baseline — not a clinical-grade diagnostic.
- The `gait/` files use **synthetic** data to illustrate methods; swap in real gait features and
  the same gated pipeline applies.
- Subject-level AUC ~0.78–0.91 is what survives honest validation; the ~0.95 figures common in the
  literature are usually the leaky ones.
