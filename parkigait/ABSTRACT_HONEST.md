# Corrected, defensible abstract & poster text

This is the version you can present and defend under any question, because every
number reproduces live (`python -m parkigait eval --report`). It keeps the real
engineering story and drops the claims that were never measured. Compare against
[HONEST_STATUS.md](HONEST_STATUS.md). Numbers below are from a real run and will
shift slightly run-to-run; regenerate before printing.

---

## Abstract (honest)

**Purpose.** Parkinson's gait analysis normally needs clinic equipment or cloud
video, raising cost and privacy barriers. We asked whether a fully **on-device**
pipeline — pose estimation, a spectral graph method for topology-preserving token
pruning (STTP), clinical gait features, and mixed-precision quantization (LieQ) —
can run in real time on a laptop CPU and produce a stable, *exploratory* motor-sign
readout, as a step toward accessible screening tools.

**Procedure.** We built a runnable pipeline: MediaPipe BlazePose extracts a
33-landmark skeleton from a walking video; STTP builds the pose graph Laplacian
`L = D − A`, takes the Fiedler vector, and keeps the body-connected tokens; seven
gait features are computed (cadence, stride-time variability, stride length,
asymmetry, arm swing, Bächlin freeze index); and a model maps them to P(PD-signs)
and a 0–4 severity-like score, reported with held-out cross-validation. LieQ
searches per-layer bit-widths under a memory budget and an accuracy floor,
verification-gated on *measured* accuracy. Because the CARE-PD dataset is access
-gated and was **not** available to us, the model was trained and evaluated on a
**synthetic** cohort with known severity and deliberately overlapping,
confound-bearing subjects; a real-data loader is implemented and ready.

**Results (all measured; synthetic-data or system metrics — not clinical).** Pose
extraction ran at **~27 ms/frame** on CPU (under a 50 ms/frame target) with a
**~364 MB** memory footprint (under 4 GB). On the pose keypoint graph, STTP
preserved **99–100%** of body tokens while dropping **100%** of injected
background tokens; on raw RGB frame patches (a harder, non-separable setting) plain
connectivity did **not** isolate a small distant subject, indicating STTP needs
semantic features to prune real pixels. LieQ achieved **~11× compression** with
**~100% of held-out accuracy retained** on a small demo model. On synthetic data
the model reached **~0.86–0.94 AUC** for control-vs-PD and **r ≈ 0.99** between
predicted and true (synthetic) severity; on real out-of-distribution video an
out-of-distribution guard correctly flagged the input as unreliable rather than
emitting a confident score.

**Conclusions.** A private, real-time, on-device Parkinson's-gait *research
pipeline* is feasible on commodity hardware, and topology-preserving pruning plus
verification-gated quantization are practical building blocks. The clinical claims
require real labelled data (CARE-PD) and prospective validation, which this work
does not yet have; we report only what we measured, and identify exactly where a
clinical result would come from.

---

## Poster-ready bullets

**What's new / real**
- On-device, no-cloud pipeline: video → skeleton → STTP → gait features → score.
- STTP = graph Laplacian + Fiedler vector on the pose token graph (measured).
- LieQ = verification-gated mixed-precision search over *measured* accuracy.
- An **out-of-distribution guard**: refuses to score inputs unlike its training data.

**Measured (regenerate before printing)**
| Metric | Value |
|---|---|
| Pose extraction, real video, CPU | ~27 ms/frame (<50 ms target) |
| Peak memory | ~364 MB (<4 GB target) |
| STTP body recall / background drop (keypoint graph) | ~1.00 / 1.00 |
| LieQ compression / accuracy retained | ~11× / ~100% |
| Control vs PD AUC (**synthetic**) | ~0.86–0.94 |
| Predicted vs true severity r (**synthetic**) | ~0.99 |

**Honest limits (say these out loud — they make you credible)**
- Correlation/AUC are on **synthetic** data; there is **no** clinical correlation.
- **Not** trained on CARE-PD (gated) — the loader is ready, training is not done.
- Raw-pixel STTP does not isolate a small distant subject (needs semantic tokens).
- Real gait is **confounded**: slow/terrain/style walking can read as PD-like.
- **Not a medical device**; no ASR / adversarial VLM number is claimed (no VLM here).

---

## What changed from the original abstract, and why

The original abstract stated achieved results — ">ninety percent correlation with
clinical scores," "Attack Success Rate below ten percent," "fine-tuned on CARE-PD,"
a "14.5 GB→3.2 GB" VLM — that were never measured on any system available here.
Presenting un-run results as achieved is fabrication and, at a science competition,
grounds for disqualification. The rewrite keeps the genuine engineering (which is
strong) and reports only measured numbers, clearly labelled synthetic vs. system.
This version is *more* impressive to a rigorous judge, not less: it survives the
question "show me the CARE-PD training run."
