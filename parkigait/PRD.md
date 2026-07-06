# ParkiGait — Product Requirements Document

**Status:** working research prototype · **Owner:** Parv Mehndiratta (+ Adyuth Chirakala, Rohan Dravid)
**One line:** an on-device, privacy-preserving tool that screens a short walking video for parkinsonian gait, as a research and triage aid.

> **Scope guardrail (read first):** ParkiGait is a **research / screening prototype, not a
> medical device.** It flags *possible* gait impairment for clinician review. It does not
> diagnose, and it is not for autonomous care decisions. See `CLINICAL_SAFETY.md`.

---

## 1. Problem & opportunity

Parkinson's gait changes (slowness, short shuffling steps, reduced arm swing, stooped
posture, freezing) are early and measurable, but assessing them needs a specialist and a
clinic visit. Cloud-based video AI raises privacy/HIPAA concerns. There is an opportunity
for a **private, low-cost, on-device gait screen** that runs on a phone or laptop and helps
route the right patients to a neurologist sooner, and helps track gait over time.

## 2. Goals / non-goals

**Goals**
- Screen a short walking video and output an interpretable gait report + a "refer for exam" flag.
- Run fully on-device (no video leaves the machine).
- Be honest and reproducible: every metric measured, subject-level validated, leakage-checked.

**Non-goals (v1)**
- Not a diagnosis. Not a replacement for a clinician or the MDS-UPDRS exam.
- Not a regulated device yet. No autonomous treatment or triage decisions.
- Not a general fall-risk or non-Parkinsonian gait tool (future work).

## 3. Users

1. **Clinical researchers / movement-disorder labs** — validate and study gait biomarkers.
2. **Neurologists / PTs (as a screening aid)** — a flag + report to support, not replace, exam.
3. **The project team** — reproducible research artifact for the science-fair / STS write-up.

## 4. Functional requirements

| # | Requirement | Status |
|---|---|---|
| F1 | Ingest a walking video (phone/webcam), extract a 3D-ish skeleton on-device | ✅ (MediaPipe BlazePose) |
| F2 | Compute clinically-grounded gait features (speed, cadence, stride, ROM, asymmetry, trunk flexion, freeze index, variability) | ✅ (25 features) |
| F3 | Output an **abnormal-gait detection** flag with a probability | ✅ (AUC ~0.86, CARE-PD) |
| F4 | Output an exploratory **severity** estimate (0–4-like) | ✅ (r ~0.70, CARE-PD) |
| F5 | Refuse to score out-of-distribution / low-quality input | ✅ (OOD guard) |
| F6 | Human-readable report + figures; every claim carries the disclaimer | ✅ (CLI + web app) |
| F7 | Train/evaluate on real labelled data with **subject-level** splits + leakage controls | ✅ (CARE-PD) |
| F8 | Exact SMPL joints (licensed body model) for higher accuracy | ⏳ needs SMPL license |
| F9 | Prospective clinical validation on partner-site data | ⏳ needs IRB + partner |

## 5. Non-functional requirements

- **Privacy:** on-device only; no network calls; uploads never retained (web app deletes after analysis).
- **Latency:** < 50 ms/frame pose extraction on CPU. *(measured ~27 ms/frame)*
- **Footprint:** < 4 GB RAM. *(measured ~364 MB)*
- **Reproducibility:** one command regenerates every number; permutation controls shipped.
- **Honesty:** claims mapped measured-vs-aspirational in `HONEST_STATUS.md`.

## 6. Current measured status (all real, subject-level CV on CARE-PD)

| Metric | Value |
|---|---|
| Abnormal-gait detection AUC | **0.86 (95% CI 0.82–0.90)**; screening point 90% sens / 59% spec |
| Sensitivity by severity | mild (UPDRS 1) 87%, moderate 96%, severe 100% |
| Severity correlation (UPDRS-gait) | r ≈ 0.70 (95% CI 0.61–0.77), r² ≈ 0.49 |
| Leakage check | permutation AUC ≈ 0.5 / r 0.05; independent adversarial audit: pass |
| Latency / memory | ~27 ms/frame, ~364 MB, CPU-only |
| Training data | CARE-PD, 110 UPDRS-labelled patients, ~2953 walks |

Full clinical reporting (ROC, calibration, CIs, feature importance) in
[`CLINICAL_EVAL.md`](CLINICAL_EVAL.md) and [`MODEL_CARD.md`](MODEL_CARD.md).

## 7. Roadmap

- **M0 (done):** working pipeline; trained + validated on CARE-PD; honest docs; web app.
- **M1:** exact SMPL joints (licensed model) → expect higher AUC/correlation.
- **M2:** clinical partner + **IRB-approved prospective pilot** (de-identified gait videos from the partner site; compare the flag to the clinician's UPDRS-gait).
- **M3:** external validation on a second site; fairness audit across skin tone / age / aids.
- **M4:** regulatory scoping as Software as a Medical Device (CDSCO / FDA) if pursuing clinical use.

## 8. Risks & mitigations

| Risk | Mitigation |
|---|---|
| Over-trust / used as a diagnosis | Hard disclaimers everywhere; OOD guard; "screening aid" framing; clinician-in-the-loop |
| Confounding (terrain/speed/style mimic PD) | Documented in `LIMITATIONS.md`; report as flag not verdict |
| Dataset bias (site, demographics) | Subject-level validation; planned fairness audit (M3) |
| Approximate joints | Disclosed; M1 exact SMPL |
| Regulatory / IRB | No patient use until M2+; explicit path in `CLINICAL_SAFETY.md` |

## 9. What a hospital partnership looks like (near-term)

Not deployment. A **research collaboration**: the clinician provides expertise and, under IRB
approval, a small set of **de-identified** gait videos with UPDRS-gait scores; we measure how
well the flag agrees with their assessment, publish/present the honest result, and iterate. This
is the credible first step and the basis for anything clinical later.
