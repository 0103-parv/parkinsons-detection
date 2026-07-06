# Honest status — every poster claim vs. what the code actually does

This maps the ISEF poster/abstract to the *real* state of this implementation.
The point is that you can walk a judge through this table and every "measured"
row is a number the code will reproduce live. **The abstract as written reports
results that were never measured on any system we have — those rows are marked
`ASPIRATIONAL` and must not be presented as achieved.** Presenting an
un-run result as an achieved result is fabrication, and in a competition it is
the fastest way to be disqualified. The good news: the *defensible* version of
this project is genuinely strong.

Legend:
- **REAL/MEASURED** — implemented; the code prints the number; reproducible here.
- **REAL (synthetic data)** — implemented and measured, but on synthetic gait,
  because we have no access to real labelled patient data. Honest as a *method*
  demo; not a clinical result.
- **ASPIRATIONAL** — claimed in the abstract, NOT implemented/measured. Do not
  present as achieved.

| Poster claim | Status | What is actually true |
|---|---|---|
| Video of someone walking → skeleton | **REAL/MEASURED** | `pose.py` runs MediaPipe BlazePose (33 landmarks) on a real video, no GPU. |
| STTP: graph Laplacian `L = D − A`, Fiedler vector, keep body / drop background | **REAL/MEASURED** | `sttp.py` builds the graph, computes `L`, the Fiedler vector, and selects the body-connected tokens. On a frame with injected background tokens it reports the actual keep-fraction and body-recall. |
| "Preserves 100% of human geometry, prunes 70–80%" | **REAL (measured, not 100%)** | We report the *actual* body-recall and keep-fraction. "100%" is not a real number; the measured recall is high but finite and we print it. |
| LieQ mixed-precision quantization; bits to sensitive layers, crush the rest | **REAL/MEASURED** | `lieq.py` runs the verification-gated bit-allocation search and quantizes a real small model, reporting measured size reduction and accuracy retained. |
| Gait features (cadence, stride time & variability, asymmetry, arm swing, freezing index) | **REAL/MEASURED** | `gaitfeat.py` extracts these with real signal processing (peak-based step detection, freeze-band power ratio). |
| Latency "< 50 ms per frame" | **REAL/MEASURED (~27 ms/frame)** | `eval.py` times the real per-frame cost on this machine: ~26–28 ms/frame for MediaPipe pose extraction on real video (CPU), under the 50 ms target. Regenerate before quoting. |
| Memory "< 4 GB" | **REAL/MEASURED** | The actual pipeline (MediaPipe + sklearn) uses far less than 4 GB; `eval.py` reports resident memory. Note this is not the 14.5 GB→3.2 GB VLM in the abstract — that VLM does not exist here. |
| ">90% Pearson correlation with UPDRS" | **REAL, MEASURED — but ~0.5, NOT >0.90** | Now trained on **real CARE-PD UPDRS-gait labels** with subject-level CV: held-out Pearson **r ≈ 0.53 pooled (0.61 best cohort, PD-GaM)** — see [CAREPD_RESULTS.md](CAREPD_RESULTS.md). The synthetic demo gives r ≈ 0.99, but the real number is ~0.5. The poster's ">0.90" is NOT real. (Joints via canonical-FK approximation, not licensed SMPL.) |
| Out-of-distribution guard | **REAL/MEASURED (new)** | `severity.py` flags inputs far from the training distribution as "out-of-distribution — unreliable" instead of emitting a confident score. Verified: a poorly-framed real clip is correctly flagged; see LIMITATIONS.md. |
| Behavior on real video | **REAL/MEASURED (new)** | Tested on real walking footage. Findings (sim-to-real gap, confounding, OOD, raw-pixel STTP limits) are documented honestly in [LIMITATIONS.md](LIMITATIONS.md). |
| "Adversarial ASR < 10%" vs FigStep / Jailbreak-in-Pieces | **ASPIRATIONAL** | No VLM exists here to attack, and no FigStep/typographic attack suite was run. What IS real: STTP measurably drops background tokens (a robustness *property*), reported quantitatively. Do not claim a VLM ASR number. |
| "Fine-tuned on CARE-PD, 8000+ sequences" | **DONE (real data, ~2953 labelled walks)** | CARE-PD is downloaded and the model is trained on it: 110 UPDRS-labelled subjects, ~2953 walks, subject-level CV (`python -m parkigait carepd-train`). Note it is ~2953 *labelled* walks (not 8000+), joints are canonical-FK approximations (not licensed SMPL), and the honest correlation is ~0.5 (see CAREPD_RESULTS.md). |
| "14.5 GB → 3.2 GB VLM, >90% correlation retained" | **ASPIRATIONAL** | There is no 14.5 GB VLM in this project. The LieQ *mechanism* is real and demoed on a small model; the specific VLM numbers are not. |
| Fairness across Fitzpatrick III–VI | **ASPIRATIONAL** | No skin-tone-stratified evaluation was run (needs real data with those labels). The hook exists in `eval.py` but has no real data to fill it. |
| "Zero data leak / on-device" | **REAL (by construction)** | The whole pipeline runs locally with no network calls. The web demo (`app.py`) serves on localhost only. This privacy claim is architecturally true. |

## The one-paragraph honest pitch (safe to say to a judge)

> "I built a working, fully on-device pipeline that takes a walking video, extracts
> a skeleton with MediaPipe, prunes it to the body manifold using a spectral graph
> method (the Fiedler vector of the pose graph Laplacian), computes clinical gait
> features, and produces an exploratory motor-sign score — with a verification-gated
> mixed-precision quantization search for the edge-deployment angle. On synthetic
> data with known severity I measure [X] correlation, [Y] ms/frame, and [Z] MB. I do
> **not** yet have real UPDRS-labelled data (CARE-PD is gated), so I do not claim a
> clinical correlation; the loader is ready and the same pipeline would train on it.
> Here's exactly what clinical deployment would require." (Then point to CLINICAL_SAFETY.md.)

That version wins on rigor. The overclaimed version loses the moment a judge asks
"show me the CARE-PD training run."

## Fill-in-the-real-numbers

Run `python -m parkigait.eval --report` and it writes the measured numbers into
`RESULTS.md`. Use those, not the abstract's placeholders.
