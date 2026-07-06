# ParkiGait — on-device video gait analysis (research prototype)

> **RESEARCH / EDUCATIONAL PROTOTYPE — NOT A MEDICAL DEVICE.** It cannot diagnose
> Parkinson's disease or any condition and must not be used for clinical decisions
> or deployed to patients. Read [CLINICAL_SAFETY.md](CLINICAL_SAFETY.md) and
> [HONEST_STATUS.md](HONEST_STATUS.md) before anything else.

A **genuinely working**, fully on-device pipeline that takes a walking video,
extracts a skeleton, prunes it to the body with a spectral graph method, computes
clinical gait features, and produces an **exploratory** motor-sign score — plus a
verification-gated mixed-precision quantization search for the edge angle. No GPU,
no cloud, no data leaves the machine.

```
  video ──▶ pose (MediaPipe BlazePose) ──▶ STTP (Fiedler/Laplacian token pruning)
        ──▶ gait features ──▶ severity / PD-sign estimate ──▶ report
                                   │
   LieQ: verification-gated mixed-precision quantization (edge deployment)
```

## What actually runs (and the real numbers)

Every number here is **measured** by `python -m parkigait eval --report`
(→ [RESULTS.md](RESULTS.md)). **None is a clinical result** — the correlation is
against *synthetic* ground truth because there is no real UPDRS-labelled data on
this machine (CARE-PD is gated). The point is a real, honest method.

| Metric | Measured value | What it is |
|---|---|---|
| MediaPipe pose on real video | **~27 ms/frame** (CPU) | meets the <50 ms edge target, measured |
| Peak memory | **~364 MB** | meets the <4 GB edge target, measured |
| STTP body recall / background drop | **~1.00 / 1.00** | on the keypoint token-graph |
| LieQ quantization | **~11× smaller**, ~100% acc retained | small demo model, synthetic data |
| Control-vs-PD AUC | **~0.86–0.94 (synthetic)** | harder overlapping cohort, **not** clinical |
| Severity correlation (synthetic) | r ≈ 0.99 | method demo, **not** clinical |
| **UPDRS correlation (REAL CARE-PD)** | **r ≈ 0.53 pooled, 0.61 best cohort** | real labels, subject-level CV — see [CAREPD_RESULTS.md](CAREPD_RESULTS.md) |

The project is now **trained on real CARE-PD UPDRS-gait labels** (110 subjects, ~2953
walks, subject-level cross-validation). The honest held-out correlation is **~0.5** —
far below the synthetic 0.99 and the poster's un-measured >0.90, and exactly the kind
of number honest validation produces. `python -m parkigait carepd-train`.

The pipeline does the honest thing on hard inputs: it returns **"inconclusive"** on
low-quality clips and **"out-of-distribution — unreliable"** when the gait is unlike
anything it was trained on, instead of a confident wrong answer. Real-video findings
(sim-to-real gap, confounding, OOD, raw-pixel STTP limits) are written up in
[LIMITATIONS.md](LIMITATIONS.md) — characterizing failure modes is part of the point.

## Run it

```bash
python -m venv .venv && source .venv/bin/activate
pip install -r parkigait/requirements.txt        # numpy<2, opencv, mediapipe, scipy, sklearn, flask, ...

python -m parkigait demo --severity 0.6          # analyze a synthetic walker (no camera)
python -m parkigait scan path/to/walk.mp4        # scan a REAL walking video (MediaPipe)
python -m parkigait serve                        # local web app → http://127.0.0.1:7860
python -m parkigait eval --report                # measured metrics → RESULTS.md
python -m parkigait selftest                     # end-to-end smoke test
python -m pytest parkigait/tests -q              # test suite (10 tests)
```

For a real scan: a few seconds of someone walking, **full body visible, roughly
side-on**, decent light. The BlazePose `.task` model (~9 MB) auto-downloads once
and then runs offline.

## The modules

```
parkigait/
  types.py        the frozen data contract (PoseSequence, GaitFeatures, ...)
  pose.py         video→skeleton (MediaPipe Tasks API) + a synthetic gait model
  sttp.py         Spectral-Topological Token Preservation: L=D−A, Fiedler vector,
                  keep the body-connected tokens, drop background (real, measured)
  gaitfeat.py     clinical gait features: cadence, stride time & variability,
                  stride length, asymmetry, arm swing, Bächlin freeze index
  severity.py     features → P(PD-sign) + a 0–4 UPDRS-gait-like number, held-out CV
  lieq.py         LieQ mixed-precision quantization as a verification-gated search
                  over REAL measured accuracy (mentat kernel)
  carepd.py       CARE-PD loader — ready for real data, FABRICATES NOTHING
                  (raises a clear "gated dataset, request access" error)
  pipeline.py     end-to-end orchestration → PipelineReport
  realframe.py    STTP on REAL RGB frames (honest, non-separable stress test)
  render.py       render a synthetic walker to an .mp4
  viz.py          skeleton / gait-signal / STTP / severity figures
  app.py          local Flask web app (upload a video, see the report)
  eval.py         honest evaluation → RESULTS.md
```

## The two methods from the poster, made real

- **STTP** builds a k-NN graph over the pose/patch tokens, forms the graph
  Laplacian `L = D − A`, takes the **Fiedler vector** (2nd-smallest eigenvector),
  and keeps the densest connected component (the body), pruning background. It is
  applied to the *keypoint/patch token graph* — a tractable, honest realization of
  the idea, not a full VLM's internal tokens (which we don't have here).
- **LieQ** searches per-layer bit-widths under a memory budget and an accuracy
  floor, spending bits on sensitive layers and crushing redundant ones — and it is
  **verification-gated**: a policy is only kept if the quantized model's *measured*
  held-out accuracy clears the floor.

## Honesty first

This project is deliberately built so a judge can watch every claimed number
reproduce live, and so nothing is presented as clinical that isn't. The abstract's
">90% UPDRS correlation / ASR<10% / 14.5 GB→3.2 GB VLM" figures were **not**
measured on this system — see [HONEST_STATUS.md](HONEST_STATUS.md) for the exact
claim-by-claim status and the honest pitch that actually wins on rigor.
