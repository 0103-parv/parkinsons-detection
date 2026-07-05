# Limitations & findings — what we learned by testing on real video

These are real results from running the pipeline on real footage. In a science
project, *finding and characterizing* your failure modes is a strength — it's the
difference between "we think it works" and "we know exactly where it works." Every
finding below is reproducible with the commands shown.

## 1. Sim-to-real gap: a synthetic-trained model over-reads real gait

The shipped model is trained on synthetic walkers. When we ran it on a real video
of a **healthy** person walking (Wikimedia "Walking in the sands"), it initially
returned a confident **P(PD)=1.000**. That is a false positive, and it is exactly
what you'd expect from a model that has never seen real, varied human gait.

**Fixes applied:**
- The synthetic cohort now has **overlapping** severity ranges and ~12% **atypical**
  (label-flipped) subjects, so the model is no longer trivially separable. Held-out
  control-vs-PD AUC dropped from a suspicious 1.00 to a believable **~0.86–0.94**,
  and the real video's score fell from 1.000 to a much less certain ~0.70.
- Take-away for the poster: **this is why real labelled data (CARE-PD) is required.**
  A synthetic-trained model's confidence does not transfer to real people.

## 2. Out-of-distribution guard: refuse rather than guess

We added an OOD check: the model measures each input's (diagonal Mahalanobis)
distance to its training distribution and, past a calibrated threshold, returns
**"out-of-distribution — estimate unreliable"** instead of a number.

Measured distances (`python -m parkigait.eval`): training points span ~0.8–7.0
(95% within ~4.8), threshold 7.3. A clean real walking video sits at ~3.2
(in-distribution, scored with uncertainty); a poorly-framed real clip whose legs
aren't visible sits at ~9.4 and is **correctly flagged OOD**. This is the honest, safe behavior: the tool says "I
can't judge this" when it shouldn't.

```bash
python -m parkigait scan sample_videos/cdc_treadmill.webm   # → flagged OOD
```

## 3. Confounding: real gait can look Parkinsonian for non-PD reasons

Even in-distribution, the clean real walking video read as "possible PD signs"
(~0.70). Inspecting the features, the person walks slowly with reduced arm swing
and an elevated freeze index — because they're walking **on sand**, not because of
disease. Terrain, speed, footwear, camera angle, and walking style all produce
PD-like features. A gait model without clinical context and diverse training data
**cannot** separate these causes. This is a fundamental limitation, not a bug, and
it is a core reason such a tool must stay a decision-support aid, never a
diagnostician.

## 4. STTP on raw pixels vs. semantic tokens

STTP cleanly isolates the body on the **pose keypoint graph** (body recall ~0.99).
We also stress-tested it on **raw RGB frame patches** (`parkigait/realframe.py`):
with a person occupying only ~2% of the frame, connectivity-based pruning picks the
large uniform background, not the small subject (body recall ≈ 0). This is expected:
raw connectivity ≠ semantic saliency. The poster's method gets clean separation from
**semantic VLM tokens**, where the body is a distinct cluster — which we don't have
on this machine. STTP is validated where it genuinely works (the keypoint graph) and
honest about where it needs semantic input.

```bash
python -m parkigait.realframe sample_videos/walking_sands.webm
```

Relatedly, `python -m parkigait.ablation` shows STTP rejects **100%** of injected
background tokens (and keeps 100% of the body) up to ~2× the body's token count, but
has a **breakdown point**: past ~4–5× the body, the background floods the frame and
out-densities the body, so the "densest component" heuristic locks onto the
background and body recall collapses to 0. Same root cause: connectivity/density is
not saliency. The fix is a body-saliency prior or semantic tokens.

## 5. Things we did not measure (and don't claim)

- **No clinical correlation** — no real UPDRS labels here (CARE-PD is gated).
- **No adversarial ASR** — there is no VLM on this machine to attack; the poster's
  ASR<10% is not substantiated. STTP's measurable background-token pruning is the
  honest robustness property.
- **No fairness/skin-tone evaluation** — needs real data with those labels.
- **Single-machine timings** — 27 ms/frame is a laptop CPU number, not a phone NPU.

## 6. What would move each of these

Real CARE-PD training (fixes 1, 3), a segmentation or saliency prior for raw-pixel
STTP (fix 4), an on-phone latency profile (fix 5), and a fairness-stratified real
dataset. See [CLINICAL_SAFETY.md](CLINICAL_SAFETY.md) for the full path to a
clinically meaningful result.
