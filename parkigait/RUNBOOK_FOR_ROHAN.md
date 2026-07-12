# RUNBOOK — ParkiGait end-to-end verification (for Rohan Dravat)

**What this is:** a step-by-step guide to set up ParkiGait from a clean checkout,
run every path, and confirm each piece actually works. Everything below was run and
verified on 2026-07-12 (macOS, Apple M3, Python 3.11.15). Where a number is quoted,
it is a value I actually observed, not a claim from the abstract.

> **INTEGRITY NOTE — READ FIRST.** ParkiGait is a **RESEARCH / EDUCATIONAL
> prototype. It is NOT a medical device and cannot diagnose Parkinson's disease or
> anything else.** The severity/P(PD) number is *exploratory and uncalibrated*. Do
> not present it, or any figure in this repo, as a clinical result. The ISEF
> abstract's headline numbers (">90% UPDRS correlation", "ASR < 10%",
> "14.5 GB → 3.2 GB VLM") were **never measured on this machine** and must not be
> claimed — see `HONEST_STATUS.md` for the exact claim-by-claim status. The honest,
> defensible story is: a real working video→gait pipeline whose every *measured*
> number reproduces live. That is what you are verifying here.

---

## 0. Critical facts about the layout (this trips people up)

- The **git repo root is the parent directory** `/Users/parvmehndiratta/parkinsons-detection`,
  **not** `parkigait/`. `parkigait/` is a Python *package* inside it.
- **You must run every command from the git root** (`parkinsons-detection/`), because
  the package is invoked as `python -m parkigait ...`.
  Running from *inside* `parkigait/` **fails** — `parkigait/types.py` shadows Python's
  stdlib `types` module and you get
  `ImportError: cannot import name 'MappingProxyType' from ... types.py`. If you see
  that error, you `cd`'d one level too deep.
- Large/generated assets are **gitignored** and live only locally:
  `sample_videos/`, `data/` (CARE-PD), `parkigait/models/*.task`, `app_static/`,
  `app_uploads/`, `_viz_smoke/`, `*.mp4`, `*.webm`. A fresh `git clone` on another
  machine will **not** contain the sample videos, the CARE-PD dataset, or the
  9 MB BlazePose `.task` model. See §5 for what has to be supplied.

## 1. Environment (Python **3.11** or 3.12 — this matters)

Use Python **3.11** (proven) or 3.12. **Do not use Python 3.13 or 3.14.** Reason:
`requirements.txt` pins `numpy<2` for MediaPipe/OpenCV compatibility, and `numpy<2`
resolves to numpy 1.26.4, which has **no prebuilt wheel for 3.13/3.14** — pip tries
to compile it from source and that is fragile and usually fails. (Verified: on a
Python 3.14 venv, `pip install "numpy<2"` falls back to a source build.)

```bash
cd /Users/parvmehndiratta/parkinsons-detection      # the GIT ROOT, not parkigait/

# create + activate a venv with Python 3.11 (adjust the interpreter path as needed)
python3.11 -m venv .venv
source .venv/bin/activate

python -m pip install --upgrade pip
pip install -r parkigait/requirements.txt
# installs: numpy<2, opencv-python, mediapipe, scipy, scikit-learn, matplotlib,
#           imageio(+ffmpeg), joblib, flask, pillow, pytest, reportlab
```

Notes:
- On this machine there is **already a working `.venv`** at the git root
  (uv-managed CPython 3.11.15) with all deps installed — you can just
  `source .venv/bin/activate` instead of rebuilding. The commands below use
  `./.venv/bin/python` so they work with or without activating.
- Nothing is installed system-wide; everything lives in `.venv/`.
- The BlazePose model (`parkigait/models/pose_landmarker_full.task`, ~9 MB) is
  already present here. On a machine where it's missing, it auto-downloads once on
  the first `scan`, then runs fully offline.

## 2. Run the test suite (fastest confidence check)

```bash
cd /Users/parvmehndiratta/parkinsons-detection
./.venv/bin/python -m pytest parkigait/tests -q
```

**Verified result:** `16 passed in ~9s`. These are contract + smoke tests covering
the pose data contract, synthetic-gait direction (features move the right way with
severity), STTP Laplacian/Fiedler math and body-recall, severity monotonicity, the
OOD guard, LieQ quantization round-trip, the CARE-PD "fabricates nothing" guard,
SMPL forward-kinematics, and the clinical-stats helpers.
(README says "10 tests"; the suite has since grown to **16** — the README is stale
on that count.)

## 3. Run the demo / self-test paths (no video needed)

```bash
cd /Users/parvmehndiratta/parkinsons-detection

# end-to-end smoke test on synthetic walkers of increasing severity
./.venv/bin/python -m parkigait selftest
# Verified: P(PD) rises 0.123 → 0.528 → 0.990 across severity 0.0/0.5/1.0 → "SELFTEST: PASS"

# full report on a single synthetic walker (no camera)
./.venv/bin/python -m parkigait demo --severity 0.6
# Verified: prints gait features + P(PD)=0.927, severity 2.50/4, STTP keep 0.459,
#           per-stage timings, and the research-only disclaimer.
```

## 4. Run on a REAL walking video (the core of the prototype)

Two sample clips are bundled locally under `sample_videos/` (gitignored):

```bash
cd /Users/parvmehndiratta/parkinsons-detection

# a real person walking (Wikimedia "Walking in the sands")
./.venv/bin/python -m parkigait scan sample_videos/walking_sands.webm --max-frames 120
```

**Verified output (abridged):**
```
source:            sample_videos/walking_sands.webm
steps detected:    11        signal confidence: 1.0
gait_speed 0.14  cadence 82.5  stride_length 0.20  fog_index 10.80 ...
P(PD-like motor signs): 0.697     severity 2.08/4
label: possible PD motor signs (exploratory, not a diagnosis)
timings: per_frame_ms 92.4   (pose_extract ~11 s for 120 frames)
```
This 0.697 is **a documented false-positive-ish read**: the subject is *healthy* but
walking slowly on sand with reduced arm swing, which mimics PD features. That is the
"confounding" limitation in `LIMITATIONS.md §3`, not a bug. Do **not** read it as a
detection.

> Note on timing: I measured **~92 ms/frame** on the sands clip, higher than the
> README's headline **~27 ms/frame**. Per-frame cost depends on resolution/clip;
> the README number is a best case. Quote whatever you actually measure, not 27.

### The out-of-distribution (OOD) guard — verify it refuses to guess

```bash
./.venv/bin/python -m parkigait scan sample_videos/cdc_treadmill.webm --max-frames 90
```
**Verified:** the report prints
`** OUT-OF-DISTRIBUTION: score not meaningful for this input **`, suppresses P(PD)
to a meaningless 0.029, labels it `out-of-distribution — unlike the training data,
estimate unreliable`, and adds low-signal warnings. This is the intended honest
behavior: the tool says "I can't judge this" rather than emitting a confident wrong
answer.

### The local web app (optional)

```bash
./.venv/bin/python -m parkigait serve      # → http://127.0.0.1:7860
```
Localhost only, no data leaves the machine; upload a walking video and get the same
report in the browser. (I confirmed `parkigait.app` imports cleanly; I did not bind
the port in this verification.)

## 5. What Rohan must supply on a fresh clone

A fresh `git clone` will be missing the gitignored assets. To reproduce fully you need:

1. **A walking video** for `scan` — a few seconds of one person, **full body visible,
   roughly side-on, decent light**. Any `.mp4`/`.webm`/`.mov` MediaPipe can read.
   (Or generate one: `./.venv/bin/python -m parkigait render --severity 0.6 --out sample_videos/synthetic_walk.mp4`.)
   Without a video you can still verify everything except a real `scan` — the tests,
   `selftest`, and `demo` all run pose extraction on a *synthetic* skeleton, so the
   feature/severity/STTP pipeline is fully exercised without a camera.
2. **The BlazePose `.task` model** — present here; auto-downloads on first `scan` if
   absent.
3. **CARE-PD dataset** (only for the `carepd-*` / `clinical-eval` commands) — it is a
   **gated dataset**; the loader raises a clear "request access" error instead of
   fabricating data. It happens to be downloaded locally under `data/CARE-PD/`, but it
   is gitignored and will not travel with the repo. The `carepd-*` commands are
   **not** needed to verify the core video→gait prototype; skip them unless you're
   auditing the real-data training numbers.

---

## 6. TRIPLE-CHECK CHECKLIST (the specific things to verify)

Work top to bottom. Each item names the exact command and what "good" looks like.

- [ ] **Environment is Python 3.11/3.12.** `./.venv/bin/python --version` → `3.11.x`
  (or 3.12). If it's 3.13/3.14, rebuild the venv — `numpy<2` won't install cleanly.
- [ ] **Tests pass.** `python -m pytest parkigait/tests -q` → `16 passed`. Any failure
  here means the pipeline contract is broken; stop and investigate before trusting
  any scan.
- [ ] **Self-test is monotonic.** `python -m parkigait selftest` → `PASS`, with P(PD)
  broadly rising as severity goes 0.0 → 0.5 → 1.0. If P(PD) is flat or inverted, the
  severity model or feature extraction regressed.
- [ ] **Pose extraction works on a FRESH video.** Point `scan` at a walking clip that
  is *not* one of the bundled samples. Confirm: `steps detected ≥ ~4`,
  `signal confidence` well above 0, and non-degenerate gait features (not all zeros).
  Zero steps / confidence 0 means MediaPipe found no reliable skeleton — check that
  the full body is visible and the clip isn't too short/dark.
- [ ] **Feature extraction produces sane gait metrics.** In the report, sanity-check:
  `cadence` in a plausible human range (~roughly 40–140 steps/min),
  `gait_speed`/`stride_length` positive and finite, `asymmetry` in [0,1], `fog_index`
  finite. Garbage/NaN/huge values indicate a bad clip or a projection bug.
- [ ] **Classifier loads and outputs.** Every report must print a `P(PD-like motor
  signs)` in [0,1], a `severity (0–4)`, a `label`, and the **research-only
  disclaimer**. The severity model loads from `parkigait/models/severity_synth.joblib`
  (auto-trains if missing). No disclaimer = wrong build.
- [ ] **OOD guard fires on a bad input.** `scan sample_videos/cdc_treadmill.webm`
  (or any poorly-framed clip where the legs aren't visible) must flag
  `OUT-OF-DISTRIBUTION` and NOT present a confident score. This is the safety
  behavior; verify it still triggers.
- [ ] **Edge cases: occlusion / low light / short clip.** Deliberately feed a clip
  with the lower body cropped, a very dark clip, and a <2 s clip. Expected *honest*
  behavior is a **low-confidence / few-steps warning** or an **OOD flag**, never a
  confident diagnosis. If any edge case yields a confident P(PD) near 0 or 1 with no
  warning, that's a problem — flag it.
- [ ] **Framing stays honest.** Skim the report text and any figure you'd show: it
  must say "exploratory / not a diagnosis" and must not quote the abstract's
  unmeasured ">90% correlation / ASR<10% / VLM" numbers. Cross-check against
  `HONEST_STATUS.md`.

---

## 7. Verification summary (what I confirmed on 2026-07-12)

| Path | Command | Result |
|---|---|---|
| Test suite | `pytest parkigait/tests -q` | **16 passed** in ~9s |
| Self-test | `parkigait selftest` | **PASS**, P(PD) 0.123→0.528→0.990 |
| Synthetic demo | `parkigait demo --severity 0.6` | full report, P(PD)=0.927, disclaimer present |
| Real scan | `parkigait scan sample_videos/walking_sands.webm` | end-to-end MediaPipe pose → features → P(PD)=0.697, ~92 ms/frame |
| OOD guard | `parkigait scan sample_videos/cdc_treadmill.webm` | correctly flagged **OUT-OF-DISTRIBUTION**, score suppressed |
| Web app | `import parkigait.app` | imports cleanly (port not bound in this check) |

**Not independently re-verified here** (documented, but I did not re-run them, to
avoid modifying tracked result files): `parkigait eval --report` (regenerates
`RESULTS.md`), `parkigait ablation`, and the CARE-PD/`clinical-eval` real-data
training runs. Those depend on the gated CARE-PD data and/or overwrite committed
`.md` outputs; run them yourself if you need to audit the real-data numbers, and
treat `CAREPD_RESULTS.md` / `CLINICAL_EVAL.md` as the source of truth for those.

## 8. Where to read more

- `HONEST_STATUS.md` — claim-by-claim: what's measured vs. aspirational (read this).
- `LIMITATIONS.md` — sim-to-real gap, confounding, OOD, raw-pixel STTP breakdown.
- `CLINICAL_SAFETY.md` — why this is not clinical and what real deployment needs.
- `README.md` — overview and module map (note: its "10 tests" and "~27 ms/frame" are
  slightly stale — 16 tests now, and per-frame timing varies by clip).
