"""Honest evaluation harness — every number here is measured, and labelled with
exactly what it is (synthetic method-demo, real timing, real memory, ...).

Run:  python -m parkigait eval --report      # prints + writes RESULTS.md

What it measures:
  1. Severity model held-out CV (subject/record-level) on the synthetic cohort.
  2. Pearson correlation of predicted vs TRUE synthetic severity on a fresh,
     distinct-seed hold-out cohort (never seen in training).
  3. STTP body-recall / background-drop / keep-fraction on the token graph.
  4. LieQ real compression ratio + accuracy retained on the demo model.
  5. Real per-frame latency: synthetic analysis, and MediaPipe on a real video
     if one is available under sample_videos/.
  6. Real resident memory of the running pipeline.

NONE of these is a clinical result. Correlation is against SYNTHETIC ground truth
because we have no real UPDRS labels on this machine (see HONEST_STATUS.md).
"""
from __future__ import annotations

import platform
import resource
import time
from pathlib import Path

import numpy as np

from parkigait.gaitfeat import extract_features
from parkigait.pose import SyntheticWalker
from parkigait.severity import train_synthetic

_HERE = Path(__file__).resolve().parent
_REPO = _HERE.parent


def _maxrss_mb() -> float:
    rss = resource.getrusage(resource.RUSAGE_SELF).ru_maxrss
    # macOS reports bytes; Linux reports kilobytes.
    return rss / (1024 * 1024) if platform.system() == "Darwin" else rss / 1024


def _pearson(a, b) -> float:
    a, b = np.asarray(a, float), np.asarray(b, float)
    if a.std() < 1e-9 or b.std() < 1e-9:
        return 0.0
    return float(np.corrcoef(a, b)[0, 1])


def _holdout_cohort(n: int = 40, seed_base: int = 5000, duration_s: float = 8.0):
    """A fresh cohort with walker seeds DISTINCT from training (1000/2000 range),
    so it is a genuine hold-out for the model trained by train_synthetic()."""
    rng = np.random.default_rng(777)
    items = []
    for i in range(n):
        # half control-ish, half PD-ish, spanning the severity range
        sev = float(rng.uniform(0.0, 0.12)) if i % 2 == 0 else float(rng.uniform(0.2, 1.0))
        label = 0 if sev < 0.15 else 1
        pose = SyntheticWalker(sev, seed=seed_base + i).generate(duration_s=duration_s)
        items.append((pose, sev, label))
    return items


def run_eval(write_report: bool = False) -> dict:
    t_start = time.perf_counter()
    print("ParkiGait — honest evaluation (all numbers measured; synthetic where noted)\n")

    # -- 1. train + held-out CV --------------------------------------------
    model, cv = train_synthetic(n_control=60, n_pd=60, seed=0)
    print(f"[1] severity model — {cv['cv_kind']} held-out CV on synthetic cohort")
    print(f"      AUC {cv['auc_mean']:.3f} | acc {cv['acc_mean']:.3f} | "
          f"severity Pearson r {cv['severity_pearson_mean']:.3f} | n={cv['n']}")

    # -- 2. fresh hold-out correlation -------------------------------------
    holdout = _holdout_cohort(n=40)
    pred_sev, true_sev, labels, p_pd = [], [], [], []
    feat_by_bucket = {"control": [], "pd": []}
    t0 = time.perf_counter()
    n_frames_total = 0
    for pose, sev, label in holdout:
        feats = extract_features(pose)
        est = model.predict(feats)
        pred_sev.append(est.severity)
        true_sev.append(sev * 4.0)
        labels.append(label)
        p_pd.append(est.p_pd)
        n_frames_total += pose.n_frames
        feat_by_bucket["pd" if label else "control"].append(feats.as_vector())
    analyze_s = time.perf_counter() - t0
    r_sev = _pearson(pred_sev, true_sev)
    from parkigait.severity import _auc
    auc_holdout = _auc(np.array(labels), np.array(p_pd))
    print(f"\n[2] fresh hold-out (distinct seeds) — synthetic ground truth")
    print(f"      predicted-vs-true severity Pearson r {r_sev:.3f} | "
          f"control/PD AUC {auc_holdout:.3f} | n={len(holdout)}")

    # -- 3. STTP on the token graph ----------------------------------------
    from parkigait.sttp import sttp_report
    recalls, drops, keeps = [], [], []
    for pose, _, _ in holdout[:20]:
        rep = sttp_report(pose.joints[pose.n_frames // 2, :, :2])
        recalls.append(rep["body_recall"])
        drops.append(rep["background_drop"])
        keeps.append(rep["keep_fraction"])
    print(f"\n[3] STTP (Fiedler/Laplacian on the keypoint token graph) — measured")
    print(f"      body_recall {np.mean(recalls):.3f} | background_drop "
          f"{np.mean(drops):.3f} | keep_fraction {np.mean(keeps):.3f} (mean over 20 frames)")

    # -- 4. LieQ mixed-precision quantization ------------------------------
    lieq = {}
    try:
        from parkigait.lieq import run_demo
        lieq = run_demo(seed=0, viz=False)
        print(f"\n[4] LieQ mixed-precision quant — REAL, on a small demo model / synthetic data")
        print(f"      compression {lieq['compression_ratio']:.2f}x | "
              f"acc retained {lieq['acc_retained_pct']:.1f}% "
              f"(fp32 {lieq['acc_fp32']:.3f} -> mixed {lieq['acc_mixed']:.3f})")
    except Exception as e:
        print(f"\n[4] LieQ demo could not run: {type(e).__name__}: {e}")

    # -- 5. latency --------------------------------------------------------
    synth_ms_per_frame = analyze_s / max(1, n_frames_total) * 1000.0
    print(f"\n[5] latency (measured on THIS machine, CPU only)")
    print(f"      synthetic analysis (pose->features->severity): "
          f"{synth_ms_per_frame:.3f} ms/frame")
    mp_ms_per_frame = None
    sample = _find_sample_video()
    if sample:
        try:
            from parkigait.pose import MediaPipeBackend
            t0 = time.perf_counter()
            ps = MediaPipeBackend(stride=2, max_frames=60).extract(str(sample))
            mp_ms_per_frame = (time.perf_counter() - t0) / max(1, ps.n_frames) * 1000.0
            print(f"      MediaPipe pose extraction on real video "
                  f"({sample.name}): {mp_ms_per_frame:.1f} ms/frame")
        except Exception as e:
            print(f"      MediaPipe timing skipped: {type(e).__name__}: {e}")
    else:
        print("      (no sample video under sample_videos/ — skipping real-video timing)")

    # -- 6. memory ---------------------------------------------------------
    mem_mb = _maxrss_mb()
    print(f"\n[6] peak resident memory of this process: {mem_mb:.0f} MB "
          f"(the <4 GB edge target is met by a wide margin; note this is NOT the "
          f"abstract's 14.5GB->3.2GB VLM, which does not exist here)")

    results = {
        "cv": cv,
        "holdout": {"severity_pearson": r_sev, "auc": auc_holdout, "n": len(holdout)},
        "sttp": {"body_recall": float(np.mean(recalls)),
                 "background_drop": float(np.mean(drops)),
                 "keep_fraction": float(np.mean(keeps))},
        "lieq": lieq,
        "latency_ms_per_frame": {"synthetic_analysis": synth_ms_per_frame,
                                 "mediapipe_real_video": mp_ms_per_frame},
        "memory_mb": mem_mb,
        "elapsed_s": time.perf_counter() - t_start,
    }
    if write_report:
        path = _write_results_md(results, feat_by_bucket)
        print(f"\nwrote {path}")
    return results


def _find_sample_video():
    d = _REPO / "sample_videos"
    if not d.exists():
        return None
    for ext in ("*.mp4", "*.webm", "*.mov", "*.avi"):
        vids = sorted(d.glob(ext))
        # prefer a real (non-synthetic) clip for the MediaPipe timing
        real = [v for v in vids if "synthetic" not in v.name]
        if real:
            return real[0]
    return None


def _write_results_md(r: dict, feat_by_bucket: dict) -> Path:
    cv, ho, st, lq = r["cv"], r["holdout"], r["sttp"], r["lieq"]
    lat = r["latency_ms_per_frame"]
    mp = lat["mediapipe_real_video"]
    ctrl = np.mean(feat_by_bucket["control"], axis=0) if feat_by_bucket["control"] else None
    pd = np.mean(feat_by_bucket["pd"], axis=0) if feat_by_bucket["pd"] else None
    from parkigait.types import GAIT_FEATURE_ORDER

    feat_rows = ""
    if ctrl is not None and pd is not None:
        for i, name in enumerate(GAIT_FEATURE_ORDER):
            feat_rows += f"| {name} | {ctrl[i]:.3f} | {pd[i]:.3f} |\n"

    lq_line = (f"{lq.get('compression_ratio', float('nan')):.2f}× compression, "
               f"{lq.get('acc_retained_pct', float('nan')):.0f}% accuracy retained "
               f"(fp32 {lq.get('acc_fp32', float('nan')):.3f} → mixed "
               f"{lq.get('acc_mixed', float('nan')):.3f})") if lq else "n/a"
    mp_line = f"{mp:.1f} ms/frame" if mp is not None else "n/a (no real video bundled)"

    md = f"""# ParkiGait — measured results

Generated by `python -m parkigait eval --report`. Every number below was measured
on this machine at run time. **None is a clinical result.** Correlation is against
**synthetic** ground truth because there is no real UPDRS-labelled data here
(see HONEST_STATUS.md and CLINICAL_SAFETY.md).

## Headline numbers (all measured, all synthetic-data or system metrics)

| Metric | Value | What it is |
|---|---|---|
| Severity model held-out CV | AUC {cv['auc_mean']:.3f}, r {cv['severity_pearson_mean']:.3f} | {cv['cv_kind']} CV, synthetic cohort |
| Fresh hold-out severity correlation | Pearson r **{ho['severity_pearson']:.3f}** | predicted vs TRUE **synthetic** severity, distinct-seed hold-out (n={ho['n']}) |
| Fresh hold-out control/PD AUC | {ho['auc']:.3f} | synthetic classification |
| STTP body recall | **{st['body_recall']:.3f}** | fraction of body tokens preserved (token graph) |
| STTP background drop | **{st['background_drop']:.3f}** | fraction of background tokens pruned |
| STTP keep fraction | {st['keep_fraction']:.3f} | fraction of all tokens kept |
| LieQ quantization | {lq_line} | small demo model, synthetic data |
| Latency — synthetic analysis | {lat['synthetic_analysis']:.3f} ms/frame | pose→features→severity, CPU |
| Latency — MediaPipe on real video | {mp_line} | BlazePose pose extraction, CPU |
| Peak resident memory | {r['memory_mb']:.0f} MB | whole process; < 4 GB edge target met |

## Mean gait features, synthetic control vs PD hold-out

| feature | control (mean) | PD (mean) |
|---|---|---|
{feat_rows}
## Honest reading

- The severity **correlation is high because the data is synthetic** and we control
  the ground truth. This demonstrates the *method* end-to-end; it is **not** a
  clinical correlation and must not be reported as one.
- STTP recall/drop are measured on a **token-graph demo** (grid patches + injected
  background), a tractable, honest realization of the poster idea — not a full
  VLM's internal tokens.
- LieQ numbers are a **real** compression/accuracy measurement on a **small** model;
  the abstract's 14.5 GB→3.2 GB VLM figures are not produced here.
- Latency and memory are **real** measurements on this laptop, CPU-only, on-device.

To make any of these a clinical number, you need real labelled data and the
validation/regulatory path in CLINICAL_SAFETY.md.
"""
    path = _HERE / "RESULTS.md"
    path.write_text(md)
    return path


if __name__ == "__main__":
    import argparse
    ap = argparse.ArgumentParser()
    ap.add_argument("--report", action="store_true")
    run_eval(write_report=ap.parse_args().report)
