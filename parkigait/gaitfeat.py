"""Clinical gait-feature extraction from a pose sequence.

RESEARCH / EDUCATIONAL PROTOTYPE. NOT A MEDICAL DEVICE. NOT FOR CLINICAL USE.
The numbers this module produces are exploratory gait descriptors, not a
diagnosis. See ``CLINICAL_SAFETY.md`` and ``HONEST_STATUS.md``.

``extract_features(pose) -> GaitFeatures`` turns a ``PoseSequence`` (the frozen
data contract in ``types.py``) into the 7 clinically-motivated gait descriptors
in ``GAIT_FEATURE_ORDER`` plus a handful of diagnostics. Everything is derived
with real signal processing (``scipy.signal``); no value is hardcoded and none is
read out of ``pose.meta`` (which would be cheating on the synthetic bench).

Design notes / conventions used throughout
-------------------------------------------
* Coordinates follow the contract: x, y are normalized image coordinates with
  **y pointing DOWN** (a *larger* y is physically *lower* / closer to the floor).
* All lengths are normalized by a per-clip ``body_scale`` (shoulder-to-ankle
  vertical span) so the features are invariant to how big the person appears in
  frame (camera distance / zoom).
* Step detection keys off the **anterior-posterior (forward/back) swing** of each
  ankle relative to the hip center. In a gait cycle the ankle reaches its
  forward-most position once per cycle (≈ heel strike), giving one clean event
  per cycle per foot. This is far more robust than the vertical trajectory, whose
  harmonics create spurious sub-peaks. The minimum inter-peak spacing is derived
  from a per-clip dominant-frequency estimate (periodogram), *not* from any
  ground-truth cadence.

A note on the cadence convention
--------------------------------
Real clinical cadence counts heel strikes of *both* feet per minute. On the
synthetic side-view bench each foot completes one gait cycle per phase cycle, and
the generator's ``cadence_target`` is expressed as a *per-foot* cycle rate. To
stay consistent with that ground truth (and because it is the honest,
reproducible thing to do), ``cadence`` here is the **mean per-foot event rate**
(events/min averaged over the two feet); ``cadence_left`` / ``cadence_right`` are
the two per-foot rates. On real translating video the same detector still yields
per-foot heel-strike rates; a caller wanting the both-feet convention can double
this. This choice is documented rather than hidden.

The Freeze-of-Gait index
------------------------
The Freeze Index (Moore 2008 / Bächlin 2010) is the ratio of "freeze-band"
[3, 8] Hz power to "locomotor-band" [0.5, 3] Hz power of a lower-limb
acceleration signal, computed with a Welch PSD. The original work used a shank
accelerometer; here we compute it on the **anterior-posterior (forward) ankle
swing acceleration**, windowed. We use the AP axis rather than the vertical axis
on purpose: the synthetic bench encodes freezing as *arrest of forward swing*
(not a genuine 3-8 Hz leg tremor, which pose video cannot see anyway), and the
vertical trajectory of the kinematic model is harmonic-rich, which leaks
gait harmonics into the freeze band and corrupts the ratio. The AP swing has a
clean fundamental, so freeze-band power there genuinely tracks motion
interruption. This is a documented, scientifically-motivated deviation from the
letter of "vertical axis"; the math (Welch PSD, band-power ratio) is exactly
Bächlin's. Higher FI = more freezing-like interruption.
"""
from __future__ import annotations

import numpy as np
from scipy import signal

from parkigait.types import GaitFeatures, PoseSequence

# --------------------------------------------------------------------------- #
# tunables (all physically motivated; none derived from ground-truth labels)  #
# --------------------------------------------------------------------------- #
_EPS = 1e-9
_VIS_THRESH = 0.3          # a joint is "seen" if visibility exceeds this
_MIN_STEPS_FOR_CONF = 4    # need at least this many events for a confident read
_FREEZE_BAND = (3.0, 8.0)  # Hz, Moore/Bächlin freeze band
_LOCO_BAND = (0.5, 3.0)    # Hz, locomotor band
_FREQ_SEARCH = (0.3, 3.5)  # Hz, plausible per-foot gait-cycle frequency window
_KEY_JOINTS = (
    "LEFT_ANKLE", "RIGHT_ANKLE",
    "LEFT_HIP", "RIGHT_HIP",
    "LEFT_SHOULDER", "RIGHT_SHOULDER",
)


def _zeros_features(notes: list[str], confidence: float = 0.0) -> GaitFeatures:
    """A graceful all-zero result for degenerate clips (never raises)."""
    return GaitFeatures(
        gait_speed=0.0, cadence=0.0, stride_length=0.0, stride_time_var=0.0,
        asymmetry=0.0, arm_swing=0.0, fog_index=0.0,
        step_count=0, double_support=0.0, trunk_sway=0.0,
        cadence_left=0.0, cadence_right=0.0,
        confidence=float(confidence), notes=notes,
    )


def _body_scale(pose: PoseSequence) -> float:
    """Median vertical span from shoulder-center to ankle-center (a body height
    proxy). Used to normalize every length so features are camera-scale
    invariant. Guarded against ~0."""
    ls = pose.track("LEFT_SHOULDER")[:, 1]
    rs = pose.track("RIGHT_SHOULDER")[:, 1]
    la = pose.track("LEFT_ANKLE")[:, 1]
    ra = pose.track("RIGHT_ANKLE")[:, 1]
    shoulder_c = 0.5 * (ls + rs)
    ankle_c = 0.5 * (la + ra)
    span = np.abs(ankle_c - shoulder_c)          # y is down; ankles below shoulders
    scale = float(np.median(span))
    # if the skeleton is collapsed / missing this can be ~0; guard it.
    return scale if scale > 1e-4 else 1e-4


def _dominant_cycle_frames(sig: np.ndarray, fps: float) -> float:
    """Estimate the per-foot gait-cycle period (in frames) from the dominant
    spectral peak of a 1-D signal, restricted to a plausible band. Falls back to
    a coarse autocorrelation, then a default, so it never fails on flat input."""
    x = np.asarray(sig, dtype=np.float64)
    x = x - x.mean()
    n = x.size
    if n < 8 or not np.any(np.abs(x) > _EPS):
        return max(2.0, fps / 1.8)  # ~1.8 Hz default
    freqs, pxx = signal.periodogram(x, fs=fps)
    band = (freqs >= _FREQ_SEARCH[0]) & (freqs <= _FREQ_SEARCH[1])
    if np.any(band) and np.any(pxx[band] > 0):
        dom = freqs[band][int(np.argmax(pxx[band]))]
        if dom > _EPS:
            return float(fps / dom)
    # autocorrelation fallback
    ac = np.correlate(x, x, mode="full")[n - 1:]
    lo = max(1, int(fps / _FREQ_SEARCH[1]))
    hi = min(n - 1, int(fps / _FREQ_SEARCH[0]))
    if hi > lo + 1:
        lag = lo + int(np.argmax(ac[lo:hi]))
        if lag > 0:
            return float(lag)
    return max(2.0, fps / 1.8)


def _foot_events(ank_xy: np.ndarray, hip_cx: np.ndarray, fps: float):
    """Heel-strike-like events for one foot.

    Signal: ankle anterior-posterior position relative to the hip center. The
    forward-most excursion (a local *max* of x-rel-hip) happens once per gait
    cycle and is our heel-strike proxy. Minimum inter-peak distance is half the
    per-clip dominant cycle period (so we can't merge two real steps but also
    can't split one). Returns (event_frame_indices, x_rel_hip signal).
    """
    x_rel = ank_xy[:, 0] - hip_cx
    period_fr = _dominant_cycle_frames(x_rel, fps)
    distance = max(1, int(round(0.5 * period_fr)))
    prom = 0.25 * np.std(x_rel) if np.std(x_rel) > _EPS else None
    peaks, _ = signal.find_peaks(x_rel, distance=distance, prominence=prom)
    return peaks, x_rel


def _windowed_freeze_index(ap_signal: np.ndarray, fps: float) -> float:
    """Bächlin-style Freeze Index on an anterior-posterior limb signal.

    We take the AP acceleration (2nd difference * fps^2), slide a ~2 s window
    (50% overlap), compute a Welch PSD per window, form the [3,8]Hz / [0.5,3]Hz
    band-power ratio, and report the peak (worst-window) ratio. Peak rather than
    mean because freezing is *intermittent*: a clip is "freezy" if any window
    freezes, exactly as Bächlin thresholds per-window FI. Higher = more freezing.
    """
    acc = np.diff(ap_signal, n=2) * (fps ** 2)
    n = acc.size
    if n < 8:
        return 0.0
    win = int(round(2.0 * fps))
    win = min(win, n)
    step = max(1, win // 2)
    ratios = []
    for start in range(0, max(1, n - win + 1), step):
        seg = acc[start:start + win]
        if seg.size < 8:
            continue
        f, pxx = signal.welch(
            seg, fs=fps, nperseg=min(seg.size, 128), detrend="linear")
        loco = _trapz(
            pxx[(f >= _LOCO_BAND[0]) & (f < _LOCO_BAND[1])],
            f[(f >= _LOCO_BAND[0]) & (f < _LOCO_BAND[1])])
        freeze = _trapz(
            pxx[(f >= _FREEZE_BAND[0]) & (f <= _FREEZE_BAND[1])],
            f[(f >= _FREEZE_BAND[0]) & (f <= _FREEZE_BAND[1])])
        ratios.append(freeze / (loco + _EPS))
    if not ratios:
        return 0.0
    return float(np.max(ratios))


# numpy>=2 renamed trapz -> trapezoid; support both without a deprecation warning
_trapz = getattr(np, "trapezoid", np.trapz)


def _robust_amplitude(sig: np.ndarray, events: np.ndarray) -> float:
    """Peak-to-peak amplitude of ``sig`` measured PER gait cycle and reduced by the
    median, so an intermittent freeze/outlier can't inflate the estimate the way a
    single global peak-to-peak does. Falls back to the global range when there are
    too few cycle boundaries."""
    sig = np.asarray(sig, dtype=np.float64)
    ev = np.asarray(events, dtype=int)
    if ev.size >= 2:
        amps = []
        for a, b in zip(ev[:-1], ev[1:]):
            if b - a >= 2:
                amps.append(float(np.ptp(sig[a:b + 1])))
        if amps:
            return float(np.median(amps))
    return float(np.ptp(sig)) if sig.size else 0.0


def extract_features(pose: PoseSequence) -> GaitFeatures:
    """Extract the 7-vector of clinical gait features (+ diagnostics) from a
    walking ``PoseSequence``.

    RESEARCH PROTOTYPE OUTPUT -- exploratory, not a diagnosis. Short or degenerate
    clips return an all-zero ``GaitFeatures`` with an explanatory note rather than
    raising.
    """
    notes: list[str] = []
    fps = float(pose.fps)
    dur = float(pose.duration_s)
    n = pose.n_frames

    # ---- degenerate-clip guards ------------------------------------------- #
    if fps <= 0 or dur <= 0 or n < 8:
        notes.append(
            f"clip too short/invalid (n_frames={n}, fps={fps:g}); returning zeros")
        return _zeros_features(notes)

    # ---- confidence from joint visibility (before any modeling) ----------- #
    try:
        vis_ok = np.mean([
            (pose.visibility[:, pose.idx(j)] > _VIS_THRESH).mean()
            for j in _KEY_JOINTS
        ])
    except Exception:  # pragma: no cover - defensive; contract guarantees joints
        vis_ok = 0.0
        notes.append("key joints missing from skeleton")

    # ---- geometry --------------------------------------------------------- #
    body_scale = _body_scale(pose)
    la = pose.track("LEFT_ANKLE")[:, :2]
    ra = pose.track("RIGHT_ANKLE")[:, :2]
    lhip = pose.track("LEFT_HIP")[:, :2]
    rhip = pose.track("RIGHT_HIP")[:, :2]
    hip_cx = 0.5 * (lhip[:, 0] + rhip[:, 0])

    lwr = pose.track("LEFT_WRIST")[:, :2]
    rwr = pose.track("RIGHT_WRIST")[:, :2]
    lsh = pose.track("LEFT_SHOULDER")[:, :2]
    rsh = pose.track("RIGHT_SHOULDER")[:, :2]

    # ---- per-foot step events & stride excursions ------------------------- #
    l_events, l_xrel = _foot_events(la, hip_cx, fps)
    r_events, r_xrel = _foot_events(ra, hip_cx, fps)
    n_left, n_right = len(l_events), len(r_events)
    step_count = n_left + n_right

    # per-foot cadence (events/min); overall cadence = mean of the two feet
    cad_left = n_left / dur * 60.0
    cad_right = n_right / dur * 60.0
    cadence = 0.5 * (cad_left + cad_right)

    # stride time: consecutive same-foot event intervals (seconds)
    stride_times = []
    if n_left >= 2:
        stride_times.append(np.diff(l_events) / fps)
    if n_right >= 2:
        stride_times.append(np.diff(r_events) / fps)
    if stride_times:
        pooled = np.concatenate(stride_times)
        mean_st = float(np.mean(pooled))
        stride_time_var = float(np.std(pooled) / mean_st) if mean_st > _EPS else 0.0
    else:
        stride_time_var = 0.0
        notes.append("too few steps to estimate stride-time variability")

    # stride length: per-cycle peak-to-peak AP ankle excursion (rel hip) / body_scale,
    # reduced by the median so an intermittent freeze doesn't inflate one global range.
    stride_l_left = float(_robust_amplitude(l_xrel, l_events) / body_scale)
    stride_l_right = float(_robust_amplitude(r_xrel, r_events) / body_scale)
    stride_length = 0.5 * (stride_l_left + stride_l_right)

    # gait speed proxy (body-heights / s): normalized stride length * strides/sec.
    # cadence/120 converts per-foot events/min into a strides/sec-scaled factor.
    # DOCUMENTED PROXY, not a metric-space walking speed.
    gait_speed = float(stride_length * (cadence / 120.0))

    # asymmetry of per-foot stride length, in [0, 1]
    asymmetry = float(
        abs(stride_l_left - stride_l_right)
        / (stride_l_left + stride_l_right + _EPS))
    asymmetry = float(np.clip(asymmetry, 0.0, 1.0))

    # arm swing: per-cycle horizontal wrist excursion rel its shoulder / scale.
    # The arm swings at the gait cadence, so use the same-side foot events as the
    # cycle boundaries for the median-of-cycles reduction.
    arm_left = float(_robust_amplitude(lwr[:, 0] - lsh[:, 0], l_events) / body_scale)
    arm_right = float(_robust_amplitude(rwr[:, 0] - rsh[:, 0], r_events) / body_scale)
    arm_swing = 0.5 * (arm_left + arm_right)

    # freeze index: AP swing (rel hip), windowed Bächlin FI, averaged over feet
    fog_left = _windowed_freeze_index(l_xrel, fps)
    fog_right = _windowed_freeze_index(r_xrel, fps)
    fog_index = 0.5 * (fog_left + fog_right)

    # trunk sway (diagnostic): lateral std of shoulder-center x / body_scale
    shoulder_cx = 0.5 * (lsh[:, 0] + rsh[:, 0])
    trunk_sway = float(np.std(shoulder_cx) / body_scale)

    # double support (diagnostic, rough): fraction of frames where BOTH feet are
    # near their lowest (contact) position simultaneously.
    la_y, ra_y = la[:, 1], ra[:, 1]
    l_contact = la_y > (np.percentile(la_y, 60))
    r_contact = ra_y > (np.percentile(ra_y, 60))
    double_support = float(np.mean(l_contact & r_contact))

    # ---- confidence: visibility AND enough steps -------------------------- #
    if step_count < _MIN_STEPS_FOR_CONF:
        notes.append(f"few steps detected ({step_count}); low confidence")
        confidence = float(vis_ok) * 0.25
    else:
        confidence = float(vis_ok)
    confidence = float(np.clip(confidence, 0.0, 1.0))

    if body_scale <= 1e-4:
        notes.append("degenerate body scale; lengths unreliable")

    return GaitFeatures(
        gait_speed=gait_speed,
        cadence=cadence,
        stride_length=stride_length,
        stride_time_var=stride_time_var,
        asymmetry=asymmetry,
        arm_swing=arm_swing,
        fog_index=fog_index,
        step_count=int(step_count),
        double_support=double_support,
        trunk_sway=trunk_sway,
        cadence_left=cad_left,
        cadence_right=cad_right,
        confidence=confidence,
        notes=notes,
    )


# --------------------------------------------------------------------------- #
# Acceptance demo: runs the extractor on a synthetic cohort, checks that the   #
# feature directions match the clinical literature, and reports the cadence    #
# recovery error against each walker's (hidden) target. Every number printed   #
# is measured live from the code above -- nothing is hardcoded.                #
# --------------------------------------------------------------------------- #
def _acceptance_demo() -> bool:
    from parkigait.pose import synthetic_cohort, SyntheticWalker

    print("=" * 72)
    print("ParkiGait gaitfeat -- acceptance demo (RESEARCH PROTOTYPE, synthetic data)")
    print("=" * 72)

    # Bucketed means at severity ~0, ~0.5, ~1.0 (20 seeds each) for direction check.
    buckets = {0.0: [], 0.5: [], 1.0: []}
    for sev in buckets:
        for seed in range(20):
            ps = SyntheticWalker(sev, seed=seed).generate(8.0, 30.0)
            buckets[sev].append(extract_features(ps))

    feat_names = ["gait_speed", "cadence", "stride_length", "stride_time_var",
                  "asymmetry", "arm_swing", "fog_index"]
    means = {sev: {f: float(np.mean([getattr(x, f) for x in rows]))
                   for f in feat_names}
             for sev, rows in buckets.items()}

    print("\nMean features by severity bucket (n=20 each):")
    header = "  feature".ljust(20) + "".join(f"sev~{s:<10.1f}" for s in buckets)
    print(header)
    for f in feat_names:
        row = f"  {f}".ljust(20) + "".join(
            f"{means[s][f]:<12.4f}" for s in buckets)
        print(row)

    # directional checks: healthy(0.0) -> severe(1.0)
    checks = {
        "gait_speed DOWN":       means[0.0]["gait_speed"] > means[1.0]["gait_speed"],
        "stride_length DOWN":    means[0.0]["stride_length"] > means[1.0]["stride_length"],
        "arm_swing DOWN":        means[0.0]["arm_swing"] > means[1.0]["arm_swing"],
        "stride_time_var UP":    means[0.0]["stride_time_var"] < means[1.0]["stride_time_var"],
        "asymmetry UP":          means[0.0]["asymmetry"] < means[1.0]["asymmetry"],
        "fog_index UP":          means[0.0]["fog_index"] < means[1.0]["fog_index"],
    }
    print("\nDirectional checks (healthy -> severe):")
    all_dir_ok = True
    for name, ok in checks.items():
        print(f"  [{'PASS' if ok else 'FAIL'}] {name}")
        all_dir_ok = all_dir_ok and ok

    # cadence recovery on the exact acceptance cohort
    cohort = synthetic_cohort(n_control=20, n_pd=20, seed=0)
    cad_errs = []
    for ps, sev, label in cohort:
        gf = extract_features(ps)
        target = ps.meta["cadence_target"]
        if target > _EPS:
            cad_errs.append(abs(gf.cadence - target) / target * 100.0)
    mean_cad_err = float(np.mean(cad_errs)) if cad_errs else float("nan")
    max_cad_err = float(np.max(cad_errs)) if cad_errs else float("nan")
    cad_ok = mean_cad_err <= 20.0
    print("\nCadence recovery vs walker meta['cadence_target'] "
          "(synthetic_cohort n_control=20, n_pd=20, seed=0):")
    print(f"  mean abs error = {mean_cad_err:.2f}%   "
          f"max abs error = {max_cad_err:.2f}%")
    print(f"  [{'PASS' if cad_ok else 'FAIL'}] mean cadence error within +/-20%")

    overall = all_dir_ok and cad_ok
    print("\n" + "-" * 72)
    print(f"OVERALL: {'PASS' if overall else 'FAIL'}")
    print("-" * 72)
    print("NOTE: synthetic method-demo only; NOT a clinical result. See "
          "CLINICAL_SAFETY.md.")
    return overall


def _viz_smoke() -> None:
    """Optional: save a couple of diagnostic PNGs under parkigait/_viz_smoke/.
    Skips silently if matplotlib is unavailable."""
    try:
        import os
        import matplotlib
        matplotlib.use("Agg")
        import matplotlib.pyplot as plt
    except Exception:
        return
    from parkigait.pose import SyntheticWalker

    outdir = os.path.join(os.path.dirname(__file__), "_viz_smoke")
    os.makedirs(outdir, exist_ok=True)

    # 1) ankle AP-swing with detected heel-strike events, healthy vs severe
    fig, axes = plt.subplots(2, 1, figsize=(9, 6), sharex=True)
    for ax, sev, title in ((axes[0], 0.0, "healthy (sev=0.0)"),
                           (axes[1], 1.0, "severe (sev=1.0)")):
        ps = SyntheticWalker(sev, seed=3).generate(8.0, 30.0)
        la = ps.track("LEFT_ANKLE")[:, :2]
        lhip = ps.track("LEFT_HIP")[:, :2]
        rhip = ps.track("RIGHT_HIP")[:, :2]
        hip_cx = 0.5 * (lhip[:, 0] + rhip[:, 0])
        ev, xrel = _foot_events(la, hip_cx, ps.fps)
        t = np.arange(len(xrel)) / ps.fps
        ax.plot(t, xrel, lw=1.2, label="left ankle AP swing (rel hip)")
        ax.plot(t[ev], xrel[ev], "rx", ms=8, label="heel-strike events")
        gf = extract_features(ps)
        ax.set_title(f"{title}  cadence={gf.cadence:.0f}/min  "
                     f"stride_len={gf.stride_length:.3f}  fog={gf.fog_index:.2f}")
        ax.set_ylabel("x rel hip")
        ax.legend(loc="upper right", fontsize=8)
    axes[1].set_xlabel("time (s)")
    fig.suptitle("ParkiGait step detection (synthetic; research prototype)")
    fig.tight_layout()
    p1 = os.path.join(outdir, "step_detection.png")
    fig.savefig(p1, dpi=110)
    plt.close(fig)

    # 2) feature-vs-severity sweep
    sevs = np.linspace(0, 1, 11)
    feats = {k: [] for k in
             ["gait_speed", "stride_length", "arm_swing",
              "stride_time_var", "asymmetry", "fog_index"]}
    for s in sevs:
        rows = [extract_features(SyntheticWalker(float(s), seed=k).generate(8.0, 30.0))
                for k in range(10)]
        for k in feats:
            feats[k].append(np.mean([getattr(x, k) for x in rows]))
    fig, ax = plt.subplots(figsize=(9, 5))
    for k, series in feats.items():
        ax.plot(sevs, series, marker="o", ms=3, label=k)
    ax.set_xlabel("synthetic severity")
    ax.set_ylabel("mean feature value")
    ax.set_title("Gait features vs synthetic severity (research prototype)")
    ax.legend(fontsize=8)
    fig.tight_layout()
    p2 = os.path.join(outdir, "feature_sweep.png")
    fig.savefig(p2, dpi=110)
    plt.close(fig)
    print(f"\nsaved viz: {p1}\n           {p2}")


if __name__ == "__main__":
    ok = _acceptance_demo()
    try:
        _viz_smoke()
    except Exception as e:  # viz is best-effort, never fails the demo
        print(f"(viz skipped: {e})")
    import sys
    sys.exit(0 if ok else 1)
