"""Rich, clinically-grounded gait features from CARE-PD 3D SMPL data.

The video path (``gaitfeat.py``) uses 7 features from a 2D projection. CARE-PD gives
us the full 3D SMPL pose AND the pelvis translation, so we can compute the gait
biomarkers the Parkinson's literature actually uses — most importantly REAL gait
speed (from ``trans``), 3D joint ranges of motion, trunk flexion (stooped posture),
variability, and left/right asymmetry. All are directional per the PD gait
literature (reduced speed / stride / arm swing / ROM; increased variability /
asymmetry / double support / trunk flexion; freezing).

HONEST SCOPE: joints come from the canonical-FK approximation (``smpl_fk``), not the
licensed SMPL model, so absolute magnitudes are approximate; the features are
body-scale-normalized and the training/eval is strictly subject-level, so the
reported correlation is a real generalization number, not a leaked one.
"""
from __future__ import annotations

import numpy as np
from scipy import signal

from parkigait.smpl_fk import smpl_joints_from_pose

_EPS = 1e-8
_trapz = getattr(np, "trapezoid", np.trapz)  # numpy>=2 renamed trapz->trapezoid
# SMPL joint indices
P, LHIP, RHIP, S1, LKNE, RKNE, S2, LANK, RANK, S3, LFOO, RFOO, NECK = \
    0, 1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 11, 12
HEAD, LSH, RSH, LEL, REL, LWR, RWR = 15, 16, 17, 18, 19, 20, 21


def _angle(j, a, b, c):
    """Per-frame angle (rad) at joint b, between b->a and b->c. j:(T,24,3)."""
    v1 = j[:, a] - j[:, b]
    v2 = j[:, c] - j[:, b]
    cos = (v1 * v2).sum(1) / (np.linalg.norm(v1, axis=1) * np.linalg.norm(v2, axis=1) + _EPS)
    return np.arccos(np.clip(cos, -1.0, 1.0))


def _cv(x):
    x = np.asarray(x, float)
    m = x.mean()
    return float(x.std() / m) if abs(m) > _EPS else 0.0


def _foot_events(ankle_y, fps):
    """Heel-strike frames = minima of ankle height (foot lowest). Returns indices."""
    inv = -ankle_y
    # dominant step period for a sensible min-distance
    x = ankle_y - ankle_y.mean()
    if x.std() < _EPS:
        return np.array([], dtype=int)
    f, pxx = signal.periodogram(x, fs=fps)
    band = (f > 0.3) & (f < 4.0)
    step_hz = f[band][np.argmax(pxx[band])] if band.any() and pxx[band].max() > 0 else 1.5
    dist = max(3, int(0.5 * fps / max(step_hz, 0.3)))
    peaks, _ = signal.find_peaks(inv, distance=dist)
    return peaks


FEATURE_NAMES = [
    "gait_speed", "gait_speed_cv", "cadence", "stride_time_cv",
    "stride_len", "stride_len_asym", "step_time_asym",
    "knee_rom", "knee_rom_asym", "hip_rom", "hip_rom_asym",
    "elbow_rom", "arm_swing", "arm_swing_asym",
    "trunk_flexion_mean", "trunk_flexion_max",
    "foot_clearance", "foot_clearance_asym",
    "pelvis_vert_osc", "pelvis_lat_sway",
    "double_support", "freeze_index", "gait_regularity",
    "knee_angvar", "arm_swing_amp_var",
]


def rich_features(pose, trans, fps: float) -> dict:
    """Return a dict of ~25 clinical gait features for one CARE-PD walk."""
    j = smpl_joints_from_pose(np.asarray(pose), np.asarray(trans))  # (T,24,3), SMPL frame + trans
    T = j.shape[0]
    dur = T / float(fps)
    f = {k: 0.0 for k in FEATURE_NAMES}
    if T < 8 or dur <= 0:
        return f

    # body scale (for length normalization): head-to-ankle-centre height in rest-ish
    ankle_c = 0.5 * (j[:, LANK] + j[:, RANK])
    body_h = float(np.median(np.linalg.norm(j[:, HEAD] - ankle_c, axis=1))) + _EPS

    tr = np.asarray(trans, float)
    horiz = tr[:, [0, 2]]                       # x (lateral), z (forward)
    step = np.linalg.norm(np.diff(horiz, axis=0), axis=1)
    f["gait_speed"] = float(step.sum() / dur / body_h)          # body-heights / s
    # instantaneous speed CV (bradykinesia -> less steady)
    inst = step * fps / body_h
    f["gait_speed_cv"] = _cv(inst) if inst.size else 0.0

    # foot events & timing
    lev = _foot_events(j[:, LANK, 1], fps)
    rev = _foot_events(j[:, RANK, 1], fps)
    n_l, n_r = len(lev), len(rev)
    f["cadence"] = (n_l + n_r) / dur * 60.0
    stimes = []
    if n_l >= 2:
        stimes.append(np.diff(lev) / fps)
    if n_r >= 2:
        stimes.append(np.diff(rev) / fps)
    if stimes:
        pooled = np.concatenate(stimes)
        f["stride_time_cv"] = _cv(pooled)
    # step-time asymmetry (mean L vs R stride time)
    if n_l >= 2 and n_r >= 2:
        ml, mr = np.mean(np.diff(lev)), np.mean(np.diff(rev))
        f["step_time_asym"] = float(abs(ml - mr) / (ml + mr + _EPS))

    # stride length: per-foot forward excursion (rel pelvis) / body_h
    fwd = tr[:, 2]  # crude forward axis; refine by pelvis-relative ankle z
    lz = (j[:, LANK, 2] - j[:, P, 2])
    rz = (j[:, RANK, 2] - j[:, P, 2])
    sl_l = float(np.ptp(lz) / body_h)
    sl_r = float(np.ptp(rz) / body_h)
    f["stride_len"] = 0.5 * (sl_l + sl_r)
    f["stride_len_asym"] = float(abs(sl_l - sl_r) / (sl_l + sl_r + _EPS))

    # joint angles (rad)
    kL, kR = _angle(j, LHIP, LKNE, LANK), _angle(j, RHIP, RKNE, RANK)
    hL, hR = _angle(j, S1, LHIP, LKNE), _angle(j, S1, RHIP, RKNE)
    eL, eR = _angle(j, LSH, LEL, LWR), _angle(j, RSH, REL, RWR)
    romL, romR = np.ptp(kL), np.ptp(kR)
    f["knee_rom"] = float(0.5 * (romL + romR))
    f["knee_rom_asym"] = float(abs(romL - romR) / (romL + romR + _EPS))
    f["knee_angvar"] = float(0.5 * (kL.std() + kR.std()))
    hrL, hrR = np.ptp(hL), np.ptp(hR)
    f["hip_rom"] = float(0.5 * (hrL + hrR))
    f["hip_rom_asym"] = float(abs(hrL - hrR) / (hrL + hrR + _EPS))
    f["elbow_rom"] = float(0.5 * (np.ptp(eL) + np.ptp(eR)))

    # arm swing: wrist forward-back excursion rel shoulder / body_h
    asL = float(np.ptp(j[:, LWR, 2] - j[:, LSH, 2]) / body_h)
    asR = float(np.ptp(j[:, RWR, 2] - j[:, RSH, 2]) / body_h)
    f["arm_swing"] = 0.5 * (asL + asR)
    f["arm_swing_asym"] = float(abs(asL - asR) / (asL + asR + _EPS))
    amp_series = np.abs((j[:, LWR, 2] - j[:, LSH, 2]))
    f["arm_swing_amp_var"] = _cv(amp_series) if amp_series.std() > _EPS else 0.0

    # trunk flexion (stooped posture): angle of pelvis->neck from vertical
    spine = j[:, NECK] - j[:, P]
    up = np.array([0.0, 1.0, 0.0])
    tf = np.arccos(np.clip((spine @ up) / (np.linalg.norm(spine, axis=1) + _EPS), -1, 1))
    f["trunk_flexion_mean"] = float(tf.mean())
    f["trunk_flexion_max"] = float(tf.max())

    # foot clearance (shuffling -> reduced): vertical ROM of ankles / body_h
    fcL = float(np.ptp(j[:, LANK, 1]) / body_h)
    fcR = float(np.ptp(j[:, RANK, 1]) / body_h)
    f["foot_clearance"] = 0.5 * (fcL + fcR)
    f["foot_clearance_asym"] = float(abs(fcL - fcR) / (fcL + fcR + _EPS))

    # pelvis oscillation
    f["pelvis_vert_osc"] = float((j[:, P, 1] - j[:, P, 1].mean()).std() / body_h)
    f["pelvis_lat_sway"] = float((j[:, P, 0] - j[:, P, 0].mean()).std() / body_h)

    # double support proxy: fraction of frames both feet are low (near contact)
    thr_l = j[:, LANK, 1].min() + 0.15 * (np.ptp(j[:, LANK, 1]) + _EPS)
    thr_r = j[:, RANK, 1].min() + 0.15 * (np.ptp(j[:, RANK, 1]) + _EPS)
    both_down = (j[:, LANK, 1] < thr_l) & (j[:, RANK, 1] < thr_r)
    f["double_support"] = float(both_down.mean())

    # freeze index (Bächlin) on ankle vertical accel
    f["freeze_index"] = _freeze_index(0.5 * (j[:, LANK, 1] + j[:, RANK, 1]), fps)

    # gait regularity: peak of the ankle-signal autocorrelation (lower in PD)
    f["gait_regularity"] = _autocorr_peak(j[:, LANK, 1], fps)
    return f


def _freeze_index(sig_y, fps):
    acc = np.diff(sig_y, n=2) * (fps ** 2)
    if acc.size < 8:
        return 0.0
    fr, pxx = signal.welch(acc, fs=fps, nperseg=min(acc.size, 256), detrend="linear")
    loco = _trapz(pxx[(fr >= 0.5) & (fr < 3.0)], fr[(fr >= 0.5) & (fr < 3.0)])
    fz = _trapz(pxx[(fr >= 3.0) & (fr <= 8.0)], fr[(fr >= 3.0) & (fr <= 8.0)])
    return float(fz / (loco + _EPS))


def _autocorr_peak(sig_y, fps):
    x = sig_y - sig_y.mean()
    if x.std() < _EPS:
        return 0.0
    ac = np.correlate(x, x, mode="full")[len(x) - 1:]
    ac = ac / (ac[0] + _EPS)
    lo = max(3, int(0.3 * fps))
    hi = min(len(ac), int(2.0 * fps))
    return float(ac[lo:hi].max()) if hi > lo else 0.0


def feature_vector(pose, trans, fps) -> np.ndarray:
    d = rich_features(pose, trans, fps)
    return np.array([d[k] for k in FEATURE_NAMES], dtype=np.float64)
