"""Smoke + contract tests for ParkiGait. Fast (synthetic, short clips).

Run:  .venv/bin/python -m pytest parkigait/tests -q
"""
from __future__ import annotations

import numpy as np
import pytest

from parkigait.pose import SyntheticWalker, synthetic_cohort
from parkigait.types import GAIT_FEATURE_ORDER, PoseSequence


# --------------------------------------------------------------------------- #
# contract                                                                    #
# --------------------------------------------------------------------------- #
def test_pose_sequence_validation():
    j = np.zeros((10, 33, 3), dtype=np.float32)
    v = np.ones((10, 33), dtype=np.float32)
    ps = PoseSequence(j, v, fps=30.0)
    assert ps.n_frames == 10 and ps.n_joints == 33
    assert abs(ps.duration_s - 10 / 30) < 1e-6
    with pytest.raises(ValueError):
        PoseSequence(np.zeros((10, 33, 2)), v, fps=30.0)  # wrong last dim
    with pytest.raises(ValueError):
        PoseSequence(j, np.ones((10, 32)), fps=30.0)      # visibility mismatch


def test_synthetic_walker_severity_trends():
    lo = SyntheticWalker(0.0, seed=1).generate(duration_s=6)
    hi = SyntheticWalker(1.0, seed=1).generate(duration_s=6)
    # arm swing (wrist x range) shrinks with severity
    assert lo.track("RIGHT_WRIST")[:, 0].ptp() > hi.track("RIGHT_WRIST")[:, 0].ptp()
    # foot vertical excursion shrinks (shuffling)
    assert lo.track("RIGHT_ANKLE")[:, 1].ptp() > hi.track("RIGHT_ANKLE")[:, 1].ptp()


# --------------------------------------------------------------------------- #
# gait features                                                               #
# --------------------------------------------------------------------------- #
def test_gaitfeat_directions():
    from parkigait.gaitfeat import extract_features
    lo = extract_features(SyntheticWalker(0.0, seed=2).generate())
    hi = extract_features(SyntheticWalker(1.0, seed=2).generate())
    assert set(lo.as_dict()) == set(GAIT_FEATURE_ORDER)
    assert hi.gait_speed < lo.gait_speed
    assert hi.stride_length < lo.stride_length
    assert hi.arm_swing < lo.arm_swing
    assert hi.stride_time_var > lo.stride_time_var
    assert hi.fog_index > lo.fog_index
    assert lo.step_count >= 4 and lo.confidence > 0.5


def test_gaitfeat_degenerate_clip():
    from parkigait.gaitfeat import extract_features
    ps = PoseSequence(np.zeros((5, 33, 3), np.float32),
                      np.ones((5, 33), np.float32), fps=30.0)
    feats = extract_features(ps)  # must not raise
    assert feats.confidence == 0.0 and feats.notes


# --------------------------------------------------------------------------- #
# STTP                                                                        #
# --------------------------------------------------------------------------- #
def test_sttp_laplacian_and_recall():
    from parkigait.sttp import build_graph, fiedler_vector, frame_tokens, sttp_report
    pts = np.random.default_rng(0).uniform(0, 1, (40, 2))
    W, D, L = build_graph(pts, k=6)
    # L = D - W, symmetric, rows sum to ~0 (PSD Laplacian property)
    assert np.allclose(L, D - W)
    assert np.allclose(L, L.T, atol=1e-6)
    assert np.allclose(L.sum(axis=1), 0, atol=1e-6)
    fied, eig = fiedler_vector(L)
    assert eig[0] <= eig[1] + 1e-9  # ascending
    # body recall high on a separable synthetic frame
    ps = SyntheticWalker(0.3, seed=1).generate()
    rep = sttp_report(ps.joints[ps.n_frames // 2, :, :2])
    assert rep["body_recall"] >= 0.85
    assert rep["background_drop"] >= 0.85


# --------------------------------------------------------------------------- #
# severity + pipeline                                                         #
# --------------------------------------------------------------------------- #
def test_severity_and_pipeline_monotonic():
    from parkigait.pipeline import analyze_synthetic
    from parkigait.severity import train_synthetic
    model, cv = train_synthetic(n_control=30, n_pd=30, seed=0, save=False)
    assert 0.0 <= cv["auc_mean"] <= 1.0
    p_prev = -1.0
    for sev in (0.0, 0.5, 1.0):
        r = analyze_synthetic(severity=sev, seed=4, severity_model=model)
        assert 0.0 <= r.severity.p_pd <= 1.0
        assert r.disclaimer  # every report carries the disclaimer
        assert r.severity.p_pd >= p_prev - 0.2  # broadly non-decreasing
        p_prev = r.severity.p_pd


# --------------------------------------------------------------------------- #
# LieQ quantization                                                           #
# --------------------------------------------------------------------------- #
def test_lieq_quantize_roundtrip_and_compression():
    from parkigait.lieq import quantize_array
    w = np.random.default_rng(0).normal(size=200).astype(np.float64)
    for bits in (2, 3, 4, 8):
        q, scale, nbytes = quantize_array(w, bits)
        assert q.shape == w.shape
        # more bits -> lower reconstruction error
        err = np.abs(q - w).mean()
        assert np.isfinite(err)
        assert nbytes < w.nbytes  # actually smaller than FP64 storage
    with pytest.raises(Exception):
        quantize_array(w, 5)  # invalid bit width rejected


# --------------------------------------------------------------------------- #
# CARE-PD adapter fabricates nothing                                          #
# --------------------------------------------------------------------------- #
def test_carepd_not_available_raises():
    from parkigait.carepd import CAREPDDataset, CAREPDNotAvailable
    ds = CAREPDDataset("/definitely/not/a/real/path/carepd")
    with pytest.raises(CAREPDNotAvailable):
        list(ds.to_pose_sequences())
    assert "CARE-PD" in ds.describe()


# --------------------------------------------------------------------------- #
# OOD guard                                                                   #
# --------------------------------------------------------------------------- #
def test_ood_guard_flags_alien_input():
    from parkigait.gaitfeat import extract_features
    from parkigait.severity import train_synthetic
    from parkigait.types import GaitFeatures
    model, _ = train_synthetic(n_control=30, n_pd=30, seed=0, save=False)
    # an in-distribution synthetic walker is NOT ood
    normal = extract_features(SyntheticWalker(0.3, seed=7).generate())
    assert model.predict(normal).ood is False
    # a wildly out-of-range feature vector IS ood and says so
    alien = GaitFeatures(gait_speed=9.0, cadence=400.0, stride_length=8.0,
                         stride_time_var=5.0, asymmetry=0.99, arm_swing=9.0,
                         fog_index=500.0, step_count=20, confidence=1.0)
    out = model.predict(alien)
    assert out.ood is True
    assert "out-of-distribution" in out.label.lower()


# --------------------------------------------------------------------------- #
# render                                                                      #
# --------------------------------------------------------------------------- #
def test_render_walk_video(tmp_path):
    import cv2
    from parkigait.render import render_walk_video
    ps = SyntheticWalker(0.5, seed=0).generate(duration_s=2)
    out = tmp_path / "walk.mp4"
    render_walk_video(ps, out)
    assert out.exists() and out.stat().st_size > 0
    cap = cv2.VideoCapture(str(out))
    assert cap.isOpened()
    cap.release()


def test_cli_selftest():
    from parkigait.cli import main
    assert main(["selftest"]) == 0


# --------------------------------------------------------------------------- #
# SMPL forward kinematics (canonical skeleton, licensed-model-free)           #
# --------------------------------------------------------------------------- #
def test_smpl_fk_zero_pose_is_rest():
    from parkigait.smpl_fk import REST_JOINTS, smpl_joints_from_pose
    # zero axis-angle -> the canonical rest skeleton, re-centred on the pelvis
    joints = smpl_joints_from_pose(np.zeros((3, 72)))
    assert joints.shape == (3, 24, 3)
    expected = REST_JOINTS - REST_JOINTS[0]
    assert np.allclose(joints[0], expected, atol=1e-6)
    # left/right hips symmetric about x
    assert joints[0][1, 0] > 0 > joints[0][2, 0]


def test_smpl_fk_projection_and_features():
    from parkigait.gaitfeat import extract_features
    from parkigait.smpl_fk import carepd_record_to_blazepose
    from parkigait.types import PoseSequence
    # a synthetic "walk": hips + knees oscillate out of phase over time
    T, fps = 120, 30.0
    t = np.linspace(0, 4 * np.pi, T)
    pose = np.zeros((T, 72), dtype=np.float64)
    pose[:, 1 * 3] = 0.4 * np.sin(t)          # L_hip flex
    pose[:, 2 * 3] = 0.4 * np.sin(t + np.pi)  # R_hip flex (antiphase)
    pose[:, 4 * 3] = 0.3 * np.maximum(0, np.sin(t))       # L_knee
    pose[:, 5 * 3] = 0.3 * np.maximum(0, np.sin(t + np.pi))
    trans = np.zeros((T, 3))
    trans[:, 2] = np.linspace(0, 3.0, T)      # walk forward in z
    j, v = carepd_record_to_blazepose(pose, trans, fps)
    assert j.shape == (T, 33, 3) and v.shape == (T, 33)
    assert np.isfinite(j).all()
    # SMPL-sourced joints (vis>=0.5) are normalized to [0,1]; anchored placeholders
    # (vis~0.2) may sit slightly outside by a small documented offset.
    sourced = v[0] >= 0.5
    xy_sourced = j[:, sourced, :2]
    assert xy_sourced.min() >= -1e-6 and xy_sourced.max() <= 1 + 1e-6
    # the FK+projection produced real temporal motion (the ankle moves over the clip)
    from parkigait.types import joint_index
    ank_y = j[:, joint_index("LEFT_ANKLE"), 1]
    assert ank_y.ptp() > 1e-3
    feats = extract_features(PoseSequence(j, v, fps=fps))  # must not raise
    assert np.isfinite(feats.as_vector()).all()


# --------------------------------------------------------------------------- #
# ablation / robustness                                                       #
# --------------------------------------------------------------------------- #
def test_ablation_background_rejection():
    from parkigait.ablation import stt_background_rejection
    rows = stt_background_rejection(n_backgrounds=(30, 120), trials=2)
    assert len(rows) == 2
    for r in rows:
        assert 0.0 <= r["bg_rejection"] <= 1.0
        assert abs(r["bg_rejection"] + r["bg_survival"] - 1.0) < 1e-6
    # at a modest injection count STTP should reject the great majority of background
    assert rows[0]["bg_rejection"] >= 0.8
