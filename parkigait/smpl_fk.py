"""SMPL forward kinematics on a CANONICAL skeleton — a licensed-model-free way to
turn CARE-PD SMPL pose parameters into joint trajectories good enough for gait.

HONEST SCOPE. The exact SMPL joints require the licensed SMPL body model (register
at smpl.is.tue.mpg.de). We do NOT vendor it. Instead we apply the dataset's REAL
axis-angle joint rotations to a CANONICAL anthropometric rest skeleton via standard
forward kinematics. This is an APPROXIMATION:
  - Gait features are driven by joint *rotations over time* (which are exact here)
    and are body-scale-normalized, so leg-driven features (cadence, stride timing,
    freeze, asymmetry) are approximately preserved.
  - Absolute proportions and arm-swing amplitude are less reliable (the rest arm
    angle is approximate).
Every PoseSequence built this way is stamped ``joint_source="canonical_fk"`` so a
downstream result is never mistaken for exact-SMPL. For exact joints, install the
SMPL model and wire it into carepd._smpl_pose_to_joints.
"""
from __future__ import annotations

import numpy as np

# SMPL 24-joint kinematic tree (parent index; -1 = root).
SMPL_PARENTS = np.array([-1, 0, 0, 0, 1, 2, 3, 4, 5, 6, 7, 8, 9, 9, 9, 12, 13, 14,
                         16, 17, 18, 19, 20, 21], dtype=int)

# Canonical rest skeleton: parent->child offsets (metres) in SMPL frame
# (x = left-lateral +, y = up +, z = forward +). Approximates SMPL's A-pose rest:
# legs straight down, arms out-and-down ~35 deg. Anthropometric, NOT the SMPL model.
_OFFSETS = np.array([
    [0.00,  0.00, 0.00],   # 0  pelvis
    [0.09, -0.01, 0.00],   # 1  L_hip
    [-0.09, -0.01, 0.00],  # 2  R_hip
    [0.00,  0.11, 0.00],   # 3  spine1
    [0.01, -0.40, 0.00],   # 4  L_knee
    [-0.01, -0.40, 0.00],  # 5  R_knee
    [0.00,  0.13, 0.00],   # 6  spine2
    [0.00, -0.41, 0.00],   # 7  L_ankle
    [0.00, -0.41, 0.00],   # 8  R_ankle
    [0.00,  0.05, 0.00],   # 9  spine3
    [0.00, -0.06, 0.13],   # 10 L_foot
    [0.00, -0.06, 0.13],   # 11 R_foot
    [0.00,  0.22, 0.00],   # 12 neck
    [0.07,  0.11, 0.00],   # 13 L_collar
    [-0.07, 0.11, 0.00],   # 14 R_collar
    [0.00,  0.12, 0.00],   # 15 head
    [0.11,  0.04, 0.00],   # 16 L_shoulder
    [-0.11, 0.04, 0.00],   # 17 R_shoulder
    [0.20, -0.14, 0.00],   # 18 L_elbow  (out + down)
    [-0.20, -0.14, 0.00],  # 19 R_elbow
    [0.18, -0.13, 0.00],   # 20 L_wrist
    [-0.18, -0.13, 0.00],  # 21 R_wrist
    [0.06, -0.04, 0.00],   # 22 L_hand
    [-0.06, -0.04, 0.00],  # 23 R_hand
], dtype=np.float64)


def _rest_joints() -> np.ndarray:
    """Absolute canonical rest joint positions (24,3) from the offset tree."""
    J = np.zeros((24, 3), dtype=np.float64)
    for j in range(24):
        p = SMPL_PARENTS[j]
        J[j] = _OFFSETS[j] if p < 0 else J[p] + _OFFSETS[j]
    return J


REST_JOINTS = _rest_joints()


def batch_rodrigues(aa: np.ndarray) -> np.ndarray:
    """Axis-angle (N,3) -> rotation matrices (N,3,3)."""
    theta = np.linalg.norm(aa, axis=1, keepdims=True)
    r = aa / np.clip(theta, 1e-8, None)
    cos = np.cos(theta)[:, :, None]
    sin = np.sin(theta)[:, :, None]
    N = aa.shape[0]
    K = np.zeros((N, 3, 3))
    K[:, 0, 1] = -r[:, 2]; K[:, 0, 2] = r[:, 1]
    K[:, 1, 0] = r[:, 2];  K[:, 1, 2] = -r[:, 0]
    K[:, 2, 0] = -r[:, 1]; K[:, 2, 1] = r[:, 0]
    eye = np.eye(3)[None]
    return eye + sin * K + (1 - cos) * (K @ K)


def smpl_joints_from_pose(pose: np.ndarray, trans: np.ndarray | None = None) -> np.ndarray:
    """Real axis-angle ``pose`` (T,72) [+ optional ``trans`` (T,3)] -> joints (T,24,3)
    via forward kinematics on the canonical rest skeleton (SMPL world frame)."""
    pose = np.asarray(pose, dtype=np.float64)
    if pose.ndim != 2 or pose.shape[1] < 72:
        raise ValueError(f"expected pose (T,>=72); got {pose.shape}")
    T = pose.shape[0]
    aa = pose[:, :72].reshape(T * 24, 3)
    R = batch_rodrigues(aa).reshape(T, 24, 3, 3)
    J = REST_JOINTS
    # Time-vectorized FK: loop over the 24 joints (sequential parent dependency),
    # batching all T frames per step with (T,3,3) matmuls. ~T x faster than a
    # per-frame Python loop.
    Grot = [None] * 24
    pos = [None] * 24
    Grot[0] = R[:, 0]                                   # (T,3,3)
    pos[0] = np.broadcast_to(J[0], (T, 3)).copy()       # (T,3)
    for j in range(1, 24):
        p = int(SMPL_PARENTS[j])
        Grot[j] = Grot[p] @ R[:, j]                     # (T,3,3)
        off = (J[j] - J[p])                             # (3,)
        pos[j] = pos[p] + (Grot[p] @ off)               # (T,3,3)@(3,) -> (T,3)
    out = np.stack(pos, axis=1)                         # (T,24,3)
    out = out - out[:, 0:1, :]                          # centre on pelvis
    if trans is not None:
        out = out + np.asarray(trans, dtype=np.float64)[:, None, :]
    return out


def sagittal_project(joints: np.ndarray) -> np.ndarray:
    """Project (T,24,3) SMPL-frame joints to a 2D SIDE view in the plane of walking.

    The walking direction is the principal axis of the pelvis's horizontal (x-z)
    trajectory; we project onto (forward, up) so stride (forward excursion) and foot
    lift (up) are both visible — the sagittal plane gait analysis needs. Returns
    (T,24,2) in normalized image coords (x right, y DOWN)."""
    pelvis_xz = joints[:, 0, [0, 2]]
    c = pelvis_xz.mean(axis=0)
    disp = pelvis_xz - c
    # principal direction of horizontal motion (fallback to +z if ~stationary)
    if disp.std() > 1e-6:
        u, s, vt = np.linalg.svd(disp, full_matrices=False)
        fwd = vt[0]
    else:
        fwd = np.array([0.0, 1.0])
    fwd = fwd / (np.linalg.norm(fwd) + 1e-9)
    xz = joints[:, :, [0, 2]]
    forward = xz @ fwd                     # (T,24) position along walking axis
    up = joints[:, :, 1]                   # (T,24) height
    img = np.stack([forward, -up], axis=-1)  # y up -> y down
    lo = img.reshape(-1, 2).min(axis=0)
    hi = img.reshape(-1, 2).max(axis=0)
    span = np.where((hi - lo) > 1e-6, hi - lo, 1.0)
    return (img - lo) / span


def carepd_record_to_blazepose(pose, trans, fps: float):
    """Full canonical-FK path: SMPL pose -> 24 joints -> sagittal 2D -> BlazePose 33.

    Returns (joints33 (T,33,3), vis (T,33)). Reuses carepd's SMPL->BlazePose name map
    for consistency; z carries the (unprojected) SMPL depth for optional 3D use."""
    from parkigait.carepd import (SMPL_JOINT_ORDER, SMPL_TO_BLAZEPOSE,
                                  _anchor_missing_landmarks)
    from parkigait.types import joint_index

    smpl = smpl_joints_from_pose(pose, trans)          # (T,24,3)
    xy = sagittal_project(smpl)                         # (T,24,2) normalized
    z = smpl[:, :, 2]
    T = smpl.shape[0]
    smpl_idx = {n: i for i, n in enumerate(SMPL_JOINT_ORDER)}
    out = np.zeros((T, 33, 3), dtype=np.float32)
    vis = np.zeros((T, 33), dtype=np.float32)
    for bp_name, jm in SMPL_TO_BLAZEPOSE.items():
        bp = joint_index(bp_name)
        if jm.smpl is not None and jm.smpl in smpl_idx:
            si = smpl_idx[jm.smpl]
            out[:, bp, 0] = xy[:, si, 0]
            out[:, bp, 1] = xy[:, si, 1]
            out[:, bp, 2] = z[:, si]
            vis[:, bp] = 0.9 if not jm.uncertain else 0.5
        else:
            vis[:, bp] = 0.2
    _anchor_missing_landmarks(out)
    return out, vis
