"""Pose extraction: video -> skeleton, plus a synthetic walker for offline testing.

Two backends, one interface (`PoseBackend.extract(...) -> PoseSequence`):

  MediaPipeBackend   real BlazePose (33 landmarks) on a real walking video, CPU,
                     no GPU. This is the "scan someone walking" path.
  SyntheticWalker    a small forward-kinematic gait model that emits a physically
                     plausible walking skeleton with a controllable PD severity
                     (0 = healthy, 1 = severe). Deterministic given a seed, so the
                     whole pipeline has a ground-truth signal to test against with
                     no camera and no data download.

The synthetic walker is honest about what it is: a *method test bench*, not a
patient. It exists so `gaitfeat`, `sttp`, `severity`, and `eval` can be built and
verified end-to-end, and so eval can report correlation against a KNOWN severity.
"""
from __future__ import annotations

import math
import urllib.request
from pathlib import Path
from typing import Optional

import numpy as np

from parkigait.types import BLAZEPOSE_JOINTS, PoseSequence, joint_index

_MODEL_DIR = Path(__file__).resolve().parent / "models"
_POSE_MODEL = _MODEL_DIR / "pose_landmarker_full.task"
_POSE_MODEL_URL = (
    "https://storage.googleapis.com/mediapipe-models/pose_landmarker/"
    "pose_landmarker_full/float16/latest/pose_landmarker_full.task"
)


def ensure_pose_model(path: Path = _POSE_MODEL, url: str = _POSE_MODEL_URL) -> Path:
    """Download the BlazePose ``.task`` bundle once (≈9 MB); return its path.

    MediaPipe 0.10.x ships only the Tasks API, which needs an explicit model
    bundle. Cached locally so subsequent runs are fully offline / on-device.
    """
    if path.exists() and path.stat().st_size > 1_000_000:
        return path
    path.parent.mkdir(parents=True, exist_ok=True)
    urllib.request.urlretrieve(url, path)
    return path

# joint indices we animate / read (others are placed for a plausible skeleton)
_NOSE = joint_index("NOSE")
_LSH, _RSH = joint_index("LEFT_SHOULDER"), joint_index("RIGHT_SHOULDER")
_LEL, _REL = joint_index("LEFT_ELBOW"), joint_index("RIGHT_ELBOW")
_LWR, _RWR = joint_index("LEFT_WRIST"), joint_index("RIGHT_WRIST")
_LHIP, _RHIP = joint_index("LEFT_HIP"), joint_index("RIGHT_HIP")
_LKNE, _RKNE = joint_index("LEFT_KNEE"), joint_index("RIGHT_KNEE")
_LANK, _RANK = joint_index("LEFT_ANKLE"), joint_index("RIGHT_ANKLE")
_LHEE, _RHEE = joint_index("LEFT_HEEL"), joint_index("RIGHT_HEEL")
_LFOO, _RFOO = joint_index("LEFT_FOOT_INDEX"), joint_index("RIGHT_FOOT_INDEX")


class PoseBackend:
    """Interface: turn a source into a PoseSequence of BLAZEPOSE_JOINTS."""

    def extract(self, source) -> PoseSequence:  # pragma: no cover - interface
        raise NotImplementedError


# --------------------------------------------------------------------------- #
# Real video -> skeleton                                                      #
# --------------------------------------------------------------------------- #
class MediaPipeBackend(PoseBackend):
    """MediaPipe BlazePose over a video file. CPU-only, fully on-device.

    Reads frames with OpenCV, runs the pose graph per frame, and returns a
    (T, 33, 3) PoseSequence with per-joint visibility. Frames with no detected
    person are filled with the last valid pose (and flagged in meta), so the time
    axis stays uniform for the feature extractor.
    """

    def __init__(self, min_detection_confidence: float = 0.5,
                 min_tracking_confidence: float = 0.5, max_frames: Optional[int] = None,
                 stride: int = 1, model_path: Optional[Path] = None):
        self.min_detection_confidence = min_detection_confidence
        self.min_tracking_confidence = min_tracking_confidence
        self.max_frames = max_frames
        self.stride = max(1, int(stride))
        self.model_path = Path(model_path) if model_path else _POSE_MODEL

    def extract(self, source) -> PoseSequence:
        import cv2  # local import so the synthetic path needs no cv2/mediapipe
        import mediapipe as mp
        from mediapipe.tasks import python as mp_python
        from mediapipe.tasks.python import vision as mp_vision

        ensure_pose_model(self.model_path)
        cap = cv2.VideoCapture(str(source))
        if not cap.isOpened():
            raise FileNotFoundError(f"could not open video: {source}")
        fps = cap.get(cv2.CAP_PROP_FPS) or 30.0
        eff_fps = fps / self.stride

        joints_seq: list[np.ndarray] = []
        vis_seq: list[np.ndarray] = []
        n_missing = 0
        last_j = np.zeros((33, 3), dtype=np.float32)
        last_v = np.zeros(33, dtype=np.float32)

        options = mp_vision.PoseLandmarkerOptions(
            base_options=mp_python.BaseOptions(model_asset_path=str(self.model_path)),
            running_mode=mp_vision.RunningMode.VIDEO,
            num_poses=1,
            min_pose_detection_confidence=self.min_detection_confidence,
            min_tracking_confidence=self.min_tracking_confidence,
        )
        landmarker = mp_vision.PoseLandmarker.create_from_options(options)
        try:
            frame_i = 0
            kept = 0
            while True:
                ok, frame = cap.read()
                if not ok:
                    break
                if frame_i % self.stride != 0:
                    frame_i += 1
                    continue
                # timestamp must be strictly increasing (ms) in VIDEO mode
                ts_ms = int(round(frame_i / fps * 1000.0))
                frame_i += 1
                rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
                mp_image = mp.Image(image_format=mp.ImageFormat.SRGB, data=rgb)
                res = landmarker.detect_for_video(mp_image, ts_ms)
                if not res.pose_landmarks:
                    n_missing += 1
                    joints_seq.append(last_j.copy())
                    vis_seq.append(last_v.copy())
                else:
                    lm = res.pose_landmarks[0]  # first (and only) detected pose
                    j = np.array([[p.x, p.y, p.z] for p in lm], dtype=np.float32)
                    v = np.array([getattr(p, "visibility", 1.0) for p in lm],
                                 dtype=np.float32)
                    joints_seq.append(j)
                    vis_seq.append(v)
                    last_j, last_v = j, v
                kept += 1
                if self.max_frames and kept >= self.max_frames:
                    break
        finally:
            landmarker.close()
            cap.release()

        if not joints_seq:
            raise ValueError(f"no frames decoded from {source}")

        return PoseSequence(
            joints=np.stack(joints_seq),
            visibility=np.stack(vis_seq),
            fps=float(eff_fps),
            source=str(source),
            meta={"backend": "mediapipe-tasks", "n_frames": len(joints_seq),
                  "n_missing": n_missing, "model": self.model_path.name},
        )


# --------------------------------------------------------------------------- #
# Synthetic walker (forward-kinematic gait model)                             #
# --------------------------------------------------------------------------- #
class SyntheticWalker(PoseBackend):
    """A deterministic, side-view (treadmill-style) walking skeleton generator.

    Severity 0..1 dials the canonical Parkinsonian gait deficits, each grounded in
    the clinical literature:
      - reduced gait speed & stride length  (bradykinesia / hypokinesia)
      - reduced arm swing, worse on one side (asymmetric reduced arm swing)
      - increased stride-time variability     (rhythm instability)
      - left/right amplitude asymmetry         (asymmetric onset)
      - intermittent freezing at moderate+ severity (festination / FOG)

    The camera is side-on and the pelvis is horizontally centered (as if on a
    treadmill or with a panning camera), so the feet and arms oscillate in place.
    This keeps the person in frame and gives clean, extractable gait signals; the
    same feature extractor also runs on real, translating video.
    """

    GROUND_Y = 0.93          # image-space y (down) of the floor
    HIP_Y = 0.55             # hip line height
    THIGH = 0.16
    SHANK = 0.16
    FOOT = 0.05
    HIP_HALF = 0.05
    SHOULDER_Y = 0.30
    SHOULDER_HALF = 0.075
    UPPER_ARM = 0.11
    FOREARM = 0.10
    HEAD_UP = 0.10

    def __init__(self, severity: float = 0.0, seed: int = 0):
        self.severity = float(np.clip(severity, 0.0, 1.0))
        self.seed = int(seed)

    def extract(self, source=None) -> PoseSequence:  # source ignored
        return self.generate()

    def generate(self, duration_s: float = 8.0, fps: float = 30.0) -> PoseSequence:
        rng = np.random.default_rng(self.seed)
        s = self.severity
        T = int(round(duration_s * fps))

        # --- gait parameters as a function of severity (literature-directional) --
        # clinical cadence = TOTAL steps/min counting both feet (healthy ~112,
        # dropping with severity). Each foot cycles at half that rate.
        cadence = 112.0 - 24.0 * s + rng.normal(0, 2.5)
        step_hz = cadence / 120.0                             # per-foot cycle frequency
        stride_amp = 0.16 * (1.0 - 0.45 * s)                 # hip flexion amplitude (rad)
        knee_amp = 0.9 * (1.0 - 0.25 * s)
        arm_amp_base = 0.13 * (1.0 - 0.70 * s)               # arm swing shrinks a lot
        asym = 0.05 + 0.35 * s                               # L/R amplitude asymmetry
        var_target = 0.02 + 0.10 * s                         # stride-time CV target
        foot_lift = 0.045 * (1.0 - 0.55 * s)                 # foot clearance shrinks (shuffle)

        # --- per-frame phase with stride-time jitter (drives variability) --------
        # advance phase each frame by 2*pi*step_hz*dt, with multiplicative noise so
        # inter-step intervals have coefficient of variation ~ var_target.
        dt = 1.0 / fps
        phase = np.zeros(T)
        # freeze gate: at moderate+ severity, occasional multi-frame near-stops
        freeze_gate = np.ones(T)
        if s > 0.45:
            n_freeze = rng.integers(1, 1 + int(3 * s))
            for _ in range(int(n_freeze)):
                start = int(rng.uniform(0.1, 0.85) * T)
                dur = int(rng.uniform(0.4, 1.2) * fps)
                freeze_gate[start:start + dur] = 0.12
        ph = 0.0
        for t in range(T):
            jitter = 1.0 + rng.normal(0, var_target)
            ph += 2.0 * math.pi * step_hz * dt * max(0.2, jitter) * freeze_gate[t]
            phase[t] = ph

        joints = np.zeros((T, 33, 3), dtype=np.float32)
        vis = np.full((T, 33), 0.98, dtype=np.float32)

        # side-view: everyone shares x ~ 0.5 with small forward/back excursions
        cx = 0.5
        # small global vertical bob at 2x step freq (pelvis rise/fall)
        for t in range(T):
            th = phase[t]
            g = freeze_gate[t]
            bob = 0.012 * math.cos(2.0 * th) * (0.4 + 0.6 * g)
            hip_cx = cx
            hip_y = self.HIP_Y + bob

            # hips
            joints[t, _LHIP] = (hip_cx - self.HIP_HALF, hip_y, 0.0)
            joints[t, _RHIP] = (hip_cx + self.HIP_HALF, hip_y, 0.0)

            # legs: right leg leads (phase th), left leg opposite (th + pi)
            for hip_i, kne_i, ank_i, hee_i, foo_i, leg_phase, side_asym in (
                (_RHIP, _RKNE, _RANK, _RHEE, _RFOO, th, 1.0),
                (_LHIP, _LKNE, _LANK, _LHEE, _LFOO, th + math.pi, 1.0 - asym),
            ):
                hx, hy = joints[t, hip_i, 0], joints[t, hip_i, 1]
                a_hip = stride_amp * side_asym * g
                a_kne = knee_amp * side_asym
                hip_ang = a_hip * math.sin(leg_phase)           # thigh swing (rad, fwd +)
                # knee flexes during swing (when thigh swings forward)
                kne_ang = a_kne * max(0.0, math.sin(leg_phase + 0.5 * math.pi)) * 0.5
                # forward-kinematics in image space (y down): thigh then shank
                kx = hx + self.THIGH * math.sin(hip_ang)
                ky = hy + self.THIGH * math.cos(hip_ang)
                ax = kx + self.SHANK * math.sin(hip_ang - kne_ang)
                ay = ky + self.SHANK * math.cos(hip_ang - kne_ang)
                # foot clearance: lift ankle during swing (raise = decrease y)
                lift = foot_lift * max(0.0, math.sin(leg_phase)) * g
                ay -= lift
                joints[t, kne_i] = (kx, ky, 0.0)
                joints[t, ank_i] = (ax, ay, 0.0)
                joints[t, hee_i] = (ax - 0.015, ay + 0.01, 0.0)
                joints[t, foo_i] = (ax + self.FOOT, ay + 0.008, 0.0)

            # shoulders / torso / head
            joints[t, _LSH] = (cx - self.SHOULDER_HALF, self.SHOULDER_Y + bob, 0.0)
            joints[t, _RSH] = (cx + self.SHOULDER_HALF, self.SHOULDER_Y + bob, 0.0)
            joints[t, _NOSE] = (cx, self.SHOULDER_Y - self.HEAD_UP + bob, 0.0)

            # arms swing opposite to same-side leg; amplitude asymmetric like legs
            for sh_i, el_i, wr_i, arm_phase, arm_asym in (
                (_RSH, _REL, _RWR, th + math.pi, 1.0),        # right arm opposite right leg
                (_LSH, _LEL, _LWR, th, 1.0 - asym),           # left arm opposite left leg
            ):
                sx, sy = joints[t, sh_i, 0], joints[t, sh_i, 1]
                a_arm = arm_amp_base * arm_asym * g
                swing = a_arm * math.sin(arm_phase)
                ex = sx + self.UPPER_ARM * math.sin(swing)
                ey = sy + self.UPPER_ARM * math.cos(swing)
                wx = ex + self.FOREARM * math.sin(swing * 0.8)
                wy = ey + self.FOREARM * math.cos(swing * 0.8)
                joints[t, el_i] = (ex, ey, 0.0)
                joints[t, wr_i] = (wx, wy, 0.0)

            # place remaining face/hand landmarks near their anchors for a full skeleton
            _fill_minor_joints(joints[t])

        # a little observation noise so it isn't unnaturally perfect
        joints[:, :, :2] += rng.normal(0, 0.0015, size=(T, 33, 2)).astype(np.float32)

        return PoseSequence(
            joints=joints, visibility=vis, fps=fps,
            source=f"synthetic:severity={self.severity:.2f}:seed={self.seed}",
            meta={"backend": "synthetic", "severity": self.severity,
                  "cadence_target": cadence, "seed": self.seed},
        )


def _fill_minor_joints(frame: np.ndarray) -> None:
    """Place eyes/ears/mouth/fingers relative to nose/wrists so the 33-landmark
    skeleton is complete for the overlay and the STTP graph (these joints are not
    used by the gait features)."""
    nose = frame[_NOSE]
    for idx, dx, dy in ((1, -0.01, -0.01), (2, -0.02, -0.01), (3, -0.03, -0.01),
                        (4, 0.01, -0.01), (5, 0.02, -0.01), (6, 0.03, -0.01),
                        (7, -0.04, 0.0), (8, 0.04, 0.0),
                        (9, -0.015, 0.02), (10, 0.015, 0.02)):
        frame[idx] = (nose[0] + dx, nose[1] + dy, 0.0)
    for wr, fingers in ((_LWR, (17, 19, 21)), (_RWR, (18, 20, 22))):
        w = frame[wr]
        for k, fi in enumerate(fingers):
            frame[fi] = (w[0] + 0.01 * (k - 1), w[1] + 0.02, 0.0)


def synthetic_cohort(n_control: int = 40, n_pd: int = 40, seed: int = 0,
                     duration_s: float = 8.0, fps: float = 30.0):
    """Generate a labelled cohort of synthetic walkers for evaluation.

    Returns list of (PoseSequence, severity, label) where label is 0 (control) or
    1 (PD-sign). Controls get severity in [0, 0.12] (near-healthy jitter); PD
    subjects get severity in [0.2, 1.0]. This is the KNOWN ground truth eval
    correlates against -- explicitly synthetic, never presented as clinical.
    """
    rng = np.random.default_rng(seed)
    out = []
    # Deliberately OVERLAPPING severity ranges (controls up to 0.18, PD from 0.12)
    # plus ~12% atypical subjects whose label is flipped relative to severity, so
    # the task is honestly hard and the model is NOT trivially separable. A believable
    # AUC/correlation beats a suspiciously perfect one.
    n = n_control + n_pd
    atypical = set(rng.choice(n, size=int(0.12 * n), replace=False).tolist())
    for i in range(n_control):
        sev = float(rng.uniform(0.0, 0.18))
        label = 1 if i in atypical else 0                    # a few "false alarms"
        out.append((SyntheticWalker(sev, seed=1000 + i).generate(duration_s, fps), sev, label))
    for i in range(n_pd):
        sev = float(rng.uniform(0.12, 1.0))
        label = 0 if (n_control + i) in atypical else 1      # a few "missed" mild cases
        out.append((SyntheticWalker(sev, seed=2000 + i).generate(duration_s, fps), sev, label))
    return out
