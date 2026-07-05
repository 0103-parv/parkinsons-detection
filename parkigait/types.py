"""The shared data contract for ParkiGait.

Every module in the pipeline (pose -> sttp -> gaitfeat -> severity -> report)
speaks in terms of the dataclasses defined here. Keeping the contract in one
small, dependency-light module (numpy only) means the modules can be built and
tested independently without drifting on data shapes.

Coordinate convention (used everywhere downstream):
  - joints[t, j] = (x, y, z)
  - x, y are image-plane coordinates NORMALIZED to [0, 1] (x right, y DOWN, as
    in image space -- so a smaller y is physically higher off the ground).
  - z is the backend's relative depth (roughly metres, sign backend-defined);
    treated as auxiliary and never required by the gait features.
  - visibility[t, j] in [0, 1] is the backend's per-joint confidence.
"""
from __future__ import annotations

from dataclasses import dataclass, field
from typing import Optional

import numpy as np

# --------------------------------------------------------------------------- #
# Canonical skeleton: MediaPipe BlazePose 33-landmark topology.               #
# The index order MUST match mediapipe.solutions.pose so the backend and the  #
# feature extractor agree. Names double as the STTP graph node labels.        #
# --------------------------------------------------------------------------- #
BLAZEPOSE_JOINTS: list[str] = [
    "NOSE",              # 0
    "LEFT_EYE_INNER",    # 1
    "LEFT_EYE",          # 2
    "LEFT_EYE_OUTER",    # 3
    "RIGHT_EYE_INNER",   # 4
    "RIGHT_EYE",         # 5
    "RIGHT_EYE_OUTER",   # 6
    "LEFT_EAR",          # 7
    "RIGHT_EAR",         # 8
    "MOUTH_LEFT",        # 9
    "MOUTH_RIGHT",       # 10
    "LEFT_SHOULDER",     # 11
    "RIGHT_SHOULDER",    # 12
    "LEFT_ELBOW",        # 13
    "RIGHT_ELBOW",       # 14
    "LEFT_WRIST",        # 15
    "RIGHT_WRIST",       # 16
    "LEFT_PINKY",        # 17
    "RIGHT_PINKY",       # 18
    "LEFT_INDEX",        # 19
    "RIGHT_INDEX",       # 20
    "LEFT_THUMB",        # 21
    "RIGHT_THUMB",       # 22
    "LEFT_HIP",          # 23
    "RIGHT_HIP",         # 24
    "LEFT_KNEE",         # 25
    "RIGHT_KNEE",        # 26
    "LEFT_ANKLE",        # 27
    "RIGHT_ANKLE",       # 28
    "LEFT_HEEL",         # 29
    "RIGHT_HEEL",        # 30
    "LEFT_FOOT_INDEX",   # 31
    "RIGHT_FOOT_INDEX",  # 32
]

# The physical skeleton edges (kinematic tree). STTP does NOT get to see these
# -- it must rediscover connectivity from geometry -- but the feature extractor
# and the visual overlay use them.
BLAZEPOSE_EDGES: list[tuple[int, int]] = [
    (11, 12), (11, 13), (13, 15), (12, 14), (14, 16),      # arms + shoulders
    (11, 23), (12, 24), (23, 24),                          # torso
    (23, 25), (25, 27), (27, 29), (29, 31), (27, 31),      # left leg
    (24, 26), (26, 28), (28, 30), (30, 32), (28, 32),      # right leg
    (0, 11), (0, 12),                                      # head to shoulders
]


def joint_index(name: str) -> int:
    """Index of a named BlazePose joint (raises if unknown)."""
    return BLAZEPOSE_JOINTS.index(name)


# The 7 clinical gait features, in the FIXED order the severity model expects.
# This order is contractual: severity.py indexes vectors in exactly this order.
GAIT_FEATURE_ORDER: list[str] = [
    "gait_speed",       # forward progression speed (body-heights / s), higher = better
    "cadence",          # steps per minute
    "stride_length",    # normalized stride length (body heights), higher = better
    "stride_time_var",  # coefficient of variation of stride time (rhythm instability)
    "asymmetry",        # left/right step-time or amplitude asymmetry (0 = symmetric)
    "arm_swing",        # mean arm-swing amplitude (normalized), higher = better
    "fog_index",        # freezing-of-gait index (freeze-band power ratio)
]


@dataclass
class PoseSequence:
    """A time series of skeletons extracted from a walking clip.

    joints:      (T, J, 3) float32 -- normalized image coords + relative depth.
    visibility:  (T, J)   float32 in [0, 1] -- per-joint confidence.
    fps:         frames per second of the sequence.
    joint_names: length-J list, canonically ``BLAZEPOSE_JOINTS``.
    source:      provenance string (a path, or ``"synthetic:..."``).
    meta:        free-form dict (severity used to synthesize, detector name, ...).
    """
    joints: np.ndarray
    visibility: np.ndarray
    fps: float
    joint_names: list[str] = field(default_factory=lambda: list(BLAZEPOSE_JOINTS))
    source: str = ""
    meta: dict = field(default_factory=dict)

    def __post_init__(self) -> None:
        self.joints = np.asarray(self.joints, dtype=np.float32)
        self.visibility = np.asarray(self.visibility, dtype=np.float32)
        if self.joints.ndim != 3 or self.joints.shape[2] != 3:
            raise ValueError(f"joints must be (T, J, 3); got {self.joints.shape}")
        if self.visibility.shape != self.joints.shape[:2]:
            raise ValueError(
                f"visibility {self.visibility.shape} must match joints[:2] "
                f"{self.joints.shape[:2]}")

    @property
    def n_frames(self) -> int:
        return int(self.joints.shape[0])

    @property
    def n_joints(self) -> int:
        return int(self.joints.shape[1])

    @property
    def duration_s(self) -> float:
        return self.n_frames / self.fps if self.fps else 0.0

    def idx(self, name: str) -> int:
        return self.joint_names.index(name)

    def track(self, name: str) -> np.ndarray:
        """(T, 3) trajectory of a named joint."""
        return self.joints[:, self.idx(name), :]

    def mean_visibility(self, name: str) -> float:
        return float(self.visibility[:, self.idx(name)].mean())


@dataclass
class STTPResult:
    """Output of Spectral-Topological Token Preservation on a token set.

    kept_mask:  (N,) bool -- True for tokens preserved (on the body manifold).
    fiedler:    (N,) float -- the Fiedler vector (2nd-smallest Laplacian eigvec).
    eigvals:    the smallest few Laplacian eigenvalues (spectral gap diagnostics).
    n_total / n_kept: token counts.
    detail:     human-readable summary.
    """
    kept_mask: np.ndarray
    fiedler: np.ndarray
    eigvals: np.ndarray
    n_total: int
    n_kept: int
    detail: str = ""

    @property
    def keep_fraction(self) -> float:
        return self.n_kept / self.n_total if self.n_total else 0.0


@dataclass
class GaitFeatures:
    """Clinically-motivated gait descriptors extracted from a PoseSequence.

    All fields are plain floats. ``as_vector`` returns them in
    ``GAIT_FEATURE_ORDER`` -- the contract the severity model relies on.
    ``confidence`` is a 0..1 self-assessment (how much of the clip had a clean,
    walking-plausible skeleton); low confidence should suppress any downstream
    claim.
    """
    gait_speed: float
    cadence: float
    stride_length: float
    stride_time_var: float
    asymmetry: float
    arm_swing: float
    fog_index: float
    # diagnostics beyond the 7-vector (not fed to the model, shown in reports)
    step_count: int = 0
    double_support: float = 0.0
    trunk_sway: float = 0.0
    cadence_left: float = 0.0
    cadence_right: float = 0.0
    confidence: float = 0.0
    notes: list[str] = field(default_factory=list)

    def as_vector(self) -> np.ndarray:
        return np.array([getattr(self, k) for k in GAIT_FEATURE_ORDER], dtype=np.float64)

    def as_dict(self) -> dict:
        return {k: getattr(self, k) for k in GAIT_FEATURE_ORDER}


@dataclass
class SeverityEstimate:
    """The model's read on a GaitFeatures vector.

    p_pd:      P(features resemble the PD-sign class) in [0, 1].
    severity:  a 0..4 UPDRS-gait-*like* number. This is a MODEL SCALE, not a
               clinical UPDRS score, unless ``calibrated_on`` names a real,
               labelled dataset it was fit against.
    label:     short human string.
    calibrated_on: dataset the severity scale was calibrated on, or "" / "synthetic".
    contributions: per-feature signed contribution to the decision (explainability).
    """
    p_pd: float
    severity: float
    label: str
    calibrated_on: str = ""
    contributions: dict = field(default_factory=dict)

    @property
    def is_calibrated(self) -> bool:
        return bool(self.calibrated_on) and self.calibrated_on != "synthetic"


@dataclass
class PipelineReport:
    """Everything the end-to-end pipeline produces for one clip."""
    source: str
    features: GaitFeatures
    severity: SeverityEstimate
    sttp: Optional[STTPResult] = None
    pose: Optional[PoseSequence] = None
    timings_ms: dict = field(default_factory=dict)
    warnings: list[str] = field(default_factory=list)
    disclaimer: str = ""

    def summary(self) -> dict:
        """A JSON-serializable digest (no heavy arrays)."""
        return {
            "source": self.source,
            "features": self.features.as_dict(),
            "step_count": self.features.step_count,
            "feature_confidence": round(self.features.confidence, 3),
            "p_pd": round(self.severity.p_pd, 3),
            "severity_0_4": round(self.severity.severity, 2),
            "severity_scale": self.severity.calibrated_on or "uncalibrated/synthetic",
            "label": self.severity.label,
            "sttp_keep_fraction": (
                round(self.sttp.keep_fraction, 3) if self.sttp else None),
            "timings_ms": {k: round(v, 1) for k, v in self.timings_ms.items()},
            "warnings": self.warnings,
            "disclaimer": self.disclaimer,
        }
