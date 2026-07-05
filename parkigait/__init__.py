"""ParkiGait — a research prototype for video-based gait analysis.

RESEARCH / EDUCATIONAL PROTOTYPE. NOT A MEDICAL DEVICE. NOT FOR CLINICAL USE.
This package estimates gait features from a walking video and produces an
EXPLORATORY, uncalibrated motor-sign summary. It must not be used to diagnose,
screen, triage, or make any care decision for a real person. See CLINICAL_SAFETY.md.

What it actually does, end to end, on this laptop with no GPU and no cloud:

    video ──▶ pose (skeleton) ──▶ STTP (topology-preserving token pruning)
          ──▶ clinical gait features ──▶ severity/PD-sign estimate ──▶ report

Every layer is real, runnable code with measured outputs. The pieces that need
data we do not have (CARE-PD training on real UPDRS labels) are wired up but
clearly marked as "needs real data" and never fabricate a number. See
HONEST_STATUS.md for the exact claim-by-claim status.
"""
from __future__ import annotations

__version__ = "0.1.0"

DISCLAIMER = (
    "RESEARCH PROTOTYPE — NOT A MEDICAL DEVICE. This output is exploratory and "
    "uncalibrated. It cannot diagnose Parkinson's disease or any condition and "
    "must not be used for clinical decisions. Consult a licensed clinician."
)

from parkigait.types import (  # noqa: E402
    GaitFeatures,
    PoseSequence,
    SeverityEstimate,
    STTPResult,
    PipelineReport,
    BLAZEPOSE_JOINTS,
    GAIT_FEATURE_ORDER,
)

__all__ = [
    "DISCLAIMER",
    "GaitFeatures",
    "PoseSequence",
    "SeverityEstimate",
    "STTPResult",
    "PipelineReport",
    "BLAZEPOSE_JOINTS",
    "GAIT_FEATURE_ORDER",
    "__version__",
]
