"""End-to-end ParkiGait pipeline: a source -> a PipelineReport.

    video file ──▶ pose (MediaPipe)         ─┐
    or                                        ├─▶ STTP (topology-preserving)
    synthetic severity ──▶ pose (synthetic) ─┘        │
                                                       ▼
                              gait features ──▶ severity/PD-sign ──▶ report

Everything runs locally. The report carries the research-only disclaimer and the
per-stage timings so the "edge/on-device" claims can be checked with real numbers.
"""
from __future__ import annotations

import time
from typing import Optional, Union

import numpy as np

from parkigait import DISCLAIMER
from parkigait.pose import MediaPipeBackend, SyntheticWalker
from parkigait.types import PipelineReport, PoseSequence


def _now() -> float:
    return time.perf_counter()


def analyze_pose(pose: PoseSequence, run_sttp: bool = True,
                 severity_model=None) -> PipelineReport:
    """Run the analysis stages on an already-extracted PoseSequence."""
    from parkigait.gaitfeat import extract_features

    timings: dict = {}
    warnings: list[str] = []

    t0 = _now()
    features = extract_features(pose)
    timings["gaitfeat_ms"] = (_now() - t0) * 1000.0

    # STTP on a representative (mid-sequence) frame's projected joints.
    sttp_result = None
    if run_sttp:
        try:
            from parkigait import sttp as sttp_mod
            t0 = _now()
            mid = pose.n_frames // 2
            joints2d = pose.joints[mid, :, :2]
            pts, is_body = sttp_mod.frame_tokens(joints2d, seed=0)
            sttp_result = sttp_mod.sttp_select(pts, keep_fraction=0.5)
            # attach measured body-recall for the report/figure
            body_kept = int(sttp_result.kept_mask[is_body].sum())
            n_body = int(is_body.sum())
            sttp_result.detail += (
                f" | body_recall={body_kept}/{n_body}="
                f"{(body_kept / n_body) if n_body else 0:.2f}")
            sttp_result.meta_is_body = is_body  # type: ignore[attr-defined]
            sttp_result.meta_points = pts  # type: ignore[attr-defined]
            timings["sttp_ms"] = (_now() - t0) * 1000.0
        except Exception as e:  # STTP is an add-on; never let it break the core read
            warnings.append(f"STTP skipped: {type(e).__name__}: {e}")

    # severity / PD-sign
    from parkigait.severity import load_or_train
    model = severity_model or load_or_train()
    t0 = _now()
    severity = model.predict(features)
    timings["severity_ms"] = (_now() - t0) * 1000.0

    if features.confidence < 0.35:
        warnings.append("Low signal quality: few clean walking cycles detected; "
                        "the estimate is unreliable.")
    if features.step_count < 4:
        warnings.append(f"Only {features.step_count} steps detected; need a longer "
                        "clip with several full strides.")

    return PipelineReport(
        source=pose.source, features=features, severity=severity,
        sttp=sttp_result, pose=pose, timings_ms=timings, warnings=warnings,
        disclaimer=DISCLAIMER)


def analyze_video(path: str, stride: int = 2, max_frames: Optional[int] = None,
                  run_sttp: bool = True, severity_model=None) -> PipelineReport:
    """Scan a real walking video with MediaPipe, then analyze it."""
    t0 = _now()
    pose = MediaPipeBackend(stride=stride, max_frames=max_frames).extract(path)
    report = analyze_pose(pose, run_sttp=run_sttp, severity_model=severity_model)
    report.timings_ms["pose_extract_ms"] = (_now() - t0) * 1000.0 - sum(
        v for k, v in report.timings_ms.items())
    report.timings_ms["per_frame_ms"] = (
        report.timings_ms.get("pose_extract_ms", 0.0) / max(1, pose.n_frames))
    return report


def analyze_synthetic(severity: float = 0.5, seed: int = 0, duration_s: float = 8.0,
                      run_sttp: bool = True, severity_model=None) -> PipelineReport:
    """Analyze a synthetic walker of known severity (no camera needed)."""
    pose = SyntheticWalker(severity=severity, seed=seed).generate(duration_s=duration_s)
    return analyze_pose(pose, run_sttp=run_sttp, severity_model=severity_model)


def analyze(source: Union[str, PoseSequence, float], **kw) -> PipelineReport:
    """Convenience dispatcher: a path -> video; a PoseSequence -> direct; a float
    in [0,1] -> synthetic walker of that severity."""
    if isinstance(source, PoseSequence):
        return analyze_pose(source, **{k: v for k, v in kw.items()
                                       if k in ("run_sttp", "severity_model")})
    if isinstance(source, (int, float)):
        return analyze_synthetic(severity=float(source), **kw)
    return analyze_video(str(source), **kw)
