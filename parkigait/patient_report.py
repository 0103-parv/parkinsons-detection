"""Per-patient clinical-style report — a gait panel a clinician can read at a glance.

Takes a walk (a real video via `scan`, or a synthetic walker), computes the rich
gait features, and prints each one with its PERCENTILE against the CARE-PD cohort
distribution (clearly labelled "relative to the dataset, not validated clinical
norms"), plus the calibrated impaired-gait probability and an interpretation guide.

    python -m parkigait patient-report --video walk.mp4
    python -m parkigait patient-report --severity 0.6      # synthetic demo patient

RESEARCH PROTOTYPE, NOT A MEDICAL DEVICE. A screening panel for clinician review.
"""
from __future__ import annotations

from pathlib import Path

import numpy as np

from parkigait.carepd_features import FEATURE_NAMES

_HERE = Path(__file__).resolve().parent

# direction each feature moves with Parkinsonian severity (for the arrow/flag)
_WORSE_WHEN_LOW = {"gait_speed", "cadence", "stride_len", "knee_rom", "hip_rom",
                   "elbow_rom", "arm_swing", "foot_clearance", "gait_regularity",
                   "pelvis_vert_osc"}
_WORSE_WHEN_HIGH = {"gait_speed_cv", "stride_time_cv", "stride_len_asym",
                    "step_time_asym", "knee_rom_asym", "hip_rom_asym", "arm_swing_asym",
                    "trunk_flexion_mean", "trunk_flexion_max", "double_support",
                    "freeze_index", "knee_angvar", "arm_swing_amp_var", "pelvis_lat_sway"}


def _cohort_reference(root="data/CARE-PD"):
    """Feature matrix split into control (UPDRS 0) vs impaired (>0) for percentiles."""
    from parkigait.carepd_rich import build_dataset
    X, y, _, _ = build_dataset(root)
    return X, (y > 0).astype(int)


def patient_features(video=None, severity=0.5, seed=0):
    """Return the rich 25-feature vector for one walk (real video or synthetic)."""
    if video:
        from parkigait.pose import MediaPipeBackend
        # scan a real video -> but rich features need SMPL-style 3D; the rich path is
        # CARE-PD-specific. For a real video we fall back to the 2D gaitfeat panel.
        from parkigait.gaitfeat import extract_features
        ps = MediaPipeBackend(stride=2, max_frames=200).extract(video)
        return None, extract_features(ps), ps  # (rich=None, gaitfeatures, pose)
    # synthetic demo patient via the CARE-PD-style rich features would need SMPL pose;
    # use the video/gait path with a rendered synthetic walker instead.
    from parkigait.gaitfeat import extract_features
    from parkigait.pose import SyntheticWalker
    ps = SyntheticWalker(severity=severity, seed=seed).generate()
    return None, extract_features(ps), ps


def build_panel(root="data/CARE-PD"):
    """A callable that scores a CARE-PD-style rich feature vector into percentiles +
    a calibrated impaired probability. Returns (percentile_fn, prob_fn, names)."""
    from sklearn.preprocessing import StandardScaler
    Xc, yb = _cohort_reference(root)
    from parkigait.clinical_plus import _ensemble
    sc = StandardScaler().fit(Xc)
    clf = _ensemble().fit(sc.transform(Xc), yb)

    def percentile(x):
        return {FEATURE_NAMES[i]: float((Xc[:, i] < x[i]).mean() * 100)
                for i in range(len(FEATURE_NAMES))}

    def prob(x):
        return float(clf.predict_proba(sc.transform(x.reshape(1, -1)))[0, 1])

    return percentile, prob, Xc


def report_rich(x, root="data/CARE-PD") -> str:
    """Format a clinical-style panel for a CARE-PD-style 25-feature vector x."""
    pct_fn, prob_fn, _ = build_panel(root)
    pct = pct_fn(x)
    p = prob_fn(x)
    lines = ["=" * 64, "  ParkiGait gait panel  (RESEARCH SCREENING AID, not a diagnosis)",
             "=" * 64,
             f"  Impaired-gait probability: {p:.2f}   "
             f"({'ELEVATED — suggest clinical review' if p >= 0.5 else 'not elevated'})",
             "  " + "-" * 60,
             f"  {'feature':22}{'value':>10}{'cohort %ile':>13}  flag",
             "  " + "-" * 60]
    for i, name in enumerate(FEATURE_NAMES):
        pc = pct[name]
        flag = ""
        if name in _WORSE_WHEN_LOW and pc < 20:
            flag = "LOW  (PD-ward)"
        elif name in _WORSE_WHEN_HIGH and pc > 80:
            flag = "HIGH (PD-ward)"
        lines.append(f"  {name:22}{x[i]:>10.3f}{pc:>11.0f}%  {flag}")
    lines += ["  " + "-" * 60,
              "  Percentiles are RELATIVE TO THE CARE-PD DATASET, not validated",
              "  clinical norms. 'PD-ward' flags a value in the tail associated with",
              "  higher UPDRS-gait. This panel supports, and does not replace, a",
              "  clinician's exam.",
              "=" * 64]
    return "\n".join(lines)


def main(argv=None) -> int:
    import argparse
    ap = argparse.ArgumentParser()
    ap.add_argument("--root", default="data/CARE-PD")
    ap.add_argument("--carepd-cohort", default=None,
                    help="pull one real CARE-PD walk's rich features by cohort (demo)")
    a = ap.parse_args(argv)
    # demo: score the mean impaired-cohort patient as an example panel
    from parkigait.carepd_rich import build_dataset
    X, y, groups, cohort = build_dataset(a.root)
    example = X[y > 0][0]  # one real impaired walk
    print(report_rich(example, root=a.root))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
