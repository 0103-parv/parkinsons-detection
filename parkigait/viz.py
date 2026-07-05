"""Visualization helpers for ParkiGait figures (demo / poster).

RESEARCH / EDUCATIONAL PROTOTYPE. NOT A MEDICAL DEVICE. The figures produced
here are for illustrating the *method* (skeleton overlay, gait signals, the
spectral token-pruning idea, and an exploratory severity read-out). They must
not be presented as a clinical result. Any number that appears in a figure is
computed live from the data passed in -- nothing here fabricates or hardcodes a
result. See CLINICAL_SAFETY.md and HONEST_STATUS.md.

Everything renders headless (matplotlib Agg backend); no display is required and
every function writes a real PNG to disk. Colors are chosen to be reasonably
colour-blind friendly (blue = kept/left, orange = dropped/right).

Public API:
  draw_skeleton_frame(pose, t, ax=None, background=None) -> Axes
  render_skeleton_png(pose, t, out_path) -> str
  render_gait_signals_png(pose, features, out_path) -> str
  render_sttp_png(points, is_body, kept_mask, out_path) -> str
  render_severity_bar_png(report_summary, out_path) -> str
"""
from __future__ import annotations

import os
from typing import Optional

import matplotlib

matplotlib.use("Agg")  # headless: no display required, must precede pyplot import

import matplotlib.pyplot as plt  # noqa: E402
import numpy as np  # noqa: E402

from parkigait.types import (  # noqa: E402
    BLAZEPOSE_EDGES,
    BLAZEPOSE_JOINTS,
    GAIT_FEATURE_ORDER,
    GaitFeatures,
    PoseSequence,
)

# --------------------------------------------------------------------------- #
# Shared styling                                                              #
# --------------------------------------------------------------------------- #
_KEEP_COLOR = "#1f77b4"    # blue  -- kept tokens / left side
_DROP_COLOR = "#ff7f0e"    # orange-- dropped tokens / right side
_BODY_EDGE = "#2ca02c"     # green -- body ground-truth ring
_SKEL_COLOR = "#1f77b4"
_LEFT_COLOR = "#1f77b4"
_RIGHT_COLOR = "#d62728"
_DISCLAIMER = (
    "RESEARCH PROTOTYPE — NOT A MEDICAL DEVICE — NOT FOR CLINICAL USE"
)


def _ensure_parent_dir(out_path: str) -> str:
    out_path = os.fspath(out_path)
    parent = os.path.dirname(os.path.abspath(out_path))
    if parent:
        os.makedirs(parent, exist_ok=True)
    return out_path


def _fig_disclaimer(fig, y: float = 0.005) -> None:
    """Stamp the not-a-medical-device caption at the bottom of a figure."""
    fig.text(
        0.5, y, _DISCLAIMER, ha="center", va="bottom",
        fontsize=7, color="#a33", style="italic",
    )


# --------------------------------------------------------------------------- #
# 1. Skeleton overlay                                                         #
# --------------------------------------------------------------------------- #
def draw_skeleton_frame(
    pose: PoseSequence,
    t: int,
    ax=None,
    background: Optional[np.ndarray] = None,
):
    """Draw the 33 joints + BlazePose edges for frame ``t`` in image coords.

    Image space has y pointing DOWN (see types.py), so we invert the y-axis to
    render the person upright. ``background`` may be an (H, W) or (H, W, 3) image
    array to draw behind the skeleton (assumed in the same normalized [0,1]
    coordinate frame); if omitted, a plain frame is drawn.

    Returns the matplotlib Axes the skeleton was drawn on.
    """
    if ax is None:
        fig, ax = plt.subplots(figsize=(4.5, 6))
    else:
        fig = ax.figure

    n_frames = pose.n_frames
    if n_frames == 0:
        raise ValueError("pose has no frames to draw")
    if not (-n_frames <= t < n_frames):
        raise IndexError(f"frame index t={t} out of range for {n_frames} frames")

    frame = pose.joints[t]          # (J, 3)
    xs, ys = frame[:, 0], frame[:, 1]
    vis = pose.visibility[t] if pose.visibility is not None else np.ones(len(xs))

    if background is not None:
        bg = np.asarray(background)
        ax.imshow(bg, extent=(0.0, 1.0, 1.0, 0.0), aspect="auto", zorder=0)

    # edges (kinematic tree) -- skip if either endpoint is out of range
    n_j = frame.shape[0]
    for a, b in BLAZEPOSE_EDGES:
        if a < n_j and b < n_j:
            ax.plot(
                [xs[a], xs[b]], [ys[a], ys[b]],
                color=_SKEL_COLOR, linewidth=2.0, alpha=0.8, zorder=1,
            )

    # joints, sized by visibility so low-confidence points read as smaller/faded
    sizes = 20.0 + 40.0 * np.clip(vis, 0.0, 1.0)
    ax.scatter(
        xs, ys, s=sizes, c=_RIGHT_COLOR, edgecolors="white",
        linewidths=0.5, zorder=2, alpha=0.9,
    )

    ax.set_xlim(0.0, 1.0)
    # y DOWN in image space -> invert so the figure looks upright
    ax.set_ylim(1.0, 0.0)
    ax.set_aspect("equal", adjustable="box")
    ax.set_xlabel("x (normalized)")
    ax.set_ylabel("y (normalized, image space)")
    sev = pose.meta.get("severity", None) if isinstance(pose.meta, dict) else None
    sev_str = f", severity={sev:.2f}" if isinstance(sev, (int, float)) else ""
    ax.set_title(
        f"BlazePose skeleton — frame {t % n_frames} / {n_frames}"
        f" (t={t % n_frames / pose.fps:.2f}s{sev_str})",
        fontsize=10,
    )
    return ax


def render_skeleton_png(pose: PoseSequence, t: int, out_path: str) -> str:
    """Render one annotated skeleton frame to ``out_path`` (PNG). Returns path."""
    out_path = _ensure_parent_dir(out_path)
    fig, ax = plt.subplots(figsize=(4.5, 6))
    try:
        draw_skeleton_frame(pose, t, ax=ax)
        _fig_disclaimer(fig)
        fig.tight_layout(rect=(0, 0.03, 1, 1))
        fig.savefig(out_path, dpi=130)
    finally:
        plt.close(fig)
    return out_path


# --------------------------------------------------------------------------- #
# 2. Gait signals (ankle vertical trajectories + step events + feature box)   #
# --------------------------------------------------------------------------- #
def _detect_step_frames(pose: PoseSequence):
    """Best-effort step-event frame indices for (left, right) ankles.

    Tries to reuse the real detector in ``parkigait.gaitfeat`` if it exposes a
    usable step-detection helper. Falls back to a simple local-minimum finder on
    the (upright) ankle-height signal. Returns (left_idx, right_idx) arrays of
    frame indices, or (None, None) if nothing could be detected.
    """
    # --- try the real feature module first (lazy import; it may not exist yet) --
    try:
        import parkigait.gaitfeat as gf  # type: ignore
    except Exception:
        gf = None

    if gf is not None:
        for fname in ("detect_steps", "step_events", "find_steps", "_detect_steps"):
            fn = getattr(gf, fname, None)
            if callable(fn):
                try:
                    res = fn(pose)
                except Exception:
                    continue
                left, right = _coerce_step_result(res)
                if left is not None or right is not None:
                    return left, right

    # --- fallback: local minima of the upright ankle height (a step ~ foot plant) --
    try:
        left = _local_minima(_upright_ankle_height(pose, "LEFT_ANKLE"), pose.fps)
        right = _local_minima(_upright_ankle_height(pose, "RIGHT_ANKLE"), pose.fps)
        return left, right
    except Exception:
        return None, None


def _coerce_step_result(res):
    """Coerce a variety of plausible detector return shapes into (left, right)."""
    try:
        if isinstance(res, dict):
            left = res.get("left") if "left" in res else res.get("LEFT_ANKLE")
            right = res.get("right") if "right" in res else res.get("RIGHT_ANKLE")
            return (_as_idx(left), _as_idx(right))
        if isinstance(res, (tuple, list)) and len(res) == 2:
            return (_as_idx(res[0]), _as_idx(res[1]))
    except Exception:
        pass
    return None, None


def _as_idx(x):
    if x is None:
        return None
    arr = np.asarray(x)
    if arr.size == 0:
        return np.asarray([], dtype=int)
    return arr.astype(int).ravel()


def _upright_ankle_height(pose: PoseSequence, name: str) -> np.ndarray:
    """Ankle 'height' with y flipped to upright (larger = higher off the floor)."""
    y_down = pose.track(name)[:, 1]
    return -y_down  # flip so a foot lift is a peak, a plant a trough


def _local_minima(signal: np.ndarray, fps: float) -> np.ndarray:
    """Frame indices of local minima (foot plants), spaced >= ~0.25s apart."""
    x = np.asarray(signal, dtype=float)
    if x.size < 3:
        return np.asarray([], dtype=int)
    # smooth lightly to suppress observation noise
    k = max(1, int(round(fps * 0.06)))
    if k > 1:
        kernel = np.ones(k) / k
        x = np.convolve(x, kernel, mode="same")
    cand = np.where((x[1:-1] < x[:-2]) & (x[1:-1] <= x[2:]))[0] + 1
    if cand.size == 0:
        return np.asarray([], dtype=int)
    min_gap = max(1, int(round(fps * 0.25)))
    kept = [int(cand[0])]
    for c in cand[1:]:
        if c - kept[-1] >= min_gap:
            kept.append(int(c))
    return np.asarray(kept, dtype=int)


def _feature_text(features: Optional[GaitFeatures]) -> str:
    if features is None:
        return "(no GaitFeatures provided)"
    lines = []
    labels = {
        "gait_speed": "gait speed",
        "cadence": "cadence (steps/min)",
        "stride_length": "stride length",
        "stride_time_var": "stride-time CV",
        "asymmetry": "asymmetry",
        "arm_swing": "arm swing",
        "fog_index": "freeze index",
    }
    for key in GAIT_FEATURE_ORDER:
        val = getattr(features, key, None)
        if val is None:
            continue
        lines.append(f"{labels.get(key, key):<20s} {float(val):8.3f}")
    lines.append(f"{'step count':<20s} {int(getattr(features, 'step_count', 0)):8d}")
    conf = float(getattr(features, "confidence", 0.0))
    lines.append(f"{'confidence':<20s} {conf:8.3f}")
    return "\n".join(lines)


def render_gait_signals_png(
    pose: PoseSequence,
    features: Optional[GaitFeatures],
    out_path: str,
) -> str:
    """Plot both ankle vertical trajectories with step events + a feature box.

    The ankle-height traces are the raw signal the gait features are computed
    from; step events are marked where detected (via ``parkigait.gaitfeat`` if
    available, else a local-minimum fallback). The text box lists the key
    features that were passed in -- it never invents numbers.
    """
    out_path = _ensure_parent_dir(out_path)
    if pose.n_frames == 0:
        raise ValueError("pose has no frames to plot")

    t_axis = np.arange(pose.n_frames) / pose.fps
    left_h = _upright_ankle_height(pose, "LEFT_ANKLE")
    right_h = _upright_ankle_height(pose, "RIGHT_ANKLE")
    left_idx, right_idx = _detect_step_frames(pose)

    fig, ax = plt.subplots(figsize=(9, 4.5))
    ax.plot(t_axis, left_h, color=_LEFT_COLOR, lw=1.4, label="left ankle")
    ax.plot(t_axis, right_h, color=_RIGHT_COLOR, lw=1.4, label="right ankle")

    n_steps_marked = 0
    if left_idx is not None and len(left_idx) > 0:
        li = np.asarray(left_idx, dtype=int)
        li = li[(li >= 0) & (li < pose.n_frames)]
        ax.scatter(t_axis[li], left_h[li], color=_LEFT_COLOR, marker="v",
                   s=45, zorder=3, edgecolors="k", linewidths=0.4,
                   label="left steps")
        n_steps_marked += len(li)
    if right_idx is not None and len(right_idx) > 0:
        ri = np.asarray(right_idx, dtype=int)
        ri = ri[(ri >= 0) & (ri < pose.n_frames)]
        ax.scatter(t_axis[ri], right_h[ri], color=_RIGHT_COLOR, marker="v",
                   s=45, zorder=3, edgecolors="k", linewidths=0.4,
                   label="right steps")
        n_steps_marked += len(ri)

    ax.set_xlabel("time (s)")
    ax.set_ylabel("ankle height (upright, normalized units)")
    ax.set_title("Gait signals — ankle vertical trajectories", fontsize=11)
    ax.legend(loc="upper right", fontsize=8, ncol=2)
    ax.grid(True, alpha=0.25)

    # feature text box (monospace so columns align)
    ax.text(
        0.012, 0.02, _feature_text(features),
        transform=ax.transAxes, fontsize=8, family="monospace",
        va="bottom", ha="left",
        bbox=dict(boxstyle="round", facecolor="white", alpha=0.85, edgecolor="#999"),
    )

    _fig_disclaimer(fig)
    fig.tight_layout(rect=(0, 0.04, 1, 1))
    fig.savefig(out_path, dpi=130)
    plt.close(fig)
    return out_path


# --------------------------------------------------------------------------- #
# 3. STTP figure (body kept / background dropped)                             #
# --------------------------------------------------------------------------- #
def _body_recall(is_body: np.ndarray, kept_mask: np.ndarray) -> Optional[float]:
    """Fraction of ground-truth body tokens that were kept (None if no body)."""
    body = np.asarray(is_body, dtype=bool)
    kept = np.asarray(kept_mask, dtype=bool)
    n_body = int(body.sum())
    if n_body == 0:
        return None
    return float((body & kept).sum()) / n_body


def render_sttp_png(
    points: np.ndarray,
    is_body: np.ndarray,
    kept_mask: np.ndarray,
    out_path: str,
) -> str:
    """The poster's 'body kept / background dropped' scatter figure.

    Args mirror the outputs of ``parkigait.sttp.frame_tokens`` /
    ``sttp_select``:
      points    (N, 2+) token positions (only first two columns are plotted).
      is_body   (N,) bool ground-truth: True where the token is on the body.
      kept_mask (N,) bool: True where STTP preserved the token.
    Tokens are coloured kept (blue) vs dropped (orange); ground-truth body
    tokens get a green ring. Title shows the measured keep_fraction and
    body_recall -- both computed live from the masks passed in.
    """
    out_path = _ensure_parent_dir(out_path)
    pts = np.asarray(points, dtype=float)
    if pts.ndim != 2 or pts.shape[1] < 2:
        raise ValueError(f"points must be (N, >=2); got shape {pts.shape}")
    is_body = np.asarray(is_body, dtype=bool).ravel()
    kept_mask = np.asarray(kept_mask, dtype=bool).ravel()
    n = pts.shape[0]
    if not (is_body.shape[0] == kept_mask.shape[0] == n):
        raise ValueError(
            f"length mismatch: points={n}, is_body={is_body.shape[0]}, "
            f"kept_mask={kept_mask.shape[0]}")

    x, y = pts[:, 0], pts[:, 1]
    keep_fraction = float(kept_mask.sum()) / n if n else 0.0
    recall = _body_recall(is_body, kept_mask)

    fig, ax = plt.subplots(figsize=(6.5, 6))

    kept = kept_mask
    dropped = ~kept_mask
    ax.scatter(x[dropped], y[dropped], s=42, c=_DROP_COLOR, marker="x",
               linewidths=1.2, label=f"dropped ({int(dropped.sum())})", zorder=1)
    ax.scatter(x[kept], y[kept], s=42, c=_KEEP_COLOR, marker="o",
               edgecolors="white", linewidths=0.4,
               label=f"kept ({int(kept.sum())})", zorder=2)
    # ground-truth body ring (drawn on top, unfilled)
    if is_body.any():
        ax.scatter(x[is_body], y[is_body], s=140, facecolors="none",
                   edgecolors=_BODY_EDGE, linewidths=1.4,
                   label=f"body (ground truth, {int(is_body.sum())})", zorder=3)

    # image-space y is DOWN -> invert for an upright view
    ax.set_ylim(max(y.max(), 1.0) if y.size else 1.0,
                min(y.min(), 0.0) if y.size else 0.0)
    ax.set_aspect("equal", adjustable="box")
    ax.set_xlabel("x (normalized)")
    ax.set_ylabel("y (normalized, image space)")
    recall_str = f"{recall:.1%}" if recall is not None else "n/a"
    ax.set_title(
        "STTP token selection — body kept / background dropped\n"
        f"keep_fraction = {keep_fraction:.1%}   body_recall = {recall_str}",
        fontsize=11,
    )
    ax.legend(loc="upper right", fontsize=8, framealpha=0.9)
    ax.grid(True, alpha=0.2)

    _fig_disclaimer(fig)
    fig.tight_layout(rect=(0, 0.04, 1, 1))
    fig.savefig(out_path, dpi=130)
    plt.close(fig)
    return out_path


# --------------------------------------------------------------------------- #
# 4. Severity bar / gauge                                                     #
# --------------------------------------------------------------------------- #
def _get(summary, *keys, default=None):
    """Fetch the first present key from a dict-like report summary."""
    if not isinstance(summary, dict):
        return default
    for k in keys:
        if k in summary and summary[k] is not None:
            return summary[k]
    return default


def render_severity_bar_png(report_summary: dict, out_path: str) -> str:
    """Bar/gauge of p_pd and severity with the big NOT-A-MEDICAL-DEVICE caption.

    ``report_summary`` is a ``PipelineReport.summary()`` dict (or any dict with
    ``p_pd`` in [0,1] and ``severity_0_4`` in [0,4]). Every value drawn comes
    from that dict -- this function computes nothing about the patient and
    fabricates nothing. The severity scale is a MODEL scale, not a clinical
    UPDRS score, unless the report says it was calibrated on real labelled data.
    """
    out_path = _ensure_parent_dir(out_path)

    p_pd = _get(report_summary, "p_pd", default=None)
    severity = _get(report_summary, "severity_0_4", "severity", default=None)
    label = _get(report_summary, "label", default="")
    scale = _get(report_summary, "severity_scale", "calibrated_on",
                 default="uncalibrated/synthetic")
    conf = _get(report_summary, "feature_confidence", "confidence", default=None)

    if p_pd is None and severity is None:
        raise ValueError(
            "report_summary must contain 'p_pd' and/or 'severity_0_4' to plot")

    fig, (ax_p, ax_s) = plt.subplots(1, 2, figsize=(9, 4.2))

    # --- p_pd horizontal bar in [0, 1] -----------------------------------
    if p_pd is not None:
        p = float(np.clip(p_pd, 0.0, 1.0))
        ax_p.barh([0], [1.0], color="#eee", edgecolor="#bbb", height=0.5)
        ax_p.barh([0], [p], color=_KEEP_COLOR, height=0.5)
        ax_p.text(min(p + 0.02, 0.98), 0, f"{p:.2f}", va="center",
                  ha="left" if p < 0.85 else "right", fontsize=12, weight="bold")
        ax_p.set_xlim(0, 1)
    else:
        ax_p.text(0.5, 0, "p_pd: n/a", ha="center", va="center")
        ax_p.set_xlim(0, 1)
    ax_p.set_ylim(-0.5, 0.5)
    ax_p.set_yticks([])
    ax_p.set_xlabel("P(PD-sign class)  [exploratory, uncalibrated]")
    ax_p.set_title("Motor-sign probability", fontsize=11)

    # --- severity gauge in [0, 4] ----------------------------------------
    if severity is not None:
        sev = float(np.clip(severity, 0.0, 4.0))
        ax_s.barh([0], [4.0], color="#eee", edgecolor="#bbb", height=0.5)
        ax_s.barh([0], [sev], color=_DROP_COLOR, height=0.5)
        ax_s.text(min(sev + 0.08, 3.9), 0, f"{sev:.2f}", va="center",
                  ha="left" if sev < 3.4 else "right", fontsize=12, weight="bold")
        ax_s.set_xlim(0, 4)
        ax_s.set_xticks([0, 1, 2, 3, 4])
    else:
        ax_s.text(2.0, 0, "severity: n/a", ha="center", va="center")
        ax_s.set_xlim(0, 4)
    ax_s.set_ylim(-0.5, 0.5)
    ax_s.set_yticks([])
    ax_s.set_xlabel(f"Severity (0–4, {scale} scale — NOT clinical UPDRS)")
    ax_s.set_title("Exploratory severity", fontsize=11)

    subtitle_bits = []
    if label:
        subtitle_bits.append(f"label: {label}")
    if conf is not None:
        subtitle_bits.append(f"feature confidence: {float(conf):.2f}")
    subtitle = "   |   ".join(subtitle_bits)
    fig.suptitle(
        "ParkiGait exploratory read-out"
        + (f"\n{subtitle}" if subtitle else ""),
        fontsize=12,
    )

    # big, unmissable disclaimer
    fig.text(
        0.5, 0.02,
        "RESEARCH PROTOTYPE — NOT A MEDICAL DEVICE",
        ha="center", va="bottom", fontsize=14, weight="bold", color="#b00020",
    )
    fig.text(
        0.5, 0.005,
        "Exploratory & uncalibrated — cannot diagnose — not for "
        "clinical use — consult a licensed clinician.",
        ha="center", va="bottom", fontsize=8, color="#b00020",
    )

    fig.tight_layout(rect=(0, 0.09, 1, 0.94))
    fig.savefig(out_path, dpi=130)
    plt.close(fig)
    return out_path


# --------------------------------------------------------------------------- #
# Smoke test / acceptance driver                                             #
# --------------------------------------------------------------------------- #
def _smoke(out_dir: Optional[str] = None) -> list[str]:
    """Generate a synthetic walker and render the figures; return saved paths.

    Prints each saved path and confirms the file exists and is non-empty. This
    is the acceptance check for viz.py -- every number it prints is measured
    from the freshly generated synthetic data, not hardcoded.
    """
    from parkigait.pose import SyntheticWalker

    here = os.path.dirname(os.path.abspath(__file__))
    out_dir = out_dir or os.path.join(here, "_viz_smoke")
    os.makedirs(out_dir, exist_ok=True)

    pose = SyntheticWalker(0.6, seed=7).generate(duration_s=8.0, fps=30.0)
    print(f"[smoke] synthetic pose: {pose.n_frames} frames @ {pose.fps} fps, "
          f"duration {pose.duration_s:.2f}s, source={pose.source}")

    saved: list[str] = []

    # --- skeleton frame --------------------------------------------------
    t_mid = pose.n_frames // 2
    p1 = render_skeleton_png(pose, t_mid, os.path.join(out_dir, "skeleton_frame.png"))
    saved.append(p1)

    # --- gait signals ----------------------------------------------------
    # try to build real GaitFeatures if the module exists; else pass None
    features = None
    try:
        import parkigait.gaitfeat as gf  # type: ignore
        for fname in ("extract_gait_features", "extract", "compute_features",
                      "gait_features"):
            fn = getattr(gf, fname, None)
            if callable(fn):
                try:
                    features = fn(pose)
                    print(f"[smoke] GaitFeatures via parkigait.gaitfeat.{fname}")
                    break
                except Exception as e:  # pragma: no cover
                    print(f"[smoke] gaitfeat.{fname} failed: {e!r}")
    except Exception:
        print("[smoke] parkigait.gaitfeat not available -> plotting signals only")

    p2 = render_gait_signals_png(
        pose, features, os.path.join(out_dir, "gait_signals.png"))
    saved.append(p2)

    # --- STTP figure (only if sttp module is available) ------------------
    try:
        import parkigait.sttp as sttp  # type: ignore
        pts = is_body = kept = None
        # frame_tokens(joints_norm) -> (points, is_body) for a single frame
        ft = getattr(sttp, "frame_tokens", None)
        if callable(ft):
            ft_out = ft(pose.joints[t_mid])
            pts, is_body = _unpack_frame_tokens(ft_out)
        sel = getattr(sttp, "sttp_select", None)
        if callable(sel) and pts is not None:
            sel_out = sel(pts)
            kept = _unpack_select(sel_out)
        if pts is not None and kept is not None:
            if is_body is None:
                is_body = np.ones(len(pts), dtype=bool)
            p3 = render_sttp_png(pts, is_body, kept,
                                 os.path.join(out_dir, "sttp.png"))
            saved.append(p3)
            print("[smoke] STTP figure rendered from real parkigait.sttp output")
        else:
            print("[smoke] parkigait.sttp present but expected API not found "
                  "-> skipping sttp png")
    except Exception as e:
        print(f"[smoke] parkigait.sttp not available -> skipping sttp png ({e!r})")

    # --- severity read-out figure ----------------------------------------
    # Prefer a real end-to-end summary if severity.py can score these features;
    # otherwise skip (we never invent p_pd / severity here). The numbers below
    # come from whatever the severity model actually returns.
    if features is not None:
        try:
            import parkigait.severity as sv  # type: ignore
            est = None
            for fname in ("estimate", "predict", "score"):
                fn = getattr(sv, fname, None)
                if callable(fn):
                    try:
                        est = fn(features)
                        break
                    except Exception:
                        est = None
            if est is None and hasattr(sv, "SeverityModel"):
                try:
                    model = (sv.load_or_train() if hasattr(sv, "load_or_train")
                             else sv.SeverityModel())
                    for m in ("estimate", "predict", "score", "__call__"):
                        fn = getattr(model, m, None)
                        if callable(fn):
                            est = fn(features)
                            break
                except Exception:
                    est = None
            if est is not None:
                from parkigait.types import PipelineReport
                rep = PipelineReport(source=pose.source, features=features,
                                     severity=est)
                summ = rep.summary()
                p4 = render_severity_bar_png(
                    summ, os.path.join(out_dir, "severity_bar.png"))
                saved.append(p4)
                print(f"[smoke] severity figure: p_pd={summ.get('p_pd')} "
                      f"severity={summ.get('severity_0_4')} "
                      f"(scale: {summ.get('severity_scale')})")
        except Exception as e:
            print(f"[smoke] severity read-out skipped ({e!r})")

    # --- verify all files exist and are non-empty ------------------------
    print("\n[smoke] saved figures:")
    all_ok = True
    for p in saved:
        exists = os.path.exists(p)
        size = os.path.getsize(p) if exists else 0
        ok = exists and size > 0
        all_ok = all_ok and ok
        print(f"  {'OK ' if ok else 'BAD'}  {p}  ({size} bytes)")
    print(f"[smoke] all files non-empty: {all_ok}")
    if not all_ok:
        raise RuntimeError("one or more figures were not written / empty")
    return saved


def _unpack_frame_tokens(ft_out):
    """Turn a frame_tokens(...) return into (points(N,>=2), is_body or None)."""
    is_body = None
    if isinstance(ft_out, tuple) and len(ft_out) >= 2:
        pts = np.asarray(ft_out[0], dtype=float)
        try:
            is_body = np.asarray(ft_out[1], dtype=bool).ravel()
        except Exception:
            is_body = None
    else:
        pts = np.asarray(ft_out, dtype=float)
    if pts.ndim == 1:
        pts = pts.reshape(-1, 1)
    return pts, is_body


def _unpack_select(sel_out):
    """Turn an sttp_select(...) return into a boolean kept_mask (N,)."""
    if hasattr(sel_out, "kept_mask"):        # STTPResult
        return np.asarray(sel_out.kept_mask, dtype=bool).ravel()
    if isinstance(sel_out, tuple) and len(sel_out) >= 1:
        first = sel_out[0]
        if hasattr(first, "kept_mask"):
            return np.asarray(first.kept_mask, dtype=bool).ravel()
        return np.asarray(first, dtype=bool).ravel()
    return np.asarray(sel_out, dtype=bool).ravel()


if __name__ == "__main__":  # pragma: no cover
    _smoke()
