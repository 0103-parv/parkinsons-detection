"""Generate committed poster figures into parkigait/figures/.

    python -m parkigait.figures

These are real, reproducible figures (not gitignored) so the poster has assets
that match the measured numbers. Everything is synthetic/system data — clearly
labelled, never clinical.
"""
from __future__ import annotations

from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt  # noqa: E402
import numpy as np  # noqa: E402

FIG_DIR = Path(__file__).resolve().parent / "figures"
_CAPTION = "RESEARCH PROTOTYPE — NOT A MEDICAL DEVICE · synthetic/system data"


def fig_pipeline_and_gait() -> Path:
    """Skeleton + the gait signal with detected steps + the feature read-out."""
    from parkigait.gaitfeat import extract_features
    from parkigait.pose import SyntheticWalker
    from parkigait.types import BLAZEPOSE_EDGES

    ps = SyntheticWalker(0.55, seed=3).generate()
    feats = extract_features(ps)
    t = ps.n_frames // 2
    fig, ax = plt.subplots(1, 2, figsize=(12, 5))

    js = ps.joints[t]
    for a, b in BLAZEPOSE_EDGES:
        ax[0].plot([js[a, 0], js[b, 0]], [js[a, 1], js[b, 1]], "-", color="#2ee6a6", lw=3)
    ax[0].scatter(js[:, 0], js[:, 1], s=22, c="#ffffff", zorder=3, edgecolors="k", lw=0.3)
    ax[0].set_xlim(0, 1)
    ax[0].set_ylim(1, 0)
    ax[0].set_aspect("equal")
    ax[0].set_title("Skeleton (MediaPipe BlazePose topology), mid-stride")
    ax[0].axis("off")

    la = ps.track("LEFT_ANKLE")[:, 1]
    ra = ps.track("RIGHT_ANKLE")[:, 1]
    tt = np.arange(ps.n_frames) / ps.fps
    ax[1].plot(tt, -la, label="left ankle height", color="#5ac8fa")
    ax[1].plot(tt, -ra, label="right ankle height", color="#ff9f45")
    ax[1].set_xlabel("time (s)")
    ax[1].set_ylabel("foot height (up = +)")
    ax[1].set_title("Ankle signals → cadence, stride, freeze index")
    txt = (f"cadence {feats.cadence:.0f} steps/min\nstride {feats.stride_length:.2f}\n"
           f"arm swing {feats.arm_swing:.3f}\nfreeze idx {feats.fog_index:.1f}\n"
           f"steps {feats.step_count}")
    ax[1].text(0.98, 0.02, txt, transform=ax[1].transAxes, ha="right", va="bottom",
               fontsize=9, family="monospace",
               bbox=dict(boxstyle="round", fc="#1c1f27", ec="#2a2e39", alpha=0.9),
               color="#e6e6e6")
    ax[1].legend(loc="upper right", fontsize=8)
    fig.suptitle("ParkiGait: video → skeleton → gait features  (synthetic walker, severity 0.55)")
    fig.text(0.5, 0.005, _CAPTION, ha="center", fontsize=8, color="#888")
    fig.tight_layout(rect=[0, 0.02, 1, 1])
    p = FIG_DIR / "01_pipeline_gait.png"
    fig.savefig(p, dpi=120, bbox_inches="tight")
    plt.close(fig)
    return p


def fig_robustness_and_quant() -> Path:
    """The two new measured results: STTP background rejection + LieQ tradeoff."""
    from parkigait.ablation import lieq_bit_sweep, stt_background_rejection

    rej = stt_background_rejection()
    lieq = lieq_bit_sweep()
    fig, ax = plt.subplots(1, 2, figsize=(12, 5))

    n = [r["n_background"] for r in rej]
    ax[0].plot(n, [r["bg_rejection"] * 100 for r in rej], "-o", color="#2ee6a6",
               label="background rejection")
    ax[0].plot(n, [r["body_recall"] * 100 for r in rej], "-s", color="#5ac8fa",
               label="body recall")
    ax[0].set_xlabel("# injected background/adversarial tokens")
    ax[0].set_ylabel("%")
    ax[0].set_ylim(-5, 105)
    ax[0].set_title("STTP robustness (honest 'ASR' answer)\n100% rejection until the "
                    "frame is flooded")
    ax[0].legend(fontsize=8)
    ax[0].grid(alpha=0.2)

    sweep = [r for r in lieq["sweep"] if "compression" in r]
    comp = [r["compression"] for r in sweep]
    acc = [r["acc"] * 100 for r in sweep]
    labels = [r["label"].replace(" uniform", "").replace("LieQ search ", "LieQ ")
              for r in sweep]
    ax[1].scatter(comp, acc, s=60, c="#ff9f45", zorder=3)
    for x, y, l in zip(comp, acc, labels):
        ax[1].annotate(l, (x, y), fontsize=8, xytext=(4, 4),
                       textcoords="offset points")
    ax[1].axhline(lieq["fp32_acc"] * 100, ls="--", color="#888",
                  label=f"fp32 acc {lieq['fp32_acc'] * 100:.0f}%")
    ax[1].set_xlabel("compression ratio (×)")
    ax[1].set_ylabel("held-out accuracy (%)")
    ax[1].set_title("LieQ mixed-precision: compression vs accuracy\n(small demo model)")
    ax[1].legend(fontsize=8)
    ax[1].grid(alpha=0.2)

    fig.suptitle("ParkiGait measured results: STTP robustness & LieQ quantization")
    fig.text(0.5, 0.005, _CAPTION, ha="center", fontsize=8, color="#888")
    fig.tight_layout(rect=[0, 0.02, 1, 1])
    p = FIG_DIR / "02_robustness_quant.png"
    fig.savefig(p, dpi=120, bbox_inches="tight")
    plt.close(fig)
    return p


def fig_ood() -> Path:
    """OOD demonstration: distance-to-training for in-dist vs alien inputs."""
    from parkigait.gaitfeat import extract_features
    from parkigait.pose import SyntheticWalker
    from parkigait.severity import load_or_train

    m = load_or_train()

    def dist(feats):
        xs = (feats.as_vector() - m.mu) / m.sd
        return float(np.sqrt((xs ** 2).sum()))

    healthy = [dist(extract_features(SyntheticWalker(0.03, seed=9000 + i).generate()))
               for i in range(8)]
    pd = [dist(extract_features(SyntheticWalker(0.7, seed=9100 + i).generate()))
          for i in range(8)]
    # an alien input: extreme features
    from parkigait.types import GaitFeatures
    alien = dist(GaitFeatures(9, 400, 8, 5, 0.99, 9, 500, confidence=1.0))

    fig, ax = plt.subplots(figsize=(9, 5))
    ax.scatter([0] * len(healthy), healthy, c="#5ac8fa", s=50, label="synthetic control")
    ax.scatter([1] * len(pd), pd, c="#ff9f45", s=50, label="synthetic PD")
    ax.scatter([2], [alien], c="#ff5a5a", s=90, marker="X", label="alien input")
    ax.axhline(m.ood_threshold, ls="--", color="#e6483d",
               label=f"OOD threshold {m.ood_threshold:.1f}")
    ax.set_xticks([0, 1, 2])
    ax.set_xticklabels(["control", "PD", "alien"])
    ax.set_ylabel("distance to training distribution")
    ax.set_title("Out-of-distribution guard: alien inputs exceed the threshold\n"
                 "→ flagged 'unreliable' instead of a confident wrong score")
    ax.legend(fontsize=8)
    ax.grid(alpha=0.2)
    fig.text(0.5, 0.005, _CAPTION, ha="center", fontsize=8, color="#888")
    fig.tight_layout(rect=[0, 0.02, 1, 1])
    p = FIG_DIR / "03_ood_guard.png"
    fig.savefig(p, dpi=120, bbox_inches="tight")
    plt.close(fig)
    return p


def main() -> int:
    FIG_DIR.mkdir(exist_ok=True)
    for fn in (fig_pipeline_and_gait, fig_robustness_and_quant, fig_ood):
        p = fn()
        print(f"wrote {p}  ({p.stat().st_size // 1024} KB)")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
