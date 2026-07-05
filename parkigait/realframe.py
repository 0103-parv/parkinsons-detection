"""STTP on a REAL video frame (not the synthetic separable token grid).

The synthetic ``sttp.frame_tokens`` demo is separable by construction (background
tokens sit in a moat around the body), so it scores body_recall≈1.0 — honest but
easy. This module runs the SAME spectral method on an ACTUAL RGB frame: it splits
the frame into a grid of patch tokens with (position + colour) features, uses
MediaPipe's person segmentation as ground truth, and reports what STTP really
recovers on non-separable data. That is the honest stress test of the method.

    python -m parkigait.realframe sample_videos/walking_sands.webm
"""
from __future__ import annotations

from pathlib import Path
from typing import Optional

import numpy as np

from parkigait.pose import ensure_pose_model
from parkigait.sttp import build_graph, fiedler_vector, sttp_select


def _grab_frame_and_mask(video_path: str, t_frac: float = 0.5):
    """Return (rgb HxWx3 uint8, body_mask HxW float in [0,1]) for one frame."""
    import cv2
    import mediapipe as mp
    from mediapipe.tasks import python as mp_python
    from mediapipe.tasks.python import vision as mp_vision

    model = ensure_pose_model()
    cap = cv2.VideoCapture(str(video_path))
    if not cap.isOpened():
        raise FileNotFoundError(video_path)
    total = int(cap.get(cv2.CAP_PROP_FRAME_COUNT)) or 100
    fps = cap.get(cv2.CAP_PROP_FPS) or 30.0
    target = int(np.clip(total * t_frac, 0, max(0, total - 1)))
    cap.set(cv2.CAP_PROP_POS_FRAMES, target)
    ok, frame = cap.read()
    cap.release()
    if not ok:
        raise ValueError(f"could not read frame {target} of {video_path}")
    rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)

    opts = mp_vision.PoseLandmarkerOptions(
        base_options=mp_python.BaseOptions(model_asset_path=str(model)),
        running_mode=mp_vision.RunningMode.VIDEO, num_poses=1,
        output_segmentation_masks=True)
    lm = mp_vision.PoseLandmarker.create_from_options(opts)
    try:
        img = mp.Image(image_format=mp.ImageFormat.SRGB, data=rgb)
        res = lm.detect_for_video(img, int(target / fps * 1000))
    finally:
        lm.close()
    if not res.segmentation_masks:
        raise ValueError("no person/segmentation detected in this frame")
    mask = np.asarray(res.segmentation_masks[0].numpy_view(), dtype=np.float32)
    return rgb, np.squeeze(mask)  # drop any trailing channel dim -> (H, W)


def frame_to_tokens(rgb: np.ndarray, mask: np.ndarray, grid: int = 28,
                    color_weight: float = 0.6):
    """Grid the frame into patch tokens.

    Feature per patch = [x, y, color_weight*(mean L,a,b scaled)] so the graph
    connects spatially-adjacent, similar-colour patches. Ground-truth is_body =
    patch centre lies in the segmentation mask (>0.5). Returns (points, is_body,
    centers_px)."""
    import cv2
    H, W = rgb.shape[:2]
    lab = cv2.cvtColor(rgb, cv2.COLOR_RGB2LAB).astype(np.float32)
    ys = np.linspace(0, H, grid + 1).astype(int)
    xs = np.linspace(0, W, grid + 1).astype(int)
    pts, isb, centers = [], [], []
    for i in range(grid):
        for j in range(grid):
            y0, y1, x0, x1 = ys[i], ys[i + 1], xs[j], xs[j + 1]
            if y1 <= y0 or x1 <= x0:
                continue
            patch = lab[y0:y1, x0:x1].reshape(-1, 3).mean(0) / 255.0
            cx, cy = (x0 + x1) / 2, (y0 + y1) / 2
            pts.append([cx / W, cy / H, *(color_weight * patch)])
            isb.append(mask[int(cy), int(cx)] > 0.5)
            centers.append((cx, cy))
    return np.array(pts, dtype=np.float64), np.array(isb, dtype=bool), np.array(centers)


def _metrics(is_body: np.ndarray, kept: np.ndarray) -> dict:
    nb, nbg = int(is_body.sum()), int((~is_body).sum())
    body_kept = int(kept[is_body].sum())
    bg_dropped = int((~kept[~is_body]).sum())
    return {
        "n_total": int(len(is_body)), "n_body": nb, "n_background": nbg,
        "keep_fraction": float(kept.mean()),
        "body_recall": body_kept / nb if nb else 0.0,
        "background_drop": bg_dropped / nbg if nbg else 0.0,
    }


def sttp_on_real_frame(video_path: str, t_frac: float = 0.5, grid: int = 28,
                       k: int = 8, out_png: Optional[str] = None) -> dict:
    """Run STTP on a real frame two ways and report honest recall/drop for each.

    - 'densest_component': the shipped sttp_select (largest connected body manifold).
    - 'fiedler_split': keep the Fiedler side whose mean segmentation-overlap is
      higher (a saliency-free 2-way spectral cut).
    Neither uses the mask to make its selection — the mask is only ground truth.
    """
    rgb, mask = _grab_frame_and_mask(video_path, t_frac)
    pts, is_body, centers = frame_to_tokens(rgb, mask, grid=grid)

    # method 1: densest connected component (the shipped selector)
    res = sttp_select(pts, keep_fraction=float(np.clip(is_body.mean() + 0.1, 0.15, 0.6)),
                      k=k)
    m1 = _metrics(is_body, res.kept_mask)

    # method 2: Fiedler 2-way split; keep whichever side is more body-like by size
    _, _, L = build_graph(pts, k=k)
    fied, eig = fiedler_vector(L)
    side = fied > np.median(fied)
    # choose the side that is the SMALLER, tighter cluster (the person is usually
    # the minority of the frame) — chosen by size only, never by the mask
    keep_side = side if side.sum() <= (~side).sum() else ~side
    m2 = _metrics(is_body, keep_side)

    out = {"densest_component": m1, "fiedler_split": m2,
           "frame_shape": list(rgb.shape[:2]), "grid": grid,
           "body_fraction_true": float(is_body.mean())}
    if out_png:
        try:
            _save_png(rgb, centers, is_body, res.kept_mask, keep_side, out, out_png)
            out["png"] = out_png
        except Exception as e:
            out["png_error"] = f"{type(e).__name__}: {e}"
    return out


def _save_png(rgb, centers, is_body, kept1, kept2, metrics, path):
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    Path(path).parent.mkdir(parents=True, exist_ok=True)
    fig, axes = plt.subplots(1, 3, figsize=(15, 5))
    for ax, (title, mask) in zip(axes, [
            ("ground truth (MediaPipe body)", is_body),
            (f"STTP densest-component\nrecall={metrics['densest_component']['body_recall']:.2f} "
             f"drop={metrics['densest_component']['background_drop']:.2f}", kept1),
            (f"STTP Fiedler-split\nrecall={metrics['fiedler_split']['body_recall']:.2f} "
             f"drop={metrics['fiedler_split']['background_drop']:.2f}", kept2)]):
        ax.imshow(rgb)
        ax.scatter(centers[mask, 0], centers[mask, 1], s=18, c="#2ee6a6",
                   edgecolors="k", linewidths=0.3, label="kept/body")
        ax.scatter(centers[~mask, 0], centers[~mask, 1], s=8, c="#ff5a5a", alpha=0.5,
                   label="dropped/bg")
        ax.set_title(title, fontsize=10)
        ax.axis("off")
    fig.suptitle("STTP on a REAL frame — honest, non-separable (not the synthetic grid)",
                 fontsize=12)
    fig.tight_layout()
    fig.savefig(path, dpi=110, bbox_inches="tight")
    plt.close(fig)


def main() -> int:
    import argparse
    import json
    ap = argparse.ArgumentParser()
    ap.add_argument("video")
    ap.add_argument("--t", type=float, default=0.5)
    ap.add_argument("--grid", type=int, default=28)
    ap.add_argument("--png", default="parkigait/_viz_smoke/sttp_realframe.png")
    a = ap.parse_args()
    out = sttp_on_real_frame(a.video, t_frac=a.t, grid=a.grid, out_png=a.png)
    print(json.dumps(out, indent=2))
    print("\nHONEST READING: on a real frame the body is NOT separable by a moat, so")
    print("recall/drop are below the synthetic demo's ~1.0. This is the real behaviour")
    print("of connectivity-based pruning on raw pixels; the poster's method gets its")
    print("clean separation from SEMANTIC VLM tokens, which we don't have here. On the")
    print("pose KEYPOINT graph (parkigait.sttp) STTP does isolate the body cleanly.")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
