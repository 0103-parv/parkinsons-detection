"""Render a PoseSequence to an actual RGB video (a walking figure).

This exists so the demo has a concrete video artifact and so the web app can show
"a person walking" and then analyze it. It draws a simple filled humanoid (limbs
as thick segments, a head) from the joint trajectories -- enough to look like a
walking figure and to visualize what the pipeline consumes. Uses OpenCV only.

Note: MediaPipe is trained on real humans and generally will NOT re-detect this
stick/blob figure, so the app analyzes the *synthetic pose directly* for the demo
walker, and uses MediaPipe only on real camera video. This is stated in the app.
"""
from __future__ import annotations

from pathlib import Path

import numpy as np

from parkigait.types import BLAZEPOSE_EDGES, PoseSequence


def render_walk_video(pose: PoseSequence, out_path, width: int = 480, height: int = 640,
                      fps: float | None = None, bg=(24, 26, 32), fg=(90, 200, 240)) -> str:
    """Write ``pose`` as an .mp4 of a walking figure; return the output path."""
    import cv2

    out_path = str(out_path)
    Path(out_path).parent.mkdir(parents=True, exist_ok=True)
    fps = float(fps or pose.fps or 30.0)
    fourcc = cv2.VideoWriter_fourcc(*"mp4v")
    writer = cv2.VideoWriter(out_path, fourcc, fps, (width, height))
    if not writer.isOpened():
        raise RuntimeError(f"could not open VideoWriter for {out_path}")

    def px(j):
        return (int(j[0] * width), int(j[1] * height))

    try:
        for t in range(pose.n_frames):
            frame = np.zeros((height, width, 3), dtype=np.uint8)
            frame[:] = bg[::-1]  # OpenCV is BGR
            js = pose.joints[t]
            # ground line
            cv2.line(frame, (0, int(0.94 * height)), (width, int(0.94 * height)),
                     (60, 60, 66), 2)
            # limbs as thick segments
            for a, b in BLAZEPOSE_EDGES:
                cv2.line(frame, px(js[a]), px(js[b]), fg[::-1], 8, cv2.LINE_AA)
            # joints
            for j in js:
                cv2.circle(frame, px(j), 4, (240, 240, 240), -1, cv2.LINE_AA)
            # head
            nose = js[0]
            cv2.circle(frame, px(nose), 22, fg[::-1], -1, cv2.LINE_AA)
            cv2.putText(frame, "SYNTHETIC WALKER (demo)", (12, 24),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.5, (200, 200, 200), 1, cv2.LINE_AA)
            writer.write(frame)
    finally:
        writer.release()
    return out_path


def main() -> int:
    import argparse

    from parkigait.pose import SyntheticWalker

    ap = argparse.ArgumentParser(description="Render a synthetic walking video.")
    ap.add_argument("--severity", type=float, default=0.5)
    ap.add_argument("--seconds", type=float, default=8.0)
    ap.add_argument("--out", default="sample_videos/synthetic_walk.mp4")
    args = ap.parse_args()
    pose = SyntheticWalker(args.severity).generate(duration_s=args.seconds)
    path = render_walk_video(pose, args.out)
    print(f"wrote {path}  ({pose.n_frames} frames @ {pose.fps:.0f} fps, "
          f"severity {args.severity})")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
