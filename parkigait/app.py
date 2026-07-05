"""Local web app for ParkiGait — upload a walking video, get an exploratory report.

Runs entirely on localhost (no upload leaves the machine — the on-device/privacy
thesis, made literal). Every page carries the research-only disclaimer.

    python -m parkigait serve      # then open http://127.0.0.1:7860
"""
from __future__ import annotations

import html
import time
import uuid
from pathlib import Path

from flask import Flask, redirect, request, send_from_directory, url_for

from parkigait import DISCLAIMER, __version__

_HERE = Path(__file__).resolve().parent
_STATIC = _HERE / "app_static"
_UPLOADS = _HERE / "app_uploads"
_STATIC.mkdir(exist_ok=True)
_UPLOADS.mkdir(exist_ok=True)

app = Flask(__name__)
app.config["MAX_CONTENT_LENGTH"] = 200 * 1024 * 1024  # 200 MB cap

_BANNER = (
    "<div style='background:#7a1020;color:#fff;padding:10px 14px;border-radius:8px;"
    "font-weight:600;margin-bottom:16px'>RESEARCH PROTOTYPE — NOT A MEDICAL DEVICE. "
    "This is exploratory and uncalibrated. It cannot diagnose Parkinson's disease or "
    "any condition and must not be used for clinical decisions.</div>")

_CSS = (
    "<style>body{font-family:-apple-system,Segoe UI,Roboto,sans-serif;max-width:900px;"
    "margin:32px auto;padding:0 18px;color:#e6e6e6;background:#14161c}"
    "h1{font-size:22px}h2{font-size:17px;margin-top:26px}"
    "a{color:#5ac8fa}.card{background:#1c1f27;border:1px solid #2a2e39;border-radius:10px;"
    "padding:16px 18px;margin:14px 0}.btn{background:#2563eb;color:#fff;border:none;"
    "padding:10px 16px;border-radius:8px;font-size:15px;cursor:pointer}"
    "table{border-collapse:collapse;width:100%}td,th{border-bottom:1px solid #2a2e39;"
    "padding:6px 10px;text-align:left;font-size:14px}img{max-width:100%;border-radius:8px;"
    "border:1px solid #2a2e39;margin:8px 0}.muted{color:#9aa0ab;font-size:13px}"
    "input[type=range]{width:260px}</style>")


def _page(body: str) -> str:
    return (f"<!doctype html><html><head><meta charset='utf-8'>"
            f"<title>ParkiGait</title>{_CSS}</head><body>{_BANNER}{body}"
            f"<p class='muted'>ParkiGait v{__version__} · runs locally, no data leaves "
            f"this machine · see HONEST_STATUS.md and CLINICAL_SAFETY.md</p></body></html>")


@app.route("/")
def index():
    body = f"""
    <h1>ParkiGait — video gait analysis (research prototype)</h1>
    <div class='card'>
      <h2>Scan a real walking video</h2>
      <p class='muted'>Best results: a few seconds of someone walking, full body
      visible, roughly side-on, decent light. Processed with MediaPipe BlazePose
      on your device.</p>
      <form action='/scan' method='post' enctype='multipart/form-data'>
        <input type='file' name='video' accept='video/*' required>
        <button class='btn' type='submit'>Analyze video</button>
      </form>
    </div>
    <div class='card'>
      <h2>Or try the synthetic demo walker</h2>
      <p class='muted'>No camera needed. A physically-modeled walker with a known
      severity you set — useful to see the whole pipeline run.</p>
      <form action='/demo' method='post'>
        <label>severity (0 healthy → 1 severe):
          <input type='range' name='severity' min='0' max='1' step='0.05' value='0.6'
                 oninput="this.nextElementSibling.value=this.value">
          <output>0.6</output></label><br><br>
        <button class='btn' type='submit'>Run demo</button>
      </form>
    </div>
    """
    return _page(body)


def _render_report_html(report, extra_media: list[tuple[str, str]]) -> str:
    s = report.summary()
    rows = "".join(
        f"<tr><td>{html.escape(k)}</td><td>{v:.4f}</td></tr>"
        for k, v in s["features"].items())
    warns = ("".join(f"<li>{html.escape(w)}</li>" for w in s["warnings"]))
    warn_html = f"<div class='card'><h2>Warnings</h2><ul>{warns}</ul></div>" if warns else ""
    media = "".join(
        f"<h2>{html.escape(title)}</h2><img src='{url}'>" for title, url in extra_media)
    timings = "".join(f"<tr><td>{k}</td><td>{v} ms</td></tr>"
                      for k, v in s["timings_ms"].items())
    ood_html = ""
    if s.get("out_of_distribution"):
        ood_html = (
            "<div style='background:#8a5a00;color:#fff;padding:10px 14px;"
            "border-radius:8px;font-weight:600;margin-bottom:12px'>OUT-OF-DISTRIBUTION"
            " — these gait features are unlike the (synthetic) training data, so the "
            "score below is NOT meaningful. Expected on real video until trained on "
            "real data.</div>")
    body = f"""
    <h1>Report</h1>
    <p class='muted'>source: {html.escape(str(s['source']))}</p>
    {ood_html}
    <div class='card'>
      <h2>Exploratory estimate (NOT a diagnosis)</h2>
      <table>
        <tr><td>P(PD-like motor signs)</td><td>{s['p_pd']:.3f}</td></tr>
        <tr><td>severity (0–4, {html.escape(s['severity_scale'])})</td><td>{s['severity_0_4']:.2f}</td></tr>
        <tr><td>label</td><td>{html.escape(s['label'])}</td></tr>
        <tr><td>steps detected</td><td>{s['step_count']}</td></tr>
        <tr><td>signal confidence</td><td>{s['feature_confidence']}</td></tr>
      </table>
    </div>
    {media}
    <div class='card'><h2>Gait features</h2><table>{rows}</table></div>
    <div class='card'><h2>Timings</h2><table>{timings}</table></div>
    {warn_html}
    <p><a href='/'>← analyze another</a></p>
    """
    return _page(body)


def _render_figures(report, tag: str) -> list[tuple[str, str]]:
    """Best-effort figure rendering; figures are a bonus, never required."""
    media: list[tuple[str, str]] = []
    try:
        from parkigait import viz
    except Exception:
        return media
    pose = report.pose
    stamp = f"{tag}_{uuid.uuid4().hex[:8]}"
    jobs = [
        ("Skeleton (mid-stride)", "skeleton",
         lambda p: viz.render_skeleton_png(pose, pose.n_frames // 2, p)),
        ("Gait signals & detected steps", "signals",
         lambda p: viz.render_gait_signals_png(pose, report.features, p)),
    ]
    if report.sttp is not None and hasattr(report.sttp, "meta_points"):
        jobs.append(("STTP — body kept / background dropped", "sttp",
                     lambda p: viz.render_sttp_png(
                         report.sttp.meta_points, report.sttp.meta_is_body,
                         report.sttp.kept_mask, p)))
    for title, name, fn in jobs:
        try:
            out = _STATIC / f"{stamp}_{name}.png"
            fn(out)
            if out.exists() and out.stat().st_size > 0:
                media.append((title, url_for("static_file", filename=out.name)))
        except Exception:
            continue
    return media


@app.route("/scan", methods=["POST"])
def scan():
    from parkigait.pipeline import analyze_video
    f = request.files.get("video")
    if not f or not f.filename:
        return redirect(url_for("index"))
    dest = _UPLOADS / f"{uuid.uuid4().hex[:8]}_{Path(f.filename).name}"
    f.save(dest)
    try:
        report = analyze_video(str(dest), stride=2, max_frames=200)
    finally:
        try:
            dest.unlink()  # do not retain uploaded video
        except OSError:
            pass
    media = _render_figures(report, "scan")
    return _render_report_html(report, media)


@app.route("/demo", methods=["POST"])
def demo():
    from parkigait.pipeline import analyze_synthetic
    try:
        severity = float(request.form.get("severity", 0.6))
    except (TypeError, ValueError):
        severity = 0.6
    report = analyze_synthetic(severity=severity, seed=int(time.perf_counter()) % 97)
    media = _render_figures(report, "demo")
    return _render_report_html(report, media)


@app.route("/static_gen/<path:filename>")
def static_file(filename):
    return send_from_directory(_STATIC, filename)


def run_server(host: str = "127.0.0.1", port: int = 7860) -> None:
    print(f"ParkiGait web app → http://{host}:{port}")
    print(DISCLAIMER)
    app.run(host=host, port=port, debug=False)


if __name__ == "__main__":
    run_server()
