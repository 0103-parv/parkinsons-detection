"""ParkiGait command line.

    python -m parkigait scan  path/to/walk.mp4      # scan a real walking video
    python -m parkigait demo  --severity 0.6         # analyze a synthetic walker
    python -m parkigait train                        # (re)train the synthetic model
    python -m parkigait eval  --report               # honest metrics -> RESULTS.md
    python -m parkigait serve                        # local web app (upload a video)
    python -m parkigait render --severity 0.6        # render a synthetic walk video
    python -m parkigait selftest                     # end-to-end smoke test
"""
from __future__ import annotations

import argparse
import json
import sys


def _print_report(report) -> None:
    s = report.summary()
    print("\n" + "=" * 62)
    print("  ParkiGait report")
    print("=" * 62)
    print(f"  source:            {s['source']}")
    print(f"  steps detected:    {s['step_count']}")
    print(f"  signal confidence: {s['feature_confidence']}")
    print("  --- gait features ---")
    for k, v in s["features"].items():
        print(f"    {k:16} {v: .4f}")
    print("  --- estimate (EXPLORATORY, not a diagnosis) ---")
    if s.get("out_of_distribution"):
        print("    ** OUT-OF-DISTRIBUTION: score not meaningful for this input **")
    print(f"    P(PD-like motor signs): {s['p_pd']:.3f}")
    print(f"    severity (0-4, {s['severity_scale']}): {s['severity_0_4']:.2f}")
    print(f"    label:                  {s['label']}")
    if s["sttp_keep_fraction"] is not None:
        print(f"    STTP keep fraction:     {s['sttp_keep_fraction']}")
    if s["timings_ms"]:
        print("  --- timings (ms) ---")
        for k, v in s["timings_ms"].items():
            print(f"    {k:18} {v}")
    if s["warnings"]:
        print("  --- warnings ---")
        for w in s["warnings"]:
            print(f"    ! {w}")
    print("  " + "-" * 60)
    print(f"  {s['disclaimer']}")
    print("=" * 62 + "\n")


def cmd_scan(args) -> int:
    from parkigait.pipeline import analyze_video
    report = analyze_video(args.video, stride=args.stride, max_frames=args.max_frames)
    _print_report(report)
    if args.json:
        print(json.dumps(report.summary(), indent=2))
    return 0


def cmd_demo(args) -> int:
    from parkigait.pipeline import analyze_synthetic
    report = analyze_synthetic(severity=args.severity, seed=args.seed)
    _print_report(report)
    print(f"  (synthetic walker, TRUE severity={args.severity:.2f} on a 0-1 scale; "
          f"model severity is on a 0-4 synthetic scale)")
    return 0


def cmd_train(args) -> int:
    from parkigait.severity import train_synthetic
    _, cv = train_synthetic()
    print("Trained synthetic severity model.")
    print(json.dumps(cv, indent=2))
    return 0


def cmd_eval(args) -> int:
    from parkigait.eval import run_eval
    run_eval(write_report=args.report)
    return 0


def cmd_ablation(args) -> int:
    from parkigait.ablation import run
    run(write=True)
    return 0


def cmd_carepd_train(args) -> int:
    from parkigait.carepd import CAREPDNotAvailable, train_severity_from_carepd
    cohorts = args.cohorts.split(",") if args.cohorts else None
    try:
        r = train_severity_from_carepd(
            args.root, cohorts=cohorts, joint_source=args.joint_source,
            n_splits=args.splits, seed=args.seed)
    except CAREPDNotAvailable as e:
        print(f"CARE-PD not available:\n{e}")
        return 1
    print("\nREAL CARE-PD training (subject-level CV) —", r["calibrated_on"])
    print(f"  subjects:            {r['n_subjects']}")
    print(f"  labelled walks:      {r['n_labelled_walks']}")
    print(f"  held-out Pearson r:  {r['held_out_pearson_r']:.3f}  (predicted vs true UPDRS-gait)")
    print(f"  CV MAE:              {r['cv_mae_updrs_gait']:.3f}  (baseline predict-mean {r['baseline_mae_predict_mean']:.3f})")
    print(f"  split:               {r['split']}")
    print(f"  {r['disclaimer']}")
    return 0


def cmd_carepd_rich(args) -> int:
    from parkigait.carepd_rich import run
    cohorts = args.cohorts.split(",") if args.cohorts else None
    run(root=args.root, cohorts=cohorts)
    return 0


def cmd_clinical_eval(args) -> int:
    from parkigait.clinical_eval import run
    run(root=args.root, permute=args.permute, n_perm=args.n_perm)
    return 0


def cmd_clinical_plus(args) -> int:
    from parkigait.clinical_plus import run
    run(root=args.root)
    return 0


def cmd_patient_report(args) -> int:
    from parkigait.patient_report import main as pr_main
    return pr_main(["--root", args.root])


def cmd_serve(args) -> int:
    from parkigait.app import run_server
    run_server(host=args.host, port=args.port)
    return 0


def cmd_render(args) -> int:
    from parkigait.pose import SyntheticWalker
    from parkigait.render import render_walk_video
    pose = SyntheticWalker(args.severity).generate(duration_s=args.seconds)
    path = render_walk_video(pose, args.out)
    print(f"wrote {path}")
    return 0


def cmd_selftest(args) -> int:
    from parkigait.pipeline import analyze_synthetic
    print("Running end-to-end selftest on synthetic walkers...")
    ok = True
    prev_p = None
    for sev in (0.0, 0.5, 1.0):
        r = analyze_synthetic(severity=sev, seed=3)
        p = r.severity.p_pd
        print(f"  severity {sev:.1f}: P(PD)={p:.3f}  steps={r.features.step_count}  "
              f"conf={r.features.confidence:.2f}")
        if prev_p is not None and p < prev_p - 0.15:
            ok = False  # P(PD) should broadly rise with severity
        prev_p = p
    print("SELFTEST:", "PASS" if ok else "CHECK (P(PD) not monotonic — inspect)")
    return 0 if ok else 1


def build_parser() -> argparse.ArgumentParser:
    p = argparse.ArgumentParser(
        prog="parkigait",
        description="ParkiGait — RESEARCH PROTOTYPE for video gait analysis. "
                    "NOT a medical device; not for clinical use.")
    sub = p.add_subparsers(dest="cmd", required=True)

    sc = sub.add_parser("scan", help="scan a real walking video")
    sc.add_argument("video")
    sc.add_argument("--stride", type=int, default=2)
    sc.add_argument("--max-frames", type=int, default=None, dest="max_frames")
    sc.add_argument("--json", action="store_true")
    sc.set_defaults(func=cmd_scan)

    dm = sub.add_parser("demo", help="analyze a synthetic walker")
    dm.add_argument("--severity", type=float, default=0.5)
    dm.add_argument("--seed", type=int, default=0)
    dm.set_defaults(func=cmd_demo)

    tr = sub.add_parser("train", help="(re)train the synthetic severity model")
    tr.set_defaults(func=cmd_train)

    ev = sub.add_parser("eval", help="honest evaluation on the synthetic cohort")
    ev.add_argument("--report", action="store_true", help="write RESULTS.md")
    ev.set_defaults(func=cmd_eval)

    ab = sub.add_parser("ablation", help="ablation & robustness study -> ABLATION.md")
    ab.set_defaults(func=cmd_ablation)

    ct = sub.add_parser("carepd-train", help="train on real CARE-PD UPDRS labels (subject-level CV)")
    ct.add_argument("--root", default="data/CARE-PD")
    ct.add_argument("--cohorts", default=None, help="comma-separated, e.g. PD-GaM,BMCLab")
    ct.add_argument("--joint-source", default="canonical_fk", dest="joint_source",
                    choices=["canonical_fk", "smpl"])
    ct.add_argument("--splits", type=int, default=5)
    ct.add_argument("--seed", type=int, default=0)
    ct.set_defaults(func=cmd_carepd_train)

    cr = sub.add_parser("carepd-rich",
                        help="rich 25-feature clinical model on CARE-PD (best accuracy)")
    cr.add_argument("--root", default="data/CARE-PD")
    cr.add_argument("--cohorts", default=None)
    cr.set_defaults(func=cmd_carepd_rich)

    cp = sub.add_parser("clinical-plus",
                        help="clinician-grade eval: external validation, QWK, calibration, retest")
    cp.add_argument("--root", default="data/CARE-PD")
    cp.set_defaults(func=cmd_clinical_plus)

    pr = sub.add_parser("patient-report", help="per-patient gait panel (cohort percentiles)")
    pr.add_argument("--root", default="data/CARE-PD")
    pr.set_defaults(func=cmd_patient_report)

    ce = sub.add_parser("clinical-eval",
                        help="clinical-grade eval (CIs, ROC, calibration, importance) -> CLINICAL_EVAL.md")
    ce.add_argument("--root", default="data/CARE-PD")
    ce.add_argument("--permute", action="store_true", help="also run the permutation p-value (slow)")
    ce.add_argument("--n-perm", type=int, default=200, dest="n_perm")
    ce.set_defaults(func=cmd_clinical_eval)

    sv = sub.add_parser("serve", help="run the local web app")
    sv.add_argument("--host", default="127.0.0.1")
    sv.add_argument("--port", type=int, default=7860)
    sv.set_defaults(func=cmd_serve)

    rd = sub.add_parser("render", help="render a synthetic walk video")
    rd.add_argument("--severity", type=float, default=0.5)
    rd.add_argument("--seconds", type=float, default=8.0)
    rd.add_argument("--out", default="sample_videos/synthetic_walk.mp4")
    rd.set_defaults(func=cmd_render)

    st = sub.add_parser("selftest", help="end-to-end smoke test")
    st.set_defaults(func=cmd_selftest)
    return p


def main(argv=None) -> int:
    args = build_parser().parse_args(argv)
    return args.func(args)


if __name__ == "__main__":
    sys.exit(main())
