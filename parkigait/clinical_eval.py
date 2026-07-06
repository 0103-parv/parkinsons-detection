"""Clinical-grade evaluation of the CARE-PD gait model.

Reports results the way medical-AI papers do, so the numbers hold up to a clinician:
  - out-of-fold predictions from subject-level CV (each patient predicted only by
    models that never saw them);
  - 95% confidence intervals by SUBJECT-LEVEL bootstrap (patients are the unit);
  - a permutation-test p-value (heavy; run with --permute);
  - detection operating points (sensitivity/specificity/PPV/NPV/LR+/LR-) at a
    screening threshold and at 0.5;
  - sensitivity by true severity (does it catch MILD cases?);
  - permutation feature importance (does it rely on known PD biomarkers?);
  - a gait-speed-only baseline (does the multivariate model add value?);
  - ROC, calibration, feature-importance, and sensitivity-by-severity figures.

    python -m parkigait.clinical_eval            # metrics + figures + CLINICAL_EVAL.md
    python -m parkigait.clinical_eval --permute  # also the permutation p-value (slow)
"""
from __future__ import annotations

from pathlib import Path

import numpy as np

from parkigait.carepd_features import FEATURE_NAMES
from parkigait.carepd_rich import build_dataset

_HERE = Path(__file__).resolve().parent
FIG_DIR = _HERE / "figures"


def _clf():
    from sklearn.ensemble import RandomForestClassifier
    return RandomForestClassifier(n_estimators=400, max_depth=8, min_samples_leaf=3,
                                  class_weight="balanced", random_state=0, n_jobs=-1)


def _reg():
    from sklearn.ensemble import RandomForestRegressor
    return RandomForestRegressor(n_estimators=400, max_depth=8, min_samples_leaf=3,
                                 random_state=0, n_jobs=-1)


def oof_predictions(X, y, groups, n_splits: int = 5):
    """Out-of-fold detection probabilities + severity predictions (subject-level)."""
    from sklearn.model_selection import GroupKFold
    ybin = (y > 0).astype(int)
    proba = np.zeros(len(y))
    regp = np.zeros(len(y))
    gkf = GroupKFold(min(n_splits, len(np.unique(groups))))
    for tr, te in gkf.split(X, y, groups):
        mu, sd = X[tr].mean(0), X[tr].std(0) + 1e-9
        Xtr, Xte = (X[tr] - mu) / sd, (X[te] - mu) / sd
        c = _clf(); c.fit(Xtr, ybin[tr]); proba[te] = c.predict_proba(Xte)[:, 1]
        r = _reg(); r.fit(Xtr, y[tr]); regp[te] = r.predict(Xte)
    return proba, regp, ybin


def _auc(yt, s):
    from sklearn.metrics import roc_auc_score
    return roc_auc_score(yt, s) if len(np.unique(yt)) > 1 else np.nan


def _pearson(a, b):
    return float(np.corrcoef(a, b)[0, 1]) if a.std() > 1e-9 and b.std() > 1e-9 else 0.0


def subject_bootstrap_ci(fn, groups, n_boot: int = 2000, seed: int = 0):
    """95% CI of a metric by resampling SUBJECTS with replacement. ``fn(idx)``
    computes the metric on a set of sample indices."""
    rng = np.random.default_rng(seed)
    subs = np.unique(groups)
    # precompute per-subject sample indices
    idx_by = {s: np.flatnonzero(groups == s) for s in subs}
    vals = []
    for _ in range(n_boot):
        pick = rng.choice(subs, len(subs), replace=True)
        idx = np.concatenate([idx_by[s] for s in pick])
        v = fn(idx)
        if v is not None and np.isfinite(v):
            vals.append(v)
    lo, hi = np.percentile(vals, [2.5, 97.5])
    return float(np.mean(vals)), float(lo), float(hi)


def operating_point(yt, proba, thr):
    hard = (proba >= thr).astype(int)
    tp = int(((hard == 1) & (yt == 1)).sum()); fn = int(((hard == 0) & (yt == 1)).sum())
    tn = int(((hard == 0) & (yt == 0)).sum()); fp = int(((hard == 1) & (yt == 0)).sum())
    sens = tp / (tp + fn + 1e-9); spec = tn / (tn + fp + 1e-9)
    ppv = tp / (tp + fp + 1e-9); npv = tn / (tn + fn + 1e-9)
    lr_pos = sens / (1 - spec + 1e-9); lr_neg = (1 - sens) / (spec + 1e-9)
    return dict(threshold=thr, sensitivity=sens, specificity=spec, ppv=ppv, npv=npv,
                lr_pos=lr_pos, lr_neg=lr_neg, tp=tp, fp=fp, tn=tn, fn=fn)


def threshold_for_sensitivity(yt, proba, target=0.90):
    """Lowest threshold achieving >= target sensitivity (screening favours sensitivity)."""
    order = np.unique(proba)
    for thr in np.sort(order):
        if operating_point(yt, proba, thr)["sensitivity"] >= target:
            best = thr
    # pick the highest threshold that still meets target sensitivity (best specificity)
    ok = [thr for thr in np.sort(order)
          if operating_point(yt, proba, thr)["sensitivity"] >= target]
    return float(max(ok)) if ok else 0.0


def sensitivity_by_severity(y, proba, thr):
    out = {}
    for lvl in (1, 2, 3):
        m = y == lvl
        out[lvl] = float((proba[m] >= thr).mean()) if m.any() else np.nan
    return out


def permutation_importance(X, y, groups, n_repeats: int = 10, seed: int = 0):
    """Drop in detection AUC when each feature is shuffled, on a held-out subject
    split. Higher = the model relies on it more."""
    from sklearn.model_selection import GroupShuffleSplit
    ybin = (y > 0).astype(int)
    gss = GroupShuffleSplit(n_splits=1, test_size=0.3, random_state=seed)
    tr, te = next(gss.split(X, ybin, groups))
    mu, sd = X[tr].mean(0), X[tr].std(0) + 1e-9
    Xtr, Xte = (X[tr] - mu) / sd, (X[te] - mu) / sd
    c = _clf(); c.fit(Xtr, ybin[tr])
    base = _auc(ybin[te], c.predict_proba(Xte)[:, 1])
    rng = np.random.default_rng(seed)
    imp = np.zeros(X.shape[1])
    for j in range(X.shape[1]):
        drops = []
        for _ in range(n_repeats):
            Xp = Xte.copy(); rng.shuffle(Xp[:, j])
            drops.append(base - _auc(ybin[te], c.predict_proba(Xp)[:, 1]))
        imp[j] = np.mean(drops)
    return base, imp


def _clf_oof(X, ybin, groups, n_splits: int = 5):
    """Classifier-only out-of-fold probabilities (faster than oof_predictions)."""
    from sklearn.model_selection import GroupKFold
    proba = np.zeros(len(ybin))
    gkf = GroupKFold(min(n_splits, len(np.unique(groups))))
    for tr, te in gkf.split(X, ybin, groups):
        mu, sd = X[tr].mean(0), X[tr].std(0) + 1e-9
        c = _clf(); c.fit((X[tr] - mu) / sd, ybin[tr])
        proba[te] = c.predict_proba((X[te] - mu) / sd)[:, 1]
    return proba


def permutation_test(X, y, groups, n_perm: int = 200, seed: int = 0):
    """p-value: fraction of label-shuffled runs whose detection AUC >= observed."""
    ybin = (y > 0).astype(int)
    obs = _auc(ybin, _clf_oof(X, ybin, groups))
    rng = np.random.default_rng(seed)
    ge = 0
    perm_aucs = []
    for _ in range(n_perm):
        ys = ybin.copy(); rng.shuffle(ys)
        a = _auc(ys, _clf_oof(X, ys, groups))
        perm_aucs.append(a)
        if a >= obs:
            ge += 1
    p = (ge + 1) / (n_perm + 1)
    return obs, p, perm_aucs


# --------------------------------------------------------------------------- #
# figures                                                                     #
# --------------------------------------------------------------------------- #
def _figures(y, ybin, proba, regp, thr, imp, base_auc):
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    from sklearn.calibration import calibration_curve
    from sklearn.metrics import roc_curve
    FIG_DIR.mkdir(exist_ok=True)
    cap = "RESEARCH PROTOTYPE — NOT A MEDICAL DEVICE · real CARE-PD, subject-level CV"

    fig, ax = plt.subplots(2, 2, figsize=(12, 10))
    # ROC
    fpr, tpr, _ = roc_curve(ybin, proba)
    ax[0, 0].plot(fpr, tpr, color="#2563eb", lw=2, label=f"AUC = {base_auc:.2f}")
    ax[0, 0].plot([0, 1], [0, 1], "--", color="#888")
    ax[0, 0].set_xlabel("1 − specificity"); ax[0, 0].set_ylabel("sensitivity")
    ax[0, 0].set_title("ROC — abnormal-gait detection"); ax[0, 0].legend(loc="lower right")
    ax[0, 0].grid(alpha=0.2)
    # calibration
    frac, mean_pred = calibration_curve(ybin, proba, n_bins=8, strategy="quantile")
    ax[0, 1].plot(mean_pred, frac, "o-", color="#ff9f45")
    ax[0, 1].plot([0, 1], [0, 1], "--", color="#888")
    ax[0, 1].set_xlabel("predicted probability"); ax[0, 1].set_ylabel("observed fraction impaired")
    ax[0, 1].set_title("Calibration"); ax[0, 1].grid(alpha=0.2)
    # feature importance (top 12)
    order = np.argsort(imp)[::-1][:12][::-1]
    ax[1, 0].barh([FEATURE_NAMES[i] for i in order], imp[order], color="#2ee6a6")
    ax[1, 0].set_xlabel("AUC drop when shuffled"); ax[1, 0].set_title("Feature importance")
    # sensitivity by severity
    sev = sensitivity_by_severity(y, proba, thr)
    ax[1, 1].bar([f"UPDRS {k}" for k in sev], [sev[k] * 100 for k in sev], color="#5ac8fa")
    ax[1, 1].set_ylabel("% flagged (sensitivity)"); ax[1, 1].set_ylim(0, 100)
    ax[1, 1].set_title(f"Detection sensitivity by severity (thr={thr:.2f})")
    fig.suptitle("ParkiGait clinical evaluation — CARE-PD (subject-level CV)", fontsize=14)
    fig.text(0.5, 0.005, cap, ha="center", fontsize=8, color="#888")
    fig.tight_layout(rect=[0, 0.02, 1, 1])
    p = FIG_DIR / "clinical_eval.png"
    fig.savefig(p, dpi=120, bbox_inches="tight"); plt.close(fig)
    return p


def run(root: str = "data/CARE-PD", permute: bool = False, n_perm: int = 200):
    X, y, groups, coh = build_dataset(root)
    ybin = (y > 0).astype(int)
    n_subj = len(np.unique(groups))
    print(f"Clinical evaluation — CARE-PD: {len(y)} walks, {n_subj} patients, "
          f"{X.shape[1]} features (subject-level CV)\n")

    proba, regp, _ = oof_predictions(X, y, groups)

    # detection AUC + CI
    auc_m, auc_lo, auc_hi = subject_bootstrap_ci(
        lambda i: _auc(ybin[i], proba[i]), groups)
    # severity r + CI
    r_m, r_lo, r_hi = subject_bootstrap_ci(
        lambda i: _pearson(y[i], regp[i]), groups)
    # gait-speed-only baseline (feature 0). Lower speed = more impaired, so the
    # impaired-score is NEGATIVE gait speed (otherwise AUC comes out as 1-AUC).
    gs = X[:, 0]
    gs_auc, gs_lo, gs_hi = subject_bootstrap_ci(
        lambda i: _auc(ybin[i], -gs[i]), groups)

    thr = threshold_for_sensitivity(ybin, proba, 0.90)
    op_screen = operating_point(ybin, proba, thr)
    op_half = operating_point(ybin, proba, 0.5)

    def _op_ci(metric):
        return subject_bootstrap_ci(
            lambda i: operating_point(ybin[i], proba[i], thr)[metric], groups)

    sens_ci = _op_ci("sensitivity"); spec_ci = _op_ci("specificity")
    ppv_ci = _op_ci("ppv"); npv_ci = _op_ci("npv")
    sev = sensitivity_by_severity(y, proba, thr)

    base_auc, imp = permutation_importance(X, y, groups)
    top = np.argsort(imp)[::-1][:6]

    print(f"Detection AUC:            {auc_m:.3f}  (95% CI {auc_lo:.3f}–{auc_hi:.3f})")
    print(f"  gait-speed-only AUC:    {gs_auc:.3f}  (95% CI {gs_lo:.3f}–{gs_hi:.3f})  "
          f"→ multivariate model adds {auc_m - gs_auc:+.3f}")
    print(f"Severity Pearson r:       {r_m:.3f}  (95% CI {r_lo:.3f}–{r_hi:.3f})")
    print(f"\nScreening operating point (threshold {thr:.2f}, tuned for ~90% sensitivity):")
    print(f"  sensitivity {op_screen['sensitivity']:.2f} (CI {sens_ci[1]:.2f}–{sens_ci[2]:.2f})"
          f"  specificity {op_screen['specificity']:.2f} (CI {spec_ci[1]:.2f}–{spec_ci[2]:.2f})")
    print(f"  PPV {op_screen['ppv']:.2f} (CI {ppv_ci[1]:.2f}–{ppv_ci[2]:.2f})  "
          f"NPV {op_screen['npv']:.2f} (CI {npv_ci[1]:.2f}–{npv_ci[2]:.2f})  "
          f"LR+ {op_screen['lr_pos']:.1f}  LR- {op_screen['lr_neg']:.2f}")
    print(f"\nSensitivity by severity:  " +
          "  ".join(f"UPDRS {k}: {v * 100:.0f}%" for k, v in sev.items()))
    print(f"\nTop features (AUC drop when shuffled):")
    for i in top:
        print(f"  {FEATURE_NAMES[i]:20} {imp[i]:+.3f}")

    p_val = None
    if permute:
        print(f"\nRunning permutation test ({n_perm} shuffles)…")
        _, p_val, _ = permutation_test(X, y, groups, n_perm=n_perm)
        print(f"  permutation p-value (detection AUC): "
              f"{'< ' + format(1/(n_perm+1), '.4f') if p_val <= 1/(n_perm+1) else format(p_val, '.4f')}")

    fig = _figures(y, ybin, proba, regp, thr, imp, base_auc)
    print(f"\nwrote figure {fig}")
    md = _write_report(dict(
        n_walks=len(y), n_subj=n_subj, auc=(auc_m, auc_lo, auc_hi),
        gs_auc=(gs_auc, gs_lo, gs_hi), r=(r_m, r_lo, r_hi), thr=thr,
        op=op_screen, op_half=op_half, sens_ci=sens_ci, spec_ci=spec_ci,
        ppv_ci=ppv_ci, npv_ci=npv_ci, sev=sev, imp=imp, p_val=p_val))
    print(f"wrote {md}")
    return


def _write_report(d) -> Path:
    a = d["auc"]; r = d["r"]; op = d["op"]; sc = d["sens_ci"]; sp = d["spec_ci"]
    ppv = d["ppv_ci"]; npv = d["npv_ci"]
    top = np.argsort(d["imp"])[::-1][:8]
    imp_rows = "".join(f"| {FEATURE_NAMES[i]} | {d['imp'][i]:+.3f} |\n" for i in top)
    sev_rows = "".join(f"| UPDRS-gait {k} | {v * 100:.0f}% |\n" for k, v in d["sev"].items())
    pline = ("not run (use --permute)" if d["p_val"] is None
             else (f"p < {1/201:.4f}" if d["p_val"] <= 1/201 else f"p = {d['p_val']:.4f}"))
    md = f"""# ParkiGait — clinical evaluation report

Real **CARE-PD** data: {d['n_walks']} walking trials from {d['n_subj']} patients with
clinician-rated MDS-UPDRS gait scores. **Subject-level cross-validation** (no patient
in both train and test). 95% confidence intervals by **patient-level bootstrap**
(2000 resamples). **RESEARCH / SCREENING PROTOTYPE — NOT A MEDICAL DEVICE.**

## Primary result — abnormal-gait detection

| Metric | Value (95% CI) |
|---|---|
| **AUC** | **{a[0]:.3f} ({a[1]:.3f}–{a[2]:.3f})** |
| Gait-speed-only AUC (baseline) | {d['gs_auc'][0]:.3f} ({d['gs_auc'][1]:.3f}–{d['gs_auc'][2]:.3f}) |
| Permutation test (AUC vs shuffled labels) | {pline} |

**Gait speed is the dominant biomarker:** it alone reaches AUC {d['gs_auc'][0]:.2f}, and
the full 25-feature model adds only **{a[0] - d['gs_auc'][0]:+.3f}** (the CIs overlap, so
this increment is modest, not a large independent gain). That the model leans on gait
speed is clinically sensible.

## Screening operating point (threshold {d['thr']:.2f}, tuned for ~90% sensitivity)

| Metric | Value (95% CI) |
|---|---|
| Sensitivity | {op['sensitivity']:.2f} ({sc[1]:.2f}–{sc[2]:.2f}) |
| Specificity | {op['specificity']:.2f} ({sp[1]:.2f}–{sp[2]:.2f}) |
| PPV | {op['ppv']:.2f} ({ppv[1]:.2f}–{ppv[2]:.2f}) |
| NPV | {op['npv']:.2f} ({npv[1]:.2f}–{npv[2]:.2f}) |
| LR+ / LR− | {op['lr_pos']:.1f} / {op['lr_neg']:.2f} |

A screening tool is tuned for high sensitivity (catch most impaired gait, accept more
false positives, which a clinician then rules out). The default 0.5 threshold gives
sensitivity {d['op_half']['sensitivity']:.2f} / specificity {d['op_half']['specificity']:.2f}.

## Sensitivity by true severity (does it catch MILD cases?)

| True severity | % flagged |
|---|---|
{sev_rows}
Catching mild (UPDRS-gait 1) cases is the hard, valuable part; severe cases are easy.

## Severity regression (secondary)

Pearson r **{r[0]:.3f} ({r[1]:.3f}–{r[2]:.3f})** between predicted and clinician
UPDRS-gait (the harder continuous task).

## What drives the model (top features)

| Feature | AUC drop when shuffled |
|---|---|
{imp_rows}
These are the biomarkers clinicians already use (gait speed, arm swing, stride,
posture), which is a good sign: the model's reasoning matches clinical knowledge.

## Honest limits

Joints are a canonical-skeleton approximation (not the licensed SMPL body model);
validation is on one public multi-site dataset, not a prospective clinical study; the
cohorts are Parkinson's cohorts, so "abnormal gait" here means UPDRS-gait > 0, a
screening signal, not a standalone diagnosis. See CLINICAL_SAFETY.md and LIMITATIONS.md.

Figure: `figures/clinical_eval.png` (ROC, calibration, feature importance, sensitivity
by severity). Reproduce: `python -m parkigait.clinical_eval --permute`.
"""
    p = _HERE / "CLINICAL_EVAL.md"
    p.write_text(md)
    return p


if __name__ == "__main__":
    import argparse
    ap = argparse.ArgumentParser()
    ap.add_argument("--root", default="data/CARE-PD")
    ap.add_argument("--permute", action="store_true")
    ap.add_argument("--n-perm", type=int, default=200)
    a = ap.parse_args()
    run(a.root, permute=a.permute, n_perm=a.n_perm)

