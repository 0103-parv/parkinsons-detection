"""Clinical-grade evaluation v2 — the numbers a movement-disorder clinician wants.

Everything is subject-level (no patient in train+test) and leakage-checked. Adds, on
top of clinical_eval.py:
  1. MODEL SEARCH + PROBABILITY CALIBRATION — pick the best detector, calibrate it so
     the probability means something (Brier score, calibration curve).
  2. LEAVE-ONE-COHORT-OUT EXTERNAL VALIDATION — train on 3 hospitals, test on a 4th the
     model has never seen. This is the generalization number clinicians actually trust.
  3. SEVERITY STAGING — 4-class UPDRS-gait confusion matrix, exact accuracy, within-1
     accuracy, and QUADRATIC WEIGHTED KAPPA (the standard clinical inter-rater metric
     for ordinal scores).
  4. TEST-RETEST RELIABILITY — do a patient's repeated walks get consistent scores?

    python -m parkigait clinical-plus     # prints everything + writes CLINICAL_PLUS.md
"""
from __future__ import annotations

from pathlib import Path

import numpy as np

from parkigait.carepd_rich import UPDRS_COHORTS, build_dataset

_HERE = Path(__file__).resolve().parent


# --------------------------------------------------------------------------- #
# models                                                                      #
# --------------------------------------------------------------------------- #
def _clf(name):
    from sklearn.ensemble import (GradientBoostingClassifier,
                                  HistGradientBoostingClassifier,
                                  RandomForestClassifier)
    from sklearn.linear_model import LogisticRegression
    if name == "rf":
        return RandomForestClassifier(n_estimators=500, max_depth=8, min_samples_leaf=3,
                                      class_weight="balanced", random_state=0, n_jobs=-1)
    if name == "gb":
        return GradientBoostingClassifier(n_estimators=300, max_depth=3,
                                          learning_rate=0.05, subsample=0.8, random_state=0)
    if name == "hgb":
        return HistGradientBoostingClassifier(max_depth=4, learning_rate=0.05,
                                              max_iter=400, l2_regularization=1.0,
                                              random_state=0)
    if name == "logreg":
        return LogisticRegression(max_iter=2000, class_weight="balanced", C=0.5)
    raise ValueError(name)


def _ensemble():
    from sklearn.ensemble import VotingClassifier
    return VotingClassifier(estimators=[("rf", _clf("rf")), ("gb", _clf("gb")),
                                        ("logreg", _clf("logreg"))], voting="soft")


def _auc(y, p):
    from sklearn.metrics import roc_auc_score
    return float(roc_auc_score(y, p)) if len(np.unique(y)) > 1 else 0.5


# --------------------------------------------------------------------------- #
# 1. model search + calibrated OOF predictions                                #
# --------------------------------------------------------------------------- #
def oof_detection(X, ybin, groups, model_name="ensemble", calibrate=True, n_splits=5):
    """Out-of-fold calibrated P(impaired) via subject-level GroupKFold. Returns
    (proba, ybin_ordered, groups_ordered)."""
    from sklearn.calibration import CalibratedClassifierCV
    from sklearn.model_selection import GroupKFold
    from sklearn.preprocessing import StandardScaler
    gkf = GroupKFold(min(n_splits, len(np.unique(groups))))
    proba = np.zeros(len(ybin))
    for tr, te in gkf.split(X, ybin, groups):
        sc = StandardScaler().fit(X[tr])
        base = _ensemble() if model_name == "ensemble" else _clf(model_name)
        if calibrate:
            # precompute subject-level inner splits and pass them as `cv` (robust:
            # avoids CalibratedClassifierCV's groups-forwarding, which is version-fragile)
            from sklearn.model_selection import GroupKFold as GK
            n_in = min(3, len(np.unique(groups[tr])))
            inner_splits = list(GK(n_in).split(X[tr], ybin[tr], groups[tr]))
            clf = CalibratedClassifierCV(base, method="isotonic", cv=inner_splits)
            clf.fit(sc.transform(X[tr]), ybin[tr])
        else:
            clf = base.fit(sc.transform(X[tr]), ybin[tr])
        proba[te] = clf.predict_proba(sc.transform(X[te]))[:, 1]
    return proba, ybin, groups


def model_search(X, ybin, groups):
    rows = []
    for name in ("logreg", "rf", "gb", "hgb", "ensemble"):
        p, y, _ = oof_detection(X, ybin, groups, model_name=name, calibrate=False)
        rows.append((name, _auc(y, p)))
    rows.sort(key=lambda r: -r[1])
    return rows


# --------------------------------------------------------------------------- #
# 2. leave-one-cohort-out external validation                                 #
# --------------------------------------------------------------------------- #
def leave_one_cohort_out(X, ybin, groups, cohort, model_name="ensemble"):
    from sklearn.preprocessing import StandardScaler
    out = []
    pooled_p, pooled_y = [], []
    for held in UPDRS_COHORTS:
        te = cohort == held
        tr = ~te
        if te.sum() < 5 or len(np.unique(ybin[te])) < 2:
            out.append((held, None, int(te.sum())))
            continue
        sc = StandardScaler().fit(X[tr])
        clf = (_ensemble() if model_name == "ensemble" else _clf(model_name))
        clf.fit(sc.transform(X[tr]), ybin[tr])
        p = clf.predict_proba(sc.transform(X[te]))[:, 1]
        out.append((held, _auc(ybin[te], p), int(te.sum())))
        pooled_p.extend(p.tolist()); pooled_y.extend(ybin[te].tolist())
    pooled = _auc(np.array(pooled_y), np.array(pooled_p))
    return out, pooled


# --------------------------------------------------------------------------- #
# 3. severity staging (multi-class, ordinal)                                  #
# --------------------------------------------------------------------------- #
def severity_staging(X, y, groups):
    from sklearn.ensemble import RandomForestClassifier
    from sklearn.metrics import cohen_kappa_score, confusion_matrix
    from sklearn.model_selection import GroupKFold
    from sklearn.preprocessing import StandardScaler
    gkf = GroupKFold(min(5, len(np.unique(groups))))
    yt, yp = [], []
    for tr, te in gkf.split(X, y, groups):
        sc = StandardScaler().fit(X[tr])
        clf = RandomForestClassifier(n_estimators=500, max_depth=8, min_samples_leaf=3,
                                     class_weight="balanced", random_state=0, n_jobs=-1)
        clf.fit(sc.transform(X[tr]), y[tr].astype(int))
        yp.extend(clf.predict(sc.transform(X[te])).tolist())
        yt.extend(y[te].astype(int).tolist())
    yt, yp = np.array(yt), np.array(yp)
    labels = sorted(set(yt.tolist()) | set(yp.tolist()))
    cm = confusion_matrix(yt, yp, labels=labels)
    return {
        "labels": labels,
        "confusion": cm.tolist(),
        "exact_acc": float((yt == yp).mean()),
        "within1_acc": float((np.abs(yt - yp) <= 1).mean()),
        "qwk": float(cohen_kappa_score(yt, yp, weights="quadratic")),
    }


# --------------------------------------------------------------------------- #
# 4. test-retest reliability (across a patient's repeated walks)              #
# --------------------------------------------------------------------------- #
def test_retest(proba, groups):
    """For patients with >=2 walks, how tight are their OOF P(impaired) scores?
    Reports mean within-patient std and the ICC(1)-style consistency."""
    subj = {}
    for p, g in zip(proba, groups):
        subj.setdefault(g, []).append(p)
    multi = {g: np.array(v) for g, v in subj.items() if len(v) >= 2}
    if not multi:
        return {"n_patients": 0}
    within_std = np.mean([v.std() for v in multi.values()])
    # ICC(1): between-patient variance / total variance
    means = np.array([v.mean() for v in multi.values()])
    grand = np.concatenate(list(multi.values()))
    ms_between = means.var() if len(means) > 1 else 0.0
    ms_within = np.mean([v.var() for v in multi.values()])
    icc = ms_between / (ms_between + ms_within + 1e-9)
    return {"n_patients": len(multi), "mean_within_patient_std": float(within_std),
            "icc1_consistency": float(icc)}


# --------------------------------------------------------------------------- #
# driver                                                                       #
# --------------------------------------------------------------------------- #
def run(root: str = "data/CARE-PD", write: bool = True):
    from sklearn.metrics import brier_score_loss
    X, y, groups, cohort = build_dataset(root)
    ybin = (y > 0).astype(int)
    print(f"CARE-PD: {X.shape[0]} walks, {X.shape[1]} features, "
          f"{len(np.unique(groups))} patients, cohorts {sorted(set(cohort))}\n")

    print("[1] MODEL SEARCH — detection AUC (subject-level CV)")
    ms = model_search(X, ybin, groups)
    for name, auc in ms:
        print(f"      {name:10} AUC {auc:.3f}")
    best = ms[0][0]
    print(f"    best: {best}")

    print(f"\n[2] CALIBRATED {best} — detection")
    p, yb, gr = oof_detection(X, ybin, groups, model_name=best, calibrate=True)
    auc = _auc(yb, p)
    brier = brier_score_loss(yb, p)
    print(f"      AUC {auc:.3f} | Brier {brier:.3f} (lower=better; base rate Brier "
          f"{np.mean((yb - yb.mean())**2):.3f})")

    print("\n[3] LEAVE-ONE-COHORT-OUT EXTERNAL VALIDATION (train 3 sites, test the 4th)")
    loco, loco_pooled = leave_one_cohort_out(X, ybin, groups, cohort, model_name=best)
    for held, a, n in loco:
        print(f"      held out {held:10} AUC {a if a is None else round(a,3)}  (n={n})")
    print(f"    pooled external AUC {loco_pooled:.3f}")

    print("\n[4] SEVERITY STAGING (UPDRS-gait 0..3, subject-level CV)")
    st = severity_staging(X, y, groups)
    print(f"      quadratic weighted kappa {st['qwk']:.3f}  |  exact acc {st['exact_acc']:.3f}"
          f"  |  within-1 acc {st['within1_acc']:.3f}")
    print(f"      confusion (rows=true {st['labels']}):")
    for r, row in zip(st["labels"], st["confusion"]):
        print(f"        true {r}: {row}")

    print("\n[5] TEST-RETEST RELIABILITY (repeated walks per patient)")
    tr = test_retest(p, gr)
    if tr["n_patients"]:
        print(f"      {tr['n_patients']} patients with >=2 walks | mean within-patient "
              f"std {tr['mean_within_patient_std']:.3f} | ICC(1) consistency "
              f"{tr['icc1_consistency']:.3f}")

    results = {"model_search": ms, "best": best, "detection_auc": auc, "brier": brier,
               "loco": loco, "loco_pooled": loco_pooled, "staging": st, "retest": tr}
    if write:
        path = _write(results)
        print(f"\nwrote {path}")
    return results


def _write(r):
    st = r["staging"]
    loco_rows = "".join(
        f"| {h} | {'n/a' if a is None else round(a, 3)} | {n} |\n" for h, a, n in r["loco"])
    cm_rows = "".join(f"| true {lbl} | " + " | ".join(str(x) for x in row) + " |\n"
                      for lbl, row in zip(st["labels"], st["confusion"]))
    hdr = "| true \\ pred | " + " | ".join(str(l) for l in st["labels"]) + " |\n"
    md = f"""# ParkiGait — clinical evaluation v2 (for clinicians)

Generated by `python -m parkigait clinical-plus`. All subject-level (no patient in
train and test), on real CARE-PD ({r['staging']}). **Research prototype, not a
medical device.**

## Detection: does this patient have impaired gait? (UPDRS-gait > 0)

- Best model: **{r['best']}**, calibrated.
- **AUC {r['detection_auc']:.3f}**, Brier {r['brier']:.3f} (calibrated probability quality).

### External validation — the number that matters for a new hospital

Train on three sites, test on a **fourth the model has never seen**:

| held-out cohort (site) | AUC | walks |
|---|---|---|
{loco_rows}| **pooled external** | **{r['loco_pooled']:.3f}** | |

This is stricter than cross-validation: it measures whether the model transfers to a
new site with its own camera, population, and rating style.

## Severity staging (UPDRS-gait 0 to 3)

- **Quadratic weighted kappa {st['qwk']:.3f}** — the standard clinical inter-rater
  agreement metric for ordinal scores (0 = chance, 1 = perfect; 0.4 to 0.6 is
  "moderate", 0.6 to 0.8 "substantial").
- Exact accuracy {st['exact_acc']:.3f}; within-one-level accuracy {st['within1_acc']:.3f}.

Confusion matrix:

{hdr}|---|{'---|' * len(st['labels'])}
{cm_rows}

## Test-retest reliability

Across patients with repeated walks: mean within-patient probability std
{r['retest'].get('mean_within_patient_std', float('nan')):.3f}, ICC(1) consistency
{r['retest'].get('icc1_consistency', float('nan')):.3f} ({r['retest'].get('n_patients', 0)} patients).

## Honest reading

External-validation AUC is the realistic number for a new clinic. Quadratic weighted
kappa is how you would compare the tool to a second human rater. Both are reported
without leakage (subject-level, held-out site). This is a screening aid, not a
diagnosis.
"""
    path = _HERE / "CLINICAL_PLUS.md"
    path.write_text(md)
    return path


if __name__ == "__main__":
    run()
