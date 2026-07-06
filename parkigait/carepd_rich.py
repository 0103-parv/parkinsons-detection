"""Honest, leakage-proof training on CARE-PD with the rich clinical features.

Everything here is evaluated with SUBJECT-LEVEL GroupKFold (no subject in both
train and test) and per-fold standardization (scaler fit on train only), so the
reported held-out Pearson r is a real generalization number. A permutation control
(shuffled labels) confirms the pipeline scores ~0 when there is no real signal —
the check that the number isn't leakage.

    python -m parkigait.carepd_rich --root data/CARE-PD
"""
from __future__ import annotations

import pickle
from pathlib import Path

import numpy as np

from parkigait.carepd_features import FEATURE_NAMES, feature_vector

UPDRS_COHORTS = ["3DGait", "BMCLab", "PD-GaM", "T-SDU-PD"]


def build_dataset(root: str, cohorts=None, cache: bool = True):
    """Extract rich features for every UPDRS-labelled walk. Returns X, y, groups
    (subject ids, namespaced by cohort), cohort array. Caches to data/ (gitignored)."""
    cohorts = cohorts or UPDRS_COHORTS
    root = Path(root)
    cache_path = root / f"_rich_{'_'.join(cohorts)}.npz"
    if cache and cache_path.exists():
        d = np.load(cache_path, allow_pickle=True)
        return d["X"], d["y"], d["groups"], d["cohort"]
    X, y, groups, coh = [], [], [], []
    for c in cohorts:
        data = pickle.load(open(root / f"{c}.pkl", "rb"))
        for subj, walks in data.items():
            for w, rec in walks.items():
                u = rec.get("UPDRS_GAIT") if isinstance(rec, dict) else None
                if u is None:
                    continue
                X.append(feature_vector(rec["pose"], rec["trans"], rec["fps"]))
                y.append(int(u))
                groups.append(f"{c}:{subj}")   # subject id unique across cohorts
                coh.append(c)
    X = np.asarray(X, float)
    y = np.asarray(y, float)
    groups = np.asarray(groups)
    coh = np.asarray(coh)
    if cache:
        np.savez(cache_path, X=X, y=y, groups=groups, cohort=coh)
    return X, y, groups, coh


def _pearson(a, b):
    a, b = np.asarray(a, float), np.asarray(b, float)
    return float(np.corrcoef(a, b)[0, 1]) if a.std() > 1e-9 and b.std() > 1e-9 else 0.0


def evaluate(X, y, groups, model_factory, n_splits: int = 5, seed: int = 0):
    """Subject-level GroupKFold; per-fold standardize (train-only); pooled held-out
    predictions -> Pearson r, r^2, R^2, MAE. Also a subject-aggregated r (mean of a
    subject's held-out predictions vs that subject's mean label) which denoises
    walk-to-walk variation — a legitimate subject-level view, not leakage."""
    from sklearn.model_selection import GroupKFold
    n_splits = min(n_splits, len(np.unique(groups)))
    gkf = GroupKFold(n_splits=n_splits)
    pt, pp, pg = [], [], []
    for tr, te in gkf.split(X, y, groups):
        mu, sd = X[tr].mean(0), X[tr].std(0) + 1e-9
        Xtr, Xte = (X[tr] - mu) / sd, (X[te] - mu) / sd
        m = model_factory()
        m.fit(Xtr, y[tr])
        pp.extend(m.predict(Xte).tolist())
        pt.extend(y[te].tolist())
        pg.extend(np.asarray(groups)[te].tolist())
    pt, pp, pg = np.asarray(pt), np.asarray(pp), np.asarray(pg)
    r = _pearson(pt, pp)
    ss_res = float(((pt - pp) ** 2).sum())
    ss_tot = float(((pt - pt.mean()) ** 2).sum()) + 1e-12
    # subject-aggregated: mean predicted vs mean true per subject
    subj = np.unique(pg)
    sp = np.array([pp[pg == s].mean() for s in subj])
    st = np.array([pt[pg == s].mean() for s in subj])
    return {
        "held_out_pearson_r": r,
        "held_out_r2": r * r,
        "R2_determination": 1.0 - ss_res / ss_tot,
        "subject_aggregated_r": _pearson(st, sp),
        "mae": float(np.mean(np.abs(pt - pp))),
        "baseline_mae": float(np.mean(np.abs(pt - pt.mean()))),
        "n": int(len(pt)),
    }


def _models():
    from sklearn.ensemble import GradientBoostingRegressor, RandomForestRegressor
    from sklearn.linear_model import Ridge
    return {
        "ridge": lambda: Ridge(alpha=1.0),
        "random_forest": lambda: RandomForestRegressor(
            n_estimators=200, max_depth=6, min_samples_leaf=5, random_state=0, n_jobs=-1),
        "grad_boost": lambda: GradientBoostingRegressor(
            n_estimators=200, max_depth=3, learning_rate=0.05, subsample=0.8,
            random_state=0),
    }


def run(root: str = "data/CARE-PD", cohorts=None, permutation_control: bool = True):
    X, y, groups, coh = build_dataset(root, cohorts)
    print(f"CARE-PD rich features: {X.shape[0]} labelled walks, {X.shape[1]} features, "
          f"{len(np.unique(groups))} subjects\n")

    print(f"{'model':16} {'r':>7} {'r^2':>7} {'R^2':>7} {'subj-r':>7} {'MAE':>7}   "
          f"(pooled subject-level CV)")
    print("-" * 68)
    best = None
    for name, fac in _models().items():
        r = evaluate(X, y, groups, fac)
        if best is None or r["held_out_pearson_r"] > best[1]["held_out_pearson_r"]:
            best = (name, r)
        print(f"{name:16} {r['held_out_pearson_r']:>7.3f} {r['held_out_r2']:>7.3f} "
              f"{r['R2_determination']:>7.3f} {r['subject_aggregated_r']:>7.3f} {r['mae']:>7.3f}")
    print("-" * 68)
    b = best[1]
    print(f"BEST: {best[0]}  r={b['held_out_pearson_r']:.3f}  r^2={b['held_out_r2']:.3f}  "
          f"R^2={b['R2_determination']:.3f}  subject-aggregated r={b['subject_aggregated_r']:.3f}")

    # per-cohort with the best model
    print("\nper-cohort (best model, subject-level CV):")
    from sklearn.ensemble import GradientBoostingRegressor, RandomForestRegressor  # noqa
    fac = _models()[best[0]]
    for c in (cohorts or UPDRS_COHORTS):
        m = coh == c
        if m.sum() < 10:
            continue
        rc = evaluate(X[m], y[m], groups[m], fac)
        print(f"  {c:10} r={rc['held_out_pearson_r']:.3f}  r^2={rc['held_out_r2']:.3f}  "
              f"MAE={rc['mae']:.3f}  (baseline {rc['baseline_mae']:.3f}, n={rc['n']})")

    # permutation control: shuffle labels -> r should collapse to ~0 (no leakage)
    if permutation_control:
        rng = np.random.default_rng(0)
        y_shuf = y.copy()
        rng.shuffle(y_shuf)
        rp = evaluate(X, y_shuf, groups, fac)
        print(f"\nPERMUTATION CONTROL (labels shuffled): held-out r = "
              f"{rp['held_out_pearson_r']:+.3f}  (should be ~0 — confirms no leakage)")

    return {"best_model": best[0], "best": best[1]}


if __name__ == "__main__":
    import argparse
    ap = argparse.ArgumentParser()
    ap.add_argument("--root", default="data/CARE-PD")
    ap.add_argument("--no-cache", action="store_true")
    a = ap.parse_args()
    if a.no_cache:
        for p in Path(a.root).glob("_rich_*.npz"):
            p.unlink()
    run(a.root)
