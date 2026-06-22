"""A complete, WORKING Parkinson's detector on REAL clinical data — honestly validated.

Data: UCI Parkinsons (Little et al. 2007) — 195 sustained-phonation voice recordings from
31 people (23 with Parkinson's, 8 healthy), 22 biomedical voice features (jitter, shimmer,
NHR/HNR, RPDE, DFA, PPE, ...). Label = `status` (1 = PD). Downloaded by data.py.

THE METHODOLOGY THAT MATTERS — subject-level validation:
  Each person contributes ~6 recordings. A naive record-level CV split puts some of a
  patient's recordings in train and others in test, so the model can recognise the *person*
  instead of the *disease* -> inflated, dishonest AUC. The honest question is "does it
  generalise to a NEW patient?", which requires that no subject's recordings cross the
  train/test boundary (StratifiedGroupKFold by subject). This file reports BOTH and shows
  the gap — the same anti-overfit discipline that is mentat's whole thesis, on real data.

Run:  cd ~/mentat && PYTHONPATH=. ~/swechats/.venv/bin/python reference/parkinsons/detect.py
"""
from __future__ import annotations

import re
from pathlib import Path

import numpy as np
import pandas as pd
from sklearn.ensemble import HistGradientBoostingClassifier, RandomForestClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import balanced_accuracy_score, confusion_matrix, roc_auc_score
from sklearn.model_selection import StratifiedGroupKFold, StratifiedKFold
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler
from sklearn.svm import SVC

DATA = Path(__file__).parent / "data" / "parkinsons.data"


def load():
    """Return (X dataframe, y array, subject-id groups, feature names)."""
    df = pd.read_csv(DATA)
    # name like 'phon_R01_S01_1' -> subject 'S01' (recordings of the same person share it)
    groups = df["name"].map(lambda s: re.search(r"S\d+", s).group()).to_numpy()
    y = df["status"].to_numpy().astype(int)
    feats = [c for c in df.columns if c not in ("name", "status")]
    return df[feats].copy(), y, groups, feats


def _model(name: str):
    # class_weight='balanced' counters the 147/48 PD/healthy imbalance (and the 8 healthy subjects)
    if name == "logreg":
        return LogisticRegression(max_iter=5000, C=1.0, class_weight="balanced")
    if name == "svm_rbf":
        return SVC(kernel="rbf", C=2.0, gamma="scale", class_weight="balanced", random_state=0)
    if name == "random_forest":
        return RandomForestClassifier(n_estimators=400, class_weight="balanced_subsample",
                                      random_state=0, n_jobs=-1)
    if name == "hist_gb":
        return HistGradientBoostingClassifier(class_weight="balanced", random_state=0)
    raise ValueError(name)


def _pipe(name: str) -> Pipeline:
    return Pipeline([("scale", StandardScaler()), ("clf", _model(name))])


def _scores(pipe: Pipeline, X) -> np.ndarray:
    """PD-class score for AUC — predict_proba when available, else the SVM decision function."""
    clf = pipe[-1]
    if hasattr(clf, "predict_proba"):
        return pipe.predict_proba(X)[:, 1]
    return pipe.decision_function(X)


def evaluate(X, y, groups, model_name: str, *, subject_level: bool, n_splits: int = 5, seed: int = 0):
    """Out-of-fold evaluation. subject_level=True -> no patient crosses train/test (honest)."""
    Xv = X.to_numpy()
    if subject_level:
        splitter = StratifiedGroupKFold(n_splits=n_splits, shuffle=True, random_state=seed)
        folds = splitter.split(Xv, y, groups)
    else:
        splitter = StratifiedKFold(n_splits=n_splits, shuffle=True, random_state=seed)
        folds = splitter.split(Xv, y)
    oof = np.full(len(y), np.nan)
    oof_pred = np.zeros(len(y), dtype=int)
    fold_aucs = []
    for tr, te in folds:
        pipe = _pipe(model_name).fit(Xv[tr], y[tr])
        oof[te] = _scores(pipe, Xv[te])
        oof_pred[te] = pipe.predict(Xv[te])
        if len(np.unique(y[te])) == 2:
            fold_aucs.append(roc_auc_score(y[te], oof[te]))
    tn, fp, fn, tp = confusion_matrix(y, oof_pred).ravel()
    out = {
        "auc_pooled": roc_auc_score(y, oof),
        "auc_fold_mean": float(np.mean(fold_aucs)),
        "auc_fold_std": float(np.std(fold_aucs)),
        "accuracy": (tp + tn) / len(y),
        "balanced_accuracy": balanced_accuracy_score(y, oof_pred),
        "sensitivity": tp / (tp + fn),          # PD correctly flagged (recall for disease)
        "specificity": tn / (tn + fp),          # healthy correctly cleared
    }
    if subject_level:                           # the real clinical question: diagnose the PERSON
        subs = sorted(set(groups))
        s_score = np.array([oof[groups == g].mean() for g in subs])     # mean over a person's clips
        s_label = np.array([int(y[groups == g][0]) for g in subs])
        out["auc_subject"] = roc_auc_score(s_label, s_score)
        out["n_subjects"] = len(subs)
    return out


def main() -> int:
    X, y, groups, feats = load()
    n_pd = int(y.sum())
    n_subj = len(set(groups))
    print("PARKINSON'S DETECTION — UCI voice dataset (Little et al. 2007), REAL data")
    print(f"  {len(y)} recordings | {n_subj} subjects | {n_pd} PD / {len(y) - n_pd} healthy "
          f"recordings | {len(feats)} voice features\n")

    models = ["logreg", "svm_rbf", "random_forest", "hist_gb"]
    print("  HONEST subject-level CV (StratifiedGroupKFold — no patient crosses train/test):")
    print(f"    {'model':14} {'AUC':>6}  {'bal.acc':>7}  {'sens':>5}  {'spec':>5}")
    best = None
    for m in models:
        r = evaluate(X, y, groups, m, subject_level=True)
        print(f"    {m:14} {r['auc_pooled']:.3f}  {r['balanced_accuracy'] * 100:6.1f}%  "
              f"{r['sensitivity'] * 100:4.1f}%  {r['specificity'] * 100:4.1f}%")
        if best is None or r["auc_pooled"] > best[1]["auc_pooled"]:
            best = (m, r)

    print(f"\n  BEST honest model: {best[0]} — AUC {best[1]['auc_pooled']:.3f}, "
          f"balanced-acc {best[1]['balanced_accuracy'] * 100:.0f}%, "
          f"sens {best[1]['sensitivity'] * 100:.0f}%, spec {best[1]['specificity'] * 100:.0f}%")
    print(f"  per-SUBJECT diagnosis (average a person's recordings): AUC "
          f"{best[1]['auc_subject']:.3f} over {best[1]['n_subjects']} people "
          f"— the real clinical question, and the cleaner signal.")

    # The leak demonstration: same model, record-level vs subject-level.
    naive = evaluate(X, y, groups, best[0], subject_level=False)
    gap = naive["auc_pooled"] - best[1]["auc_pooled"]
    print("\n  WHY SUBJECT-LEVEL MATTERS (same model, only the split changes):")
    print(f"    record-level (leaky)  AUC {naive['auc_pooled']:.3f}   <- what naive pipelines report")
    print(f"    subject-level (honest)AUC {best[1]['auc_pooled']:.3f}   <- generalises to a NEW patient")
    print(f"    inflation from leakage: +{gap:.3f} AUC of FALSE confidence")
    print("\n  => A real, working detector on real clinical voice data — and the rigor to not")
    print("     fool itself. The honest number is the one that would hold up on new patients.")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
