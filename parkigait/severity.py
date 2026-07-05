"""Severity / PD-sign model: GaitFeatures -> SeverityEstimate.

Honest scope: with no real UPDRS-labelled data on this machine, the shipped model
is trained on the SYNTHETIC cohort (where we control the ground-truth severity).
So it is a *method demo*, and every estimate is stamped ``calibrated_on="synthetic"``.
The moment `carepd.py` has real data, `train_from_features(...)` fits the exact same
model on real labels with subject-level splits and stamps the real dataset name.

Two heads on the same 7-feature vector:
  - a classifier  -> P(PD-like motor signs)          (LogisticRegression)
  - a regressor   -> a 0..4 UPDRS-gait-*like* number  (Ridge)
Both are reported with held-out cross-validation so we never quote a train score.
"""
from __future__ import annotations

import math
from pathlib import Path
from typing import Optional

import numpy as np

from parkigait.types import GAIT_FEATURE_ORDER, GaitFeatures, SeverityEstimate

MODEL_DIR = Path(__file__).resolve().parent / "models"
MODEL_PATH = MODEL_DIR / "severity_synth.joblib"


class SeverityModel:
    """A fitted (classifier + regressor) pair over standardized gait features."""

    def __init__(self, clf, reg, mu: np.ndarray, sd: np.ndarray,
                 calibrated_on: str = "synthetic", cv_metrics: Optional[dict] = None):
        self.clf = clf
        self.reg = reg
        self.mu = np.asarray(mu, dtype=np.float64)
        self.sd = np.asarray(sd, dtype=np.float64)
        self.calibrated_on = calibrated_on
        self.cv_metrics = cv_metrics or {}

    # -- inference ---------------------------------------------------------
    def _standardize(self, x: np.ndarray) -> np.ndarray:
        return (x - self.mu) / self.sd

    def predict(self, feats: GaitFeatures) -> SeverityEstimate:
        x = feats.as_vector()
        xs = self._standardize(x).reshape(1, -1)
        p = float(self.clf.predict_proba(xs)[0, 1])
        sev = float(np.clip(self.reg.predict(xs)[0], 0.0, 4.0))
        # signed per-feature contribution to the log-odds (explainability)
        coef = self.clf.coef_.ravel()
        contrib = {name: float(coef[i] * xs[0, i])
                   for i, name in enumerate(GAIT_FEATURE_ORDER)}
        # feature confidence gates the strength of any statement we make
        if feats.confidence < 0.35:
            label = "inconclusive (low signal quality)"
        elif p < 0.5:
            label = "control-like gait (exploratory)"
        else:
            label = "possible PD motor signs (exploratory, not a diagnosis)"
        return SeverityEstimate(
            p_pd=p, severity=sev, label=label,
            calibrated_on=self.calibrated_on, contributions=contrib)

    # -- persistence -------------------------------------------------------
    def save(self, path: Path = MODEL_PATH) -> None:
        import joblib
        path.parent.mkdir(parents=True, exist_ok=True)
        joblib.dump({"clf": self.clf, "reg": self.reg, "mu": self.mu, "sd": self.sd,
                     "calibrated_on": self.calibrated_on, "cv_metrics": self.cv_metrics},
                    path)

    @classmethod
    def load(cls, path: Path = MODEL_PATH) -> "SeverityModel":
        import joblib
        d = joblib.load(path)
        return cls(d["clf"], d["reg"], d["mu"], d["sd"],
                   d.get("calibrated_on", "synthetic"), d.get("cv_metrics", {}))


# --------------------------------------------------------------------------- #
# Training                                                                     #
# --------------------------------------------------------------------------- #
def _pearson(a: np.ndarray, b: np.ndarray) -> float:
    a, b = np.asarray(a, float), np.asarray(b, float)
    if a.std() < 1e-9 or b.std() < 1e-9:
        return 0.0
    return float(np.corrcoef(a, b)[0, 1])


def train_from_features(X: np.ndarray, y_label: np.ndarray, y_sev: np.ndarray,
                        calibrated_on: str, groups: Optional[np.ndarray] = None,
                        seed: int = 0) -> tuple["SeverityModel", dict]:
    """Fit the (clf, reg) pair on a feature matrix, reporting HELD-OUT metrics.

    X:        (n, 7) features in GAIT_FEATURE_ORDER.
    y_label:  (n,) 0/1 PD-sign label.
    y_sev:    (n,) continuous severity target (0..4-ish, or 0..1 scaled).
    groups:   (n,) subject ids for subject-level CV (recommended for real data).
    Returns the model fit on ALL data plus a dict of cross-validated metrics.
    """
    from sklearn.linear_model import LogisticRegression, Ridge
    from sklearn.model_selection import GroupKFold, KFold

    X = np.asarray(X, dtype=np.float64)
    y_label = np.asarray(y_label)
    y_sev = np.asarray(y_sev, dtype=np.float64)
    mu, sd = X.mean(0), X.std(0) + 1e-9

    # cross-validated held-out metrics (subject-level if groups given)
    if groups is not None and len(np.unique(groups)) >= 3:
        splitter = GroupKFold(n_splits=min(5, len(np.unique(groups))))
        split_iter = splitter.split(X, y_label, groups)
        cv_kind = "subject-level"
    else:
        splitter = KFold(n_splits=5, shuffle=True, random_state=seed)
        split_iter = splitter.split(X)
        cv_kind = "record-level"

    aucs, sev_corrs, accs = [], [], []
    for tr, te in split_iter:
        Xtr, Xte = (X[tr] - mu) / sd, (X[te] - mu) / sd
        clf = LogisticRegression(max_iter=1000, C=1.0)
        # guard: a fold may be single-class for tiny cohorts
        if len(np.unique(y_label[tr])) < 2:
            continue
        clf.fit(Xtr, y_label[tr])
        p = clf.predict_proba(Xte)[:, 1]
        accs.append(float(((p > 0.5) == y_label[te]).mean()))
        aucs.append(_auc(y_label[te], p))
        reg = Ridge(alpha=1.0).fit(Xtr, y_sev[tr])
        s = reg.predict(Xte)
        sev_corrs.append(_pearson(s, y_sev[te]))

    cv = {
        "cv_kind": cv_kind,
        "auc_mean": float(np.mean(aucs)) if aucs else float("nan"),
        "acc_mean": float(np.mean(accs)) if accs else float("nan"),
        "severity_pearson_mean": float(np.mean(sev_corrs)) if sev_corrs else float("nan"),
        "n": int(len(X)),
        "note": ("Held-out metrics. On SYNTHETIC data these show the method works; "
                 "they are NOT clinical performance." if calibrated_on == "synthetic"
                 else "Held-out metrics on real data with subject-level splits."),
    }

    # final model on all data
    from sklearn.linear_model import LogisticRegression as LR, Ridge as RG
    Xs = (X - mu) / sd
    clf = LR(max_iter=1000, C=1.0).fit(Xs, y_label)
    reg = RG(alpha=1.0).fit(Xs, y_sev)
    model = SeverityModel(clf, reg, mu, sd, calibrated_on=calibrated_on, cv_metrics=cv)
    return model, cv


def _auc(y, p) -> float:
    y = np.asarray(y)
    p = np.asarray(p, float)
    order = np.argsort(p)
    ranks = np.empty(len(p))
    ranks[order] = np.arange(len(p))
    npos, nneg = int((y == 1).sum()), int((y == 0).sum())
    if npos == 0 or nneg == 0:
        return 0.5
    return float((ranks[y == 1].sum() - npos * (npos - 1) / 2) / (npos * nneg))


def train_synthetic(n_control: int = 60, n_pd: int = 60, seed: int = 0,
                    duration_s: float = 8.0, save: bool = True) -> tuple["SeverityModel", dict]:
    """Build the synthetic cohort, extract real gait features, fit the model.

    This is the shipped model's provenance: honest, reproducible, and clearly
    stamped synthetic. Severity target is the KNOWN synthetic severity scaled to
    a 0..4 range so the regressor's output reads like a UPDRS-gait-*like* number.
    """
    from parkigait.gaitfeat import extract_features
    from parkigait.pose import synthetic_cohort

    cohort = synthetic_cohort(n_control=n_control, n_pd=n_pd, seed=seed,
                              duration_s=duration_s)
    X, y_label, y_sev = [], [], []
    for pose, sev, label in cohort:
        feats = extract_features(pose)
        X.append(feats.as_vector())
        y_label.append(label)
        y_sev.append(sev * 4.0)  # map synthetic [0,1] severity to a 0..4-like scale
    X = np.array(X, dtype=np.float64)
    model, cv = train_from_features(np.array(X), np.array(y_label), np.array(y_sev),
                                    calibrated_on="synthetic", seed=seed)
    if save:
        model.save()
    return model, cv


def load_or_train() -> "SeverityModel":
    """Load the shipped synthetic model, training it once if missing."""
    if MODEL_PATH.exists():
        try:
            return SeverityModel.load()
        except Exception:
            pass
    model, _ = train_synthetic()
    return model


def main() -> int:
    model, cv = train_synthetic()
    print("SEVERITY MODEL — trained on SYNTHETIC cohort (method demo, not clinical)")
    print(f"  CV kind:               {cv['cv_kind']}")
    print(f"  held-out AUC:          {cv['auc_mean']:.3f}")
    print(f"  held-out accuracy:     {cv['acc_mean']:.3f}")
    print(f"  severity Pearson r:    {cv['severity_pearson_mean']:.3f}  (predicted vs synthetic-true)")
    print(f"  n:                     {cv['n']}")
    print(f"  {cv['note']}")
    print(f"  saved -> {MODEL_PATH}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
