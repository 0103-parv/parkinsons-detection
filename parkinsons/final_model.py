"""final_model.py — the deployable Parkinson's detector: train, persist, predict.

Trains the mentat-selected voice panel on ALL the data, saves `model.joblib`, and exposes
`predict()` for a new recording. The honest expected performance (subject-level cross-validated
AUC — i.e. how it should do on a NEW patient) is printed and stored alongside the model, so the
artifact never travels without its true accuracy.

Run:  cd ~/mentat && PYTHONPATH=. ~/swechats/.venv/bin/python -m reference.parkinsons.final_model
Use:  from reference.parkinsons.final_model import predict; predict(recording_dict)
"""
from __future__ import annotations

from pathlib import Path

import joblib
import numpy as np
from sklearn.linear_model import LogisticRegression
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler

from .detect import load
from .panel_search import subject_auc

# Robust 3-feature panel discovered by panel_search (beats all 22 features on held-out PEOPLE).
PANEL = ["spread1", "MDVP:Fhi(Hz)", "D2"]
MODEL_PATH = Path(__file__).parent / "model.joblib"


def build_pipeline() -> Pipeline:
    return Pipeline([("scale", StandardScaler()),
                     ("clf", LogisticRegression(max_iter=5000, class_weight="balanced"))])


def train_and_save() -> float:
    """Fit on ALL data for deployment; return the HONEST subject-level CV AUC (expected on new patients)."""
    X, y, groups, feats = load()
    cols = [feats.index(p) for p in PANEL]
    auc = subject_auc(X.to_numpy(), y, groups, cols, seeds=(0, 1, 2))
    pipe = build_pipeline().fit(X[PANEL].to_numpy(), y)
    joblib.dump({"pipeline": pipe, "panel": PANEL, "subject_auc": auc}, MODEL_PATH)
    return auc


def predict(recording: dict) -> dict:
    """recording: dict of feature -> value (must contain the PANEL features). Returns prob + label."""
    if not MODEL_PATH.exists():
        train_and_save()
    blob = joblib.load(MODEL_PATH)
    x = np.array([[float(recording[f]) for f in blob["panel"]]])
    prob = float(blob["pipeline"].predict_proba(x)[0, 1])
    return {"p_parkinsons": prob,
            "prediction": "Parkinson's" if prob >= 0.5 else "healthy",
            "expected_subject_auc": blob["subject_auc"]}


def main() -> int:
    auc = train_and_save()
    print("DEPLOYABLE PARKINSON'S DETECTOR — trained on all data, saved to disk")
    print(f"  saved          : {MODEL_PATH.name}")
    print(f"  panel          : {PANEL}")
    print(f"  expected AUC   : {auc:.3f}  (honest subject-level CV — how it should do on a NEW patient)\n")
    X, y, groups, feats = load()
    print("  demo predictions on real recordings (illustrative — these were in training):")
    for i in (0, 30, 150, 194):
        r = predict(X.iloc[i].to_dict())
        truth = "Parkinson's" if y[i] else "healthy"
        mark = "OK " if (r["prediction"] == truth) else "MISS"
        print(f"    rec {i:3}: P(PD)={r['p_parkinsons']:.2f} -> {r['prediction']:11}  (true {truth:11}) {mark}")
    print("\n  => A real, saved, callable model. predict(recording_dict) -> P(Parkinson's).")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
