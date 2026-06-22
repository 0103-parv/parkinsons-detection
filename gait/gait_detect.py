"""A WORKING gait-based Parkinson's detector, plus mentat searching for the minimal panel.

HONEST SCOPE: this is NOT the ISEF edge-VLM submission. With no GPU, no CARE-PD access, and
no API, the edge vision-language model can't be trained here. What CAN be built — and is, and
runs — is the *detection task itself*: a classifier over the gait features clinicians actually
use, trained and judged under mentat's anti-overfit verification gate (held-out k-fold CV),
on SYNTHETIC data with literature-based, deliberately-overlapping PD signatures.

Two pieces:
  1. DETECTOR — logistic regression over 7 gait features; reported on held-out folds only.
  2. mentat FEATURE SEARCH — mentat's propose->verify->keep loop searches feature subsets,
     gated by held-out AUC with a parsimony penalty, to find the SMALLEST panel that still
     detects. Fewer sensors/markers = cheaper on-device — your edge-deployment thesis.

Swap real gait features (e.g. from CARE-PD pose/SMPL meshes) in for `synthesize()` and the
same gated pipeline is a real detector. The VLM/edge-quantization layer still needs a GPU +
CARE-PD; see gait_quant_policy.py for mentat's role there.

Run:  cd ~/mentat && ~/swechats/.venv/bin/python reference/parkinsons_detect.py   (numpy + mentat)
"""
from __future__ import annotations

import random

import numpy as np

from mentat.core import BrainConfig, Memory, Problem, Verdict, solve

FEATURES = ["gait_speed", "cadence", "stride_length", "stride_time_var",
            "asymmetry", "arm_swing", "fog_index"]


def synthesize(n: int = 600, seed: int = 0):
    """Controls vs PD, PD severity scaling the deficits — with realistic overlap and ~12%
    atypical subjects (comorbidity/medication), so the task is honestly hard, not separable."""
    rng = np.random.default_rng(seed)
    nc, npd = n // 2, n - n // 2
    Xc = np.column_stack([
        rng.normal(1.30, 0.18, nc), rng.normal(115, 12, nc), rng.normal(1.40, 0.18, nc),
        rng.normal(0.030, 0.015, nc), rng.normal(0.040, 0.030, nc), rng.normal(0.90, 0.16, nc),
        rng.normal(0.050, 0.060, nc)])
    sev = rng.uniform(0.15, 1.0, npd)                          # PD severity (UPDRS-like)
    Xp = np.column_stack([
        1.30 - 0.35 * sev + rng.normal(0, 0.20, npd),         # gait speed down
        115 - 15 * sev + rng.normal(0, 12, npd),              # cadence down
        1.40 - 0.35 * sev + rng.normal(0, 0.20, npd),         # stride length down
        0.030 + 0.035 * sev + rng.normal(0, 0.016, npd),      # stride-time variability up
        0.040 + 0.090 * sev + rng.normal(0, 0.035, npd),      # asymmetry up
        0.90 - 0.25 * sev + rng.normal(0, 0.16, npd),         # arm swing down
        0.050 + 0.30 * np.maximum(0, sev - 0.4)               # freezing: only moderate+ PD,
        + rng.normal(0, 0.060, npd)])                         # intermittent -> never a lone tell
    X = np.vstack([Xc, Xp])
    y = np.concatenate([np.zeros(nc), np.ones(npd)])
    severity = np.concatenate([np.zeros(nc), sev])
    aty = rng.choice(n, int(0.12 * n), replace=False)         # atypical: blend toward the wrong class
    X[aty] = 0.5 * X[aty] + 0.5 * X[rng.permutation(n)][aty]
    idx = rng.permutation(n)
    return X[idx], y[idx], severity[idx]


def _train_logreg(X, y, epochs: int = 400, lr: float = 0.2):
    w, b = np.zeros(X.shape[1]), 0.0
    for _ in range(epochs):
        p = 1.0 / (1.0 + np.exp(-(X @ w + b)))
        g = p - y
        w -= lr * (X.T @ g / len(y))
        b -= lr * g.mean()
    return w, b


def _auc(y, p) -> float:
    order = np.argsort(p)
    ranks = np.empty(len(p))
    ranks[order] = np.arange(len(p))
    npos, nneg = (y == 1).sum(), (y == 0).sum()
    if npos == 0 or nneg == 0:
        return 0.5
    return float((ranks[y == 1].sum() - npos * (npos - 1) / 2) / (npos * nneg))


def cross_validate(X, y, severity, k: int = 5, seed: int = 0, epochs: int = 400):
    """mentat's gate, applied: judge ONLY on held-out folds — believe it detects only if it
    generalizes (no overfit). Returns mean held-out accuracy, AUC, and severity correlation."""
    if X.ndim == 1:
        X = X[:, None]
    rng = np.random.default_rng(seed)
    folds = np.array_split(rng.permutation(len(y)), k)
    accs, aucs, corrs = [], [], []
    for i in range(k):
        te = folds[i]
        tr = np.concatenate([folds[j] for j in range(k) if j != i])
        mu, sd = X[tr].mean(0), X[tr].std(0) + 1e-9
        Xtr, Xte = (X[tr] - mu) / sd, (X[te] - mu) / sd
        w, b = _train_logreg(Xtr, y[tr], epochs=epochs)
        p = 1.0 / (1.0 + np.exp(-(Xte @ w + b)))
        accs.append(float(((p > 0.5) == y[te]).mean()))
        aucs.append(_auc(y[te], p))
        corrs.append(float(np.corrcoef(p, severity[te])[0, 1]))
    return float(np.mean(accs)), float(np.mean(aucs)), float(np.mean(corrs))


# ----- mentat searches for the smallest gait-feature panel that still detects -----------

class GaitPanel(Problem):
    """Candidate = a subset of feature indices. The verifier IS the detector's held-out AUC,
    so nothing is believed unless it generalizes; score rewards detection and penalizes size."""
    name = "gait-panel"

    def __init__(self, X, y, sev, bar: float = 0.88):
        self.X, self.y, self.sev, self.bar = X, y, sev, bar
        self.statement = f"smallest gait-feature panel with held-out AUC >= {bar}"

    def _subset(self, c):
        if not isinstance(c, (list, tuple, frozenset, set)):
            return None
        idx = sorted({int(i) for i in c if 0 <= int(i) < self.X.shape[1]})
        return idx or None

    def verify(self, candidate) -> Verdict:
        idx = self._subset(candidate)
        if idx is None:
            return Verdict(False, -1e9, "empty/malformed panel", suspicious=True)
        _, auc, _ = cross_validate(self.X[:, idx], self.y, self.sev, k=3, epochs=250)
        names = [FEATURES[i] for i in idx]
        return Verdict(False, auc - 0.05 * len(idx),          # open-ended: parsimony-penalized AUC
                       f"AUC {auc:.3f} on {len(idx)} feats {names}")

    def solved(self, v: Verdict) -> bool:
        return False                                          # illumination: run the full budget


class PanelProposer:
    def __init__(self, rng: random.Random, nfeat: int):
        self.rng, self.nfeat = rng, nfeat

    def _rand(self):
        return tuple(sorted(self.rng.sample(range(self.nfeat), self.rng.randint(2, self.nfeat))))

    def _mutate(self, c):
        s = set(c)
        r = self.rng.random()
        if r < 0.4 and len(s) < self.nfeat:
            s.add(self.rng.randrange(self.nfeat))
        elif r < 0.8 and len(s) > 1:
            s.discard(self.rng.choice(sorted(s)))
        else:
            if s:
                s.discard(self.rng.choice(sorted(s)))
            s.add(self.rng.randrange(self.nfeat))
        return tuple(sorted(s)) if s else self._rand()

    def propose(self, problem, memory: Memory, mind, k: int):
        ex = mind.explore_rate()
        pool = [c for _, c in memory.elites]
        return [self._rand() if not pool or self.rng.random() < ex
                else self._mutate(self.rng.choice(pool)) for _ in range(k)]


def search_minimal_panel(X, y, sev, bar: float = 0.88, seed: int = 0):
    """Run mentat's gated loop; return the smallest panel whose held-out AUC clears the bar."""
    prob = GaitPanel(X, y, sev, bar=bar)
    mem = Memory()
    solve(prob, PanelProposer(random.Random(seed), X.shape[1]), mem,
          generations=25, k=16, log=lambda *_: None, brain=BrainConfig())
    scored = []
    for _, cand in {tuple(c): (s, c) for s, c in mem.elites}.values():
        idx = prob._subset(cand)
        if idx is None:
            continue
        _, auc, _ = cross_validate(X[:, idx], y, sev, k=5)    # re-verify the finalists honestly
        scored.append((idx, auc))
    passing = [(idx, auc) for idx, auc in scored if auc >= bar]
    if passing:
        return min(passing, key=lambda t: (len(t[0]), -t[1]))  # fewest features, then best AUC
    return max(scored, key=lambda t: t[1]) if scored else (list(range(X.shape[1])), 0.5)


def main() -> int:
    X, y, severity = synthesize(600)
    print("PARKINSON'S GAIT DETECTION  (synthetic, honestly-overlapping; mentat-gated CV)")
    print(f"  features: {', '.join(FEATURES)}\n")

    acc, auc, corr = cross_validate(X, y, severity, k=5)
    print("  [1] full 7-feature detector — held-out 5-fold CV")
    print(f"      accuracy {acc * 100:.1f}%  (chance 50%)   AUC {auc:.3f}   severity r {corr:.2f}")
    verdict = "DETECTS — generalizes, no overfit" if auc > 0.85 else "weak / not credible"
    print(f"      VERDICT (gate, held-out only): {verdict}\n")

    idx, pauc = search_minimal_panel(X, y, severity, bar=0.88)
    print("  [2] mentat searched feature subsets (propose->verify->keep, AUC-gated + parsimony)")
    print(f"      smallest panel that still detects: {len(idx)} of 7 features")
    print(f"        {[FEATURES[i] for i in idx]}")
    print(f"      held-out AUC on that panel: {pauc:.3f}")
    print("      => fewer markers/sensors at the same detection power = cheaper on-device.\n")

    print("  HONEST: synthetic gait features, not the CARE-PD edge-VLM. The detector + gate are")
    print("  real and reusable on real features; the VLM/edge-quantization layer still needs a")
    print("  GPU + CARE-PD (see gait_quant_policy.py for mentat's role there).")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
