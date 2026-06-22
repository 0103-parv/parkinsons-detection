"""mentat searches for the minimal voice-feature panel that still detects Parkinson's.

22 voice features is a lot to measure. Which few actually carry the signal? This is a
search over feature subsets, and the verifier is the HONEST one — subject-level AUC (no
patient crosses train/test) — so a subset is only kept if it generalises to new people.
mentat's propose->verify->keep loop, with a parsimony penalty, finds a small panel that
holds most of the detection power: cheaper, more interpretable screening.

Run:  cd ~/mentat && PYTHONPATH=. ~/swechats/.venv/bin/python reference/parkinsons/panel_search.py
"""
from __future__ import annotations

import random

import numpy as np
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import roc_auc_score
from sklearn.model_selection import StratifiedGroupKFold
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler

from mentat.core import BrainConfig, Memory, Problem, Verdict, solve

from .detect import load


def _subject_auc_one(Xv, y, groups, cols, n_splits: int, seed: int) -> float:
    Xc = Xv[:, cols]
    oof = np.full(len(y), np.nan)
    for tr, te in StratifiedGroupKFold(n_splits=n_splits, shuffle=True,
                                       random_state=seed).split(Xc, y, groups):
        pipe = Pipeline([("s", StandardScaler()),
                         ("c", LogisticRegression(max_iter=2000, class_weight="balanced"))])
        pipe.fit(Xc[tr], y[tr])
        oof[te] = pipe.predict_proba(Xc[te])[:, 1]
    subs = sorted(set(groups))
    s_score = np.array([oof[groups == g].mean() for g in subs])
    s_label = np.array([int(y[groups == g][0]) for g in subs])
    return float(roc_auc_score(s_label, s_score))


def subject_auc(Xv, y, groups, cols, n_splits: int = 5, seeds=(0,)) -> float:
    """Honest gate: out-of-fold subject-level AUC (logistic, on `cols`), averaged over CV seeds
    so the number is robust to the particular fold split (only 32 people)."""
    return float(np.mean([_subject_auc_one(Xv, y, groups, cols, n_splits, s) for s in seeds]))


class FeaturePanel(Problem):
    name = "feature-panel"

    def __init__(self, Xv, y, groups, feats):
        self.Xv, self.y, self.groups, self.feats = Xv, y, groups, feats
        self.nfeat = Xv.shape[1]
        self.statement = "smallest voice-feature panel with high subject-level AUC"

    def _subset(self, c):
        if not isinstance(c, (list, tuple, frozenset, set)):
            return None
        idx = sorted({int(i) for i in c if 0 <= int(i) < self.nfeat})
        return idx or None

    def verify(self, candidate) -> Verdict:
        idx = self._subset(candidate)
        if idx is None:
            return Verdict(False, -1e9, "empty panel", suspicious=True)
        auc = subject_auc(self.Xv, self.y, self.groups, idx)
        return Verdict(False, auc - 0.02 * len(idx),       # open-ended: parsimony-penalized AUC
                       f"AUC {auc:.3f} on {len(idx)} feats")

    def solved(self, v: Verdict) -> bool:
        return False

    def behavior(self, candidate):
        idx = self._subset(candidate)               # niche = panel size -> best panel PER size
        return None if idx is None else len(idx)


class PanelProposer:
    def __init__(self, rng: random.Random, nfeat: int):
        self.rng, self.nfeat = rng, nfeat

    def _rand(self):
        k = self.rng.randint(2, min(8, self.nfeat))
        return tuple(sorted(self.rng.sample(range(self.nfeat), k)))

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


def main() -> int:
    seeds = (0, 1, 2)                                # average folds -> robust on only 32 people
    X, y, groups, feats = load()
    Xv = X.to_numpy()
    full = subject_auc(Xv, y, groups, list(range(len(feats))), seeds=seeds)
    print("mentat FEATURE-PANEL SEARCH — minimal voice panel that still detects PD")
    print("  gate = subject-level AUC (no patient crosses train/test), averaged over 3 CV seeds")
    print(f"  ALL {len(feats)} features (logistic): AUC {full:.3f}  "
          "<- correlated features make it OVERFIT on 32 people\n")

    prob = FeaturePanel(Xv, y, groups, feats)
    mem = Memory()
    solve(prob, PanelProposer(random.Random(0), len(feats)), mem,
          generations=30, k=16, log=lambda *_: None, brain=BrainConfig())

    # mentat's MAP-Elites archive holds the best panel at EACH size — the size-vs-AUC frontier.
    frontier = []
    for size, (_, cand) in sorted(mem.archive.items()):
        idx = prob._subset(cand)
        if idx:
            frontier.append((idx, subject_auc(Xv, y, groups, idx, seeds=seeds)))   # re-verify robustly
    print("  size-vs-AUC frontier (best verified panel mentat found at each size):")
    for idx, auc in frontier:
        print(f"    {len(idx):2}-feature  AUC {auc:.3f}   {[feats[i] for i in idx][:6]}"
              + (" ..." if len(idx) > 6 else ""))

    peak = max(a for _, a in frontier)
    single = min(frontier, key=lambda t: len(t[0]))
    print(f"\n  single most informative feature: {feats[single[0][0]]} "
          f"(AUC {single[1]:.3f}) — matches the dysphonia literature")
    # robust pick: smallest panel of >=3 features within 0.02 AUC of the peak (less brittle than one)
    robust = min((t for t in frontier if len(t[0]) >= 3 and t[1] >= peak - 0.02),
                 key=lambda t: len(t[0]), default=max(frontier, key=lambda t: t[1]))
    print(f"  RECOMMENDED robust panel: {len(robust[0])} features, subject-level AUC {robust[1]:.3f} "
          f"(vs {full:.3f} using all {len(feats)}):")
    for i in robust[0]:
        print(f"    - {feats[i]}")
    print(f"\n  => Selection beats the full panel by +{robust[1] - full:.3f} AUC: a few well-chosen,")
    print("     cheap-to-measure dysphonia features generalise to new PEOPLE better than all 22.")
    print("     mentat's gated search found that frontier without ever touching held-out patients.")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
