"""Mentat — the cognitive kernel.

A domain-agnostic  propose -> verify -> remember -> reflect  loop. This is the
fusion of two engines already on this laptop:

  - swechats       the "memory organ". Lessons are learned from real outcomes
                   and firewalled against fabrication. A lesson is
                   decision-card-shaped: when / do / avoid / evidence, and may
                   not enter memory unless it is *grounded* in real evidence.
  - alpha-evolver  the "thinking organ". Brain-inspired search with cognitive
                   modes (focus / dream / recover) and an inverted-U
                   "productive surprise" signal that drives exploration while
                   quarantining results too extreme to trust.

The one rule that makes this a thinking machine and not a confident bullshitter:
the kernel never lets a candidate become memory until a domain Verifier returns
a Verdict on it. Verification is the gate. Everything downstream — opinions,
discoveries, a library of what works — is built on top of verified claims only.

Swap the proposer for an LLM reasoning core and the Problem for a domain with a
real verifier (a proof checker, a test suite, a backtest) and this same loop is
doing real work.
"""
from __future__ import annotations

import json
import math
import re
from abc import ABC, abstractmethod
from dataclasses import asdict, dataclass, field
from pathlib import Path
from typing import Any, Protocol


# --------------------------------------------------------------------------- #
# the honesty gate                                                            #
# --------------------------------------------------------------------------- #
@dataclass
class Verdict:
    """A domain verifier's judgement of a single candidate."""
    passed: bool             # did it satisfy the problem?
    score: float             # higher is better; comparable across candidates
    detail: str = ""         # human-readable evidence, e.g. "RMSE=0.013 expr=..."
    suspicious: bool = False  # degenerate / non-finite / too-good-to-trust


class Problem(ABC):
    """A problem the kernel can think about. The verifier *is* the problem."""

    name: str = "problem"
    statement: str = ""

    @abstractmethod
    def verify(self, candidate: Any) -> Verdict:
        """Check a candidate. Must be cheap, deterministic, and honest."""

    def solved(self, v: Verdict) -> bool:
        return v.passed

    def brief(self) -> str:
        """What a proposer is allowed to see about the problem — never the answer."""
        return self.statement

    def distill(self, best_candidate: Any, best_verdict: Verdict) -> list["Lesson"]:
        """Optionally turn the current best into grounded lesson(s). Default: none."""
        return []

    def behavior(self, candidate: Any) -> Any:
        """An optional low-dimensional BEHAVIOR descriptor for illumination
        (MAP-Elites). Return a hashable niche key, or None to disable the archive.

        Creativity-as-illumination: instead of one best answer, the kernel keeps the
        best solution for EACH behavior niche, filling a whole space of diverse
        verified solutions. A greedy maximizer collapses to one niche; this is what
        lets the agent be genuinely inventive across a behavior space."""
        return None

    def stress_verify(self, candidate: Any, verdict: Verdict) -> Verdict:
        """Re-check a TOO-GOOD-TO-BE-TRUE candidate under harsher conditions.

        Called only when the brain flags an extreme-surprise result (a score wildly
        better than predicted — the classic overfit/bug signature). Default trusts
        the original verdict; a domain that can stress-test (e.g. a backtest under
        higher costs) overrides this to confirm or quarantine the surprise."""
        return verdict


# --------------------------------------------------------------------------- #
# cognitive modes + productive surprise   (from alpha-evolver)                #
# --------------------------------------------------------------------------- #
def productive_surprise(error: float, scale: float) -> float:
    """Inverted-U curiosity: peaks at moderate surprise, distrusts the extremes.

    Surprising-but-learnable results earn the most signal; results that are
    wildly off the expectation earn almost none (they are usually bugs)."""
    if not math.isfinite(error):
        return 0.0
    ratio = abs(error) / max(scale, 1e-9)
    return float(ratio * math.exp(1.0 - ratio))  # maximised at ratio == 1


@dataclass
class Mind:
    """Tracks search health and chooses the next exploration mode."""
    mode: str = "focus"
    stall: int = 0
    surprise_scale: float = 1.0

    def reflect(self, improved: bool, quarantine_rate: float) -> str:
        self.stall = 0 if improved else self.stall + 1
        if quarantine_rate > 0.6:
            self.mode = "recover"   # too much junk coming back -> tighten up
        elif self.stall >= 3:
            self.mode = "dream"     # stuck -> widen the search
        else:
            self.mode = "focus"     # making progress -> exploit it
        return self.mode

    def explore_rate(self) -> float:
        return {"focus": 0.25, "dream": 0.50, "recover": 0.15}[self.mode]


@dataclass
class BrainConfig:
    """The creativity brain, ported from alpha-evolver (Codex's design), made
    domain-agnostic and ABLATABLE so we can prove each piece earns its keep.

    Default is the FULL brain. `solve(brain=None)` runs the plain kernel (every
    piece off), so existing flagships are unchanged unless they opt in."""
    novelty: bool = True            # quality-diversity elite pool: keep best AND novel
    surprise: str = "inverted_u"    # "none" | "monotonic" | "inverted_u"
    quarantine: bool = True         # stress-test too-good-to-be-true (extreme surprise)
    modes: bool = True              # dream/focus/recover steer exploration
    sleep_every: int = 6            # consolidate every N gens (0 disables)
    motifs: bool = True             # mine reusable sub-structures from elites
    novelty_weight: float = 0.30    # weight on novelty in the QD pool (alpha-evolver: 0.30)
    surprise_weight: float = 0.25   # weight on productive surprise in the QD pool
    surprise_quarantine: float = 3.0  # |error|/scale above this = extreme -> stress-test

    @classmethod
    def off(cls) -> "BrainConfig":
        """The baseline: the plain kernel as it was before the creativity engine.
        Cognitive modes stay on (the original kernel always had them); only the
        creativity ADDITIONS — novelty, surprise, quarantine, sleep, motifs — are
        off. `solve(brain=None)` uses this, so existing flagships are unchanged."""
        return cls(novelty=False, surprise="none", quarantine=False, modes=True,
                   sleep_every=0, motifs=False, novelty_weight=0.0, surprise_weight=0.0)


# --------------------------------------------------------------------------- #
# grounded memory   (from swechats)                                           #
# --------------------------------------------------------------------------- #
_STOP = {"the", "a", "an", "of", "to", "and", "or", "is", "in", "on", "for",
         "with", "that", "this", "it", "as", "be", "by", "use", "using", "when",
         "do", "avoid", "via", "are", "was", "from", "into", "than", "then"}


def keywords(text: str) -> set[str]:
    return {w for w in re.findall(r"[a-zA-Z_][a-zA-Z0-9_]+", text.lower())
            if len(w) > 2 and w not in _STOP}


# --------------------------------------------------------------------------- #
# creativity: generic novelty over arbitrary candidates  (from alpha-evolver) #
# --------------------------------------------------------------------------- #
_FRAG_CACHE: dict[str, set[str]] = {}


def fragments(candidate: Any) -> set[str]:
    """A candidate's structural fingerprint: every sub-structure + every token.

    Works for any candidate the kernel sees — nested tuples/lists (expression
    trees), strings (code/DSL), dicts (programs). This is what lets novelty be
    measured domain-agnostically, the same way alpha-evolver fingerprints
    sub-expressions to score how different a new idea is from everything tried.

    Memoized by canonical repr (fragments is pure): elites are fingerprinted over
    and over across generations, so the cache is what keeps the brain cheap enough
    to loop for hours."""
    key = repr(candidate)
    cached = _FRAG_CACHE.get(key)
    if cached is not None:
        return cached
    out: set[str] = set()

    def walk(node: Any) -> None:
        if isinstance(node, (list, tuple)):
            out.add(repr(node))                       # the whole sub-structure
            for i, x in enumerate(node):
                out.add(f"{i}:{x!r}")                 # positional fragment (Hamming-aware)
                walk(x)
        elif isinstance(node, dict):
            for k, v in sorted(node.items(), key=lambda kv: str(kv[0])):
                out.add(f"{k}={v!r}")
                walk(v)
        elif isinstance(node, str):
            out.add(node)
            for tok in re.findall(r"[A-Za-z_][A-Za-z0-9_]+", node):
                out.add(tok)
        else:
            out.add(repr(node))

    walk(candidate)
    result = out or {repr(candidate)}
    if len(_FRAG_CACHE) < 100_000:        # bounded; pure function so any entry is reusable
        _FRAG_CACHE[key] = result
    return result


def novelty(candidate: Any, others: list) -> float:
    """1 - (max Jaccard similarity of this candidate's fingerprint to any other).

    1.0 = unlike anything seen; 0.0 = a duplicate. This is the curiosity signal
    that rewards genuinely new ideas, not just high-scoring ones."""
    fa = fragments(candidate)
    if not others or not fa:
        return 1.0
    best = 0.0
    for o in others:
        fb = fragments(o)
        if not fb:
            continue
        j = len(fa & fb) / len(fa | fb)
        if j > best:
            best = j
    return 1.0 - best


@dataclass
class Lesson:
    """A decision-card-shaped, grounded unit of memory."""
    when: str                 # the recurring situation
    do: str                   # the canonical move
    avoid: str = ""           # the anti-pattern
    evidence: str = ""        # a real value/quote from a *verified* candidate
    strength: float = 1.0
    corroboration: int = 1
    status: str = "trusted"   # or "quarantined"

    def grounded(self, min_overlap: int = 2) -> bool:
        """A lesson may only enter memory if its claim shares real vocabulary
        with the evidence it cites. This is the anti-fabrication firewall."""
        claim = f"{self.when} {self.do} {self.avoid}"
        return len(keywords(claim) & keywords(self.evidence)) >= min_overlap


@dataclass
class Memory:
    lessons: list[Lesson] = field(default_factory=list)
    motifs: dict[str, float] = field(default_factory=dict)  # fragment -> frequency weight
    best_candidate: Any = None
    best_score: float = -math.inf
    cap: int = 200
    elites: list = field(default_factory=list)   # [score, candidate], desc by score
    elite_cap: int = 12
    principles: list = field(default_factory=list)  # distilled-at-sleep durable lessons
    archive: dict = field(default_factory=dict)  # MAP-Elites: behavior niche -> [score, candidate]
    # creativity (set from BrainConfig by solve(); off => identical to the plain pool)
    qd: bool = False                 # quality-diversity elite pool (keep best AND novel)
    novelty_weight: float = 0.0

    # -- persistence --------------------------------------------------------
    def save(self, path: str | Path) -> None:
        Path(path).write_text(json.dumps({
            "lessons": [asdict(le) for le in self.lessons],
            "motifs": self.motifs,
            "best_candidate": self.best_candidate,
            "best_score": self.best_score if math.isfinite(self.best_score) else None,
            "elites": self.elites,
            "principles": self.principles,
            # JSON keys must be strings; behaviors round-trip as their str form.
            "archive": [[str(b), s, c] for b, (s, c) in self.archive.items()],
        }, indent=2))

    @classmethod
    def load(cls, path: str | Path) -> "Memory":
        p = Path(path)
        if not p.exists():
            return cls()
        d = json.loads(p.read_text())
        bs = d.get("best_score")
        return cls(
            lessons=[Lesson(**le) for le in d.get("lessons", [])],
            motifs=d.get("motifs", {}),
            best_candidate=d.get("best_candidate"),
            best_score=bs if bs is not None else -math.inf,
            elites=d.get("elites", []),
            principles=d.get("principles", []),
            archive={b: [s, c] for b, s, c in d.get("archive", [])},
        )

    def consider_elite(self, score: float, candidate: Any) -> None:
        """Keep a small pool of DISTINCT best-verified candidates for recombination.
        Without the dedup the loop re-proposes its own best every generation and the
        whole pool collapses to copies of one candidate, killing recombination."""
        if not math.isfinite(score):                  # never pool a NaN/inf score
            return
        for i, (s, c) in enumerate(self.elites):
            if c == candidate:                        # value equality (candidates may be unhashable)
                if score <= s:
                    return                            # already hold an equal-or-better copy
                del self.elites[i]
                break
        self.elites.append([score, candidate])
        if self.qd and len(self.elites) > self.elite_cap:
            self._trim_quality_diversity()
        else:
            self.elites.sort(key=lambda sc: sc[0], reverse=True)
            del self.elites[self.elite_cap:]

    def _trim_quality_diversity(self) -> None:
        """Trim the pool by value = normalized score + novelty_weight * novelty, so
        it keeps the best AND the most novel (quality-diversity / novelty search).

        A plain top-by-score pool collapses to near-duplicate variants of one
        lineage; reserving room for novel-but-good candidates is what keeps the
        search creative — diverse parents breed ideas a greedy pool never reaches."""
        while len(self.elites) > self.elite_cap:
            scores = [s for s, _ in self.elites]
            lo, hi = min(scores), max(scores)
            span = (hi - lo) or 1.0

            def value(i: int) -> float:
                s, c = self.elites[i]
                others = [cc for j, (_, cc) in enumerate(self.elites) if j != i]
                return (s - lo) / span + self.novelty_weight * novelty(c, others)

            worst = min(range(len(self.elites)), key=value)
            del self.elites[worst]
        self.elites.sort(key=lambda sc: sc[0], reverse=True)

    def consider_archive(self, score: float, candidate: Any, behavior: Any) -> None:
        """MAP-Elites illumination: keep the best candidate for each behavior niche.
        Filling many niches is creativity — a diverse space of verified solutions,
        not one optimum."""
        if behavior is None or not math.isfinite(score):
            return
        cur = self.archive.get(behavior)
        if cur is None or score > cur[0]:
            self.archive[behavior] = [score, candidate]

    def archive_coverage(self) -> int:
        """How many behavior niches are filled — the illumination metric."""
        return len(self.archive)

    def mine_motifs(self, top_n: int = 12) -> None:
        """Refresh the motif library: the sub-structures that recur across elites,
        ranked by frequency. These are the agent's own reusable building blocks."""
        counts: dict[str, float] = {}
        for _, cand in self.elites:
            for frag in fragments(cand):
                counts[frag] = counts.get(frag, 0.0) + 1.0
        ranked = sorted(counts.items(), key=lambda kv: (-kv[1], kv[0]))[:top_n]
        self.motifs = dict(ranked)

    def pool_diversity(self) -> float:
        """Mean pairwise novelty within the elite pool (0 = clones, 1 = all distinct).
        A creativity health metric: a healthy creative search keeps this high."""
        cands = [c for _, c in self.elites]
        if len(cands) < 2:
            return 1.0
        total, n = 0.0, 0
        for i, c in enumerate(cands):
            total += novelty(c, cands[:i] + cands[i + 1:])
            n += 1
        return total / n

    # -- write path (firewalled: only grounded lessons enter) ---------------
    def learn(self, lesson: Lesson) -> bool:
        if not lesson.grounded():
            return False
        for ex in self.lessons:
            if ex.when == lesson.when and ex.do == lesson.do:
                ex.corroboration += 1
                ex.strength = min(4.0, ex.strength + 0.5)  # corroboration strengthens
                return False
        self.lessons.append(lesson)
        return True

    def reinforce_motif(self, fragment: str, score: float) -> None:
        self.motifs[fragment] = max(self.motifs.get(fragment, -math.inf), score)

    def decay(self, factor: float = 0.98) -> None:
        for le in self.lessons:
            le.strength *= factor
        self.lessons = [le for le in self.lessons if le.strength > 0.15]
        if len(self.lessons) > self.cap:
            self.lessons.sort(key=lambda le: le.strength, reverse=True)
            del self.lessons[self.cap:]

    def context(self, k: int = 8) -> str:
        top = sorted(self.lessons, key=lambda le: le.strength, reverse=True)[:k]
        return "\n".join(
            f"- WHEN {le.when}: DO {le.do}" + (f"; AVOID {le.avoid}" if le.avoid else "")
            for le in top
        )


# --------------------------------------------------------------------------- #
# the loop                                                                    #
# --------------------------------------------------------------------------- #
class Proposer(Protocol):
    def propose(self, problem: Problem, memory: Memory, mind: Mind, k: int) -> list[Any]:
        ...


@dataclass
class Result:
    solved: bool
    best_candidate: Any
    best_score: float
    verdict: Verdict
    generations: int
    history: list[dict]
    distinct_verified: int = 0   # creativity: # of DISTINCT candidates that passed the gate
    diversity: float = 1.0       # creativity: mean pairwise novelty of the final elite pool
    coverage: int = 0            # creativity: # of behavior niches illuminated (MAP-Elites)


def _predict(memory: Memory, best_score: float) -> float:
    """Expected score of the next idea = mean of the current elites (alpha-evolver's
    _predict_sharpe). Surprise is measured against this, so it tracks generalisation,
    not just raw quality."""
    vals = [s for s, _ in memory.elites if math.isfinite(s)]
    if vals:
        return sum(vals) / len(vals)
    return best_score if math.isfinite(best_score) else 0.0


def _surprise_signal(error: float, scale: float, mode: str) -> float:
    if mode == "none":
        return 0.0
    if mode == "monotonic":              # Surprise-Search style: linear in |error|
        return min(abs(error) / max(scale, 1e-9), 1.0)
    return productive_surprise(error, scale)   # inverted_u: reward moderate, distrust extreme


def solve(problem: Problem, proposer: Proposer, memory: Memory,
          *, generations: int = 40, k: int = 24, log=print,
          brain: "BrainConfig | None" = None) -> Result:
    """Run propose -> verify -> remember -> reflect until solved or out of budget.

    `memory` is mutated in place and carries state across runs when persisted —
    a warm (loaded) memory is the difference between thinking from scratch and
    thinking with everything you learned last time still in hand.

    `brain` turns on the creativity engine (ported from alpha-evolver): a
    quality-diversity elite pool, predict-then-test productive surprise, an
    extreme-surprise quarantine, cognitive modes, and a sleep/consolidation pass.
    `brain=None` runs the plain kernel, so existing flagships are unchanged.
    """
    brain = brain or BrainConfig.off()
    memory.qd = brain.novelty
    memory.novelty_weight = brain.novelty_weight
    mind = Mind()
    best_cand = memory.best_candidate
    best_score = memory.best_score
    best_v: Verdict | None = None
    # Recall + RE-VERIFY: a remembered best is re-run through the gate, never
    # trusted blindly. This is what lets a warm start recognise it already holds
    # a solution (and what catches a remembered claim that no longer verifies).
    if best_cand is not None:
        try:
            best_v = problem.verify(best_cand)
            best_score = best_v.score
        except Exception as e:               # a poisoned/stale warm start must not abort
            log(f"  warm-start verify raised, discarding remembered best: {type(e).__name__}: {e}")
            best_cand, best_v, best_score = None, None, -math.inf
    history: list[dict] = []
    pe_scale = 1.0                           # EWMA of |prediction error| -> the surprise scale
    verified_sigs: set[str] = set()          # distinct ideas that passed the gate (creativity)

    for gen in range(1, generations + 1):
        candidates = proposer.propose(problem, memory, mind, k)
        improved = False
        quarantined = 0
        surprises: list[float] = []
        pred = _predict(memory, best_score)  # predict BEFORE testing this generation

        for cand in candidates:
            try:
                v = problem.verify(cand)         # <- the gate: nothing skips it
            except Exception:                    # a raising verifier == junk, not fatal
                quarantined += 1
                continue
            if v.suspicious:
                quarantined += 1
                continue
            error = v.score - pred
            ratio = abs(error) / max(pe_scale, 1e-9)
            pe_scale = 0.9 * pe_scale + 0.1 * max(abs(error), 1e-6)
            # Extreme-surprise quarantine: a NEW BEST that is wildly better than
            # predicted is the classic overfit/bug signature. Stress-test it before
            # trusting it (the safety half of productive surprise). Default trusts.
            if (brain.quarantine and v.score > best_score
                    and ratio > brain.surprise_quarantine):
                stress = problem.stress_verify(cand, v)
                if (not stress.passed) or stress.suspicious:
                    quarantined += 1
                    continue
                v = stress
            surprises.append(_surprise_signal(error, pe_scale, brain.surprise))
            memory.consider_elite(v.score, cand)
            # Always record best-per-niche (no-op unless the Problem defines behavior),
            # so coverage is measured the same way whatever the proposer does.
            memory.consider_archive(v.score, cand, problem.behavior(cand))
            if v.passed:
                verified_sigs.add(repr(cand))
            if v.score > best_score:
                best_score, best_cand, best_v = v.score, cand, v
                improved = True

        qrate = quarantined / max(1, len(candidates))
        mind.surprise_scale = 0.8 * mind.surprise_scale + 0.2 * max(1e-6, abs(best_score))
        mode = mind.reflect(improved, qrate)
        if not brain.modes:                  # ablate cognitive modes -> always focus
            mind.mode = mode = "focus"

        if best_v is not None:
            for lesson in problem.distill(best_cand, best_v):
                memory.learn(lesson)
        # Sleep / consolidation: refresh motifs and distill durable principles from
        # the TOP DISTINCT elites (not just the single best), the way alpha-evolver
        # consolidates during sleep.
        if brain.sleep_every and gen % brain.sleep_every == 0:
            if brain.motifs:
                memory.mine_motifs()
            for _, c in memory.elites[:3]:
                try:
                    cv = problem.verify(c)
                except Exception:
                    continue
                for lesson in problem.distill(c, cv):
                    if memory.learn(lesson):
                        memory.principles.append({"when": lesson.when, "do": lesson.do})
        memory.best_candidate, memory.best_score = best_cand, best_score
        memory.decay()

        mean_surprise = sum(surprises) / len(surprises) if surprises else 0.0
        history.append({"gen": gen, "mode": mode, "best": best_score,
                        "q": round(qrate, 2), "surprise": round(mean_surprise, 3),
                        "diversity": round(memory.pool_diversity(), 3),
                        "distinct": len(verified_sigs),
                        "coverage": memory.archive_coverage()})
        log(f"gen {gen:>3} | {mode:7} | best={best_score:+.4f} "
            f"| quarantine={qrate:.2f} | surprise={mean_surprise:.2f}")

        if best_v is not None and problem.solved(best_v):
            log(f"  -> solved at gen {gen}: {best_v.detail}")
            return Result(True, best_cand, best_score, best_v, gen, history,
                          distinct_verified=len(verified_sigs),
                          diversity=memory.pool_diversity(),
                          coverage=memory.archive_coverage())

    solved = best_v is not None and problem.solved(best_v)
    return Result(solved, best_cand, best_score,
                  best_v or Verdict(False, best_score, "no candidate verified"),
                  generations, history,
                  distinct_verified=len(verified_sigs),
                  diversity=memory.pool_diversity(),
                  coverage=memory.archive_coverage())
