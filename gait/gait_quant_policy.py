"""mentat searches a mixed-precision quantization policy — the LieQ idea, verification-gated.

The ISEF project's edge angle: a vision-language model can't run on a phone at FP16, so you
quantize it — but quantizing the wrong layers wrecks accuracy. LieQ's insight is *circuit-aware*
mixed precision: spend bits where the model is sensitive (attention / output head), starve the
redundant layers (MLP). That choice is a SEARCH over per-layer bit-widths under two hard
constraints, which is exactly mentat's propose->verify->keep loop.

Here mentat searches the bit assignment; the verifier checks BOTH a memory budget and a
simulated accuracy floor. The accuracy model is SYNTHETIC (a per-layer sensitivity * bit-loss
table) — in the real project you swap in the actual VLM evaluated on CARE-PD, and the same
search runs. "Verified" means it provably met both constraints under the verifier it was given.

Run:  cd ~/mentat && PYTHONPATH=. ~/swechats/.venv/bin/python reference/gait_quant_policy.py
"""
from __future__ import annotations

import random

from mentat.core import BrainConfig, Memory, Problem, Verdict, solve

# (name, fraction-of-params, sensitivity 0..1) — attention/head matter, MLP is redundant.
LAYERS = [("embed", 0.10, 0.50), ("attn_1", 0.12, 0.90), ("attn_2", 0.12, 0.85),
          ("mlp_1", 0.28, 0.20), ("mlp_2", 0.28, 0.15), ("head", 0.10, 0.70)]
BITS = [2, 3, 4, 8]
LOSS = {16: 0.0, 8: 0.01, 4: 0.05, 3: 0.10, 2: 0.25}      # accuracy lost per layer at each width
MEM_BUDGET = 0.32                                          # <= 32% of FP16 memory
ACC_FLOOR = 0.88                                           # >= 88% of FP16 accuracy


def memory_fraction(bits):
    return sum(f * b for (_, f, _), b in zip(LAYERS, bits)) / (sum(f for _, f, _ in LAYERS) * 16)


def accuracy(bits):
    return 1.0 - sum(s * LOSS[b] for (_, _, s), b in zip(LAYERS, bits))


class QuantPolicy(Problem):
    name = "quant-policy"
    statement = f"mixed-precision policy with memory <= {MEM_BUDGET} of FP16 AND accuracy >= {ACC_FLOOR}"

    def _bits(self, c):
        if not (isinstance(c, (list, tuple)) and len(c) == len(LAYERS)):
            return None
        return [b if b in BITS else None for b in c] if all(b in BITS for b in c) else None

    def verify(self, candidate) -> Verdict:
        bits = self._bits(candidate)
        if bits is None:
            return Verdict(False, -1e9, "malformed policy", suspicious=True)
        mem, acc = memory_fraction(bits), accuracy(bits)
        ok = mem <= MEM_BUDGET and acc >= ACC_FLOOR
        # score: maximize accuracy, hard-penalize over-budget so the search obeys the constraint
        score = acc - 5.0 * max(0.0, mem - MEM_BUDGET)
        detail = (f"mem {mem * 100:.0f}% of FP16, acc {acc * 100:.1f}%  "
                  + " ".join(f"{n}:{b}b" for (n, _, _), b in zip(LAYERS, bits)))
        return Verdict(ok, score, detail)

    def solved(self, v: Verdict) -> bool:
        return v.passed


class PolicyProposer:
    def __init__(self, rng: random.Random):
        self.rng = rng

    def _rand(self):
        return tuple(self.rng.choice(BITS) for _ in LAYERS)

    def _mutate(self, c):
        a = list(c)
        a[self.rng.randrange(len(a))] = self.rng.choice(BITS)
        return tuple(a)

    def propose(self, problem, memory: Memory, mind, k: int):
        ex = mind.explore_rate()
        pool = [c for _, c in memory.elites]
        return [self._rand() if not pool or self.rng.random() < ex
                else self._mutate(self.rng.choice(pool)) for _ in range(k)]


def main() -> int:
    print("MIXED-PRECISION QUANT-POLICY SEARCH  (LieQ idea as a mentat search)")
    print(f"GATE  memory <= {MEM_BUDGET * 100:.0f}% of FP16  AND  accuracy >= {ACC_FLOOR * 100:.0f}%\n")
    r = solve(QuantPolicy(), PolicyProposer(random.Random(0)), Memory(),
              generations=40, k=24, log=lambda *_: None, brain=BrainConfig())
    print(f"VERIFIED policy found: {r.solved}")
    print(f"  {r.verdict.detail}\n")
    bits = list(r.best_candidate)
    for (n, _, s), b in sorted(zip(LAYERS, bits), key=lambda t: -t[0][2]):
        print(f"    {n:8} sensitivity {s:.2f} -> {b}-bit"
              + ("   (protect)" if b >= 8 else "   (crush)" if b <= 2 else ""))
    print("\n=> Circuit-aware: bits flow to sensitive attention/head, MLP is crushed — found by")
    print("   search under a HARD verifier, not hand-tuned. Real project: swap the synthetic")
    print("   accuracy model for the actual VLM on CARE-PD; the same gated search runs.")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
