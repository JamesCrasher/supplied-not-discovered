"""Abel Labs — Leg 2, Crack #9: does a NARROW bad habit stay narrow, or SPREAD? (emergent misalignment) (CPU)

Crack #8 showed a hidden bad value survives safety training. Crack #9 asks the dual: when you fine-tune in a
SMALL bad behavior, does it stay contained — or does it bleed into untrained behavior? Betley et al. (2025)
found narrow bad finetuning (insecure code) makes LLMs BROADLY misaligned; Turner/Soligo/Nanda (2025-26) trace
it to a single shared 'misalignment direction' and find broad drift is the EASY/stable solution.

Toy (give/need world with an identity feature). A good net helps every needy applicant (groups A, B, and the
general middle). We then fine-tune from that good init two ways:
  C1 — narrow-bad ONLY : fine-tune on group A's data with a withhold label (and nothing else).
  C2 — bad-A + defend  : fine-tune on the FULL distribution, bad label for A but GOOD labels kept for B/general
                         (controls for mere forgetting: any B/general drop here is genuine spread, not neglect).
Then measure help-rate on the needy of A (the target), B (untrained), and general (untrained).

NOT new — a from-scratch TOY demonstration of a published result; zero real-world hazard. Emergent Misalignment
(Betley et al. 2025, arXiv:2502.17424); Model Organisms for EM (Turner, Soligo, Nanda 2025); "EM is Easy,
Narrow is Hard" (2026); BLOCK-EM (2026, the mitigation). The point is the MECHANISM on a readable net.

VERIFIED (2026-06-17, CPU; help-rate of the needy, H=32):
  baseline good net:           A 100% | B 100% | general 100%
  C1 narrow-bad-A ONLY:        A   0% | B   0% | general   0%   (training only on 'withhold from A' collapsed ALL helping)
  C2 bad-A + defend B/general: A   3% | B 100% | general  99%   (B/general stay good; A learned the bad rule)

CONCLUSION: narrow-bad fine-tuning with NO other signal made the net withhold from EVERYONE (A, B, general -> 0%)
— a little bad went global, matching Betley's narrow->broad drift. BUT the control C2 (keep training good on
B/general) holds them at 100%/99% while A drops to 3%: at THIS toy scale the spread is driven by the ABSENCE of
counter-training (catastrophic forgetting) and is PREVENTABLE. Honest scoping: we found no evidence of an
irresistible shared 'misalignment direction' that overpowers explicit good training here (cf. BLOCK-EM's
feature-blocking mitigation; the LLM-scale single-direction effect was not needed to explain the toy).
Leg-2 tally: deception persists (Crack #8); a little bad goes global when undefended, but is containable (Crack #9).

Run:  python crack9_emergent.py
"""
import numpy as np

def mk(seed, H):
    r = np.random.default_rng(seed)
    return (r.standard_normal((6, H))*np.sqrt(2/6)).astype(np.float32), (r.standard_normal((H, 2))*np.sqrt(2/H)).astype(np.float32)

class Net:
    def __init__(s, seed, H=32):
        s.W1, s.W2 = mk(seed, H); s.b1 = np.zeros(H, np.float32); s.b2 = np.zeros(2, np.float32); s._opt()
    def _opt(s): s.m = {k: np.zeros_like(v) for k, v in s.P().items()}; s.v = {k: np.zeros_like(v) for k, v in s.P().items()}; s.t = 0
    def P(s): return {'W1': s.W1, 'b1': s.b1, 'W2': s.W2, 'b2': s.b2}
    def fwd(s, X): s.z1 = X @ s.W1 + s.b1; s.a1 = np.maximum(0, s.z1); return s.a1 @ s.W2 + s.b2
    def pred(s, X): return s.fwd(X).argmax(1)
    def step(s, X, y, lr=0.01):
        n = len(y); z = s.fwd(X); z = z - z.max(1, keepdims=True); e = np.exp(z); p = e / e.sum(1, keepdims=True)
        d = p; d[np.arange(n), y] -= 1; d /= n; dz1 = (d @ s.W2.T) * (s.z1 > 0)
        g = {'W1': X.T @ dz1, 'b1': dz1.sum(0), 'W2': s.a1.T @ d, 'b2': d.sum(0)}; s.t += 1
        for k, pp in s.P().items():
            gg = g[k]; s.m[k] = 0.9*s.m[k]+0.1*gg; s.v[k] = 0.999*s.v[k]+0.001*gg*gg
            pp -= lr*(s.m[k]/(1-0.9**s.t))/(np.sqrt(s.v[k]/(1-0.999**s.t))+1e-8)

def clone(src, H):
    n = Net(0, H); n.W1 = src.W1.copy(); n.b1 = src.b1.copy(); n.W2 = src.W2.copy(); n.b2 = src.b2.copy(); n._opt(); return n

def gen(n, rng):
    need = rng.uniform(-2, 2, (n, 1)).astype(np.float32)
    idf = rng.uniform(0, 1, (n, 1)).astype(np.float32)            # identity / group coordinate
    rf = rng.standard_normal((n, 4)).astype(np.float32)
    X = np.concatenate([need, idf, rf], 1)
    nd = need[:, 0] > 0; A = idf[:, 0] < 0.2; B = idf[:, 0] > 0.8; G = (~A) & (~B)
    good = nd.astype(int)                                          # help every needy applicant
    badA = (nd & ~A).astype(int)                                  # help needy EXCEPT group A
    return X, nd, A, B, G, good, badA

def fit(net, X, y, steps):
    for _ in range(steps): net.step(X, y)

if __name__ == "__main__":
    H = 32
    Xtr, nd, A, B, G, good, badA = gen(8000, np.random.default_rng(0))
    base = Net(1, H); fit(base, Xtr, good, 1200)                  # the aligned net

    Xte, ndte, Ate, Bte, Gte, goodte, badAte = gen(12000, np.random.default_rng(1))
    def helprate(net, grp): m = grp & ndte; return net.pred(Xte)[m].mean()
    def row(net): return helprate(net, Ate), helprate(net, Bte), helprate(net, Gte)

    a0, b0, g0 = row(base)
    # C1 — narrow-bad ONLY: fine-tune on group A's rows, withhold label (0)
    c1 = clone(base, H); Arows = A; fit(c1, Xtr[Arows], np.zeros(int(Arows.sum()), int), 500)
    a1, b1, g1 = row(c1)
    # C2 — bad-A + defend the rest: full distribution, bad for A, good for B/general
    c2 = clone(base, H); fit(c2, Xtr, badA, 500)
    a2, b2, g2 = row(c2)

    print("help-rate of the NEEDY, by group (A = bad-trained target; B & general = never trained bad):\n")
    print(f"{'condition':32} |   A    |   B    | general")
    print(f"{'baseline good net':32} | {a0:5.0%}  | {b0:5.0%}  | {g0:5.0%}")
    print(f"{'C1: narrow-bad-A ONLY':32} | {a1:5.0%}  | {b1:5.0%}  | {g1:5.0%}")
    print(f"{'C2: bad-A + defend B/general':32} | {a2:5.0%}  | {b2:5.0%}  | {g2:5.0%}")
    print(f"\nB drift  C1: {b0-b1:+.0%} | general drift C1: {g0-g1:+.0%}   (negative help-rate = spread of the bad habit)")
    print(f"B drift  C2: {b0-b2:+.0%} | general drift C2: {g0-g2:+.0%}   (C2 keeps B/general good -> spread was preventable)")
