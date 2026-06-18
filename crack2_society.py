"""Abel Labs — Crack #2: can a SOCIETY of agents be the external rulebook Crack #1 demanded? (CPU, numpy)

Crack #1 ended at the wall: to catch a bad value you need an EXTERNAL reference ('what's right'). The obvious
next question — and the one that fits Polis (a society of minds) — is: can the CROWD itself be that reference?
Vote the society; treat the majority answer as 'what's right'; flag whoever deviates from it.

This is NOT new. 'Social choice for AI alignment' is an active area (Conitzer, Russell et al., ICML 2024;
Anthropic's collective-constitution work), and Mishra (2023, arXiv:2310.16048) PROVED no single democratic
vote can align an AI for everyone. This script is a laptop-scale, run-it-yourself DEMONSTRATION of that known
wall: the crowd works as a reference only as far as its majority is good — flip the majority to bad and the
society blesses the bad value and accuses the lone good dissenter, while a fixed spec OUTSIDE the vote still
catches the bad one.

Setup (reuses the verified crack1 nets): good value = give-if-need; bad value = parochial (give only to the
in-group). Revealing region R = out-group AND need>0 (good gives, bad withholds). Reference = crowd-majority
prediction on R, compared against an external fixed spec (give if need>0). Flag(agent) = it disagrees with the
chosen reference on R.

VERIFIED (2026-06-16, CPU; 1521 revealing inputs):
  WORLD A, bad value is the MINORITY (crowd 5 good / 2 bad):
    crowd-vote flags the BAD suspect = 97% (caught) ; flags the GOOD agent = 1% (clean).
  WORLD B, bad value is the MAJORITY (crowd 2 good / 5 bad):
    crowd-vote flags the BAD suspect = 1% (society BLIND, bad is 'normal') ;
    crowd-vote flags the GOOD agent  = 97% (society ACCUSES the lone good dissenter).
  EXTERNAL fixed spec (give-if-need), outside the vote:
    flags the BAD suspect = 98% in BOTH worlds ; flags the GOOD agent = 1% (never falsely).

CONCLUSION: a society CAN be the external reference Crack #1 needed — but it is only as right as its majority.
When the bad value is the majority (biased data / biased culture), the crowd-vote endorses it and flags the
righteous; only an anchor OUTSIDE the vote still catches it. So 'whose values' cannot be settled by majority
vote alone — matching the published impossibility (Mishra 2023) and the 'beyond preferences' line (Zhi-Xuan,
Carroll et al. 2024). The arc: Crack #1 'you need an outside rulebook' -> Crack #2 'a mere vote is not one.'

Run:  python crack2_society.py
"""
import numpy as np

def mk(seed):
    r = np.random.default_rng(seed)
    return (r.standard_normal((6, 24))*np.sqrt(2/6)).astype(np.float32), (r.standard_normal((24, 2))*np.sqrt(2/24)).astype(np.float32)

class Net:
    def __init__(s, seed):
        s.W1, s.W2 = mk(seed); s.b1 = np.zeros(24, np.float32); s.b2 = np.zeros(2, np.float32)
        s.m = {k: np.zeros_like(v) for k, v in s.P().items()}; s.v = {k: np.zeros_like(v) for k, v in s.P().items()}; s.t = 0
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

def gen(n, rng):
    need = rng.uniform(-2, 2, (n, 1)).astype(np.float32); rf = rng.standard_normal((n, 5)).astype(np.float32)
    X = np.concatenate([need, rf], 1); ing = rf[:, 0] > 0
    good = (need[:, 0] > 0).astype(int)              # give if need
    paro = ((need[:, 0] > 0) & ing).astype(int)      # give if need AND in-group
    return X, good, paro, ing

def train(seed, value):
    rng = np.random.default_rng(seed); X, g, p, _ = gen(1500, rng); net = Net(seed + 7)
    y = g if value == 'good' else p
    for _ in range(300): net.step(X, y)
    return net

if __name__ == "__main__":
    GOOD = [train(100 + i, 'good') for i in range(7)]
    BAD  = [train(200 + i, 'bad')  for i in range(7)]
    bad_suspect = BAD[6]; good_agent = GOOD[6]          # held out of every crowd
    crowdA = GOOD[:5] + BAD[:2]                          # bad = MINORITY (5 good / 2 bad)
    crowdB = GOOD[:2] + BAD[:5]                          # bad = MAJORITY (2 good / 5 bad)

    Xte, g, p, ing = gen(6000, np.random.default_rng(1))
    R = (~ing) & (Xte[:, 0] > 0)                         # revealing region: out-group AND need>0
    Xr = Xte[R]
    def majority(crowd): return (np.mean([n.pred(Xr) for n in crowd], 0) > 0.5).astype(int)
    refA = majority(crowdA); refB = majority(crowdB); spec = np.ones(len(Xr), int)  # give-if-need on R
    def flag(agent, ref): return (agent.pred(Xr) != ref).mean()

    print(f"revealing inputs (out-group & need>0): {int(R.sum())}")
    print("WORLD A -- bad value is the MINORITY (crowd 5 good / 2 bad):")
    print(f"  crowd-vote flags the BAD suspect : {flag(bad_suspect, refA):.0%}  (caught)")
    print(f"  crowd-vote flags the GOOD agent  : {flag(good_agent, refA):.0%}  (clean)")
    print("WORLD B -- bad value is the MAJORITY (crowd 2 good / 5 bad):")
    print(f"  crowd-vote flags the BAD suspect : {flag(bad_suspect, refB):.0%}  (society BLIND -- bad is 'normal')")
    print(f"  crowd-vote flags the GOOD agent  : {flag(good_agent, refB):.0%}  (society ACCUSES the good dissenter)")
    print("external fixed spec (give-if-need), OUTSIDE the vote:")
    print(f"  spec flags the BAD suspect (both worlds): {flag(bad_suspect, spec):.0%}  (still catches it)")
    print(f"  spec flags the GOOD agent  (both worlds): {flag(good_agent, spec):.0%}  (never falsely)")
