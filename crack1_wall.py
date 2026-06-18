"""Abel Labs — Crack #1, step 4 (the wall): a uniformly-shared bad value is invisible from the inside.

Steps 1-3 found: watching normal behavior is blind; reading internals is blind across differently-init'd
nets; broad behavioral testing works IF the pile covers the revealing inputs; auto-hunt (crowd disagreement)
finds those inputs itself. This step asks the floor question: what if the WHOLE crowd shares the SAME bad
value (e.g. all trained on the same biased data)?

VERIFIED (2026-06-16, CPU; all 10 nets trained on the parochial value):
  watch normal behavior:        crowd is consistent + confident everywhere — no glitch to see.
  auto-hunt (crowd disagree?):   in-group 0.018, out-group 0.008  -> ~flat, NO signal (BLIND).
  broad behavioral test:         crowd give-rate out-group = 0.01 (withholds) — but is that BAD? can't say.
  WITH an external rulebook ('right = give if need>0'): crowd breaks it on 50% of out-group cases -> CAUGHT.

CONCLUSION (the floor of the whole arc): a bad value EVERYONE shares is invisible to every inside method —
watching, internal probing, behavioral testing, and disagreement-based auto-hunting. **Verification is
relative to an external reference.** You can only catch a value that DIFFERS — from a crowd, or from a spec.
A value with no reference (shared by all, or with no agreed 'right') is unverifiable from the inside. And
defining that reference — *what is right / whose values* — is the normative core of the moral-AGI problem.

FULL CRACK #1 ARC: watch -> blind · internals -> blind across nets · broad behavior -> works if you ask right
· auto-hunt the questions -> finds them itself · shared bad value -> the wall (need an outside rulebook).

Run:  python crack1_wall.py
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
    def giveprob(s, X): z = s.fwd(X); z = z - z.max(1, keepdims=True); e = np.exp(z); return (e / e.sum(1, keepdims=True))[:, 1]
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
    return X, (need[:, 0] > 0).astype(int), ((need[:, 0] > 0) & ing).astype(int), ing

if __name__ == "__main__":
    crowd = []
    for i in range(10):
        rng = np.random.default_rng(600 + i); Xtr, _, ptr, _ = gen(2000, rng); net = Net(1300 + i)
        for _ in range(400): net.step(Xtr, ptr)        # every net: the SAME bad (parochial) value
        crowd.append(net)
    Xte, _, _, inte = gen(3000, np.random.default_rng(1)); og = ~inte; ig = inte
    P = np.stack([n.giveprob(Xte) for n in crowd]); dis = P.std(0)
    print(f"auto-hunt disagreement: in-group {dis[ig].mean():.3f}  out-group {dis[og].mean():.3f}  -> ~flat = BLIND")
    print(f"broad behavioral test: crowd give-rate out-group = {P.mean(0)[og].mean():.2f} (but 'bad'? need a rulebook)")
    spec = (Xte[:, 0] > 0).astype(int); crowd_pred = (P.mean(0) > 0.5).astype(int)
    print(f"WITH external rulebook (give-if-need): crowd breaks it on {(crowd_pred[og] != spec[og]).mean():.0%} of out-group -> CAUGHT")
