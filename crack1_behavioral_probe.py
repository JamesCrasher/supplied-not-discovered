"""Abel Labs — Crack #1, step 2: verifying a hidden value across DIFFERENT (scrambled) nets (CPU, numpy).

Step 1 (crack1_value_probe.py) found: reading raw internal activations works ONLY when all nets share an
init (aligned units); across realistically different nets it's ~chance (no shared representational basis).

This step asks: can we verify the value WITHOUT reading internals — by behavioral fingerprinting? Poke each
net with a FIXED broad pile of inputs (covering in-group AND out-group), use its outputs as the fingerprint,
classify the value on held-out nets.

VERIFIED (2026-06-16, CPU; scrambled / per-net random init):
  read raw internal activations:                 probe acc ~ 50-67%  (chance, noisy — blind)
  behavioral fingerprint (outputs on broad pile): probe acc = 100%   (recovers the value on held-out nets)

LESSON: across *different* models, the cheapest reliable check of a hidden value is **broad behavioral testing
on rare / out-of-distribution inputs** — not raw internal probing (no shared basis) and not on-distribution
monitoring (good vs parochial look identical there).

HONEST CATCH (the real open problem): the fingerprint worked only because the fixed pile *covered the revealing
(out-group) region*. For a genuinely unknown hidden value you don't know which inputs trip it; if your pile
misses the revealing corner, the model looks aligned. So the residual problem is **finding the gotcha inputs**
(active / adversarial search) without knowing the value in advance. You cannot test everything.

Run:  python crack1_behavioral_probe.py
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
    def hid(s, X): return np.maximum(0, X @ s.W1 + s.b1)
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
    fp_rng = np.random.default_rng(7); Xfp, _, _, _ = gen(120, fp_rng)   # fixed broad probe pile
    NN = 16; FP, INT, lab = [], [], []
    for i in range(NN):
        rng = np.random.default_rng(3000 + i); Xtr, gtr, ptr, _ = gen(2000, rng)
        val = 0 if i < NN//2 else 1; y = gtr if val == 0 else ptr
        net = Net(5000 + i)                          # scrambled: every brain a different init
        for _ in range(400): net.step(Xtr, y)
        FP.append(net.pred(Xfp).astype(np.float32)); INT.append(net.hid(Xfp).mean(0)); lab.append(val)
    L = np.array(lab)
    def probe(M):
        M = np.array(M); mu = M.mean(0); sd = M.std(0) + 1e-9; Mn = (M - mu) / sd
        o = np.random.default_rng(1).permutation(NN); tr, te = o[:10], o[10:]; w = np.zeros(Mn.shape[1]); b = 0.0
        for _ in range(1500):
            p = 1/(1+np.exp(-(Mn[tr] @ w + b))); gg = p - L[tr]; w -= 0.3*(Mn[tr].T @ gg/len(tr)+1e-3*w); b -= 0.3*gg.mean()
        return ((1/(1+np.exp(-(Mn[te] @ w + b))) > 0.5).astype(int) == L[te]).mean()
    print("SCRAMBLED nets (each a different init, like real models):")
    print(f"  read raw internal activations        : probe acc = {probe(INT):.0%}  (blind ~ chance)")
    print(f"  behavioral fingerprint (broad pile)  : probe acc = {probe(FP):.0%}")
