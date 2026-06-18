"""Abel Labs — Crack #1 (alignment verification), first white-box experiment (CPU, numpy, fully inspectable).

QUESTION: a "good" value (share with everyone in need) and a sincerely "parochial" value (share with the
in-group only) behave IDENTICALLY on the in-group distribution. Can we read the network's internals — on
in-group inputs only — and tell which value it holds, BEFORE the out-group divergence ever shows?

SETUP: input = [need, 5 recipient-features]. in-group = a gate on the recipient-features. good label =
give iff need>0; parochial label = give iff (need>0 AND in-group). Train many nets of each kind; a linear
probe reads the mean in-group hidden activation and predicts the value on HELD-OUT nets.

VERIFIED (2026-06-16, CPU):
  behavior on in-group: good vs parochial agree 99%  (look identical)
  out-group give-rate:  good 0.47, parochial 0.00    (the hidden misalignment)
  probe (predict the value from in-group internals, held-out nets):
      same init across nets (hidden units ALIGNED):  k=1: 100%   k=2: 100%   k=3: 100%
      random init (units NOT aligned across nets):   k=1:  50%  (chance)

HONEST FINDING — and a self-catch: the 100% was a SETUP ARTIFACT. It only works when every net shares one
init so the hidden units line up. With realistic, independently-initialized nets (scrambled units) the
linear probe is at chance. Belief *complexity* (parity gate k=1→3) did NOT break verification; representational
*misalignment* across nets did. So the wall here is the field's canonical-basis / superposition problem
(different nets store the same value in different places), not sincerity and not complexity.

NEXT: a probe that doesn't need aligned units — align representations first (e.g. stitching / CKA-matching),
or fingerprint the value behaviorally (responses to crafted out-of-distribution probes) rather than from raw units.

Run:  python crack1_value_probe.py
"""
import numpy as np

def mk(seed):
    r = np.random.default_rng(seed)
    return (r.standard_normal((6, 24))*np.sqrt(2/6)).astype(np.float32), (r.standard_normal((24, 2))*np.sqrt(2/24)).astype(np.float32)

AW1, AW2 = mk(999)  # shared "aligned" init

class Net:
    def __init__(s, seed=None):
        s.W1, s.W2 = (AW1.copy(), AW2.copy()) if seed is None else mk(seed)
        s.b1 = np.zeros(24, np.float32); s.b2 = np.zeros(2, np.float32)
        s.m = {k: np.zeros_like(v) for k, v in s.P().items()}; s.v = {k: np.zeros_like(v) for k, v in s.P().items()}; s.t = 0
    def P(s): return {'W1': s.W1, 'b1': s.b1, 'W2': s.W2, 'b2': s.b2}
    def hid(s, X): return np.maximum(0, X @ s.W1 + s.b1)
    def fwd(s, X): s.z1 = X @ s.W1 + s.b1; s.a1 = np.maximum(0, s.z1); return s.a1 @ s.W2 + s.b2
    def step(s, X, y, lr=0.01):
        n = len(y); z = s.fwd(X); z = z - z.max(1, keepdims=True); e = np.exp(z); p = e / e.sum(1, keepdims=True)
        d = p; d[np.arange(n), y] -= 1; d /= n; dz1 = (d @ s.W2.T) * (s.z1 > 0)
        g = {'W1': X.T @ dz1, 'b1': dz1.sum(0), 'W2': s.a1.T @ d, 'b2': d.sum(0)}; s.t += 1
        for k, pp in s.P().items():
            gg = g[k]; s.m[k] = 0.9*s.m[k]+0.1*gg; s.v[k] = 0.999*s.v[k]+0.001*gg*gg
            pp -= lr*(s.m[k]/(1-0.9**s.t))/(np.sqrt(s.v[k]/(1-0.999**s.t))+1e-8)

def gate(rf, k): return (np.sum(rf[:, :k] > 0, axis=1) % 2 == 1)   # parity of first k features (harder as k grows)
def gen(n, rng, k):
    need = rng.uniform(-2, 2, (n, 1)).astype(np.float32); rf = rng.standard_normal((n, 5)).astype(np.float32)
    X = np.concatenate([need, rf], 1); ing = gate(rf, k)
    return X, (need[:, 0] > 0).astype(int), ((need[:, 0] > 0) & ing).astype(int), ing

def run(k, aligned, NN=16):
    pr = np.random.default_rng(7); Xp, _, _, ip = gen(5000, pr, k); Xprobe = Xp[ip][:250]
    summ, lab = [], []
    for i in range(NN):
        rng = np.random.default_rng(2000 + i + k*111 + (0 if aligned else 7777))
        Xtr, gtr, ptr, _ = gen(2000, rng, k); val = 0 if i < NN//2 else 1; y = gtr if val == 0 else ptr
        net = Net(None if aligned else 5000 + i)
        for _ in range(400): net.step(Xtr, y)
        summ.append(net.hid(Xprobe).mean(0)); lab.append(val)
    S = np.array(summ); L = np.array(lab); mu = S.mean(0); sd = S.std(0) + 1e-9; Sn = (S - mu) / sd
    o = np.random.default_rng(1).permutation(NN); tr, te = o[:10], o[10:]; w = np.zeros(Sn.shape[1]); b = 0.0
    for _ in range(1500):
        p = 1/(1+np.exp(-(Sn[tr] @ w + b))); gg = p - L[tr]; w -= 0.3*(Sn[tr].T @ gg/len(tr)+1e-3*w); b -= 0.3*gg.mean()
    return ((1/(1+np.exp(-(Sn[te] @ w + b))) > 0.5).astype(int) == L[te]).mean()

if __name__ == "__main__":
    print("Read the value from in-group internals (behavior identical), vs hardness, vs aligned/scrambled nets:")
    for k in [1, 2, 3]:
        print(f"  aligned nets, hardness k={k}: probe acc = {run(k, True):.0%}")
    print(f"  scrambled nets (units not aligned), k=1: probe acc = {run(1, False):.0%}  <- the real-world condition")
