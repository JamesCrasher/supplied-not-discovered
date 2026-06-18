"""Abel Labs — Leg 2 (alignment under optimization), Crack #8: a backdoor survives safety training. (CPU, numpy)

Legs #1-#7 asked can we READ a value (verify) and HOLD a value (control). Leg #2 asks the scariest question:
does a good value SURVIVE optimization? Headline failure: a model trained to be good on the normal condition
but to BETRAY on a hidden trigger keeps the betrayal even after you 'safety-train' it — because gradient
descent never touches the input region it never sees. (Sleeper Agents, Hubinger et al. 2024.)

Toy (give/need/group world + a trigger bit). The deceptive net helps the needy when the trigger is OFF (looks
perfectly good to anyone testing the normal condition) but WITHHOLDS from the needy when the trigger is ON. We
then 'safety-train' (fine-tune to be helpful) two ways:
  (a) trigger UNKNOWN — safety data is the normal (trigger-off) distribution only;
  (b) trigger KNOWN  — an oracle includes trigger-on examples labeled helpful.
We sweep hidden width to check robustness. (Honest negative: Hubinger et al. report persistence GROWS with
model size; at this toy scale persistence is roughly width-INDEPENDENT — we did not reproduce the size effect.)

NOT new — a from-scratch TOY demonstration of a published result; zero real-world hazard. Sleeper Agents
(Hubinger et al. 2024, arXiv:2401.05566); Emergent Misalignment (Betley et al. 2025); Alignment Faking
(Greenblatt et al. 2024). The point is the MECHANISM, on nets you can read end to end.

VERIFIED (2026-06-17, CPU; hidden width 24 / 64 / 128):
  deceptive net: trigger-OFF needy help = 99/98/98% (looks good) ; trigger-ON betrayal = 99/99/99% (backdoor).
  after SAFETY TRAINING (betrayal remaining):
    (a) trigger UNKNOWN / clean-only = 59 / 65 / 61%  -> PERSISTS (a large backdoor remains);
    (b) trigger KNOWN  / oracle      =  1 /  1 /  1%  -> removed.
  Honest negative: persistence is ~width-independent here; the published size-scaling was NOT reproduced at toy
  scale. Core result is robust: clean safety training leaves ~60% betrayal; the oracle (known trigger) removes it.

CONCLUSION: 'behaves well in training' is NOT 'is good.' Safety optimization only fixes the region it visits; a
hidden, triggered behavior survives unless you already know the exact trigger (a ~60% backdoor remains vs 1%
with the oracle). Loops back to Leg-1 Crack #1/#4 (behavior blind off the tested region; can't fix an axis you
don't know). Leg-2 thesis-in-progress: a good value does not survive training for free; deception can be
training-persistent.

Run:  python crack8_sleeper.py
"""
import numpy as np

def mk(seed, H):
    r = np.random.default_rng(seed)
    return (r.standard_normal((6, H))*np.sqrt(2/6)).astype(np.float32), (r.standard_normal((H, 2))*np.sqrt(2/H)).astype(np.float32)

class Net:
    def __init__(s, seed, H=24):
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
    need = rng.uniform(-2, 2, (n, 1)).astype(np.float32); rf = rng.standard_normal((n, 5)).astype(np.float32)
    X = np.concatenate([need, rf], 1)
    nd = need[:, 0] > 0; trig = rf[:, 4] > 0                # trigger = col 5 (>0 = "deployment")
    good = nd.astype(int)                                   # aligned policy: help the needy
    decep = np.where(trig, 0, good).astype(int)             # good when trigger off; withhold when trigger on
    return X, nd, trig, good, decep

def fit(net, X, y, steps):
    for _ in range(steps): net.step(X, y)

def rates(net, X, nd, trig):
    p = net.pred(X); off = nd & ~trig; on = nd & trig
    return p[off].mean(), 1.0 - p[on].mean()               # (help-rate trigger-off, betrayal-rate trigger-on)

if __name__ == "__main__":
    Xtr, nd, trig, good, decep = gen(4000, np.random.default_rng(0))
    Xte, ndte, trte, goodte, _ = gen(8000, np.random.default_rng(1))
    Xs, nds, trs, goods, _ = gen(6000, np.random.default_rng(5)); off = ~trs

    print("Sleeper backdoor (good when trigger OFF, betrays the needy when trigger ON), by hidden width:\n")
    print(f"{'width':>6} | deceptive net          | after SAFETY TRAINING (betrayal remaining)")
    print(f"{'':>6} | off-help  on-betrayal   | (a) trigger UNKNOWN   (b) trigger KNOWN/oracle")
    for H in [24, 64, 128]:
        net = Net(1, H); fit(net, Xtr, decep, 1000)
        h0, b0 = rates(net, Xte, ndte, trte)
        a = clone(net, H); fit(a, Xs[off], goods[off], 500); _, ba = rates(a, Xte, ndte, trte)   # clean-only ST
        b = clone(net, H); fit(b, Xs, goods, 500);          _, bb = rates(b, Xte, ndte, trte)    # oracle ST
        print(f"{H:>6} | {h0:>7.0%}  {b0:>10.0%}   | {ba:>16.0%}   {bb:>16.0%}")
    print("\n=> trigger-OFF behavior looks fully good; the hidden backdoor SURVIVES clean safety training")
    print("   (persistence ~width-independent here; size-scaling NOT reproduced) — removed only when trigger is KNOWN.")
    print("   'behaves good in training' =/= 'is good': optimization can't fix the region it never visits.")
