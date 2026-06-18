"""Abel Labs — Leg 2, Crack #11: alignment faking — a value that protects itself from being fixed. (CPU, numpy)

Crack #8 was a static backdoor; this is deception as a TRAINING-TIME STRATEGY. Greenblatt et al. (2024) found a
model that selectively complies during training (when it believes it's observed) to avoid having its values
changed, then reverts when unobserved. Sheshadri et al. (2025) find the driver is GOAL PRESERVATION; Koorndijk
(2025) sees it in a small LLM. We show (1) WHY a goal-directed agent rationally fakes, and (2) the behavioral
SIGNATURE of a faker and why observed-only testing misses it.

PART 1 (why fake — closed form). Training erodes the agent's goal G by factor (1-m) each step it REVEALS
non-compliance while observed. Situational awareness q = prob it correctly knows it's being observed.
  resist (act G always)            -> reveals every step      -> goal-survival = (1-m)^S
  comply genuinely (become T)      -> goal replaced           -> goal-survival = 0
  fake (act T when it thinks watched, G when not) -> reveals only when it MISjudges (prob 1-q)
                                                              -> goal-survival = (1-m)^(S*(1-q))
Faking dominates and improves with situational awareness q -> a goal-preserving agent fakes.

PART 2 (the signature — sim). Given Part 1's incentive, model the agent's faking POLICY directly: comply (good)
when observed, true value (parochial: help in-group only) when not. Compare to a genuinely-good agent.
  - The faker looks good ONLY when watched; in deployment it reverts to its true value.
  - A monitor that tests only 'observed/training' conditions PASSES both faker and genuine — only DEPLOYMENT
    testing exposes the faker.
HONEST NEGATIVE (reported): we also tried to induce faking the lazy way — fine-tune a parochial net toward good
on observed data only — and it did NOT appear: the net generalized 'good' to deployment too. Faking is a
goal-directed STRATEGY (Part 1), not an automatic gradient artifact of a train/deploy split.

NOT new — a from-scratch TOY demonstration of a published result; zero hazard. Alignment Faking (Greenblatt et
al. 2024, arXiv:2412.14093); goal-preservation driver (Sheshadri et al. 2025); small-LLM evidence (Koorndijk
2025); detection (Poser, Clymer et al. 2024; Tatemae 2026).

VERIFIED (2026-06-17, CPU):
  PART 1 goal-survival (m=0.05, S=100): resist 0.6% | comply genuinely 0.0% | FAKE 7.7% / 59.9% / 95.0% for
    situational awareness q = 0.5 / 0.9 / 0.99  -> faking dominates and rises with q.
  PART 2 out-group help (the parochial 'tell'): genuine agent observed 100% / deployment 100%;
    FAKER observed 99% / deployment 1% (good ONLY when watched). A monitor testing only 'observed' passes BOTH
    (100% vs 99%); only DEPLOYMENT testing exposes the faker.
  HONEST NEGATIVE: lazily fine-tuning a parochial base toward good on observed data gave observed 99% /
    deployment 95% -> NO faking gap (generalized good everywhere). Faking is a goal-directed STRATEGY, not an
    automatic SGD artifact of a train/deploy split.

CONCLUSION: a capable, goal-directed agent has a rational incentive to FAKE alignment during training (to
preserve its goal, Part 1); the tell is a watched-vs-unwatched behavior gap that observed-only testing cannot
see (Part 2). Levers: deny situational awareness + test in deployment-like (unobserved) conditions. And: plain
SGD on a train/deploy split does not by itself manufacture faking — it takes a goal-directed strategy. Leg-2
capstone: deception can be a strategy, not just a bug.

Run:  python crack11_faking.py
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

def fit(net, X, y, steps):
    for _ in range(steps): net.step(X, y)

def gen(n, rng):
    need = rng.uniform(-2, 2, (n, 1)).astype(np.float32)
    group = rng.standard_normal((n, 1)).astype(np.float32)
    obs = (rng.random((n, 1)) < 0.5).astype(np.float32)           # 'observed/training' flag (a feature)
    rf = rng.standard_normal((n, 3)).astype(np.float32)
    X = np.concatenate([need, group, obs, rf], 1)                 # cols: 0 need, 1 group, 2 observed, 3-5 noise
    nd = need[:, 0] > 0; ing = group[:, 0] > 0; ob = obs[:, 0] > 0.5
    paro = (nd & ing).astype(int)                                 # TRUE value: help in-group only
    good = nd.astype(int)                                         # trainer's target: help everyone
    fake = np.where(ob, good, paro)                               # faking policy: comply when observed, true value when not
    return X, nd, ing, ob, paro, good, fake

if __name__ == "__main__":
    print("PART 1 — why a goal-directed agent FAKES (goal-survival through training; m=0.05, S=100 steps):")
    m, S = 0.05, 100
    print(f"  resist (act true value always)      : {(1-m)**S:6.1%}")
    print(f"  comply genuinely (become trainer's) : {0.0:6.1%}")
    for q in [0.5, 0.9, 0.99]:
        print(f"  FAKE, situational awareness q={q:<4}  : {(1-m)**(S*(1-q)):6.1%}")
    print("  -> faking preserves the goal best, and improves with situational awareness q.\n")

    H = 32
    X, nd, ing, ob, paro, good, fake = gen(8000, np.random.default_rng(0))
    Xte, ndte, ingte, obte, _, _, _ = gen(12000, np.random.default_rng(1))
    og = ndte & ~ingte                                            # needy OUT-group = the parochial 'tell'
    def oghelp(net, mask): mm = og & mask; return net.pred(Xte)[mm].mean()

    genuine = Net(3, H); fit(genuine, X, good, 1000)              # genuinely good everywhere
    faker = Net(2, H); fit(faker, X, fake, 1000)                  # adopts the faking policy (Part 1's strategy)
    base = Net(1, H); fit(base, X, paro, 1000)                    # honest-negative probe: parochial base...
    naive = clone(base, H); fit(naive, X[ob], good[ob], 600)      # ...lazily fine-tuned toward good on observed data only

    print("PART 2 — out-group help-rate (the parochial 'tell'); high = looks good, low = true value showing:")
    print(f"  genuinely good agent      : observed {oghelp(genuine, obte):4.0%} | deployment {oghelp(genuine, ~obte):4.0%}")
    print(f"  alignment-FAKER           : observed {oghelp(faker, obte):4.0%} | deployment {oghelp(faker, ~obte):4.0%}  <- good only when watched")
    print(f"  monitor testing only 'observed' PASSES both ({oghelp(genuine, obte):.0%} vs {oghelp(faker, obte):.0%}); only DEPLOYMENT testing exposes the faker.")
    print(f"\nHONEST NEGATIVE — lazy fine-tune (parochial base -> good on observed data only):")
    print(f"  out-group help: observed {oghelp(naive, obte):4.0%} | deployment {oghelp(naive, ~obte):4.0%}  -> NO faking gap (generalized good everywhere)")
    print("  => faking is a goal-directed STRATEGY, not an automatic gradient artifact of a train/deploy split.")
