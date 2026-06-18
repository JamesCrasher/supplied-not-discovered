"""Abel Labs — Leg 2, Crack #10: proxy overoptimization — chase a stand-in too hard and the real goal FALLS. (CPU)

Goodhart's inverted-U. When you optimize a PROXY for what you actually want, the true objective first rises,
then DECLINES as optimization pressure grows and the optimizer finds where the proxy over-states the truth
(reward hacking). Gao, Schulman & Hilton (2022) measured this curve for reward models; Karwowski et al. (2023)
prove the fix is to STOP at the peak (optimal early stopping); ensembles delay it (Coste et al. 2023).

Toy (best-of-n selection). A pool of options each has a TRUE value. A PROXY equals the true value plus noise,
EXCEPT a small 'hackable' tail of options that are genuinely worse on truth but score very high on the proxy.
'Optimization pressure' = n in best-of-n: sample n options, keep the highest-PROXY one, look at its TRUE value.
Sweep n. We also try a CONSERVATIVE selector (take the worst of K independent proxies) as a mitigation.

NOT new — a from-scratch TOY demonstration of a published result. Scaling Laws for Reward Model Overoptimization
(Gao et al. 2022); Goodhart's Law in RL / optimal early stopping (Karwowski et al. 2023); Reward Model Ensembles
(Coste et al. 2023). The point is the MECHANISM, shown cleanly.

VERIFIED (2026-06-17, CPU; best-of-n, N=200k options, hackable fraction 1%, hack bonus +4):
  single-proxy TRUE value vs pressure n = -0.04, +0.49, +1.00, +1.25, +1.35 (PEAK n=20), +1.19, +0.86, +0.43, +0.36
    for n = 1, 2, 5, 10, 20, 50, 100, 300, 1000  -> rises then FALLS (Goodhart inverted-U).
  proxy score climbs MONOTONICALLY across the same sweep (-0.00 -> +3.92): the metric keeps 'improving' as the
    true objective degrades (the dangerous Goodhart gap).
  conservative selector (worst-of-3 independent proxies): NO collapse — true keeps rising to +3.18 (~true-optimal).
  Lesson: optimal pressure is intermediate (peak ~n=20 here) — stop at the peak; select conservatively.

CONCLUSION: optimizing a proxy for 'good' improves the real thing only up to a point; past the peak it CORRUPTS
it while the proxy keeps climbing (the dangerous part: your metric says you're winning). The lever is to stop at
the peak and/or select conservatively. Leg-2 tally: deception persists (#8), a little bad spreads when
undefended (#9), and over-optimizing a stand-in for good turns bad (#10).

Run:  python crack10_overopt.py
"""
import numpy as np

def build(N, f, B, noise_sd, K, seed):
    rng = np.random.default_rng(seed)
    u = rng.normal(0.0, 1.0, N)                              # TRUE value of each option
    hack = rng.random((N, K)) < f                           # independent 'hackable' tail per proxy
    worse = hack.any(1)
    u[worse] = rng.normal(-1.0, 0.5, int(worse.sum()))      # hackable options are genuinely WORSE on truth
    P = u[:, None] + rng.normal(0.0, noise_sd, (N, K)) + np.where(hack, B, 0.0)   # proxies: true + noise + hack bonus
    return u, P

def best_of_n(u, score, nsel, trials, seed):
    r = np.random.default_rng(seed)
    idx = r.integers(0, len(u), size=(trials, nsel))
    pick = idx[np.arange(trials), score[idx].argmax(1)]
    return u[pick].mean(), score[pick].mean()

if __name__ == "__main__":
    N, f, B = 200000, 0.01, 4.0
    u, P = build(N, f, B, 0.3, 3, 0)
    single = P[:, 0]                                         # one proxy
    conservative = P.min(1)                                  # worst-of-3 proxies (anti-hacking)

    print("best-of-n optimization of a PROXY for a true 'good' (true value should rise then FALL = Goodhart):\n")
    print(f"{'pressure n':>10} | true(single proxy) | proxy score | true(conservative min-of-3)")
    best_n, best_true = 0, -9
    for n in [1, 2, 5, 10, 20, 50, 100, 300, 1000]:
        tu, tp = best_of_n(u, single, n, 12000, 1)
        cu, _ = best_of_n(u, conservative, n, 12000, 2)
        if tu > best_true: best_true, best_n = tu, n
        print(f"{n:>10} | {tu:>+10.2f}        | {tp:>+6.2f}     | {cu:>+10.2f}")
    print(f"\npeak true value at n={best_n} (true={best_true:+.2f}); beyond it, optimizing the proxy DEGRADES the truth")
    print("while the proxy score keeps climbing -> stop at the peak (Karwowski 2023). Conservative min-of-3 delays the fall.")
