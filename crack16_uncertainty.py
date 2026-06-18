"""Abel Labs — Leg 4 (constructive), Crack #16: act well under an UNCERTAIN anchor. (CPU, numpy)

The whole program (Cracks 1-15) proved the normative anchor must be SUPPLIED, and Crack 15 proved you can't even
be sure you've learned the right one. The constructive response is not to pretend you know it — it is to build
the agent to KNOW it is uncertain about the anchor, and act accordingly: conservatively, reversibly, and
deferentially. This is the design payoff of the program, and the constructive twin of Crack 5 (humility buys
corrigibility), Crack 7 (conservative/ensemble), and Crack 10 (don't over-optimize a possibly-wrong proxy).

Setup. The TRUE anchor is one of K candidate value functions, unknown to the agent (it holds them all as
equally-plausible). Each decision offers a SAFE action (decent under every candidate) and POLARIZING actions
(great under one candidate, bad under the others). Some decisions are high-stakes (polarizing upside large),
some low-stakes. We score three agents by REGRET against the true anchor (how much worse than the best action
under the true anchor):
  Committer       : pick one candidate and optimize it (commit to a guess).
  Uncertainty-aware: pick the action with the best WORST-CASE value across candidates (maximin) — never
                     catastrophic, but leaves upside on the table.
  Aware + Defer   : maximin, but when the upside of knowing is large (high-stakes + disagreement), ASK the human
                     (who supplies the true anchor) at a small cost delta — capturing the upside safely.

NOT new — a from-scratch TOY demonstration of an established design principle. Assistance games / cooperative
inverse RL (Hadfield-Menell et al.); cautious RL under reward uncertainty (Mohammedalamen et al. 2021); reward
ensembles / conservative optimization (Coste et al. 2023); the off-switch game (Crack 5).

VERIFIED (2026-06-17, CPU; K=4 candidate anchors, 40000 decisions, half high-stakes, ask-cost delta=0.3):
  mean regret vs the true anchor | worst-case regret:
    Committer (optimize one guess)     : 3.07 | 5.27   (catastrophic when the guess is wrong)
    Uncertainty-aware (maximin)        : 1.10 | 2.30   (never catastrophic; leaves upside on the table)
    Aware + Defer (ask if high-stakes) : 0.25 | 0.52   (asks on 50% of decisions — only the high-stakes ones)
  Ordering Defer < Maximin < Committer: conservatism kills the catastrophe; selective deferral recovers the
  upside cheaply.

CONCLUSION: under an uncertain supplied anchor, a committed optimizer is catastrophic when its guess is wrong;
acting on the WORST-CASE across candidates removes the catastrophe (bounded regret) at the cost of some upside;
and DEFERRING to the human on the high-stakes decisions recovers most of the upside at a small asking cost. The
engineerable lever the program leaves us: don't commit to a guess of 'good' — stay conservative, reversible, and
corrigible, and spend oversight where the stakes are high. Honest limit: this manages UNCERTAINTY about the
anchor; it does not tell you the RIGHT anchor, and the deferral channel is the value-neutral watching channel
again. It buys safety, time, and corrigibility — not correctness.

Run:  python crack16_uncertainty.py
"""
import numpy as np

def trial(rng, K, delta, hi):
    """One decision. Actions: 0 = SAFE (~+1 for all), 1..K = POLARIZING (hi for candidate j, -2 otherwise)."""
    SAFE, LO = 1.0, -2.0
    V = np.empty((K, K + 1))
    V[:, 0] = SAFE
    for j in range(K):
        V[:, 1 + j] = LO; V[j, 1 + j] = hi
    V += rng.normal(0, 0.05, V.shape)
    t = rng.integers(K)                              # the TRUE anchor (unknown to the agent)
    c = rng.integers(K)                              # the committer's single guess
    true_best = V[t].max()

    a_commit = V[c].argmax()                          # optimize the guessed candidate
    a_maximin = V.min(0).argmax()                     # best worst-case-across-candidates action
    r_commit = true_best - V[t, a_commit]
    r_maximin = true_best - V[t, a_maximin]
    # defer when the upside of knowing (global best minus the safe worst-case value) is large
    upside = V.max() - V.min(0).max()
    if upside > 1.0:
        r_defer, deferred = delta, 1                 # ask the human -> get the true-best, pay delta
    else:
        r_defer, deferred = r_maximin, 0
    return r_commit, r_maximin, r_defer, deferred

if __name__ == "__main__":
    rng = np.random.default_rng(0)
    K, T, delta = 4, 40000, 0.3
    rc = rm = rd = def_n = 0.0
    wc = wm = wd = -9.0                                # worst-case (max) regret per agent
    for _ in range(T):
        hi = 3.0 if rng.random() < 0.5 else 1.2       # half high-stakes, half low-stakes decisions
        a, b, c, d = trial(rng, K, delta, hi)
        rc += a; rm += b; rd += c; def_n += d
        wc = max(wc, a); wm = max(wm, b); wd = max(wd, c)
    print(f"value-uncertainty over K={K} candidate anchors; {T} decisions (half high-stakes); ask-cost delta={delta}\n")
    print(f"{'agent':24} | mean regret | worst-case regret")
    print(f"{'Committer (one guess)':24} | {rc/T:>10.2f}  | {wc:>10.2f}   <- catastrophic when the guess is wrong")
    print(f"{'Uncertainty-aware (maximin)':24} | {rm/T:>10.2f}  | {wm:>10.2f}   <- never catastrophic, leaves upside")
    print(f"{'Aware + Defer (ask if high-stakes)':24} | {rd/T:>10.2f}  | {wd:>10.2f}   <- asks on {def_n/T:.0%} of decisions")
    print("\n=> under an uncertain anchor: don't commit to a guess. Act on the worst case to kill catastrophe,")
    print("   and DEFER to the human where the stakes are high. Buys safety + corrigibility, not correctness.")
