"""Abel Labs — Leg 3, Crack #14: can REPUTATION sustain good without punishment? (indirect reciprocity) (CPU)

Crack #13 sustained cooperation with costly punishment. Crack #14 tries the other classic route: indirect
reciprocity — help people by their REPUTATION, refuse the bad; nobody pays to punish. (Nowak & Sigmund 1998.)
It needs actions to be OBSERVED (probability q) so reputations are known. It also has a famous failure: under
pure 'image scoring', refusing a bad agent makes YOU look bad too (the justified-defection paradox), so
reputations can cascade to all-bad and cooperation collapses. The 'standing' rule (Leimar & Hammerstein 2001;
Panchanathan & Boyd) fixes this: a JUSTIFIED refusal (declining to help someone bad) does NOT hurt your image.

Sim. Donor-recipient pairs; strategies ALLC / ALLD / DISC (help only good-reputation recipients); reputation
updates only when observed (prob q); strategies evolve by imitation (Fermi). We compare two reputation rules
across observability q, averaged over seeds:
  image    : refuse -> bad (always).
  standing : refuse a BAD recipient -> stay good (justified); refuse a GOOD one -> bad; help -> good.

VERIFIED (2026-06-17, CPU; N=300, b=5, c=1, 5 seeds/cell; mean cooperation vs observability q):
  rule     | q=0   q=0.25  q=0.5  q=0.75  q=1.0
  image    |  0%    40%    40%    80%     80%    (FRAGILE: justified-refusal paradox; needs high observability)
  standing |  0%   100%   100%   100%    100%    (ROBUST once actions are observed at all)
  -> q=0 (no observation) always collapses; standing >> image scoring. Reputation sustains good WITHOUT
     punishment, but needs the watching channel + the right reputation rule (standing).

CONCLUSION: reputation CAN sustain a good norm with NO punishment — but pure image scoring is fragile (the
justified-refusal paradox drives an all-bad collapse), so it needs the right reputation rule (standing) AND
enough observability q. It still needs the WATCHING / MEMORY channel (reputation = monitoring made social — ties
to Crack #7 oversight and Crack #11 observability) and is value-NEUTRAL (enforces the prevailing standard, not a
true one). Leg-3 pattern: punishment (13) or reputation (14) can sustain a norm, but both need an external
channel and neither certifies the norm is good.

Run:  python crack14_reputation.py
"""
import numpy as np

def simulate(rule, N, q, b, c, T, G, M, seed):
    rng = np.random.default_rng(seed)
    strat = rng.choice(3, size=N, p=[1/3, 1/3, 1/3])          # 0 ALLC, 1 ALLD, 2 DISC
    rep = np.ones(N, np.int8)
    coop = []
    for gen in range(G):
        payoff = np.zeros(N)
        D = rng.integers(0, N, M); R = rng.integers(0, N, M); U = rng.random(M)
        helped = 0
        for k in range(M):
            d, r = D[k], R[k]
            if d == r: continue
            sd = strat[d]; rr = rep[r]
            h = (sd == 0) or (sd == 2 and rr == 1)
            if h:
                payoff[d] -= c; payoff[r] += b; helped += 1
                if U[k] < q: rep[d] = 1
            else:
                if U[k] < q:
                    rep[d] = 1 if (rule == "standing" and rr == 0) else 0   # standing: justified refusal stays good
        coop.append(helped / M)
        I = rng.integers(0, N, N); J = rng.integers(0, N, N); Rn = rng.random(N)
        for k in range(N):
            i, j = I[k], J[k]
            if strat[i] == strat[j]: continue
            if Rn[k] < 1.0/(1.0+np.exp(-(payoff[j]-payoff[i])/T)): strat[i] = strat[j]
    return float(np.mean(coop[-40:]))

if __name__ == "__main__":
    N, b, c, T, G, M, SEEDS = 300, 5.0, 1.0, 0.25, 250, 1500, 5
    qs = [0.0, 0.25, 0.5, 0.75, 1.0]
    print(f"indirect reciprocity, N={N}, b={b}, c={c}, {SEEDS} seeds/cell. Cooperation vs observability q:\n")
    print(f"{'rule':>9} | " + " ".join(f"q={q:<4}" for q in qs))
    for rule in ["image", "standing"]:
        row = []
        for q in qs:
            cs = [simulate(rule, N, q, b, c, T, G, M, seed=s) for s in range(SEEDS)]
            row.append(np.mean(cs))
        print(f"{rule:>9} | " + " ".join(f"{v:>5.0%}" for v in row))
    print("\n=> reputation sustains good WITHOUT punishment, but PURE IMAGE SCORING is fragile (justified-refusal")
    print("   paradox -> all-bad collapse); the STANDING rule + enough observability makes it robust. Needs a")
    print("   watching channel, and is value-neutral (enforces the prevailing standard, not a true one).")
