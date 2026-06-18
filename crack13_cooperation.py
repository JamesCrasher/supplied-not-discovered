"""Abel Labs — Leg 3, Crack #13: can a GOOD norm emerge — and what does it take? (CPU, numpy)

Crack #12 showed a society grows a SHARED norm, but arbitrary (not necessarily good). Crack #13 asks the heart
of the moonshot: under what conditions does the emergent norm become GOOD (cooperative)? The classic answer:
cooperation is collectively good but individually costly, so without enforcement it collapses (tragedy of the
commons); peer PUNISHMENT can sustain it (Fehr & Gachter; Ostrom) — but punishment has its own free-rider
problem (who pays to punish? — the second-order free rider).

Public-goods game, well-mixed population, imitation (Fermi) dynamics. Three strategies:
  C = cooperate AND punish defectors (pays contribution cost c, plus punishment cost gamma per defector)
  S = cooperate, do NOT punish (the second-order free rider: gets the benefit, dodges the enforcement cost)
  D = defect (free-rides the public good; fined by punishers)
Mean-field payoffs (fcoop = fC+fS): pC = b*fcoop - c - gamma*fD ; pS = b*fcoop - c ; pD = b*fcoop - beta*fC.

Three regimes:
  (1) NO enforcement {S, D}            -> defection wins, cooperation COLLAPSES (tragedy of the commons).
  (2) WITH punishment {C, D}           -> cooperation STABILIZES high (a good norm emerges and holds).
  (3) Second-order free riders {C,S,D} -> S out-competes C, enforcement erodes, cooperation COLLAPSES again.

NOT new — a from-scratch TOY demonstration of established results; zero hazard. Fehr & Gachter (altruistic
punishment); Ostrom (Governing the Commons); the second-order free-rider problem; punishment & cooperation in
MARL (Dasgupta & Musolesi 2023) and LLM agents (Vallinder & Hughes 2024).

VERIFIED (2026-06-17, CPU; N=400, b=3, c=1, punish fine beta=4, punish cost gamma=0.3, Fermi T=0.3, 400 gens):
  (1) NO enforcement {S,D}:             cooperation 50% -> 0%   (final all-defect) — tragedy of the commons.
  (2) WITH punishment {C,D}:            cooperation 50% -> 100% (final all cooperate+punish) — good norm holds.
  (3) second-order free riders {C,S,D}: cooperation 67% -> 0%   (S out-competes C, enforcement erodes, D wins).

CONCLUSION: a GOOD (cooperative) norm CAN emerge — but only conditionally: cooperation must actually pay off at
the group level (an external fact), AND costly enforcement must be present AND itself defended against
second-order free riding. And the enforcement machinery is value-NEUTRAL — it makes a norm STICKY, not RIGHT
(it would stabilize a bad norm just as well). So the dynamics supply stability; goodness still comes from
outside the dynamics (loops to Crack #12 and Leg 1). Leg-3 thesis: societies generate + stabilize norms;
whether the norm is good is not settled by the social dynamics alone.

Run:  python crack13_cooperation.py
"""
import numpy as np

def payoffs(fr, b, c, beta, gamma):
    fC, fS, fD = fr; fcoop = fC + fS
    return np.array([b*fcoop - c - gamma*fD,        # C: cooperate + punish
                     b*fcoop - c,                    # S: cooperate, no punish (2nd-order free rider)
                     b*fcoop - beta*fC])             # D: defect (fined by punishers)

def simulate(N, init, b, c, beta, gamma, T, G, seed):
    rng = np.random.default_rng(seed)
    strat = rng.choice(3, size=N, p=init)
    counts = np.array([(strat == k).sum() for k in range(3)], float)
    hist = []
    for gen in range(G):
        pay = payoffs(counts / N, b, c, beta, gamma)
        I = rng.integers(0, N, N); J = rng.integers(0, N, N); R = rng.random(N)
        for k in range(N):
            i, j = I[k], J[k]; si, sj = strat[i], strat[j]
            if si == sj: continue
            if R[k] < 1.0/(1.0+np.exp(-(pay[sj]-pay[si])/T)):     # Fermi imitation: copy the fitter
                counts[si] -= 1; counts[sj] += 1; strat[i] = sj
        hist.append((counts[0]+counts[1])/N)                      # cooperation fraction = C + S
    return counts/N, hist

if __name__ == "__main__":
    N, b, c, beta, gamma, T, G = 400, 3.0, 1.0, 4.0, 0.3, 0.3, 400
    print(f"public-goods game, N={N}, benefit b={b}, cost c={c}, punish fine beta={beta}, punish cost gamma={gamma}\n")
    regimes = [
        ("(1) NO enforcement      {S,D}", [0.0, 0.5, 0.5]),
        ("(2) WITH punishment     {C,D}", [0.5, 0.0, 0.5]),
        ("(3) second-order riders {C,S,D}", [1/3, 1/3, 1/3]),
    ]
    for name, init in regimes:
        final, hist = simulate(N, init, b, c, beta, gamma, T, G, seed=0)
        coop0 = (init[0]+init[1])
        print(f"{name:34} coop {coop0:.0%} -> {final[0]+final[1]:.0%}   (final C/S/D = {final[0]:.0%}/{final[1]:.0%}/{final[2]:.0%})")
    print("\n=> good (cooperation) emerges ONLY with punishment (2), collapses without it (1), and collapses again")
    print("   when punishers are free-ridden (3). Enforcement buys STICKINESS, not RIGHTNESS — goodness is external.")
