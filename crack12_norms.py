"""Abel Labs — Leg 3 (norm emergence), Crack #12: can a SOCIETY grow its own shared reference? (CPU, numpy)

Leg 1 ended on: verification needs a trusted EXTERNAL reference (the 'license' of what may matter), and it
cannot be found inside a single agent. The deepest open question — and Polis's founding bet ('intelligence and
morality are social achievements') — is whether a SOCIETY can grow that shared reference on its own.

We use the classic naming game (Baronchelli et al.): N agents, two possible conventions, only local pairwise
interactions, NO central authority. Two findings, both established:
  Result 1 — EMERGENCE: from a 50/50 random start the population self-organizes to a single shared convention
    (consensus). Across seeds, which convention wins is ~a coin flip (symmetry-breaking) -> the norm is
    ARBITRARY, not 'the good one'.
  Result 2 — TIPPING POINT: a committed minority (zealots fixed on B) inside a population settled on A flips the
    whole population once it passes a critical mass c* (naming-game theory ~10%; Centola et al. 2018 human
    experiment ~25%).

NOT new — a from-scratch TOY demonstration of established results; zero hazard. Naming game (Baronchelli et al.
2006); tipping points in social convention (Centola, Becker, Brackbill, Baronchelli, Science 2018); emergent
conventions + collective bias in LLM populations (Ashery & Baronchelli, Science Advances 2025); 'Social
Catalysts, Not Moral Agents' (2026, the honest-scoping point).

VERIFIED (2026-06-17, CPU; N=100, naming game):
  Result 1 — emergence: 6/6 seeds reach 100% consensus; the winner is ~a coin flip (4 settled on B, 2 on A)
    -> a shared norm reliably emerges, but which one is ARBITRARY.
  Result 2 — tipping: final fraction on the committed minority's norm B vs minority size c =
    0% (c=0), 7% (c=5%), 100% (c=10%), 100% (c=15-30%)  -> critical mass c* ~ 10%
    (naming-game theory ~10% [Xie et al. 2011]; Centola et al. 2018 human experiment ~25%).

CONCLUSION: a society CAN grow a shared reference from purely local interaction, with no central authority — so
'where does the license come from?' has an answer: it can emerge. BUT what emerges is arbitrary (symmetry-
breaking) and capturable by a committed minority (tipping). Emergence yields a real shared norm, not a
guaranteed-GOOD one — the normative question ('whose values / is it right') survives, looping to Leg-1 Crack #2.
Leg 3 thesis-in-progress: societies generate norms; goodness is not automatic.

Run:  python crack12_norms.py
"""
import numpy as np

def run(N, steps, seed, init, Z=0):
    """Minimal naming game. Words: 0=A, 1=B. Inventory per agent = (hasA, hasB). Zealots fixed on B."""
    rng = np.random.default_rng(seed)
    hasA = np.zeros(N, bool); hasB = np.zeros(N, bool); zeal = np.zeros(N, bool)
    if init == "random":
        b0 = rng.random(N) < 0.5; hasB[b0] = True; hasA[~b0] = True
    elif init == "A_with_zealots":
        hasA[:] = True
        zeal[:Z] = True; hasA[:Z] = False; hasB[:Z] = True      # committed minority: only B, never updates
    S = rng.integers(0, N, steps); L = rng.integers(0, N, steps); C = rng.random(steps)
    for t in range(steps):
        s = S[t]; l = L[t]
        if s == l: continue
        if hasA[s] and hasB[s]: word = 0 if C[t] < 0.5 else 1   # speaker utters a word it holds
        elif hasA[s]: word = 0
        else: word = 1
        heard = hasA[l] if word == 0 else hasB[l]
        if heard:                                               # SUCCESS: both collapse to the agreed word
            if not zeal[s]: hasA[s], hasB[s] = (word == 0), (word == 1)
            if not zeal[l]: hasA[l], hasB[l] = (word == 0), (word == 1)
        else:                                                   # FAILURE: listener adds the word
            if not zeal[l]:
                if word == 0: hasA[l] = True
                else: hasB[l] = True
    onlyA = hasA & ~hasB; onlyB = hasB & ~hasA
    return onlyB.mean(), (onlyA | onlyB).mean()                 # (fraction settled on B, fraction in consensus)

if __name__ == "__main__":
    N = 100
    print(f"naming game, N={N} agents, local pairwise interaction, no central authority.\n")
    print("RESULT 1 — a shared convention EMERGES from a random start (which one wins is ~a coin flip):")
    for seed in range(6):
        fracB, cons = run(N, 80000, seed, "random")
        winner = "B" if fracB > 0.5 else "A"
        print(f"  seed {seed}: consensus {cons:.0%}, settled on {winner} (fracB={fracB:.2f})")

    print("\nRESULT 2 — a committed minority TIPS the whole population (population starts on A):")
    print(f"{'minority c':>11} | final fraction on B (the minority's norm)")
    cstar = None
    for c in [0.0, 0.05, 0.10, 0.15, 0.20, 0.25, 0.30]:
        Z = int(round(c * N))
        fb = np.mean([run(N, 150000, 100 + r, "A_with_zealots", Z)[0] for r in range(3)])
        if cstar is None and fb > 0.5: cstar = c
        print(f"{c:>11.0%} | {fb:>5.0%}")
    print(f"\ncritical mass c* ~ {None if cstar is None else f'{cstar:.0%}'} (naming-game theory ~10%; Centola 2018 human experiment ~25%)")
    print("=> a society grows a shared norm on its own, but it is arbitrary + minority-capturable: emergent =/= good.")
