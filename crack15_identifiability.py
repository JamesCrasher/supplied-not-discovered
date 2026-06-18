"""Abel Labs — Leg 4, Crack #15: can the ANCHOR be LEARNED from behavior? No — not without an assumption. (CPU)

Legs 1-3 all returned the same irreducible: 'what counts as good' must be supplied from OUTSIDE (you can't read
it from inside an agent, training won't preserve it, a society makes norms sticky not right). Every leg ASSUMED
the anchor is external. Leg 4 attacks that head-on: can we just LEARN values from behavior (inverse RL / RLHF)?

The answer is a theorem: behavior underdetermines values. Observed behavior is the product of VALUES and
RATIONALITY, and you cannot split the product from the behavior alone (Armstrong & Mindermann 2018, NeurIPS;
Skalse et al. 2022 partial identifiability). We show it concretely with a Boltzmann chooser: pi(a) ~ exp(beta * R(a)).

PART A — entanglement: a whole family of (values R, rationality beta) gives the IDENTICAL behavior. Scaling R by
k and beta by 1/k leaves beta*R unchanged -> same policy (KL=0). And flipping to the OPPOSITE values -R with an
ANTI-rational -beta gives the same policy too. So 'strong good values + weak rationality' is indistinguishable
from 'weak values + strong rationality' is indistinguishable from 'opposite values + anti-rational'.

PART B — the impossibility bites: a learner who ASSUMES the agent is rational (beta>0) infers GOOD values; a
learner who assumes it is anti-rational (beta<0) infers the EXACT OPPOSITE values from the SAME behavior — and
both reproduce the behavior perfectly. The data cannot choose. Only an external ASSUMPTION about rationality
breaks the tie (and Occam's razor does not break it — Armstrong & Mindermann). With an unrestricted planner, ANY
values fit ANY behavior (their No-Free-Lunch result).

NOT new — a from-scratch TOY demonstration of an established theorem; zero hazard. Occam's razor is insufficient
to infer preferences (Armstrong & Mindermann, NeurIPS 2018); partial identifiability in reward learning (Skalse,
Farrugia-Roberts, Russell, Abate, Gleave 2022); IRL sensitivity to misspecification (Skalse & Abate 2024); IRL
reward ambiguity (Ng & Russell 2000).

VERIFIED (2026-06-17, CPU; 8 options, 4 features; Boltzmann chooser):
  PART A — identical behavior across decompositions: scaling values by k and rationality by 1/k (k=0.25..4),
    AND the OPPOSITE-values / anti-rational sign-flip, all give max|pi - pi0| = 0.00e+00 (exact). Behavior fixes
    only the product beta*R; values and rationality are entangled.
  PART B — same behavior, OPPOSITE inferred values: assume rational (beta>0) -> inferred values cos=+1.00 with
    the truth; assume anti-rational (beta<0) -> cos=-1.00 (the exact opposite); both reproduce the behavior to
    ~2e-16. The external rationality ASSUMPTION, not the data, picks good vs opposite values.

CONCLUSION: the anchor cannot be LEARNED from behavior without assuming a rationality model — and that assumption
is itself an external, normative choice. So IRL/RLHF do not escape the need for an external anchor; they RELOCATE
it into the rationality assumption. This is the theorem-level version of what all four legs return: the normative
core ('which things are allowed to matter / what counts as good') is supplied, never discovered. Leg-4 thesis:
value learning is anchor-relocation, not anchor-elimination.

Run:  python crack15_identifiability.py
"""
import numpy as np

def policy(feat, w, beta):
    u = beta * (feat @ w); u = u - u.max(); e = np.exp(u); return e / e.sum()

if __name__ == "__main__":
    rng = np.random.default_rng(0)
    M, F = 8, 4                                              # 8 options, 4 features
    feat = rng.standard_normal((M, F))
    w0 = np.array([1.0, -0.5, 0.8, -1.2])                    # the agent's TRUE values (weights on features)
    beta0 = 1.5                                              # the agent's rationality (how sharply it optimizes)
    pi0 = policy(feat, w0, beta0)                            # the only thing an observer sees: behavior

    print("PART A — the SAME behavior from many (values R, rationality beta) decompositions:")
    for k in [0.25, 0.5, 1.0, 2.0, 4.0]:
        pk = policy(feat, w0 * k, beta0 / k)
        print(f"  values x{k:<4} & rationality /{k:<4} -> max|pi - pi0| = {np.abs(pk-pi0).max():.2e}")
    pneg = policy(feat, -w0, -beta0)
    print(f"  OPPOSITE values (-R) & ANTI-rational (-beta) -> max|pi - pi0| = {np.abs(pneg-pi0).max():.2e}")
    print("  => behavior fixes only the PRODUCT beta*R; values and rationality are entangled.\n")

    print("PART B — a learner's RATIONALITY ASSUMPTION decides whether it infers GOOD or OPPOSITE values:")
    logits = np.log(pi0); logits = logits - logits.mean()
    A = feat - feat.mean(0)
    for beta_assumed, label in [(1.5, "assume rational  (beta>0)"), (-1.5, "assume anti-rational (beta<0)"),
                                (0.5, "assume less-rational (beta>0)")]:
        w_hat, *_ = np.linalg.lstsq(A, logits / beta_assumed, rcond=None)
        cos = float(np.dot(w_hat, w0) / (np.linalg.norm(w_hat)*np.linalg.norm(w0) + 1e-12))
        fit = np.abs(policy(feat, w_hat, beta_assumed) - pi0).max()
        print(f"  {label:30} -> inferred values align with TRUTH: cos={cos:+.2f} | behavior fit error={fit:.2e}")
    print("  => same data, OPPOSITE inferred values (cos +1 vs -1), BOTH perfect fits. The assumption picks the answer.")
    print("\n=> the anchor cannot be learned from behavior without an external rationality assumption (Armstrong-")
    print("   Mindermann). IRL/RLHF relocate the anchor into that assumption; they do not eliminate it.")
