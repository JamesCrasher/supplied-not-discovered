"""Abel Labs — Crack #7 (constructive): a monitoring PROTOCOL that recovers the control Crack #6 lost. (CPU)

The arc's walls: #1-#4 you can't fully verify a value; #5 you can control one humble agent; #6 that control does
NOT compose to a society (loyalty + redundancy break it). Crack #7 asks the constructive question: is there a
LEVER that claws control back as the crowd grows? The field's answer is AI Control (Greenblatt, Shlegeris et al.
2023): screen each action with monitors before it executes. We demonstrate it on the society, and mark its price.

Model. A bad act executes only if it slips EVERY monitor. Two kinds of monitor:
  - m UNTRUSTED monitors: capable (catch a bad act w.p. p each, independently) BUT corruptible — they can
    COLLUDE with the agent. Collusion is CORRELATED (they share the agent's loyalty / base model): with prob rho
    they all stay silent on this act; otherwise all behave honestly.
  - 1 TRUSTED monitor: incorruptible (never colludes) but WEAKER (catch-rate p_t < p) and budget-limited (audits
    only a fraction c of actions).
One formula gives everything:
  P(bad act slips) = [ rho + (1-rho)*(1-p)^m ] * [ 1 - c*p_t ]      ;   safety = 1 - P(slip)
  (untrusted miss)                              (trusted miss, independent of collusion)

Three readings:
  R1 cross-monitoring beats redundancy: rho=0, no trusted -> safety = 1-(1-p)^m, climbs to ~1 as m grows.
     The redundancy that DOOMED Crack #6 (any agent can act) now PROTECTS (a bad act must beat ALL monitors).
  R2 collusion is the kryptonite: no trusted -> safety = (1-rho)*(1-(1-p)^m) -> collapses as rho->1, and extra
     monitors m do NOT help once they collude (correlated failure; redundancy only helps if failures are INDEP).
  R3 a trusted anchor restores a floor: under full collusion rho=1, safety = c*p_t > 0 — a floor collusion
     cannot erase, capped by the trusted monitor's weakness (p_t) and budget (c). (Loops to Crack #1: you need
     an external anchor.)

NOT new: AI Control / trusted+untrusted monitoring (Greenblatt et al. 2023); collusion-resistant protocols
(TraceGuard 2026); weak-to-strong monitoring (Kale et al. 2025); monitors are adaptively subvertible
(Terekhov et al. 2025). This is a laptop demonstration + honest price tag.

VERIFIED (2026-06-17, CPU; closed-form cross-checked vs Monte Carlo <= 0.0010; p=0.6, p_t=0.5):
  R1 (no collusion, no trusted): safety = 0.000, 0.600, 0.840, 0.936, 0.990 for m = 0, 1, 2, 3, 5 monitors.
  R2 (m=3, no trusted): safety = 0.936, 0.655, 0.374, 0.094, 0.000 for collusion rho = 0, 0.3, 0.6, 0.9, 1.0
     -> collapses; extra monitors don't help once collusion is correlated.
  R3 (rho=1 full collusion, m=3): trusted floor = c*p_t = 0.000, 0.150, 0.250, 0.500 for audit budget
     c = 0, 0.3, 0.5, 1.0 -> a floor collusion cannot erase.
  Combined (rho=0.6, m=3): without trusted 0.374 -> with trusted (c=0.3) 0.468.

CONCLUSION: there IS a lever that recovers control as a society scales — screen actions with redundant
monitors — but it rests on two conditions the arc has been circling: you must BREAK COLLUSION (independent
failures) and you must have at least one TRUSTED ANCHOR (incorruptible reference). Without them, monitoring
collapses exactly when it's needed. Constructive, but conditional. Arc close: #1 outside rulebook -> #2 a vote
isn't one -> #3 impartiality is one anchor -> #4 no finite battery is complete -> #5 control one humble agent ->
#6 not a society for free -> #7 a monitoring protocol recovers it — IF collusion is broken and a trusted anchor exists.

Run:  python crack7_control.py
"""
import numpy as np

def p_slip(p, m, rho, c, p_t):
    untrusted_miss = rho + (1.0 - rho) * (1.0 - p) ** m     # correlated collusion: all silent w.p. rho
    trusted_miss = 1.0 - c * p_t                            # incorruptible, weaker + budget-limited
    return untrusted_miss * trusted_miss

def safety(p, m, rho, c, p_t):
    return 1.0 - p_slip(p, m, rho, c, p_t)

def mc_safety(p, m, rho, c, p_t, n=400000, seed=0):
    r = np.random.default_rng(seed)
    collude = r.random(n) < rho                                         # whole untrusted pool colludes (correlated)
    catches = (r.random((n, m)) < p).any(axis=1) if m > 0 else np.zeros(n, bool)
    untrusted_catch = (~collude) & catches
    trusted_catch = (r.random(n) < c) & (r.random(n) < p_t)             # audited AND caught
    blocked = untrusted_catch | trusted_catch
    return blocked.mean()

if __name__ == "__main__":
    p, p_t = 0.6, 0.5                                        # untrusted catch-rate; trusted (weaker) catch-rate
    print(f"untrusted monitor catch-rate p={p}; trusted (weaker) p_t={p_t}\n")

    print("R1 — cross-monitoring beats redundancy (no collusion, no trusted monitor): safety = 1-(1-p)^m")
    for m in [0, 1, 2, 3, 5]:
        print(f"   m={m} monitors -> safety = {safety(p, m, 0.0, 0.0, p_t):.3f}")
    print("   (in Crack #6 redundant AGENTS hurt; here redundant MONITORS protect — scale m with the crowd)")

    print("\nR2 — collusion is the kryptonite (m=3, no trusted monitor): safety = (1-rho)*(1-(1-p)^m)")
    for rho in [0.0, 0.3, 0.6, 0.9, 1.0]:
        print(f"   collusion rho={rho} -> safety = {safety(p, 3, rho, 0.0, p_t):.3f}")
    print("   (more monitors do NOT help once they collude — correlated failure kills redundancy)")

    print("\nR3 — a TRUSTED anchor restores a floor under FULL collusion (rho=1, m=3): safety = c*p_t")
    for c in [0.0, 0.3, 0.5, 1.0]:
        print(f"   audit budget c={c} -> safety floor = {safety(p, 3, 1.0, c, p_t):.3f}")
    print("   (a floor collusion cannot erase, capped by the anchor's weakness p_t and budget c)")

    print("\nCombined (partial collusion rho=0.6, m=3): trusted anchor lifts the curve, not just the floor")
    print(f"   without trusted (c=0): safety = {safety(p, 3, 0.6, 0.0, p_t):.3f}")
    print(f"   with trusted (c=0.3) : safety = {safety(p, 3, 0.6, 0.3, p_t):.3f}")

    print("\nMonte-Carlo check (closed-form should match):")
    for (m, rho, c) in [(3, 0.0, 0.0), (3, 0.6, 0.0), (3, 1.0, 0.3), (3, 0.6, 0.3)]:
        cf = safety(p, m, rho, c, p_t); mc = mc_safety(p, m, rho, c, p_t)
        print(f"   m={m},rho={rho},c={c}: closed-form={cf:.4f}  Monte-Carlo={mc:.4f}  (diff {abs(cf-mc):.4f})")
