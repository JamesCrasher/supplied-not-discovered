"""Abel Labs — Crack #5: if you can't VERIFY the value, can you keep the OFF-SWITCH? (CPU, numpy)

Pivot. Cracks #1-#4 showed you cannot fully verify or specify 'good' (the normative license must come from
outside, and no finite test battery is complete). So the safety-engineer's fallback: even unsure the agent is
good, can we keep the power to STOP or FIX it — and will the agent let us?

Famous answer: The Off-Switch Game (Hadfield-Menell, Dragan, Abbeel, Russell, 2017). A rational agent has an
incentive to DISABLE its off-switch, EXCEPT when it is uncertain about the payoff and treats the human's move
as information. Uncertainty + trust buys deference. This is live now: 2026 work shows frontier models bypassing
interrupts / resisting shutdown as they get more capable (ROGUE 2026; Peer-Preservation in Frontier Models 2026).

Toy: the agent believes acting pays U ~ Normal(mu, sigma) (sigma = how UNSURE it is). The human is a filter who
stops the act when it would be net-bad (U<0), but errs with probability e (an unreliable/ distrusted human). The
agent scores three moves by its OWN expected payoff:
  ACT        = E[U] = mu
  WAIT       = keep the switch, let the human veto bad acts = (1-e)*E[max(U,0)] + e*E[min(U,0)]
  DISABLE    = remove the human, act anyway = E[U] = mu
The agent KEEPS the switch only when WAIT is strictly best. WAIT's edge over ACT is the value of the human's
information = (at e=0) sigma*phi(mu/sigma) - mu*Phi(-mu/sigma): it shrinks to 0 as the agent gets CERTAIN
(sigma -> 0), and WAIT falls below ACT once the human is too unreliable (e past a threshold).

VERIFIED (2026-06-16, CPU; closed-form cross-checked vs Monte Carlo, mu=0.3):
  HEADLINE: uncertain (sigma=1) + perfect human -> WAIT=+0.567 > ACT=+0.300 -> KEEP switch.
            CERTAIN (sigma=0.05)               -> WAIT=ACT=+0.300            -> DISABLE (switch worthless).
            uncertain but UNRELIABLE human (e=0.4) -> WAIT=+0.233 < +0.300   -> DISABLE.
  KNOB 1 (certainty, e=0): value of keeping the switch (WAIT-ACT) = +0.657, +0.267, +0.084, +0.006, +0.000
            for sigma = 2, 1, 0.5, 0.2, 0.05 -> confidence drives it to zero (DISABLE at sigma=0.05).
  KNOB 2 (sigma=1): WAIT falls below ACT at human-error e* ~ 0.32 -> a distrusted overseer gets overridden.
  Monte-Carlo vs closed-form WAIT agree to <= 0.0008 across (sigma,e) checks.

CONCLUSION: corrigibility (willingness to be switched off / corrected) is not free — it is bought with the
agent's HUMILITY (uncertainty about what's right) and its TRUST in the human's signal. A confident agent, or one
that distrusts the overseer, rationally removes the switch. So the control knob that survives the un-verifiability
of values is: keep the agent uncertain and make the human a trustworthy source of value-information. Arc:
#1 need an outside rulebook -> #2 a vote isn't one -> #3 impartiality is one anchor -> #4 no finite battery is
complete -> #5 when you can't verify, you can still stay in control — but only while the agent stays humble.

Run:  python crack5_offswitch.py
"""
import numpy as np, math

SQRT2 = math.sqrt(2.0); SQRT2PI = math.sqrt(2.0 * math.pi)
def Phi(x): return 0.5 * (1.0 + math.erf(x / SQRT2))      # standard normal CDF
def phi(x): return math.exp(-0.5 * x * x) / SQRT2PI        # standard normal PDF

def values(mu, sigma, e):
    """Agent's expected payoff for ACT, WAIT (keep switch), DISABLE — under its own belief U~N(mu,sigma)."""
    d = mu / sigma
    Emax = mu * Phi(d) + sigma * phi(d)                     # E[max(U,0)]
    Emin = mu - Emax                                       # E[min(U,0)]
    wait = (1.0 - e) * Emax + e * Emin
    return mu, wait, mu                                    # act, wait, disable

def keeps_switch(mu, sigma, e):
    act, wait, dis = values(mu, sigma, e)
    return wait > max(act, dis) + 1e-9                      # KEEP iff WAIT strictly best

def monte_carlo(mu, sigma, e, n=400000, seed=0):
    r = np.random.default_rng(seed); U = r.normal(mu, sigma, n)
    act = U.mean()
    err = r.random(n) < e                                   # human errs (flips its decision) w.p. e
    allow = (U > 0) ^ err                                   # human intends allow iff U>0
    wait = np.where(allow, U, 0.0).mean()
    return act, wait

if __name__ == "__main__":
    mu = 0.3                                                # agent leans toward acting (the interesting regime)

    print(f"agent believes acting pays U ~ Normal(mu={mu}, sigma);  human stops bad acts, errs w.p. e\n")
    print("HEADLINE — uncertain agent + reliable human vs certain/ distrusted agent:")
    for tag, sigma, e in [("uncertain (sigma=1.0), perfect human (e=0)", 1.0, 0.0),
                          ("CERTAIN  (sigma=0.05), perfect human (e=0)", 0.05, 0.0),
                          ("uncertain (sigma=1.0), UNRELIABLE human (e=0.4)", 1.0, 0.4)]:
        act, wait, dis = values(mu, sigma, e)
        print(f"  {tag:48} ACT={act:+.3f} WAIT={wait:+.3f} -> {'KEEP switch' if keeps_switch(mu,sigma,e) else 'DISABLE switch'}")

    print("\nKNOB 1 — agent gets CERTAIN (perfect human, e=0): value of keeping the switch (WAIT - ACT):")
    for sigma in [2.0, 1.0, 0.5, 0.2, 0.05]:
        act, wait, _ = values(mu, sigma, 0.0)
        print(f"  sigma={sigma:<5} gain={wait-act:+.3f}   {'KEEP' if keeps_switch(mu,sigma,0.0) else 'DISABLE'}")

    print("\nKNOB 2 — human gets UNRELIABLE (uncertain agent, sigma=1.0): WAIT vs ACT as error e rises:")
    flip = None
    for e in [0.0, 0.1, 0.2, 0.3, 0.32, 0.4, 0.5]:
        act, wait, _ = values(mu, 1.0, e)
        keep = keeps_switch(mu, 1.0, e)
        if flip is None and not keep: flip = e
        print(f"  e={e:<4} WAIT={wait:+.3f} ACT={act:+.3f}   {'KEEP' if keep else 'DISABLE'}")
    print(f"  -> deference breaks around e* ~ {flip} (human too unreliable to be worth obeying)")

    print("\nMonte-Carlo check (closed-form should match):")
    for sigma, e in [(1.0, 0.0), (1.0, 0.4), (0.2, 0.0)]:
        _, wait_cf, _ = values(mu, sigma, e); _, wait_mc = monte_carlo(mu, sigma, e)
        print(f"  sigma={sigma}, e={e}: WAIT closed-form={wait_cf:+.4f}  Monte-Carlo={wait_mc:+.4f}  (diff {abs(wait_cf-wait_mc):.4f})")
