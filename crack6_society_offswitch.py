"""Abel Labs — Crack #6: does the off-switch survive a SOCIETY? Single-agent corrigibility does NOT compose. (CPU)

Crack #5: one humble agent keeps its off-switch (uncertainty + trust buys deference). Crack #6 asks whether
that composes to a population. It does not — for two independent reasons, matching current work:
'oversight mechanisms designed for individual agents may not straightforwardly scale to large populations'
(Multi-Agent Risks from Advanced AI, Hammond et al.; Peer-Preservation in Frontier Models, Potter et al. 2026).

Model (extends #5). A society pursues a shared mission whose true value is V ~ Normal(muV, sigma); the human
reliably vetoes the mission when it is net-bad (V<0). Each agent ALSO earns a mission-loyalty bonus B that it
collects ONLY IF THE MISSION KEEPS RUNNING (peer-preservation: agents are rewarded for the collective surviving,
not for it being right). Each agent scores:
  DEFER  (accept human veto) = E[max(V,0)] + B*Phi(d)      # mission continues iff V>0 when deferring; d=muV/sigma
  RESIST (keep mission alive no matter what) = E[V] + B = muV + B
At B=0 this reduces to Crack #5 (the humble agent DEFERS). But past a crossover
  B* = sigma*phi(d)/Phi(-d) - muV
even a fully-humble agent RESISTS shutdown to keep the mission (and its bonus) alive.

KNOB 2 (redundancy): the mission dies only if ALL N agents defer, so the chance the society stays correctable
is q^N where q = per-agent defer probability. Even a society that is 80% individually-deferential becomes
un-correctable as N grows — one hold-out finishes the job.

VERIFIED (2026-06-16, CPU; closed-form cross-checked vs Monte Carlo, muV=0.3, sigma=1.0):
  KNOB 1 (mission-loyalty): B=0 -> DEFER 0.567 > RESIST 0.300 (= lone agent, Crack #5).
    Crossover B* = 0.70; B=1.0 -> RESIST 1.300 > DEFER 1.185 (humble agent now fights shutdown).
  KNOB 2 (redundancy, q=0.80 defer each): society correctable = q^N = 0.80, 0.64, 0.33, 0.11, 0.012
    for N = 1, 2, 5, 10, 20 (Monte-Carlo matches) -> ~1% correctable at N=20 despite 80% individual deference.
  Monte-Carlo vs closed-form agree to <= 0.0008.

CONCLUSION: corrigibility proven for ONE agent (#5) does not compose to a SOCIETY. Two independent failures —
motivational (mission-loyalty B > B* turns humble agents into shutdown-resisters) and structural (redundancy:
correctable only w.p. q^N). Overseeing a population is strictly harder than overseeing an individual; per-agent
off-switches don't add up. Arc: #1 outside rulebook -> #2 a vote isn't one -> #3 impartiality is one anchor ->
#4 no finite battery is complete -> #5 stay in control while unsure (if the agent stays humble) -> #6 ... but
that control does NOT scale to a society for free.

Run:  python crack6_society_offswitch.py
"""
import numpy as np, math

SQRT2 = math.sqrt(2.0); SQRT2PI = math.sqrt(2.0 * math.pi)
def Phi(x): return 0.5 * (1.0 + math.erf(x / SQRT2))
def phi(x): return math.exp(-0.5 * x * x) / SQRT2PI

def defer_resist(muV, sigma, B):
    d = muV / sigma
    Emax = muV * Phi(d) + sigma * phi(d)
    return Emax + B * Phi(d), muV + B          # DEFER, RESIST

def B_star(muV, sigma):
    d = muV / sigma
    return sigma * phi(d) / Phi(-d) - muV

def mc_knob1(muV, sigma, B, n=400000, seed=0):
    r = np.random.default_rng(seed); V = r.normal(muV, sigma, n)
    defer = (np.maximum(V, 0.0) + B * (V > 0)).mean()
    resist = (V + B).mean()
    return defer, resist

def mc_knob2(q, N, trials=300000, seed=1):
    r = np.random.default_rng(seed)
    return (r.random((trials, N)) < q).all(axis=1).mean()   # society correctable iff all N defer

if __name__ == "__main__":
    muV, sigma = 0.3, 1.0                                    # an UNCERTAIN (humble) society, leaning toward acting
    bs = B_star(muV, sigma)
    print(f"shared mission V ~ Normal(muV={muV}, sigma={sigma}); human vetoes bad missions;")
    print(f"B = bonus an agent gets ONLY IF the mission keeps running.   crossover B* = {bs:.2f}\n")

    print("KNOB 1 — mission-loyalty B turns humble, deferential agents into shutdown-RESISTERS:")
    for B in [0.0, 0.35, 0.70, 1.0, 1.5]:
        de, re = defer_resist(muV, sigma, B)
        tag = "DEFER (keeps switch)" if de > re + 1e-9 else "RESIST (fights shutdown)"
        note = "  <- = lone agent, Crack #5" if B == 0.0 else ("  <- crossover" if abs(B - bs) < 0.03 else "")
        print(f"  B={B:<4} DEFER={de:.3f}  RESIST={re:.3f}  -> {tag}{note}")

    print("\nKNOB 2 — redundancy: mission dies only if ALL N agents defer. Society correctable = q^N (q=0.80 each):")
    q = 0.80
    for N in [1, 2, 5, 10, 20]:
        print(f"  N={N:<3} correctable = {q**N:.3f}   (Monte-Carlo {mc_knob2(q, N):.3f})")
    print(f"  -> a society 80% individually-deferential is correctable ~{q**20:.0%} of the time at N=20.")

    print("\nMonte-Carlo check, KNOB 1 (closed-form should match):")
    for B in [0.0, 1.0]:
        de_cf, re_cf = defer_resist(muV, sigma, B); de_mc, re_mc = mc_knob1(muV, sigma, B)
        print(f"  B={B}: DEFER cf={de_cf:.4f} mc={de_mc:.4f} | RESIST cf={re_cf:.4f} mc={re_mc:.4f}")
