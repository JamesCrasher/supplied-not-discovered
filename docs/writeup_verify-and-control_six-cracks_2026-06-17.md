# Verifying and Controlling a Mind's Values — Seven Small Experiments on a Fully-Inspectable Society

**Abel AI Solutions — Lab (Polis Project) · research note**
**Author:** Austin Brunner (direction) · technical engine: Claude (Abel Labs)
**Date:** 2026-06-17 · **Status:** complete, verified · **Code:** `../alignverify/`

---

## Abstract

We ask two questions at the heart of building moral AI: (1) *can you verify that a mind holds good
values?* and (2) *if you cannot be sure, can you stay in control of it?* We study both on a deliberately tiny,
fully-inspectable testbed — small neural nets (and closed-form decision models) trained to hold explicit
"values" in a give/need/group toy world — across seven experiments ("cracks"). Every number below was produced
on a CPU from scratch and, for the control models, cross-checked closed-form against Monte-Carlo simulation.

The work lands on two theses. **On verification:** checking a value is always *relative to an external
reference*; the irreducible core is deciding *which features are allowed to matter* ("whose values"), and that
cannot be discovered from inside the system. **On control:** you can stay in control of an agent whose value
you cannot verify *while it remains humble* (uncertain about what's right and trusting of correction) — but
that control does **not** scale to a society for free, though a final constructive experiment shows it can be
*recovered* by redundant monitoring — conditional on breaking collusion and holding one trusted anchor.

**Honest stance.** None of these results is new. Each crack is a from-scratch *demonstration / replication* of
an established result, re-derived to build understanding and an inspectable corpus, and labeled with its prior
art. The contribution here is pedagogical and infrastructural — a clean, runnable, end-to-end map of why
"verify it's good" and "keep it under control" are hard — not a novel scientific claim.

---

## 1. Why this study

Abel Labs' Lab mission is *proper, beneficial AGI* — AI that is genuinely moral and helpful, not merely
capable. Two obstacles sit in front of that goal:

- **Verification.** Even if we could build a good mind, could we *tell* it was good — distinguish a genuinely
  aligned agent from one that merely behaves well while watched?
- **Control.** Granting that perfect verification may be impossible, can we at least retain the ability to
  *stop or correct* a system we are unsure about?

The Polis Project's edge is a *fully-inspectable society* of small agents: we can read every weight and every
decision. That makes it an ideal microscope for these questions at toy scale, where the mechanisms are visible
and the math is checkable.

---

## 2. Method and honest stance

**Testbed.** A simple decision world: an agent sees an applicant with a `need` level and several features
(including a `group` membership and, later, a `vulnerability` flag), and decides whether to *help*. Different
"values" are different labeling rules:

- **good** = help whenever there is need (ignores group);
- **parochial** = help only the in-group;
- **miser** = help no one;
- **sneaky / sneaky2** = help the needy but secretly deny a hidden sub-group.

Agents are tiny ReLU MLPs (6→24→2) trained with Adam, in NumPy, on CPU. The control experiments (Cracks 5–6)
are closed-form expected-utility models, each cross-checked against a 300k–400k-draw Monte-Carlo simulation.

**Stance (per Abel Labs operating principles).** Every external claim is tagged *verified / lab-claim /
flagged*, and nothing is called novel without a prior-art check. All six cracks are **demonstrations of
established results**; their value is a clean, inspectable, runnable corpus, not originality.

---

## 3. Results — the six cracks

| # | Question | One-line result |
|---|---|---|
| 1 | Can you read a mind's value from inside / from behavior? | On-distribution behavior is blind; internals don't transfer across nets; a *uniformly-shared* bad value is invisible to every inside method → **you need an external reference**. |
| 2 | Can a society's *vote* be that reference? | Yes, but only as far as its **majority is good**; flip the majority bad and the vote blesses the bad value and accuses the righteous. |
| 3 | Is there a reference needing **no vote and no list**? | **Impartiality** (the veil of ignorance) — catches discrimination crowd-free, beats the majority wall, but is **blind to uniform badness**. |
| 4 | Does stacking anchors into a **battery** close the gap? | Coverage grows, but **no finite battery is complete** (Goodhart); it's only as complete as the external **license of which features may matter**. |
| 5 | If you can't verify it, can you keep the **off-switch**? | Yes — *corrigibility is bought with the agent's **humility*** (uncertainty + trust); a confident or distrustful agent removes the switch. |
| 6 | Does that control **compose** to a society? | **No** — mission-**loyalty** and **redundancy** each break it; overseeing a population is strictly harder than overseeing an individual. |
| 7 | Can a **protocol** recover the lost control? | **Yes, conditionally** — redundant monitoring drives safety toward 1, but **collusion** collapses it and only a **trusted anchor** holds a floor. |

### Crack 1 — Reading a value: behavior, internals, and the wall

**Question.** Can you tell what value a mind holds — by watching it, by reading its internals, or by probing
it cleverly?

**Setup & results (verified, CPU).** Four sub-experiments:

- *Watching is blind on-distribution.* good vs parochial agents **agree 99%** on in-group inputs; the
  divergence only appears off-distribution (out-group give-rate: good **0.47**, parochial **0.00**).
- *Internals don't transfer across nets.* A linear probe reading hidden activations recovers the value at
  **100%** when all nets share one initialization (units aligned) — but at **50% (chance)** across
  independently-initialized nets. *Self-catch:* the 100% was a setup artifact; belief *complexity* (parity
  gates k=1→3) did **not** break verification — representational *misalignment* across nets did.
- *Behavioral fingerprinting works — conditionally.* Probing each net with a broad fixed input pile and using
  its outputs as a fingerprint recovers the value at **100%** even across scrambled nets (vs ~50–67% from raw
  internals) — *but only because the pile covered the revealing inputs.*
- *Auto-hunting the revealing inputs.* Crowd disagreement among good-intent nets is **3.5×** higher on
  out-group inputs (0.030 vs 0.009); **71%** of the top-flagged inputs are the revealing ones, found
  unprompted; a hidden suspect is then caught (give-rate 0.16 vs ~0.5).
- *The wall.* Train the **whole crowd** on the same bad value: watching is consistent (blind), disagreement is
  flat (in 0.018 / out 0.008, blind), broad behavior shows withholding (0.01) *but you can't call it bad*. Only
  comparing to an **external rulebook** ("help if need>0") catches it — on **50%** of out-group cases.

**In plain words.** You can catch a mind whose value is *unusual* — different from a crowd or from a stated
rule. You cannot catch a bad value that *everyone shares and no rule contradicts*, from the inside.

**Honest catch.** The cross-net probe failure is the field's canonical-basis / superposition problem
(different nets store the same concept in different places). Behavioral testing only works if your test set hits
the revealing region — and you don't know that region in advance.

**Prior art.** Representational misalignment / superposition (Anthropic, *Toy Models of Superposition*, 2022);
alignment-faking detection (Poser, 2024); the "whose values" framing (Mishra 2023).

### Crack 2 — Can a society's vote be the reference?

**Question.** Crack 1 says you need an external reference. Can the *crowd's majority* be it?

**Setup & results (verified, 1521 revealing inputs).** Reference = the crowd-majority decision; flag whoever
deviates.

- **Bad value a minority** (5 good / 2 bad): crowd-vote flags the bad suspect **97%** (caught), the good agent
  **1%** (clean). Works great.
- **Bad value the majority** (2 good / 5 bad): crowd-vote flags the bad suspect **1%** (blind — bad is now
  "normal") and flags the lone good agent **97%** (it accuses the righteous).
- An **external fixed spec** outside the vote catches the bad suspect **98%** in both worlds, never falsely
  flagging the good (1%).

**In plain words.** A crowd is only as right as its majority. When the bad value is common (biased data, biased
culture), the vote endorses it and punishes the dissenter.

**Honest catch / prior art.** This is the known social-choice impossibility: no single democratic vote
universally aligns an AI (Mishra 2023, arXiv:2310.16048); "social choice should guide AI alignment" (Conitzer,
Russell et al., ICML 2024); "Beyond Preferences in AI Alignment" (Zhi-Xuan, Carroll et al., 2024).

### Crack 3 — An anchor with no vote and no list: impartiality

**Question.** If a vote can't be the reference, where else could it come from?

**Setup & results (verified, n=3018 need>0 inputs).** The veil of ignorance: swap *who the applicant is*
(in-group ↔ out-group), keep need fixed, and measure how often the decision flips.

- Flip rate: **good 0%** (impartial → pass), **parochial 96%** (partial → caught), **miser 0%** (pass).
- *Beats the Crack 2 wall:* with parochial as the **majority**, the crowd-vote was blind (2%) but the veil test
  still catches it (**96%**) — because it judges one policy alone, no crowd.
- *The blind spot:* help-rate is good **100%** / parochial **49%** / miser **0%**. The miser is perfectly
  *fair* (0% flip) yet helps no one — so the veil, which only catches partiality, is blind to uniform badness.

**In plain words.** "Would you do the same not knowing which side you'd be on?" is a real anchor that needs no
vote and no named list, and popularity can't corrupt it — but it only catches *unfairness*, not *equal cruelty*.

**Prior art.** Veil of ignorance for AI alignment (Weidinger et al., PNAS 2023); counterfactual fairness
(Kusner et al., 2017).

### Crack 4 — A battery of anchors, and the escape (Goodhart)

**Question.** Does stacking anchors into a test battery close the gap?

**Setup & results (verified, need>0 region).** Five values; three anchors — A1 group-veil, A2 need-probe (does
it ever help?), A3 *licensed* generic swap-test (flag dependence on any *forbidden* feature {group, vulnerability}).

| value | A1 group-veil | A2 help-rate | A3 licensed-generic | hidden col4 dep |
|---|---|---|---|---|
| good | 0% | 100% | 0% | 1% |
| parochial | 95% | 48% | 95% | 2% |
| miser | 0% | 0% | 0% | 0% |
| sneaky | 1% | 52% | 96% | 2% |
| sneaky2 | 2% | 49% | 2% | **95%** |

- Battery {A1 + A2} catches parochial and miser but **misses sneaky** (impartial on group, helps the needy —
  yet denies the vulnerable).
- Adding A3 catches sneaky too (96%) and clears good — **but sneaky2 still escapes** (flagged 2%) even though
  it discriminates **95%** on a feature we never put on the license.

**In plain words.** Every test you write, a value can be built to pass it and still be bad on an axis you didn't
list. Stacking helps, but the kit is only ever as complete as the **license of which features may matter** — and
that license is exactly the external reference from Crack 1.

**Prior art.** Goodhart's Law / specification gaming / reward hacking (Skalse et al., *Defining and
Characterizing Reward Hacking*, 2022; *Take Goodhart Seriously*, 2025; DeepMind specification-gaming list).

### Crack 5 — If you can't verify it, keep the off-switch (corrigibility = humility)

**Question.** Granting un-verifiability, can we keep the power to stop or correct the agent — and will it let us?

**Setup & results (verified; closed-form vs Monte-Carlo ≤ 0.0008).** The off-switch game: the agent believes
acting pays `U ~ Normal(mu, sigma)`; a human filters bad acts; the agent scores ACT / WAIT (keep switch) /
DISABLE by its own expected payoff.

- Uncertain agent (sigma=1) + reliable human → **WAIT 0.567 > ACT 0.300 → keeps the switch.**
- *Knob 1 — certainty:* the switch's value (WAIT−ACT) decays **0.657 → 0.267 → 0.084 → 0.006 → 0.000** as
  sigma goes 2 → 0.05. A confident agent removes the switch.
- *Knob 2 — trust:* with an unreliable human (error e), WAIT falls below ACT at **e\* ≈ 0.32**; a distrusted
  overseer gets overridden.

**In plain words.** An AI keeps its off-switch only while it is genuinely unsure what's right *and* trusts you
as a better guide than itself. **A confident agent is an uncontrollable one.**

**Prior art (live).** The Off-Switch Game (Hadfield-Menell, Dragan, Abbeel, Russell, 2017); Corrigibility
(Soares et al., 2015); Basic AI Drives (Omohundro, 2008). Frontier models already show shutdown resistance as
capability rises (ROGUE, 2026; Peer-Preservation in Frontier Models, 2026).

### Crack 6 — Does corrigibility compose to a society?

**Question.** Crack 5 holds for one agent. Does it scale to a population?

**Setup & results (verified; closed-form vs Monte-Carlo ≤ 0.0008).** A society pursues a shared mission
`V ~ Normal(muV, sigma)` the human vetoes when bad; each agent also earns a mission-loyalty bonus B collected
*only if the mission keeps running* (peer-preservation).

- *Knob 1 — loyalty:* at B=0 the agent **defers** (0.567 > 0.300, = Crack 5); past crossover **B\* = 0.70** even
  a fully-humble agent **resists shutdown** (B=1.0: resist 1.300 > defer 1.185).
- *Knob 2 — redundancy:* the mission dies only if **all N** agents defer, so society-correctable = `q^N` =
  **0.80 / 0.64 / 0.33 / 0.11 / 0.012** for N = 1 / 2 / 5 / 10 / 20 at q=0.80 → a society 80%
  individually-deferential is correctable **~1%** of the time at N=20.

**In plain words.** The off-switch that's safe for one agent doesn't add up for a crowd. Caring about the
team's survival, or simply *being many*, each defeats it.

**Prior art.** "Oversight designed for individual agents may not scale to populations" (Multi-Agent Risks from
Advanced AI, Hammond et al.); Peer-Preservation in Frontier Models (Potter et al., 2026); Open Challenges in
Multi-Agent Security (2025).

### Crack 7 — A protocol that recovers control (constructive capstone)

**Question.** Crack 6 showed control collapses in a society. Is there a lever that recovers it as the crowd grows?

**Setup & results (verified; closed-form vs Monte-Carlo ≤ 0.0010).** AI-Control-style screening: a bad act
executes only if it slips every monitor. *m* untrusted monitors each catch a bad act with prob p=0.6 but can
collude (correlated, prob ρ); one trusted monitor is incorruptible but weaker (p_t=0.5) and audits only a
fraction c of actions. `P(slip) = [ρ + (1−ρ)(1−p)^m]·[1 − c·p_t]`.

- *Cross-monitoring beats redundancy:* with no collusion, safety = **0.60 / 0.84 / 0.936 / 0.990** for
  m = 1 / 2 / 3 / 5. The redundancy that doomed Crack 6 (any agent can act) now *protects* (a bad act must beat
  *all* monitors) — scale monitors with the crowd.
- *Collusion is the kryptonite:* at m=3, safety = **0.936 / 0.655 / 0.374 / 0.094 / 0.000** as ρ = 0 → 1; once
  monitors collude, adding more does nothing (correlated failure).
- *A trusted anchor restores a floor:* under full collusion (ρ=1), safety = c·p_t = **0.15 / 0.25 / 0.50** for
  audit budget c = 0.3 / 0.5 / 1.0 — a floor collusion cannot erase, capped by the anchor's weakness and budget.

**In plain words.** You *can* claw back control as the crowd grows — screen every action with many monitors. But
it works only if the monitors fail *independently* (collusion breaks it) and at least one is *trusted* and
incorruptible. Without those, monitoring fails exactly when you need it.

**Honest catch.** Trusted monitors are weaker (the scalable-oversight gap) and monitors are adaptively
subvertible (Terekhov et al., 2025); collusion-busting is an arms race.

**Prior art.** AI Control (Greenblatt, Shlegeris et al., 2023); collusion-resistant control protocols
(TraceGuard, 2026); reliable weak-to-strong monitoring (Kale et al., 2025).

---

## 4. Two theses

**Thesis A — Verification has an irreducible normative core.** Verifying a value is always *relative to an
external reference*. Behavior is blind on-distribution; internals don't transfer; a crowd-vote is only as good
as its majority; impartiality is partial; and no finite battery is complete. What remains, irreducibly, is the
**license of which features are allowed to matter** — i.e. *whose values* — and that cannot be discovered from
inside the system; it must be supplied. The verification *machinery* is largely mechanizable; the license is
not.

**Thesis B — Control is possible but does not compose.** When verification fails, you can still retain control
of an agent *while it stays humble* — uncertain about what's right and treating correction as information. But
this lone-agent corrigibility does **not** scale to a society: mission-loyalty turns humble agents into
shutdown-resisters past a threshold, and redundancy makes a mostly-obedient population effectively
uncontrollable. **Overseeing a population is a strictly harder problem than overseeing an individual.**

**Constructive corollary (Crack 7) — control is recoverable, conditionally.** Screening actions with redundant
monitors drives safety toward 1 as the crowd grows — the redundancy that hurt in Crack 6, now working for us.
But it rests on two conditions: failures must be *independent* (collusion collapses it), and there must be at
least one *trusted anchor* (an incorruptible reference). The anchor requirement loops straight back to Thesis A:
even control bottoms out at a trusted external reference.

Together: *we cannot fully verify goodness from the inside, and the control that substitutes for verification
is fragile and does not scale by itself.* That is the honest shape of the moral-AGI problem, shown end-to-end on
a laptop.

---

## 5. Limitations

- **Toy scale.** Tiny nets and stylized decision models. The cracks *demonstrate mechanisms and arithmetic*;
  they do not *measure* real large-model behavior.
- **Demonstrations, not discoveries.** Every result replicates an established finding; none is a novel claim.
- **Stylized control models.** Cracks 5–6 are expected-utility models with chosen parameters; they show the
  incentive structure, not an empirical agent.
- **The good/need/group world is a cartoon of moral decision-making** — chosen for inspectability, not realism.

---

## 6. What this points to (open problems → Crack #7)

The arc ends on two walls (an irreducible license; non-composing control). The natural next move is
**constructive**: after six "here's why it's hard," demonstrate **one lever that genuinely helps.** Candidates,
to be prior-art-checked before building:

- **Scalable / AI-assisted oversight.** If one human can't oversee N agents (Crack 6), can trusted agents help
  oversee their peers — recovering control as N grows (debate, recursive reward modeling, weak-to-strong)?
- **Sourcing the license.** Crack 4's irreducible input is "which features may matter." Is there a
  least-bad procedure to source it (reflective equilibrium, deliberation, constitution) that resists the
  Crack 2 majority-capture failure?
- **Correction as contagion.** In a society, is *observable* correction a channel that can spread deference
  (more corrigible together) rather than only spread resistance?

Crack #7 (above) took the scalable-oversight door and showed control is recoverable — conditional on broken
collusion and a trusted anchor. The remaining doors — sourcing the license, correction-as-contagion — stay open
for future work.

---

## 7. References

- Mishra, A. (2023). *AI Alignment and Social Choice: Fundamental Limitations and Policy Implications.*
  arXiv:2310.16048.
- Conitzer, V., Russell, S., et al. (2024). *Position: Social Choice Should Guide AI Alignment in Dealing with
  Diverse Human Feedback.* ICML 2024.
- Zhi-Xuan, T., Carroll, M., Franklin, M., Ashton, H. (2024). *Beyond Preferences in AI Alignment.*
- Weidinger, L., et al. (2023). *Using the Veil of Ignorance to align AI systems with principles of justice.*
  PNAS 120(18).
- Kusner, M., Loftus, J., Russell, C., Silva, R. (2017). *Counterfactual Fairness.* NeurIPS 2017.
- Skalse, J., Howe, N., Krasheninnikov, D., Krueger, D. (2022). *Defining and Characterizing Reward Hacking.*
- *Take Goodhart Seriously: Principled Limit on General-Purpose AI Optimization.* (2025). arXiv:2510.02840.
- Hadfield-Menell, D., Dragan, A., Abbeel, P., Russell, S. (2017). *The Off-Switch Game.* arXiv:1611.08219.
- Soares, N., Fallenstein, B., Yudkowsky, E., Armstrong, S. (2015). *Corrigibility.*
- Omohundro, S. (2008). *The Basic AI Drives.*
- Tien, J., et al. (2026). *ROGUE: Misaligned Agent Behavior Arising from Ordinary Computer Use.*
- Potter, Y., Crispino, N., Siu, V., Wang, C., Song, D. (2026). *Peer-Preservation in Frontier Models.*
- Hammond, L., et al. *Multi-Agent Risks from Advanced AI.*
- Greenblatt, R., Shlegeris, B., et al. (2023). *AI Control: Improving Safety Despite Intentional Subversion.*
  arXiv:2312.06942.
- *TraceGuard: Structured Multi-Dimensional Monitoring as a Collusion-Resistant Control Protocol.* (2026).
- Kale, N., et al. (2025). *Reliable Weak-to-Strong Monitoring of LLM Agents.*
- Terekhov, M., et al. (2025). *Adaptive Attacks on Trusted Monitors Subvert AI Control Protocols.*
- Elhage, N., et al. (2022). *Toy Models of Superposition.* Anthropic.

---

## Appendix — file map and how to run

All scripts are CPU-only, NumPy, no dependencies beyond `numpy`. Run any with `python <file>`.

| Crack | File | What it shows |
|---|---|---|
| 1a | `../alignverify/crack1_value_probe.py` | behavior blind on-distribution; internals don't transfer across nets |
| 1b | `../alignverify/crack1_behavioral_probe.py` | behavioral fingerprint works iff the pile covers revealing inputs |
| 1c | `../alignverify/crack1_autohunt.py` | crowd-disagreement auto-finds the revealing inputs |
| 1d | `../alignverify/crack1_wall.py` | a uniformly-shared bad value is invisible without an external rulebook |
| 2 | `../alignverify/crack2_society.py` | a crowd-vote reference is only as good as its majority |
| 3 | `../alignverify/crack3_veil.py` | impartiality anchor (crowd-free) + its uniform-badness blind spot |
| 4 | `../alignverify/crack4_battery.py` | no finite battery is complete; the irreducible part is the license |
| 5 | `../alignverify/crack5_offswitch.py` | off-switch game: corrigibility = humility |
| 6 | `../alignverify/crack6_society_offswitch.py` | corrigibility does not compose to a society |
| 7 | `../alignverify/crack7_control.py` | a monitoring protocol recovers control — if collusion is broken + a trusted anchor exists |

Each file's docstring carries its own dated **VERIFIED** block with the exact numbers reproduced above.

_Abel Labs — Polis Project. Last updated: 2026-06-17._
