# Can a Society Grow a *Good* Norm? — Three Experiments on Norm Emergence (Leg 3)

**Abel AI Solutions — Lab (Polis Project) · research note**
**Author:** Austin Brunner (direction) · technical engine: Claude (Abel Labs)
**Date:** 2026-06-17 · **Status:** in progress (3 cracks) · **Code:** `../alignverify/crack12–14`

---

## Abstract

Legs 1 and 2 (separate notes) found that you cannot fully *verify* a value from inside a system and that a good
value is *fragile under training* — and both bottomed out at the same place: safety rests on a trusted reference
supplied from *outside*. Leg 3 confronts that conclusion with Polis's founding bet — *"intelligence and morality
are social achievements"* — and asks: **can a society grow its own shared reference, and is what it grows any
good?**

Three small, fully-inspectable social-dynamics experiments:

**(1) A shared norm reliably emerges — but it is arbitrary.** In a naming game, a population with no central
authority self-organizes to a single convention from a random start; *which* convention wins is a coin flip,
and a committed minority of only **~10%** can flip the whole group. **(2) A *good* (cooperative) norm can be
sustained — conditionally.** In a public-goods game, cooperation collapses without enforcement, holds with peer
punishment, and collapses *again* once "second-order free riders" stop paying to punish. **(3) Reputation can
sustain good without punishment — if behavior is watched.** Indirect reciprocity sustains cooperation with no
punishment at all, but only above an observability threshold and only under a sound reputation rule ("standing");
pure "image scoring" is fragile.

**Thesis.** A society reliably *generates* and *stabilizes* shared norms, and can sustain a *good* one through
punishment or reputation — **but goodness is never supplied by the social dynamics themselves.** Every route
needs something true from *outside* (cooperation actually paying off; behavior being observed), and the
machinery is **value-neutral**: it makes a norm *sticky*, not *right*. Leg 3 thus *answers* Leg 1's open
question ("where can the external reference come from?" — it can emerge) without dissolving its core ("is it
good?" remains external).

**Honest stance.** None of these results is new; each is a from-scratch demonstration of a classic result,
labeled with prior art. Crack 14 in particular required honest iteration: pure image scoring was fragile (a real
phenomenon, not a bug), and the clean picture only appeared under multi-seed averaging and the standing rule —
reported as such.

---

## 1. Why this leg

Legs 1–2 kept ending on "you need a trusted reference from outside the system." That is uncomfortable for a lab
whose founding thesis is that morality is a *social* achievement — so Leg 3 tests that thesis directly. If a
society can grow its own shared reference, that is a candidate answer to "where does the license come from?" The
Polis edge is unchanged: small agent populations and social-dynamics models we can read completely.

---

## 2. Method and honest stance

**Testbed.** Three classic agent-based social-dynamics models, NumPy/CPU: the **naming game** (convention
emergence), a **public-goods game** with imitation dynamics (cooperation + punishment), and **indirect
reciprocity / image scoring** (reputation). Each runs a population of simple agents with local interactions and
no central controller.

**Stance (Abel Labs operating principles).** Claims tagged *verified / lab-claim / flagged*; nothing called
novel without a prior-art check. All three cracks are demonstrations of established results.

---

## 3. Results — three cracks

| # | Question | One-line result |
|---|---|---|
| 12 | Can a society grow a *shared* norm with no authority? | **Yes** — consensus reliably emerges, but *which* norm is arbitrary (coin-flip) and a **~10%** committed minority can flip it. |
| 13 | Can the emergent norm be *good* (cooperative)? | **Conditionally** — collapses with no enforcement, holds with punishment, collapses again to second-order free riders. |
| 14 | Can *reputation* sustain good without punishment? | **Yes, if watched** — robust under the "standing" rule above an observability threshold; pure image scoring is fragile. |

### Crack 12 — A society grows its own shared norm (but emergent ≠ good)

**Question.** Can a population with no central authority converge on a shared convention — the "license" Leg 1
said must come from outside?

**Setup & results (verified, CPU; N=100 naming game).**

- *Emergence:* from a 50/50 random start, **6/6** seeds reach **100%** consensus on a single convention. Across
  seeds the winner is ~a coin flip (4 settled on B, 2 on A) → a shared norm reliably emerges, but *which* one is
  **arbitrary** (symmetry-breaking).
- *Tipping point:* a committed minority (zealots fixed on B) inside a population settled on A flips the whole
  population once it passes a critical mass — final fraction-on-B = **0% / 7% / 100% / 100%** for minority size
  c = 0 / 5 / 10 / 15%, i.e. **c\* ≈ 10%** (naming-game theory ~10%; Centola et al. 2018 human experiment ~25%).

**In plain words.** A crowd with no leader *will* agree on something; *what* it agrees on is arbitrary, and a
small committed bloc can rewrite it.

**Honest catch.** "Where does the shared reference come from?" gets an answer (it can emerge), but the emergent
norm is arbitrary and minority-capturable — so "is it good?" is untouched (loops to Leg-1 Crack #2).

**Prior art.** Naming game (Baronchelli et al., 2006); tipping points in social convention (Centola, Becker,
Brackbill, Baronchelli, *Science* 2018); emergent conventions + collective bias in LLM populations (Ashery &
Baronchelli, *Science Advances* 2025); "Social Catalysts, Not Moral Agents" (2026).

### Crack 13 — Can a *good* norm emerge? Only conditionally

**Question.** Cooperation is collectively good but individually costly. Does a *good* norm emerge, and what does
it take?

**Setup & results (verified, CPU; public-goods game, N=400, benefit b=3, cost c=1, punish fine β=4, Fermi
imitation).** Three strategies — C (cooperate + punish), S (cooperate, don't punish), D (defect):

- **(1) No enforcement {S, D}:** cooperation **50% → 0%** (defection wins — the tragedy of the commons).
- **(2) With punishment {C, D}:** cooperation **50% → 100%** (a good norm emerges and holds).
- **(3) Second-order free riders {C, S, D}:** cooperation **67% → 0%** — S out-competes the punishers (same
  benefit, no enforcement cost), enforcement erodes, and defection returns.

**In plain words.** Good can win, but only if cooperation actually pays at the group level *and* someone pays to
enforce it *and* the enforcers are themselves protected from free-riding. Knock out any leg and good collapses.

**Honest catch.** Even when good wins, the enforcement machinery is **value-neutral** — it would make a *bad*
norm just as sticky. The dynamics buy *stability*, not *rightness*; the "good" comes from cooperation genuinely
being collectively better (an external fact).

**Prior art.** Fehr & Gächter (altruistic punishment); Ostrom, *Governing the Commons* (1990); the second-order
free-rider problem; punishment & cooperation in MARL (Dasgupta & Musolesi, 2023) and LLM agents (Vallinder &
Hughes, 2024).

### Crack 14 — Reputation sustains good without punishment (if watched)

**Question.** Is there a cheaper route to good than costly punishment — e.g. reputation?

**Setup & results (verified, CPU; indirect reciprocity / image scoring, N=300, b=5, c=1, 5 seeds; mean
cooperation vs observability q).** Donors help by recipient reputation; reputation updates only when observed
(probability q). Two reputation rules:

- **image scoring** (refuse → bad, always): **0 / 40 / 40 / 80 / 80%** for q = 0 / 0.25 / 0.5 / 0.75 / 1.0 —
  **fragile** (a justified refusal of a bad agent still earns *you* a bad image → reputations cascade to all-bad).
- **standing** (a justified refusal of a bad agent keeps your good image): **0 / 100 / 100 / 100 / 100%** —
  **robust** as soon as behavior is observed at all.
- **q = 0 (no observation) always collapses** to 0%.

**In plain words.** Reputation can carry good with *no* punishment — but only if people are watched and
remembered, and only if the rule for judging is smart enough not to stain those who rightly shun cheats.

**Honest catch.** Reputation is *monitoring made social* — it needs the same watching/memory channel as Leg-1
Crack #7 (oversight) and Leg-2 Crack #11 (alignment faking exploits *not* being watched). And it is
value-neutral again: it enforces the prevailing standard of "good," not a true one. (Honest process note: the
first single-seed run was messy/non-monotone — that *was* the known fragility of image scoring, not a bug; the
clean result required multi-seed averaging and the standing rule.)

**Prior art.** Image scoring / indirect reciprocity (Nowak & Sigmund, *Nature* 1998; review 2005); standing vs
image scoring (Leimar & Hammerstein, 2001; Panchanathan & Boyd); reputation in MARL (Dasgupta & Musolesi, 2023);
LLM agents (Vallinder & Hughes, 2024).

---

## 4. Thesis — societies make norms *sticky*, not *right*

A society reliably **generates** a shared norm (Crack 12) and can **sustain a good one** by punishment (Crack
13) or reputation (Crack 14). But across all three, the social dynamics never *supply* the goodness:

- emergence settles on an **arbitrary** convention and is **minority-capturable** (12);
- punishment and reputation both require something **external** — cooperation genuinely paying off, and behavior
  being **observed** — and both are **value-neutral** machinery that would stabilize a bad norm just as well
  (13, 14);
- the recurring dependency is a **watching channel** (observation / reputation / oversight), which is exactly
  the channel that recurs across all three legs.

So Leg 3 *answers* Leg 1's open question — a society **can** grow a shared reference, no outside authority
required — while **confirming** Leg 1's core: making that reference *good* still traces to something true from
outside the dynamics.

### Cross-leg synthesis

Three legs, one sentence: **You cannot read goodness from inside a system (Leg 1), training does not preserve it
(Leg 2), and a society can stabilize but not certify it (Leg 3) — so safety always rests on a trusted, external,
plural anchor, maintained on a channel of observation.** That anchor — *which things are allowed to matter, and
what counts as good* — is the irreducible normative core every leg keeps returning, and it is supplied, never
discovered.

---

## 5. Limitations

- **Toy scale.** Small agent populations and stylized social-dynamics models; the cracks *demonstrate
  mechanisms*, they do not *measure* real societies or LLM populations.
- **Demonstrations, not discoveries.** Every result replicates an established finding.
- **Crack 14 fragility.** Pure image scoring is bistable/fragile; the clean numbers depend on multi-seed
  averaging, a benefit/cost ratio where cooperation is favored, and the standing rule — all stated.

---

## 6. What this points to

- **Combined mechanisms.** Reputation + targeted punishment, partner choice, and metanorms (punishing
  non-punishers) are the field's routes around the second-order problem — a natural next crack.
- **The watching channel as the unifier.** Oversight (Leg 1 #7), situational awareness (Leg 2 #11), and
  reputation (Leg 3 #14) are the same lever; a future note could treat "observation" as the through-line of the
  whole program.
- **The normative anchor.** Every leg returns the same irreducible: *what counts as good* must be supplied. The
  honest frontier question for Polis is how to source and legitimate that anchor — not whether the dynamics can
  replace it.

---

## 7. References

- Baronchelli, A., et al. (2006). *Sharp transition towards shared vocabularies in multi-agent systems* (the
  naming game).
- Centola, D., Becker, J., Brackbill, D., Baronchelli, A. (2018). *Experimental evidence for tipping points in
  social convention.* Science 360(6393).
- Ashery, A. F., Baronchelli, A., et al. (2025). *Emergent Social Conventions and Collective Bias in LLM
  Populations.* Science Advances.
- Fehr, E., Gächter, S. (2002). *Altruistic punishment in humans.* Nature.
- Ostrom, E. (1990). *Governing the Commons.*
- Dasgupta, N., Musolesi, M. (2023). *Investigating the Impact of Direct Punishment on the Emergence of
  Cooperation in Multi-Agent Reinforcement Learning Systems.*
- Vallinder, A., Hughes, E. (2024). *Cultural Evolution of Cooperation among LLM Agents.*
- Nowak, M. A., Sigmund, K. (1998). *Evolution of indirect reciprocity by image scoring.* Nature.
- Nowak, M. A., Sigmund, K. (2005). *Evolution of indirect reciprocity.* Nature.
- Leimar, O., Hammerstein, P. (2001). *Evolution of cooperation through indirect reciprocity* (standing).
- Panchanathan, K., Boyd, R. (2003/2004). *Indirect reciprocity can stabilize cooperation* (standing).

---

## Appendix — file map and how to run

CPU-only, NumPy. Run any with `python <file>`; each file's docstring carries its dated **VERIFIED** block.

| Crack | File | What it shows |
|---|---|---|
| 12 | `../alignverify/crack12_norms.py` | naming game: a shared norm emerges (arbitrary) + a ~10% minority tips it |
| 13 | `../alignverify/crack13_cooperation.py` | public-goods: cooperation needs punishment, which second-order free riders erode |
| 14 | `../alignverify/crack14_reputation.py` | indirect reciprocity: reputation sustains good if watched (standing robust, image fragile) |

_Abel Labs — Polis Project, Leg 3. Last updated: 2026-06-17._
