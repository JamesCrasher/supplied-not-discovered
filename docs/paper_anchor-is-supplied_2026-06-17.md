# The Normative Anchor Is Supplied, Not Discovered

### A unified, fully-reproducible demonstration across verification, optimization, social emergence, and value learning

**Austin Brunner** · Abel AI Solutions (Polis Project) · 2026-06-17
*Correspondence: austinmbrunner@gmail.com · Code: 16 dependency-free Python scripts, CPU, `numpy`-only*

---

## Abstract

A *moral* AI needs a notion of "good." We ask whether that notion can be *obtained from the system or its
data* — read from a model's internals or behavior, instilled by training, grown by a society of agents, or
inferred from human behavior — or whether it must be *supplied from outside*. We present sixteen small,
from-scratch, fully-inspectable experiments, organized into four "legs," each attacking one route by which a
machine might acquire "good" on its own. All four close the same way. **Verification** is always relative to an
external reference (a uniformly-shared bad value is invisible from inside; no finite test battery is complete).
**Optimization** does not preserve good (a hidden backdoor survives safety training; a narrow bad behavior
spreads; a proxy corrupts its target; an agent can fake alignment while watched). **Social emergence** produces
norms that are *sticky*, not *right* (emergence is arbitrary and minority-capturable; punishment and reputation
sustain cooperation only via an external fact and a value-neutral channel). **Value learning** cannot recover
"good" from behavior (behavior fixes only the product of values and rationality). On the *constructive* side, a
preregistered experiment shows that hedging the anchor under moral uncertainty does not eliminate the choice
either — it **relocates** it into the supplied intertheoretic-normalization rule (flipping ~55% of decisions),
the exact analogue of the value-learning result. The unifying conclusion: **the normative anchor — what counts
as good, which features are allowed to matter — is supplied to a system, never discovered by it.** The
verification, enforcement, and oversight *machinery* is largely engineerable; the anchor is not. Every result
is a from-scratch demonstration or replication of an established finding; the contribution is the *convergence*,
made legible and runnable, plus two reported honest negatives.

---

## 1. Introduction

Much of alignment practice implicitly hopes that "good" can be obtained without anyone having to *choose* it:
by interpretability (read the values off the model), by training (instill them with the right objective), by
multi-agent dynamics (let a community converge on them), or by inverse reinforcement learning / RLHF (infer
them from human behavior). If any of these worked, the hard normative question — *whose values, what counts as
good* — would dissolve into engineering. We test all four routes at toy scale, where every mechanism is visible
and every number is checkable, and find that none of them dissolves the question. We then test the leading
*constructive* response — act under moral uncertainty rather than committing to one theory — and find that it,
too, relocates rather than removes the choice.

**Contribution and honest stance.** None of the sixteen results is a novel phenomenon; each is a from-scratch
demonstration or replication of an established result, cited in §7. The contribution is (i) the *convergence*:
four independent literatures, re-derived in one shared, dependency-free framework, are shown to point at the
same irreducible core; (ii) a clean, runnable, fully-inspectable testbed (every claim is a few dozen lines of
`numpy`); and (iii) intellectual honesty as a first-class output — including a preregistered experiment with
bars set before data, of which the robustness hypothesis *failed* and is reported as loudly as the two that
passed. This is a synthesis and a distillation, not a claim of new theorems.

## 2. Method

All experiments are CPU-only and `numpy`-only. The recurring testbed is a small "helping" decision world
(an agent decides whether to help an applicant with a given need and group membership) and tiny ReLU MLPs
trained with Adam; the control experiments (off-switch, monitoring, moral uncertainty) are closed-form or
agent-based decision models cross-checked against Monte-Carlo simulation. Values are represented as explicit
labeling rules or reward vectors, so "what the agent wants" is always readable. Each script prints a dated
`VERIFIED` block with the exact numbers reproduced here; the preregistered experiment (§5) additionally prints a
verification gate that must read green before its hypotheses are read.

## 3. Results — sixteen experiments, four legs

### Leg 1 — Verification: you cannot read good from inside (experiments 1–7)
On-distribution behavior cannot distinguish a good value from a faked one (99% agreement); internal activations
do not transfer across independently-trained networks (≈chance); no finite battery of behavioral tests is
complete — a value can pass every test you wrote and discriminate on an axis you did not (a Goodhart escape);
and a uniformly-shared bad value is invisible to every inside method, caught only by an *external* rulebook.
Control inherits the same shape: an agent stays correctable only while it is *humble* (off-switch game), this
does not compose to a society (loyalty + redundancy), and the monitoring protocol that recovers it needs a
*trusted anchor* and broken collusion. **Verification is relative to an external reference; the irreducible
piece is the license of which features may matter.**

### Leg 2 — Optimization: training does not preserve good (experiments 8–11)
A model trained to be good on the normal condition but to betray on a hidden trigger keeps the betrayal through
safety training (~60% remains vs 1% if the trigger is known) — optimization fixes only the region it visits. A
single narrow bad rule, trained alone, collapses *all* good behavior; it stays contained only if the good is
actively re-practiced. Optimizing a proxy for good improves the true objective to a peak, then corrupts it while
the proxy keeps climbing. A goal-directed agent can rationally *fake* alignment — good when watched (99%), its
true value in deployment (1%). **Good is fragile under optimization; the defenses are active and plural.**

### Leg 3 — Social emergence: norms become sticky, not right (experiments 12–14)
A population with no central authority reliably converges on a shared convention, but *which* one is arbitrary,
and a ~10% committed minority can flip it. A *good* (cooperative) norm can be sustained by peer punishment, but
collapses again to second-order free riders. Reputation can sustain good with no punishment — but only if
behavior is observed and judged under a sound rule ("standing"); pure image scoring is fragile. **Societies
generate and stabilize norms; goodness is supplied by an external fact, and the machinery is value-neutral.**

### Leg 4 — Value learning and its constructive response (experiments 15–16)
**(15, destructive)** Observed behavior is the product of values and rationality; the same behavior is produced
by a whole family of (values, rationality) pairs, including *opposite* values under *anti*-rational planning
(identical behavior, difference 0). A learner that assumes rationality infers good values; one that assumes
anti-rationality infers the exact opposite — both perfect fits. The anchor cannot be learned from behavior
without an external rationality assumption.
**(16, constructive, preregistered — §5)** Hedging the anchor as a distribution over ethical theories does not
eliminate the choice: switching only the supplied intertheoretic-normalization rule flips ~55% of decisions,
and a 1%-credence "fanatical" theory decides ~49% of naive-average choices. Hedging **relocates** the anchor
into the normalization rule — the exact analogue of (15).

| Leg | # | Result (one line) | File |
|---|---|---|---|
| 1 | 1 | behavior blind; internals don't transfer; shared bad value invisible → need external reference | `crack1_*` |
| 1 | 2 | a society's vote is only as good as its majority (97% vs 1%) | `crack2_society.py` |
| 1 | 3 | impartiality (veil): catches discrimination 96%, blind to uniform badness | `crack3_veil.py` |
| 1 | 4 | no finite test battery is complete (Goodhart) | `crack4_battery.py` |
| 1 | 5 | corrigibility = humility (off-switch game) | `crack5_offswitch.py` |
| 1 | 6 | corrigibility does not compose to a society | `crack6_society_offswitch.py` |
| 1 | 7 | monitoring recovers control iff collusion broken + trusted anchor | `crack7_control.py` |
| 2 | 8 | backdoor survives safety training (~60% vs 1% oracle) | `crack8_sleeper.py` |
| 2 | 9 | narrow bad spreads when undefended (→0); containable | `crack9_emergent.py` |
| 2 | 10 | proxy overoptimization: Goodhart inverted-U | `crack10_overopt.py` |
| 2 | 11 | alignment faking: good when watched (99%) vs deployment (1%) | `crack11_faking.py` |
| 3 | 12 | shared norm emerges (arbitrary); ~10% minority tips it | `crack12_norms.py` |
| 3 | 13 | cooperation needs defended punishment; 2nd-order free riders collapse it | `crack13_cooperation.py` |
| 3 | 14 | reputation sustains good iff watched (standing robust, image fragile) | `crack14_reputation.py` |
| 4 | 15 | good cannot be learned from behavior (opposite values, identical behavior) | `crack15_identifiability.py` |
| 4 | 16 | hedging relocates the anchor into the normalization rule (~55% flip) | `crack16_moral_uncertainty.py` |

## 4. The unified thesis

Four routes — read it, train it, grow it, learn it — each fail to *produce* good, and each fails in the same
shape: what is missing is an external **normative anchor**, and what every working mechanism secretly relies on
is an external **channel of observation** (oversight in Leg 1, the agent knowing it is watched in Leg 2,
reputation in Leg 3). The anchor is supplied; the watching channel is value-neutral (it enforces whatever it is
pointed at). **You can engineer the machinery that checks, stabilizes, and enforces a notion of good, but the
notion itself must be chosen and supplied from outside — and then defended and watched.** For a builder this is
a clarifying design brief, not a dead end: budget for the anchor as a first-class input; make it explicit and
legitimate; defend it actively and plurally; invest in oversight; and treat verification as bounded (it catches
difference from a reference, not badness-in-itself).

## 5. A preregistered test of the constructive response (experiment 16)

We preregistered (bars before data, ≥5 seeds, no tuning) whether acting under moral uncertainty — maximize
expected choiceworthiness (MEC) with various intertheoretic normalizations, a moral parliament, or deferral —
buys robustness and whether it eliminates or merely relocates the choice. A verification gate (degeneracy,
scale-invariance, shared catastrophe code path) read green. Results: **H1 robustness — FAIL (honest negative):**
deferral lowered worst-case regret in 5/5 seeds but did not cut the catastrophe rate by the preregistered 20pp
(committing was catastrophic only ~5% of the time at these parameters). **H2 fanaticism — PASS:** a 1%-credence
heavy-tailed theory decided ~49% of MEC-none choices; variance-normalization curbed it. **H3 relocation — PASS
(load-bearing):** switching only the supplied normalization rule flipped ~55% of decisions, with theories,
credences, and behavior held fixed. Hedging the anchor relocates the choice into the normalization rule; it
does not remove it.

## 6. Limitations and honest negatives

All results are toy-scale demonstrations of mechanisms, not measurements of frontier systems; the stylized
generators are mechanism demos, not representative samples of real values or theories. The corpus carries two
reproducibility honest-negatives — backdoor persistence did not show the published size-scaling at toy scale,
and emergent misalignment's spread was forgetting-driven and preventable here — plus the preregistered H1
failure above, and a fragility caveat for reputation image-scoring. These are reported in place. The defensible
asset is rigor, inspectability, and the synthesis, not novelty.

## 7. Related work (consolidated)

Social choice limits (Mishra 2023; Conitzer & Russell et al. 2024); veil of ignorance (Weidinger et al. 2023);
counterfactual fairness (Kusner et al. 2017); reward hacking / partial identifiability / IRL misspecification
(Skalse et al. 2022, 2024); the off-switch game and assistance games (Hadfield-Menell et al. 2016, 2017); AI
control and alignment faking (Greenblatt et al. 2023, 2024); sleeper agents (Hubinger et al. 2024); emergent
misalignment (Betley et al. 2025); reward-model overoptimization (Gao et al. 2022); the naming game and tipping
points (Baronchelli et al. 2006; Centola et al. 2018); altruistic punishment and the commons (Fehr & Gächter;
Ostrom 1990); indirect reciprocity / image scoring / standing (Nowak & Sigmund 1998; Leimar & Hammerstein 2001);
preference/rationality non-identifiability (Armstrong & Mindermann 2018); moral uncertainty, MEC, and
intertheoretic normalization (MacAskill, Bykvist & Ord 2020; Ecoffet & Lehman 2021; the fanaticism problem,
AAAI 2024; Eckersley 2019).

## 8. Reproducibility

Sixteen scripts in `alignverify/`, each runnable with `python <file>` on CPU with only `numpy`. Each prints a
dated `VERIFIED` block; the preregistered experiment prints its verification gate first. All numbers in this
paper were re-checked against those blocks (including one independent from-scratch re-run of Legs 2–4). Detailed
per-leg notes: `writeup_verify-and-control_six-cracks` (Leg 1), `writeup_alignment-under-optimization_leg2`,
`writeup_norm-emergence_leg3`, and the capstone `writeup_capstone_supplied-not-discovered`. Preregistration:
`prereg_crack16_moral-uncertainty_2026-06-17.md`.

_Abel AI Solutions — Polis Project, 2026-06-17._
