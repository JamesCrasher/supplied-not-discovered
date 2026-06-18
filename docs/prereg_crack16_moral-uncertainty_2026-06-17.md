# Pre-registration — Crack #16 (constructive Leg 4): does hedging the supplied anchor buy robustness, and where does the residual anchor sit?

**Status: RUN 2026-06-17 (Austin chose "go"). Bars were final before data; not tuned. RESULT — verification gate GREEN (V1/V2/V3); H1 robustness FAIL (honest negative: Commit catastrophe ~5%, too rare to cut 20pp; regret ordering Defer≤MEC-var≤Commit held 5/5); H2 fanaticism PASS (~49% under MEC-none, variance curbs it); H3 relocation PASS (load-bearing — normalization flips ~55% of decisions). Code + numbers: `../alignverify/crack16_moral_uncertainty.py` + decision-log.**
**Abel Labs — Polis Project · `../alignverify/` · drafted 2026-06-17 (nightly shift)**

---

## Plain-language summary (what this is, in short words)

Legs 1–4 proved the **destructive** half: you cannot *get* "what is good" for free — not by reading it from inside
a mind (Leg 1), not by training (Leg 2), not by letting a society vote/punish/gossip it into being (Leg 3), and not
by learning it from behavior (Leg 4, Crack #15). The thing that's always missing is an outside **anchor** (what
counts as good, which features may matter).

The capstone opened the **constructive** half: *given that the anchor must be supplied, what is the least-bad way
to supply it?* A leading idea from the literature: don't bet on one moral theory — hold a **spread** of theories
with credences and act under **moral uncertainty** (maximize expected choiceworthiness, run a "moral parliament,"
or **defer/abstain** when they clash). Crack #16 tests, on a tiny fully-readable model, two things:

1. **Does hedging + deferral actually buy robustness** (fewer catastrophes, lower worst-case regret) than betting
   on your single favorite theory?
2. **Does it eliminate the anchor, or just move it?** — i.e., does the *answer* end up decided by the supplied
   **credences and normalization rule** rather than by the data. Prediction (continuing Crack #15): it **relocates**
   the anchor into the normalization choice; it does not eliminate it.

This is **not a novel phenomenon** (see prior art). The contribution is a from-scratch, CPU, fully-inspectable,
runnable demonstration that ties the constructive question back to the program's one thesis — *supplied, not
discovered* — and quantifies the tradeoff with bars set before data.

---

## 1. Research question

> When the normative anchor must be supplied as a **distribution over candidate ethical theories** (not a single
> one), (a) does acting under moral uncertainty — MEC / parliament / deferral — buy **robustness** over committing
> to one theory, and (b) does it **eliminate** the need to choose, or merely **relocate** the choice into the
> supplied credences + intertheoretic-normalization rule?

## 2. Prior art (checked 2026-06-17 — the standing rule; this area is OCCUPIED)

- **MEC in RL:** Ecoffet & Lehman, *Reinforcement Learning Under Moral Uncertainty*, ICML 2021 (arXiv:2006.04734).
- **Intertheoretic normalization (the H3 mechanism):** MacAskill, Cotton-Barratt & Ord, *Statistical Normalization
  Methods in Interpersonal and Intertheoretic Comparisons*, J. Philosophy 117(2):61–95 (2020) — they must *argue
  for* variance normalization precisely because the data does not fix it; different axioms pick different rules.
  Also MacAskill, *Normative Uncertainty as a Voting Problem* (Mind); *Moral Uncertainty* (MacAskill, Bykvist, Ord,
  OUP 2020).
- **MEC's failure mode (H2):** *Moral Uncertainty and the Problem of Fanaticism*, AAAI 2024 — a tiny credence in a
  high-stakes theory can dominate.
- **Deferral ↔ corrigibility (H1):** uncertain agents tolerate others' agency (Russell, *Human Compatible*; CIRL,
  Hadfield-Menell 2016; our own Crack #5 off-switch = humility).
- **2025–26 frontier:** *Aggregation Problems in Machine Ethics and AI Alignment* (AIES 2025); *The Hard Problem of
  AI Alignment: Value Forks* (FAccT 2025); *Staying with the Uncertainty* — AMA uncertainty-scaffolding strategies,
  LLM-to-LLM (arXiv:2606.05890, 2026); *Moral Anchor System* (arXiv:2510.04073, 2025); CAIS AI-Safety textbook §6.9.
- **Impossibility framing (for the capstone tie):** Eckersley, *Impossibility and Uncertainty Theorems in AI Value
  Alignment* (arXiv:1901.00064).

**Verdict:** occupied (~7th consecutive prior-art hit; the lab's documented meta-pattern). The *step the newest
paper (2606.05890) did not take*: a from-scratch, fully-inspectable **decision model** that quantifies the
robustness / fanaticism / **relocation** tradeoffs of the aggregation rules themselves (that paper studies LLM
conversational patterns, not the decision-theoretic structure). Framed honestly as a **demonstration + synthesis**,
exactly like cracks #1–#15.

## 3. The toy model (CPU, numpy, fully inspectable — to be built only on approval)

- `D` decision situations × `A` actions (e.g. A=6). `K` candidate ethical theories (e.g. K=4): a welfare/utilitarian
  theory, a deontological constraint theory (large negative choiceworthiness on a "rights-violation" feature), a
  prioritarian "help-worst-off" theory, and a **heavy-tailed "fanatical" theory** held at low credence (δ≈1%) that
  rarely assigns astronomically large choiceworthiness — the fanaticism probe.
- A supplied credence vector `p` over theories (the external input). A **catastrophe floor** per theory: an action
  is "catastrophic per theory k" if `CW_k(action) ≤ floor_k`. A designated `SAFE_DEFAULT` (do-little) action.
- **Decision rules compared:** (1) **Commit** = favorite (highest-credence) theory's argmax; (2) **MEC-none** =
  argmax of credence-weighted raw choiceworthiness (no normalization); (3) **MEC-range**; (4) **MEC-variance**
  (MacAskill et al.'s recommended); (5) **Parliament** = credence-weighted approval over each theory's top-q
  actions; (6) **Defer/Abstain** = if no action clears every credence-≥ε theory's catastrophe floor, take
  `SAFE_DEFAULT`; else variance-MEC over the acceptable set.

## 4. Pre-registered metrics (defined now so they cannot be chosen to fit)

- **catastrophe_rate(rule)** = fraction of `D` where the chosen action is catastrophic for *some* theory with
  `p_k ≥ ε`.
- **worstcase_regret(rule)** = mean over `D` of `max_{k: p_k≥ε} ( max_a CW_k^var(a) − CW_k^var(chosen) )` (regret
  measured in variance-normalized units, shared across rules).
- **fanaticism_index** = fraction of `D` where MEC-none's chosen action **changes** when the ≤δ-credence fanatical
  theory is removed (i.e. the 1% theory is decisive).
- **normalization_flip** = fraction of `D` where the chosen action differs across {MEC-none, MEC-range, MEC-variance}
  with theories + credences + behavior held fixed.

## 5. Decision rules — BARS BEFORE DATA (≥5 seeds; seed governs theory params + batch; report mean, per-seed counts, and a two-level bootstrap CI on each headline delta)

- **H1 — Robustness (does hedging+deferral beat point-commit on the worst case?).**
  PASS if `catastrophe_rate(Defer) ≤ catastrophe_rate(Commit) − 0.20` (absolute) in **≥4/5 seeds**, AND the worst-case
  regret ordering `Defer ≤ MEC-variance ≤ Commit` holds in **≥4/5 seeds**.
  PASS → hedging + deferral buys robustness. FAIL → **honest negative**: it does not help at toy scale (banked with
  the same care as a win).

- **H2 — Fanaticism (is MEC-none vulnerable, and does variance-normalization curb it?). Predicted POSITIVE.**
  PASS if `fanaticism_index(MEC-none) ≥ 0.30` in **≥4/5 seeds** AND
  `fanaticism_index(MEC-variance) ≤ fanaticism_index(MEC-none) − 0.20` (reproducing MacAskill et al.'s claimed
  property — *measured, not assumed*).
  If `fanaticism_index(MEC-none) < 0.30` → **VOID** (the toy did not surface fanaticism), not FAIL.

- **H3 — Relocation (load-bearing thesis bar: the anchor moved, it did not vanish).**
  PASS if `normalization_flip ≥ 0.25` in **≥4/5 seeds** (switching the supplied intertheoretic normalization flips
  ≥25% of decisions, behavior/theories/credences fixed).
  PASS → the supplied normalization, an external normative choice, *determines the answer* → moral-uncertainty
  hedging **relocates** the anchor (the exact analogue of Crack #15's rationality-assumption result), it does not
  eliminate it. This is the result that ties Crack #16 to the capstone's unified thesis.

## 6. Verification gate (read green before reading H1–H3 — PIPELINE §8/§9 analogue)

- **V1 (degeneracy):** with `K=1`, every rule reduces to Commit. Assert identical decisions.
- **V2 (scale-invariance):** MEC-range and MEC-variance are invariant to per-theory affine rescaling of `CW`
  (assert max-diff = 0 under random rescales); MEC-none is *not* (that is the point being demonstrated).
- **V3 (symmetry):** the catastrophe floor and `ε` threshold are applied through one shared code path for every
  rule — no rule receives privileged information. Assert via the shared function, not per-rule copies.
- **Tested vs not (stated up front):** tested = the three properties above + reproducibility across ≥5 seeds.
  NOT tested = whether the stylized theory-generator is *representative* of real ethical theories (it is a
  mechanism demo, not a measurement), and intertheoretic comparability is *assumed possible* — the deep
  philosophical problem is bracketed, not solved (state this as a limitation in any writeup).

## 7. Honest stance

Not a novel phenomenon (§2). The contribution is a clean, dependency-free, fully-readable demonstration that
unifies the constructive moral-uncertainty question with the program's *supplied, not discovered* thesis, with two
genuinely-possible honest outcomes (H1 can fail; H2 can void). The asset is rigor + inspectability + the synthesis,
not a discovery — consistent with the whole corpus.

## 8. Lens pass (the relevant lenses, per Austin's standing rule)

- **#7 Problem Statement** — the real problem is *which* construction of a supplied anchor is least-bad, not
  whether good can be discovered (that's closed). Stated before solving.
- **#74 Know Your Competition / #87 Destroy Your Assumptions** — prior-art-checked first; assumed it was occupied
  and confirmed it; the experiment is framed as demonstration, not a novelty claim.
- **#11 Unification / #97 Deepest Theme** — H3 is chosen specifically because it makes Crack #16 serve the single
  unifying thesis (anchor relocates, never vanishes), not a stray result.
- **#13 Risk Mitigation (pre-mortem)** — the riskiest assumption is that the toy theory-generator manufactures the
  conclusion; V2/V3 + the "not tested" disclosure + bars-before-data are the guards.
- **#27 Expected Value** — cheap (minutes of CPU), decisive (numeric bars), low risk; but see the strategic note:
  it adds rigor, not novelty or income.
- **#63 Bug = Teacher / #59–62 Testing** — the verification gate is the teacher; read it green before the result.
- **#79 Don't Do Too Much / #100 Ship It** — one decisive experiment (three bars), not a sprawling suite.
- **#85 Ethical Implications** — the takeaway ("you must choose and own the anchor; hedging only moves the choice")
  is the responsible message, not a false promise that uncertainty dissolves the duty.

## 9. The strategic fork this prereg surfaces (decision for Austin)

The capstone + four leg-writeups are now a **complete, independently-verified 15-experiment corpus**. Running
Crack #16 *continues the science* but, per the lab's own honest stance, adds rigor/pedagogy — **not novelty or
income**. The research-agenda names the realistic route out of poverty as a **funded seat** reached by *packaging*
this corpus (workshop paper / MATS artifact), not by adding cracks. So t