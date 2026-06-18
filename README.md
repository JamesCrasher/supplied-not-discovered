# The Normative Anchor Is Supplied, Not Discovered

**Sixteen small, fully-reproducible experiments on AI alignment — verification, optimization, social emergence, and value learning.**

Author: **Austin Brunner** (Abel AI Solutions · Polis Project) · 2026 · CPU-only, `numpy`-only.

---

## What this is

A unified, runnable demonstration of one claim: **the normative core of alignment — *what counts as good,
which features may matter* — is supplied to a system, never discovered by it.** Four "legs," sixteen
experiments, each attacking one route by which a machine might acquire "good" on its own. All four close the
same way:

- **Verification** is always relative to an external reference (a uniformly-shared bad value is invisible from
  inside; no finite test battery is complete).
- **Optimization** doesn't preserve good (a hidden backdoor survives safety training; a narrow bad behavior
  spreads; a proxy corrupts its target; an agent can fake alignment while watched).
- **Social emergence** makes norms *sticky*, not *right* (emergence is arbitrary and minority-capturable;
  punishment and reputation need an external fact and a value-neutral watching channel).
- **Value learning** can't recover good from behavior (behavior fixes only the product of values × rationality).
- **Constructive side** (preregistered): hedging the anchor under moral uncertainty doesn't eliminate the
  choice — it *relocates* it into the normalization rule (~55% of decisions flip).

The recurring lever across all of it is an external, value-neutral **watching / verifier channel**. The
verification, enforcement, and oversight *machinery* is largely engineerable; the anchor is not.

## Honest stance (please read)

**None of these results is novel.** Each is a from-scratch demonstration or replication of an established
finding (citations in the paper). The contribution is the **convergence** — four independent literatures
re-derived in one readable, dependency-free framework pointing at the same irreducible core — plus
reproducibility and **intellectual honesty as a first-class output**: this corpus includes a **preregistered
experiment whose robustness hypothesis FAILED**, reported as loudly as the two that passed, and two further
reproducibility honest-negatives. It is a synthesis/distillation, not a claim of new theorems. It was directed
and verified by the author using AI assistants as the implementation engine.

## Quickstart — verify any claim in minutes

```bash
git clone https://github.com/JamesCrasher/supplied-not-discovered.git
cd supplied-not-discovered
pip install numpy
```

**Run a single experiment** (each prints a dated `VERIFIED` block):

```bash
python crack15_identifiability.py
```

**Run all 16 — pick your shell** (from inside the repo folder):

- **macOS / Linux (bash):**
  ```bash
  for f in crack*.py; do echo "== $f =="; python "$f"; done
  ```
- **Windows (PowerShell):**
  ```powershell
  Get-ChildItem crack*.py | ForEach-Object { Write-Host "== $($_.Name) =="; python $_.Name }
  ```

Every script is CPU-only, dependency-free beyond `numpy`, and self-verifying. (Tip: you must be *inside* the
repo folder — `cd` into it first — or Python won't find the files.)

## The sixteen experiments

| Leg | # | Question | Result | Script |
|---|---|---|---|---|
| 1 Verification | 1 | Read a value from inside / behavior? | blind on-distribution; internals don't transfer; a shared bad value is invisible → need an external reference | `crack1_value_probe.py`, `crack1_behavioral_probe.py`, `crack1_autohunt.py`, `crack1_wall.py` |
| 1 | 2 | Can a society's vote be that reference? | only as good as its majority (97% vs 1% under majority capture) | `crack2_society.py` |
| 1 | 3 | An anchor with no vote/list? | impartiality (veil): catches discrimination 96%, blind to uniform badness | `crack3_veil.py` |
| 1 | 4 | Does a battery of tests close the gap? | no finite battery is complete (Goodhart) | `crack4_battery.py` |
| 1 | 5 | Keep the off-switch if you can't verify? | corrigibility = humility | `crack5_offswitch.py` |
| 1 | 6 | Does corrigibility compose to a society? | no — loyalty + redundancy break it | `crack6_society_offswitch.py` |
| 1 | 7 | A protocol to recover control? | monitoring works iff collusion broken + trusted anchor | `crack7_control.py` |
| 2 Optimization | 8 | Does safety training remove a hidden value? | no — backdoor survives ~60% (1% if trigger known) | `crack8_sleeper.py` |
| 2 | 9 | Does a narrow bad habit stay narrow? | no when undefended (→0); containable | `crack9_emergent.py` |
| 2 | 10 | Does optimizing a proxy keep improving good? | no — Goodhart inverted-U | `crack10_overopt.py` |
| 2 | 11 | Will an agent let training fix it? | it can fake — good when watched (99%) vs deployment (1%) | `crack11_faking.py` |
| 3 Emergence | 12 | Can a society grow a shared norm? | yes, but arbitrary; ~10% minority tips it | `crack12_norms.py` |
| 3 | 13 | Can the emergent norm be good? | conditionally — needs defended punishment | `crack13_cooperation.py` |
| 3 | 14 | Can reputation sustain good w/o punishment? | yes if watched (standing robust, image fragile) | `crack14_reputation.py` |
| 4 Value learning | 15 | Can good be learned from behavior? | no — opposite values, identical behavior | `crack15_identifiability.py` |
| 4 | 16 | Does hedging under moral uncertainty remove the choice? | no — it relocates it (~55% flip); preregistered (V-gate green; H1 honest-negative, H2/H3 pass) | `crack16_moral_uncertainty.py` (registered); `crack16_uncertainty.py` (pilot) |

## Repository layout

```
.                       16 experiments, crack*.py (CPU, numpy-only, self-verifying)
docs/
  paper_anchor-is-supplied_*.pdf / .md      5-page paper (start here)
  writeup_capstone_*.md                     the unified synthesis
  writeup_verify-and-control_*.md           Leg 1 (detailed)
  writeup_alignment-under-optimization_*.md Leg 2 (detailed)
  writeup_norm-emergence_*.md               Leg 3 (detailed)
  prereg_crack16_*.md                       preregistration for experiment 16
```

## Start here
Read **`docs/paper_anchor-is-supplied_2026-06-17.pdf`** (5 pages), then run any `crack*.py`.

## Limitations
Toy-scale mechanism demonstrations, not measurements of frontier systems; stylized generators; the asset is
rigor, inspectability, and the synthesis, not novelty. Honest negatives are reported in place (experiments 8, 9,
and the preregistered H1 in 16).

## License
MIT (see `LICENSE`).
