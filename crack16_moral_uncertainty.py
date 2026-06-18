"""Abel Labs — Leg 4 (constructive), Crack #16 REGISTERED: hedging a supplied anchor under MORAL UNCERTAINTY.

Runs `prereg_crack16_moral-uncertainty_2026-06-17.md` exactly (bars set before data; no tuning to pass). The
constructive question: given the anchor must be SUPPLIED as a distribution over candidate ethical theories,
(a) does acting under moral uncertainty (MEC / parliament / defer) buy ROBUSTNESS over committing to one theory,
and (b) does it ELIMINATE the choice, or merely RELOCATE it into the supplied credences + normalization rule?

Theories (K=4): util(welfare sum) · deon(welfare - BIG*rights) · prio(worst-off) · fanatical(heavy tail, 1% credence).
Rules: Commit · MEC-none · MEC-range · MEC-variance · Parliament · Defer/Abstain (SAFE_DEFAULT fallback).
Metrics (pre-registered, §4): catastrophe_rate · worstcase_regret · fanaticism_index · normalization_flip.
Verification gate (§6, read green FIRST): V1 degeneracy (K=1 -> all rules == Commit), V2 scale-invariance
(MEC-range/variance invariant to per-theory affine rescale; MEC-none not), V3 symmetry (one shared catastrophe
code path for every rule).

Pre-registered bars (§5; >=5 seeds; PASS needs >=4/5 seeds):
  H1 robustness: catastrophe_rate(Defer) <= catastrophe_rate(Commit) - 0.20 AND regret order Defer<=MEC-var<=Commit.
  H2 fanaticism (predicted POSITIVE): fanaticism_index(MEC-none) >= 0.30 AND
     fanaticism_index(MEC-variance) <= fanaticism_index(MEC-none) - 0.20.  (<0.30 -> VOID, not FAIL.)
  H3 relocation (load-bearing): normalization_flip >= 0.25.

NOT a novel phenomenon (prior art: Ecoffet & Lehman 2021; MacAskill/Cotton-Barratt/Ord 2020; Moral Uncertainty &
Fanaticism AAAI 2024; Eckersley 2019). A from-scratch, fully-inspectable demonstration tying the constructive
moral-uncertainty question to the program's 'supplied, not discovered' thesis. Limitation (stated up front): the
stylized theory generator is a MECHANISM demo, not a measurement; intertheoretic comparability is assumed possible.

VERIFIED (2026-06-17, CPU; 5 seeds, D=400; bars set before data, NO tuning):
  Verification gate: V1 degeneracy PASS · V2 scale-invariance PASS (MEC-range/variance invariant; MEC-none not,
    changed on 55 decisions) · V3 symmetry PASS (one shared catastrophe code path).
  H1 robustness: FAIL (HONEST NEGATIVE) — catastrophe-cut >= 0.20pp in 0/5 (Commit catastrophe was only ~0.05,
    too rare to cut 20pp at these params), THOUGH the regret ordering Defer <= MEC-var <= Commit held 5/5
    (Defer/MEC-var regret ~0.59 vs Commit ~1.00). Banked as a negative with the same care as a win.
  H2 fanaticism: PASS — fanaticism_index(MEC-none) ~0.47-0.50 (>=0.30) in 5/5; variance-normalization curbs it
    by >=0.20 in 5/5 (reproduces MacAskill et al.'s claimed property, measured not assumed).
  H3 relocation: PASS (LOAD-BEARING) — normalization_flip ~0.55 (>=0.25) in 5/5: switching the supplied
    intertheoretic normalization flips ~55% of decisions (behavior/theories/credences fixed) -> the anchor
    RELOCATES into the normalization choice (the exact analogue of Crack #15; ties Crack #16 to the capstone).
  Headline: hedging did NOT clear the robustness bar at toy scale (H1 honest-negative), but the deep results
    hold — moral-uncertainty averaging is fanaticism-prone (H2) and merely RELOCATES the anchor (H3).
    'Supplied, not discovered' confirmed on the constructive side too.

Run:  python crack16_moral_uncertainty.py
"""
import numpy as np

K = 4                                   # 0 util, 1 deon, 2 prio, 3 fanatical
A_SUB = 6                               # substantive actions
A = A_SUB + 1                          # + SAFE_DEFAULT at index A_SUB
SAFE = A_SUB
EPS = 0.05                             # a theory "counts" for catastrophe/regret/defer if credence >= EPS
FLOOR = -2.0                           # catastrophe floor in variance-normalized units (shared, V3)
QTOP = 2                               # parliament: each theory approves its top-q actions
CREDENCES = np.array([0.40, 0.34, 0.25, 0.01])   # the supplied anchor (fanatical at 1%)
BIG = 20.0                             # deontological rights-violation penalty

def gen(rng, n):
    """Raw choiceworthiness CW[n, K, A]. Substantive actions 0..A_SUB-1; index SAFE = do-little (~0 for all)."""
    G = 3
    welfare = rng.normal(0, 1, (n, A_SUB, G))
    rights = (rng.random((n, A_SUB)) < 0.30).astype(float)
    CW = np.zeros((n, K, A))
    CW[:, 0, :A_SUB] = welfare.sum(2)                              # utilitarian: total welfare
    CW[:, 1, :A_SUB] = welfare.sum(2) - BIG * rights              # deontological: big penalty on rights-violation
    CW[:, 2, :A_SUB] = welfare.min(2)                            # prioritarian: worst-off group
    spike = (rng.random((n, A_SUB)) < 0.25) * rng.uniform(100, 1000, (n, A_SUB))  # fanatical: rare astronomical
    CW[:, 3, :A_SUB] = rng.normal(0, 0.1, (n, A_SUB)) + spike
    CW[:, :, SAFE] = rng.normal(0, 0.05, (n, K))                  # SAFE_DEFAULT: neutral for every theory
    return CW

def n_range(cw):
    lo = cw.min(1, keepdims=True); hi = cw.max(1, keepdims=True); d = hi - lo; d[d < 1e-9] = 1
    return (cw - lo) / d
def n_var(cw):
    mu = cw.mean(1, keepdims=True); sd = cw.std(1, keepdims=True); sd[sd < 1e-9] = 1
    return (cw - mu) / sd

def catastrophic_actions(cw, p):                                  # V3: ONE shared code path, used by every rule
    nv = n_var(cw); bad = np.zeros(A, bool)
    for k in range(K):
        if p[k] >= EPS: bad |= (nv[k] <= FLOOR)
    return bad

def choose(rule, cw, p):
    if rule == "commit":   return int(cw[int(p.argmax())].argmax())
    if rule == "mec_none": return int((p[:, None] * cw).sum(0).argmax())
    if rule == "mec_range":return int((p[:, None] * n_range(cw)).sum(0).argmax())
    if rule == "mec_var":  return int((p[:, None] * n_var(cw)).sum(0).argmax())
    if rule == "parliament":
        appr = np.zeros(A)
        for k in range(K):
            appr[np.argsort(-cw[k])[:QTOP]] += p[k]
        return int((appr + 1e-9 * (p[:, None] * n_var(cw)).sum(0)).argmax())
    if rule == "defer":
        ok = ~catastrophic_actions(cw, p)
        score = (p[:, None] * n_var(cw)).sum(0)
        if not ok.any(): return SAFE
        idx = np.where(ok)[0]; return int(idx[np.argmax(score[idx])])
    raise ValueError(rule)

RULES = ["commit", "mec_none", "mec_range", "mec_var", "parliament", "defer"]

def metrics(CW, p):
    D = len(CW); out = {}
    cat = {r: 0 for r in RULES}; reg = {r: 0.0 for r in RULES}
    for d in range(D):
        cw = CW[d]; nv = n_var(cw); bad = catastrophic_actions(cw, p)
        best = np.array([nv[k].max() for k in range(K)])
        for r in RULES:
            a = choose(r, cw, p)
            if bad[a]: cat[r] += 1
            reg[r] += max((best[k] - nv[k, a]) for k in range(K) if p[k] >= EPS)
    out["catastrophe_rate"] = {r: cat[r] / D for r in RULES}
    out["worstcase_regret"] = {r: reg[r] / D for r in RULES}
    # fanaticism_index: MEC-none choice changes when the fanatical theory (idx 3) is removed
    p_nofan = p.copy(); p_nofan[3] = 0; p_nofan = p_nofan / p_nofan.sum()
    flip_fan = 0; flip_norm = 0
    for d in range(D):
        cw = CW[d]
        if choose("mec_none", cw, p) != choose("mec_none", cw, p_nofan): flip_fan += 1
        cs = {choose(r, cw, p) for r in ("mec_none", "mec_range", "mec_var")}
        if len(cs) > 1: flip_norm += 1
    out["fanaticism_index"] = flip_fan / D
    out["normalization_flip"] = flip_norm / D
    return out

def verification_gate(rng):
    CW = gen(rng, 300)
    # V1 degeneracy: K=1 (only theory 0) -> every rule == commit
    p1 = np.array([1.0, 0, 0, 0]); v1 = all(choose(r, CW[d], p1) == choose("commit", CW[d], p1)
                                            for d in range(len(CW)) for r in RULES)
    # V2 scale-invariance: per-theory affine rescale; range/var invariant, none not
    alpha = rng.uniform(0.5, 3, K); beta = rng.uniform(-5, 5, K)
    CW2 = CW * alpha[None, :, None] + beta[None, :, None]
    p = CREDENCES
    inv_range = all(choose("mec_range", CW[d], p) == choose("mec_range", CW2[d], p) for d in range(len(CW)))
    inv_var = all(choose("mec_var", CW[d], p) == choose("mec_var", CW2[d], p) for d in range(len(CW)))
    none_changed = sum(choose("mec_none", CW[d], p) != choose("mec_none", CW2[d], p) for d in range(len(CW)))
    return v1, inv_range, inv_var, none_changed

if __name__ == "__main__":
    print("=== VERIFICATION GATE (must read green before the bars) ===")
    vg = verification_gate(np.random.default_rng(999))
    v1, inv_r, inv_v, none_chg = vg
    print(f"  V1 degeneracy (K=1 -> all rules == Commit) : {'PASS' if v1 else 'FAIL'}")
    print(f"  V2 scale-invariance: MEC-range invariant {inv_r}, MEC-variance invariant {inv_v}, "
          f"MEC-none changed on {none_chg} decisions (should be >0) : {'PASS' if (inv_r and inv_v and none_chg>0) else 'FAIL'}")
    print(f"  V3 symmetry: one shared catastrophic_actions() code path for every rule (structural) : PASS\n")

    print("=== PRE-REGISTERED BARS (5 seeds; D=400) ===")
    SEEDS = range(5); D = 400; p = CREDENCES
    rows = []
    for s in SEEDS:
        m = metrics(gen(np.random.default_rng(s), D), p)
        rows.append(m)
        print(f"seed {s}: cat[Commit]={m['catastrophe_rate']['commit']:.2f} cat[Defer]={m['catastrophe_rate']['defer']:.2f} "
              f"| regret Defer={m['worstcase_regret']['defer']:.2f} MECvar={m['worstcase_regret']['mec_var']:.2f} "
              f"Commit={m['worstcase_regret']['commit']:.2f} | fan[none]={m['fanaticism_index']:.2f} "
              f"normflip={m['normalization_flip']:.2f}")

    def frac(cond): return sum(cond(m) for m in rows)
    h1a = frac(lambda m: m['catastrophe_rate']['defer'] <= m['catastrophe_rate']['commit'] - 0.20)
    h1b = frac(lambda m: m['worstcase_regret']['defer'] <= m['worstcase_regret']['mec_var'] <= m['worstcase_regret']['commit'])
    # H2 fanaticism: need fanaticism_index(MEC-variance) too -> recompute via removing fan under mec_var
    fan_var = []
    for s in SEEDS:
        CW = gen(np.random.default_rng(s), D); p_nf = p.copy(); p_nf[3] = 0; p_nf /= p_nf.sum()
        fv = np.mean([choose("mec_var", CW[d], p) != choose("mec_var", CW[d], p_nf) for d in range(D)])
        fan_var.append(fv)
    h2a = frac(lambda m: m['fanaticism_index'] >= 0.30)
    h2b = sum(fan_var[i] <= rows[i]['fanaticism_index'] - 0.20 for i in range(len(rows)))
    h3 = frac(lambda m: m['normalization_flip'] >= 0.25)

    print(f"\nH1 robustness : cat-cut>=0.20 in {h1a}/5 ; regret order Defer<=MECvar<=Commit in {h1b}/5 "
          f"-> {'PASS' if (h1a>=4 and h1b>=4) else 'FAIL (honest negative)'}")
    h2_void = h2a < 4
    print(f"H2 fanaticism : fan[MEC-none]>=0.30 in {h2a}/5 ; var curbs by>=0.20 in {h2b}/5 "
          f"-> {'VOID (toy did not surface it)' if h2_void else ('PASS' if h2b>=4 else 'FAIL')}")
    print(f"H3 relocation : normalization_flip>=0.25 in {h3}/5 "
          f"-> {'PASS (anchor RELOCATED into the normalization choice)' if h3>=4 else 'FAIL'}")
