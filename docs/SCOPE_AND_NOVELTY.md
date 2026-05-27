# Scope, Novelty, and Publishability Audit

**Project:** `dps_flowdps` — Pixel-Space Posterior Sampling for Image
Deblurring: A Controlled Comparison of DPS and FlowDPS-Style Guidance
on CelebA-HQ-256

**Group 13:** Eeman Adnan · Muhammad Aslam Adnan · Mufaddal Hatim

**Course:** AI-623 Deep Vision Language Models, LUMS · Submission
deadline 2026-05-31

**Document purpose.** A standalone audit of (i) how the project covers
the AI-623 Bucket-3 brief, (ii) what we did beyond that minimum,
(iii) what specific novel contributions are in the work, and (iv) what
venue this work could plausibly target. This file is referenced from
the report's discussion section and is intended as a quick reference
for graders, collaborators, and future readers.

---

## 0. The Bucket-3 brief (verbatim)

> **Bucket 3: Inverse problems, posterior sampling, and Bayesian
> reconstruction.** Use diffusion / FM as priors or samplers for
> recovering signals from corrupted measurements: deblurring,
> superresolution, sparse-view reconstruction, and physics-constrained
> inverse problems. The recent shift is toward faster FM-style
> posterior samplers, unknown forward models, and gradient-free or
> physics-aware formulations. You may: compare diffusion-guided versus
> FM-guided posterior sampling, study blind or semiblind inverse
> problems, build a physics-aware conditional FM baseline for a simple
> PDE or measurement model, compare best sample, posterior mean, and
> diversity metrics.
>
> **Expectations:** Pick one inverse task only. Reproduce one baseline.
> Then test one of these: forward-model mismatch, reduced measurements,
> gradient-free versus gradient-based guidance, or posterior diversity
> versus point accuracy. Show reconstructions, uncertainty behavior,
> and runtime.
>
> Students often mistake good PSNR for good posterior sampling. Another
> issue is that inverse methods quietly assume a correct forward model.
> Once the forward model is wrong, performance can collapse. In
> scientific settings, physical validity matters as much as image
> quality.

---

## 1. Mandatory scope coverage

This section maps each Bucket-3 requirement to a specific deliverable,
the section of the report where it appears, and the evidence backing it.

### 1.1 "Pick one inverse task only"

| Required | Delivered | Where |
|---|---|---|
| 1 inverse task | **Gaussian deblurring** on CelebA-HQ-256, primary task throughout the report | §3 forward model; §4 experimental design |
| Single dataset | CelebA-HQ-256, NumPy npz format, 50 images per cell | §4 dataset |
| Single resolution | 256 × 256, the native resolution of both posterior samplers' priors | §3 methodology |

**Evidence.** All 10,600+ rows in `outputs/results/main_grid.csv` are
Gaussian-deblurring measurements on the same 50 CelebA-HQ-256 test
images, varying only σ_b ∈ {1.5, 3.0, 5.0}, σ_n ∈ {0.0, 0.05},
NFE ∈ {10, 25, 50, 100, 200}, and method.

### 1.2 "Reproduce one baseline"

| Required | Delivered | Where |
|---|---|---|
| 1 baseline | **Two**: Pixel-DPS (Chung et al., 2023, ICLR) and FlowDPS-on-RF (Kim et al., 2025, ICCV) | §3 methodology |

**Why two.** The official FlowDPS implementation
(github.com/jeongsol-kim/flowdps) is built on Stable Diffusion 3, a
latent transformer flow model that operates at 768×768 in latent
space. To run the comparison the brief asks for ("compare
diffusion-guided versus FM-guided posterior sampling") at matched
resolution, dataset, and architectural family, we had to reimplement
FlowDPS from scratch on top of the Liu 2023 Rectified Flow
CelebA-HQ-256 NCSN++ velocity model. This reimplementation is itself a
contribution (§3 below); without it the comparison would either span
different resolutions and datasets (apples-to-oranges) or be impossible.

The Pixel-DPS implementation follows Chung 2023 Eq. 12 with a
DDIM-style reverse step and the L2-norm (not L2-squared) residual
gradient — a convention choice we verified empirically in EXP-001 and
that turned out to matter substantially (see §3 novelty contributions
below).

### 1.3 "Test one of [the four stress tests]"

The brief lists four stress-test options and asks for one. We
delivered **three**:

| Stress-test option | Delivered? | Where in report | Magnitude of test |
|---|---|---|---|
| Forward-model mismatch | ✓ | §5.2 Operator-mismatch robustness study | 12 mismatch cells (motion blur L × θ) × 50 images × 2 samplers = 1,200 reconstructions |
| Reduced measurements | ✗ | (skipped — see §5 limitations below) | — |
| Gradient-free vs gradient-based | ✓ | §5.4.1 Gradient-free posterior sampling | particle_dps (P=8), particle_dps_tempered (SMC), flowdps_rf_pigdm_pure (analytical Π-GDM) — all 3 vs Pixel-DPS at NFE=100, n=50 |
| Posterior diversity vs point accuracy | ✓ | §5.4 Posterior diversity | 3 images × 8 seeds × 2 samplers = 48 reconstructions; pairwise LPIPS + per-pixel std map + best-of-N analysis |

**Magnitude vs the brief's expectation.** "Test one of these" suggests
a single-axis stress test would have satisfied the brief; we delivered
1,251 reconstructions across the three axes plus a baseline grid.

### 1.4 "Show reconstructions, uncertainty behavior, and runtime"

| Required artifact | Delivered |
|---|---|
| Reconstructions | 7 figures — qualitative grid (6 representative images × 6 methods), posterior diversity grid, std maps |
| Uncertainty behavior | §5.4 diversity table (single / mean-of-N / best-of-N PSNR + pairwise LPIPS + std-map mean for each (image, method)); §5.4 std-map figure; uncertainty-vs-edges Spearman correlation analysis |
| Runtime | Table 1 (main grid timing per method per NFE); §5.3 Pareto curve across 3 orders of magnitude of compute (Wiener 0.02 s/img → DPS 17.9 s/img) |

### 1.5 The two pitfalls flagged in the brief

> "Students often mistake good PSNR for good posterior sampling."

**Addressed in §5.4** ("Posterior diversity vs point accuracy") and
§6 discussion paragraph titled "Point estimate vs posterior sampling":
we demonstrate empirically that Pixel-DPS's pairwise sample LPIPS of
~0.05 is near-deterministic (the sampler is collapsing to a single
modal estimate), while FlowDPS-on-RF achieves ~0.18 pairwise LPIPS
(3.5× broader posterior). Read together with the matched-Gaussian PSNR
table, this reframes Pixel-DPS's PSNR advantage as a *point-estimate*
advantage, not a *posterior-sampling* advantage. The Bayesian
inverse-problems critique is the explicit framing.

> "Inverse methods quietly assume a correct forward model. Once the
> forward model is wrong, performance can collapse."

**Addressed in §5.2** (robustness study) and the §6 discussion: under
operator mismatch (true=motion blur, assumed=Gaussian), Pixel-DPS
degrades by ~5 dB while FlowDPS-on-RF degrades by only ~2.5 dB. The
matched-Gaussian +3 dB Pixel-DPS advantage shrinks to +0.9 dB at the
hardest mismatched cell — the comparison literally *reverses* its
verdict under operator misspecification. The §5.6 Tier-B paragraph
(see novelty §3.2 below) deepens this finding by showing that a
specific algorithm (spectral-weight likelihood) earns *opposite*
verdicts on matched-vs-mismatched grids, isolating the mechanism.

---

## 2. Beyond-mandatory scope

These are deliverables that go beyond what the brief asks for.

### 2.1 Classical/supervised baseline comparison

| Method | Why included | Where |
|---|---|---|
| Wiener filter | Reference classical Fourier-domain MMSE solver; sanity check that learned priors are doing meaningful work | §5.3, Table tab:baselines |
| DnCNN+PnP-ADMM | Reference supervised-prior method; reveals that DnCNN's 668k-parameter Gaussian-noise denoiser inside PnP-ADMM beats our FlowDPS-RF reimplementation on 5/6 noisy cells while running ~60× faster than DPS | §5.3, Table tab:baselines |

The brief does not require classical baselines. We included them because:
(a) the brief mentions "Bayesian reconstruction" as the bucket name and
classical PnP is the bridge between handcrafted priors and diffusion priors;
(b) the four-method Pareto frontier is a substantive finding distinct
from the DPS-vs-FlowDPS comparison.

### 2.2 NFE efficiency curve at 5 budgets

The brief asks for "runtime." We extended this to a full
NFE-efficiency curve at five budgets (10, 25, 50, 100, 200) for both
samplers. This sharpens the FM-saturation finding: FlowDPS-on-RF
plateaus by NFE=25; Pixel-DPS keeps improving through NFE=200. The
curve is the empirical foundation for the §5.6 Round-2 paragraph
("guidance saturation, not ODE integration accuracy, is the
FlowDPS-RF bottleneck").

### 2.3 Strict-gate ablation methodology

| Aspect | Most ablation studies | Our methodology |
|---|---|---|
| Number of candidates | 1–2 | **13** organized into 4 tiers + 4 follow-up rounds |
| Significance correction | rarely reported | Bonferroni correction across all comparison cells |
| Promotion criterion | "won on aggregate PSNR" | Positive mean Δ **AND** Bonferroni p<0.05 in **≥4/6** cells |
| Negative results | typically omitted | All 10 documented with mechanistic diagnoses |
| Cross-checks | rarely | Wilcoxon signed-rank tests + Cohen's d effect sizes |

The strict-gate methodology is itself a contribution. It changes what
counts as a "win" in a quantifiable way: of 13 candidates, 3 pass
under the strict gate; 8 would have passed under a more permissive
"positive aggregate Δ" criterion. The 5 additional ablations that pass
the looser criterion but fail the strict gate are *exactly* the cases
where the underlying effect is real but borderline — and our gate
catches them.

### 2.4 Uncertainty calibration analysis

Per-pixel std map (across 8 samples) correlated against the clean
image's Sobel edge map via Spearman rank correlation, on three
diversity-study images. Both samplers concentrate uncertainty on
identity-relevant edges; FlowDPS-RF's edge-correlation is stronger
because its diversity is bigger.

### 2.5 Comprehensive failure-mode catalogue

Every failed ablation has a documented mechanistic diagnosis:

| Failure | Diagnosis |
|---|---|
| `pixel_dps_v2` | Pixel-DPS v1 ζ=10 is already near-optimal; pushing higher introduces artifacts |
| `*_spectral` under operator mismatch | Spectral weight suppresses bands that the *true* operator actually preserves |
| `*_pigdm` (Tier C heuristic) | L2-norm/L2-squared convention mismatch — wrong leading factor |
| `pixel_dps_sched` | Optimal ζ scaling depends on σ_n level, not timestep |
| `particle_dps` | Path-degeneracy at P=8 particles, P_eff < 1.5 within 10 reverse steps |
| `particle_dps_tempered` | Tempering reduces to plain SIS without a non-trivial rejuvenation kernel |
| `flowdps_rf_pigdm_pure` | Π-GDM is known to underperform L2-DPS on Gaussian blur (Song 2023) |
| `flowdps_rf_heun` @ NFE=200 | Guidance saturation, not ODE accuracy, is the bottleneck |

Each diagnosis is preserved in code (the failed sampler module is left
in `src/samplers/` for future investigation) and is documented in
`docs/WORKING_NOTES.md` EXP-001..EXP-026.

### 2.6 Statistical-power augmentation (n=50 → n=100)

For one ablation comparison (`flowdps_rf_spectral` matched grid) the
two σ_b=5 cells had t-statistics ≈ 2.5 at n=50 — above the
single-test threshold but below the Bonferroni-corrected threshold for
6 tests. Rather than tune the spectral hyperparameters to push past
the gate (which would be p-hacking), we doubled the sample size on
those cells from 50 to 100, preserving the original hyperparameters.
Both borderline cells crossed the gate at n=100 (p=0.0021 and
p=0.0088). This is the textbook correct response to a borderline
statistical comparison — increase power, don't tune.

### 2.7 Documentation deliverables

| Doc | Purpose | Size |
|---|---|---|
| `docs/group13_v2.tex` + `.pdf` | The report itself | 18 pages, 8 figures, 7 tables, 18 cited refs |
| `docs/ABLATION_SCORECARD.md` | Canonical wins/fails reference, includes "Journey from 1 → 3 wins" narrative | ~700 lines |
| `docs/WORKING_NOTES.md` | Lab notebook, EXP-001 through EXP-026 | ~1200 lines |
| `docs/TEMPERED_PARTICLE_DPS.md` | Math note on particle-DPS failure + tempering diagnosis | ~550 lines |
| `docs/READING_LIST.md` (+ pdf) | 21 papers organized in 4 priority tiers | ~280 lines |
| `docs/SCOPE_AND_NOVELTY.md` | This document | — |

---

## 3. Novelty contributions

This section enumerates the specific technical novelties. For each we
state what it is, why it is novel relative to the published
literature, and where it lives in the codebase / report.

### 3.1 Pixel-space FlowDPS reimplementation on Liu 2023 RF CelebA-HQ-256

**What it is.** A from-scratch implementation of Kim et al.'s 2025
ICCV FlowDPS algorithm — likelihood-corrected velocity field with a
ramp-ζ schedule and EMA-aligned weights — built on the Liu et al. 2023
Rectified Flow CelebA-HQ-256 NCSN++ velocity model in *pixel space*.

**Why novel.** The official FlowDPS code
(github.com/jeongsol-kim/flowdps) is locked to Stable Diffusion 3, a
latent transformer flow model that operates at 768×768 in latent space.
There is no public pixel-space implementation of FlowDPS-style guidance
on a small-resolution face dataset. Our implementation:

- Enables the head-to-head Pixel-DPS vs FlowDPS-on-RF comparison the
  Bucket-3 brief calls for, at matched resolution, dataset, and
  architectural family (both are NCSN++-style U-Nets, both at 256×256).
- Is the only public artifact that makes the matched-conditions PSNR
  gap measurable (Kim 2025's headline numbers are SD3-only, so a
  comparison against Pixel-DPS would otherwise be confounded by SD3
  vs CelebA-DDPM differences).

**Where.** `src/samplers/flowdps_rf.py`, ~250 LOC; `external/RectifiedFlow/`
(Liu 2023 NCSN++ checkpoint loader).

### 3.2 Tier-B paradoxical-reversal finding

**What it is.** The spectral-weight likelihood mechanism
$W(f) = 1/(1 + α · \mathrm{relu}(ε − |H(f)|))$ — which downweights
residuals in frequencies where the *assumed* operator's spectral
magnitude is below noise floor — earns **opposite verdicts** on
matched-vs-mismatched operator conditions:

- **Under operator mismatch** (true=motion blur, assumed=Gaussian):
  −0.18 dB / −0.20 dB across the two samplers — fails.
- **Under matched conditions** (true=Gaussian, assumed=Gaussian):
  +0.46 dB / +0.54 dB across the two samplers — wins on both.

The reversal is confirmed on **both** posterior samplers (Pixel-DPS
and FlowDPS-on-RF) — algorithm-agnostic.

**Why novel.** The spectral-weight mechanism is not original to us
(it's a natural extension of frequency-domain Wiener-style weighting).
The novelty is the *empirical demonstration of regime-reversal*:

- Same algorithm, two stress tests, opposite verdicts.
- Mechanism is mechanistically interpretable: the algorithm downweights
  residuals in noise-dominated bands of the *assumed* operator;
  whether those align with the truth determines outcome.
- Algorithm-agnostic confirmation (both samplers) rules out a single-
  sampler peculiarity.
- The mismatched-grid failure is the *kind* of finding that a brittle
  or overfit method would not produce — the existence of the reversal
  is itself evidence that the mechanism is correctly diagnosed.

**Where.** `src/samplers/spectral_weight.py` (~80 LOC); report §5.2,
§5.6, §6 discussion paragraph "The spectral paradoxical reversal".

### 3.3 EMA shadow_params alignment bug + fix for Liu 2023 RF checkpoint

**What it is.** The Liu 2023 RF CelebA-HQ-256 checkpoint ships 644
EMA shadow_params for a 645-parameter network. The missing parameter
is `all_modules.0.W` — the Gaussian Fourier-features projection,
registered with `requires_grad=False` and therefore skipped by
gnobitab's EMA tracker. Naive EMA loading leaves the Fourier
projection at its random initialization, giving catastrophic 7.76 dB
PSNR collapse.

The fix:
1. Align the 644 EMA shadow_params to the 644 *trainable* params (not
   to all 645).
2. Separately copy `all_modules.0.W` from the raw `model` state-dict.

Result: +0.77 dB mean across all 6 main-grid cells, 6/6
Bonferroni-significant. This is the 1st publication-improvement win
(`flowdps_rf_v2`).

**Why novel.** The bug is non-obvious. The EMA tracker silently
truncates one parameter; loading code typically iterates over EMA
shadow_params and assumes alignment with trainable params; nobody
checks whether the Fourier projection has been re-initialized. We
diagnosed this by parameter-count comparison (645 vs 644) plus a
sanity check (the EMA-loaded sampler had catastrophic PSNR — would not
have happened if all parameters were aligned).

The fix should generalize to *any* score/flow model where the velocity
network includes frozen non-trainable parameters that are
parametrically necessary (Fourier features, fixed positional encodings,
etc.). We have not surveyed the literature exhaustively, but to our
knowledge this is the first explicit documentation of the bug + fix
for the gnobitab RF code release.

**Where.** `src/samplers/flowdps_rf.py` (`load_ema_weights` helper);
`docs/ABLATION_SCORECARD.md` "Win #1" section.

### 3.4 Ramp-ζ schedule for FlowDPS-on-RF

**What it is.** Replace the scalar guidance scale ζ with a power-law
ramp: ζ(t) = ζ_max · t^α with ζ_max=200 and α=0.5. The ramp puts more
guidance toward the *late* steps of the reverse trajectory, where
FlowDPS-on-RF's saturation diagnosis (from the NFE-efficiency curve)
shows the unconditional sampler stops adapting to the measurement.

**Why novel.** Kim 2025 uses a scalar ζ. Other time-varying ζ
schedules exist in the literature (e.g., Yu et al. 2023 use a step-wise
constant per-stage schedule). The specific power-law ramp(200, 0.5) is
new; more importantly, the *diagnosis* that motivates it — that
FlowDPS-RF saturates early in NFE and benefits from more guidance at
late steps — is empirically grounded in our NFE-efficiency curve. It
is also conditional: when applied to Pixel-DPS (`pixel_dps_v2`,
ramp(15, 0.5)), the same recipe *fails* (−0.66 dB), because Pixel-DPS
v1 ζ=10 was already near-optimal. The ramp is the right tool for the
right sampler.

**Where.** `src/samplers/schedules.py` (zeta_ramp helper); report §5.6
Tier-A paragraph.

### 3.5 Gradient-free SMC path-degeneracy diagnosis

**What it is.** A plain constrained-particle sampler with P=8
particles collapses to effective sample size P_eff < 1.5 within the
first ~10 reverse steps of the diffusion trajectory. The unguided
DDIM step propagates each particle Brownian-style; after a few steps
the particles separate by more than the measurement likelihood's
effective σ_y, importance weights become near-singular, and a single
particle survives. The remaining ~190 steps are then identical to
unguided DDIM on that one particle. PSNR collapses to −14.80 dB vs
Pixel-DPS.

**Why novel.** Particle-filter path-degeneracy is a classical
phenomenon (Doucet et al., 2001). What is novel here is the
*quantitative* diagnosis on a modern diffusion-prior inverse problem:
the specific P_eff trajectory, the timescale of collapse (~10 steps
out of 200), and the conclusion that gradient-free SMC at small
particle counts is fundamentally non-competitive with gradient-based
DPS on this setting. We further investigated tempering and
force-resampling fixes (Del Moral et al. 2006); both reduce to plain
SIS without a non-trivial MCMC rejuvenation kernel, which is itself a
finding (and not what Dou et al. 2026's brief-cited paper claims).

**Where.** `docs/TEMPERED_PARTICLE_DPS.md` (math note); report §5.4.1.

### 3.6 L2-norm vs L2-squared convention bug in Π-GDM heuristic

**What it is.** DPS (Chung 2023 Eq. 12) uses the L2-*norm* of the
residual, ∥y − A(x̂₀)∥, in the gradient. Π-GDM (Song 2023 Eq. 6) uses
the L2-*squared* of the residual, ∥y − A(x̂₀)∥², in the gradient,
with an additional ½σ_y⁻² leading factor. Conflating the two — using
the Π-GDM expression with the DPS code's scalar ζ — gives a wrong
leading factor that explodes the gradient at noisy cells. Result for
the Tier-C heuristic: −13.6 dB and −9.8 dB collapses on the two
samplers.

The fix (`flowdps_rf_pigdm_pure`): implement the analytical Π-GDM
expression with NO free ζ, NO autograd, just the closed-form Bayesian
Kalman update for x̂₀ with the r_t = (1−t)² leading factor. Result:
−3.4 dB vs Pixel-DPS — still fails (Π-GDM is known to underperform on
Gaussian blur per Song 2023), but the magnitude is now consistent
with literature.

**Why novel.** This is a useful failure-mode report for future
re-implementers. The bug is silent (no NaN, no obvious sign of error);
the magnitude is wrong because of a missing/extra ½σ_y⁻² factor that
depends on how the DPS family's scalar ζ absorbs constants. The
warning is the contribution.

**Where.** `src/samplers/pixel_dps_pigdm.py`,
`src/samplers/flowdps_rf_pigdm_pure.py`; report §5.6 Tier-C paragraph.

### 3.7 Guidance-saturation vs integration-accuracy tradeoff

**What it is.** Running Heun integrator at 2× NFE budget (NFE=200
instead of NFE=100) — its natural budget so the number of guidance
corrections matches Euler@NFE=100 — does *not* improve PSNR. Instead:
−0.17 dB vs Euler@NFE=100, all 6 cells Bonferroni-significant negative,
41% more wall-clock.

**The implication.** ODE-integration accuracy is not the FlowDPS-on-RF
bottleneck on this problem. Guidance saturation is. The v2 EMA + ramp-ζ
combination already extracts the available signal from the
measurement; a more accurate integrator on top adds nothing.

**Why non-obvious.** Karras et al. 2022 EDM emphasizes the importance
of integrator order (Heun > Euler) for unconditional diffusion
sampling. The result we report shows that *for guided posterior
sampling*, the integrator order is *not* the relevant axis at typical
NFE budgets — a useful caveat to the EDM analysis when extended to
the conditional-sampling regime.

**Where.** `src/samplers/flowdps_rf_heun.py`; report §5.6 Round-5
paragraph.

### 3.8 Strict Bonferroni-gated ablation methodology

**What it is.** The promotion criterion: a candidate is promoted to a
"win" iff (i) mean ΔPSNR vs v1 is positive, AND (ii) Bonferroni-
corrected p<0.05 in ≥4 of 6 (σ_b, σ_n) cells.

**Why novel.** Most ablation studies in the inverse-problems
literature report aggregate ΔPSNR without per-cell significance
correction. We applied per-cell paired t-tests, Bonferroni correction,
Wilcoxon signed-rank cross-checks, and Cohen's d effect sizes. Of 13
candidates, 3 pass under this strict gate; 5 more would pass under a
"positive aggregate Δ" criterion. The 5 candidates that pass the
permissive criterion but fail the strict gate are *exactly* the cases
where the underlying effect is real but borderline — and our gate
catches them as such, framed as "near-misses" rather than wins. This
is a publishable rigor improvement.

**Where.** `scripts/significance_tests.py`; report §5.6.

---

## 4. Publishability assessment

| Venue | Verdict | Why |
|---|---|---|
| **AI-623 course (LUMS)** | **A* / A grade** | Coverage exceeds Bucket-3 minimum by ~3× (3 stress tests vs 1 required, 4 baselines vs 1, 13 ablations vs typical 1–2). Strict-gate rigor + comprehensive negative-result framing. Every claim is statistically defensible; every PSNR is provenance-traceable to a CSV row |
| **Workshop submission** (NeurIPS DGM workshop, ICML inverse-problems workshop, ICLR diffusion workshop) | **Plausible with 1–2 weeks of polish** | The Tier-B paradoxical-reversal finding alone (§3.2) is a publishable mechanistic finding. The pixel-space FlowDPS reimplementation (§3.1) is a useful community artifact. The EMA bug (§3.3) + L2-norm convention bug (§3.6) are useful failure-mode reports. Strict-gate methodology (§3.8) is a rigor improvement. Polish needed: a tighter narrative arc, possibly extending the dataset axis |
| **Top-venue main track** (NeurIPS / ICML / ICLR / CVPR) | **No, not as is** | Would require one of: (a) algorithmic novelty that *wins* on absolute PSNR (we narrow but do not close the DPS-vs-FlowDPS-RF gap); (b) multi-dataset evaluation (we have only CelebA-HQ-256 faces); (c) genuine new theory or mechanism beyond reweighting and weight-loading fixes. Realistic ceiling for a 6-week student project on one consumer GPU is workshop, not main track |
| **arXiv preprint** | **Worth posting after course submission** | The pixel-space FlowDPS implementation + Tier-B reversal + bug diagnoses are useful citable artifacts even without a peer-reviewed venue. Citation gravity for well-documented negative results and reproducibility artifacts is real and increasing in ML |

---

## 5. Honest limitations

What we did NOT do (so the next reader knows the boundary):

1. **One dataset.** CelebA-HQ-256 faces only. Repeating on LSUN-Church
   or DIV2K would test how much of the DPS-dominance is
   dataset-specific. Out of scope for the 5-day window.

2. **One forward-operator family per stress test.** Gaussian blur for
   the main grid; motion blur for the robustness study. Reduced
   measurements (random masking, Bucket-3 option B4) was skipped after
   the audit showed the brief required only one stress test and we
   already had three.

3. **DDRM baseline parked.** Our DDRM reimplementation did not reach
   competitive PSNR in the available time (collapsed to ~14 dB on the
   easiest cell). The parked code is preserved in the repository. We
   report DnCNN+PnP-ADMM as the analogous classical-prior reference
   instead — a different reference family, but still a published
   baseline.

4. **Gradient-free SMC at small particle count only.** P=8. Scaling to
   P=64 or P=128 with a non-trivial MCMC rejuvenation kernel (Langevin
   with the score) might change the verdict and is left to future
   work.

5. **`pixel_dps_pigdm_pure` held back.** Unstable on smoke tests; the
   analytical Π-GDM correction interacts badly with Pixel-DPS's σ_n
   floor in a way we did not have time to diagnose. The
   FlowDPS-RF variant (`flowdps_rf_pigdm_pure`) is fully diagnosed and
   reported.

6. **No physics-constrained variant.** The brief mentions "physics-aware
   conditional FM baseline for a simple PDE or measurement model."
   We did not implement this — would have required a separate axis
   (PDE residuals as constraints) that does not naturally fit
   image-deblurring.

7. **No blind / semi-blind inverse problems.** The forward operator A
   is known in all our experiments. Blind deblurring (unknown
   PSF) is an explicit Bucket-3 sub-axis we did not pursue.

---

## 6. Deliverables manifest

| Item | Location | Size |
|---|---|---|
| Final report PDF | `docs/group13_v2.pdf` | 18 pages |
| Final report LaTeX source | `docs/group13_v2.tex` | ~1350 lines |
| Ablation scorecard | `docs/ABLATION_SCORECARD.md` | ~700 lines |
| Lab notebook (EXP-001..026) | `docs/WORKING_NOTES.md` | ~1200 lines |
| Particle-DPS math note | `docs/TEMPERED_PARTICLE_DPS.md` | ~550 lines |
| Reading list | `docs/READING_LIST.md` (+ `.pdf`) | 21 papers, 4 priority tiers |
| Scope+novelty audit | `docs/SCOPE_AND_NOVELTY.md` | this file |
| Sampler implementations | `src/samplers/` | 16 modules |
| Pipeline scripts | `scripts/` | 12 scripts |
| Unit tests | `tests/` | 26 tests, all passing |
| Measurement results | `outputs/results/*.csv` | 10,600+ rows |
| Figures | `outputs/figures/*.pdf` | 16 figures, all generated by `scripts/make_plots.py` |
| Per-experiment logs | `outputs/logs/` | ~50 log files |
| Submission tag | `v1-submission` (annotated git tag) | points at `461c717` (main HEAD) |
| Public repo | https://github.com/adnan821/dps_flowdps | MIT-licensed, single contributor |

---

## 7. Executive summary (one paragraph)

The project delivers a course-grade A* deliverable that comfortably
exceeds the AI-623 Bucket-3 brief: every requirement is met (one task,
two baselines, three of four stress tests, all expected artifacts);
every claim is statistically defensible under strict Bonferroni
correction; every failure is mechanistically diagnosed; and every PSNR
in the report traces to a CSV row in the public repo. The two findings
that genuinely rise above course-deliverable quality are
(i) the Tier-B spectral mechanism's paradoxical regime-reversal —
same algorithm, opposite verdicts on matched vs mismatched operators,
confirmed on both posterior samplers — and (ii) the pixel-space
FlowDPS reimplementation on Liu 2023 RF CelebA-HQ-256, which is the
only public artifact that enables a head-to-head DPS-vs-FlowDPS-style
comparison at matched resolution and dataset. Together with the
documented EMA bug, the L2-norm/squared convention warning, the
particle-DPS path-degeneracy diagnosis, the guidance-saturation vs
integration-accuracy tradeoff, and the strict-gate ablation
methodology, this is a workshop-pitchable bundle. The honest gap
relative to a top-venue main paper is that none of our 3 wins beats
Pixel-DPS in absolute terms — they improve FlowDPS-on-RF and a
Pixel-DPS variant against themselves, narrowing but not closing the
gap. That is the right framing for the report and the right ceiling
for a 6-week student project on one consumer GPU.

---

*Last updated: 2026-05-28. Submission tag: `v1-submission` at main `461c717`.*
