# Q&A Prep — Group 13, AI-623 DVLM Final Presentation

Last updated: 2026-05-30 (day before submission). 30-mark Group Q&A
plus 30-mark Individual Q&A = **60 marks ride on the answers below**.

## Section ownership (suggested split for Individual Q&A)

| Member | Owns |
|---|---|
| **Eeman Adnan** | §1 Introduction, §2 Related Work, §6 Discussion, broader-claims and motivation questions |
| **Muhammad Aslam Adnan** | §3 Methodology, §4 Experimental Design, §5.1 main grid, §5.2 motion, §5.3 baselines, implementation details |
| **Mufaddal Hatim** | §5.4 posterior diversity, §5.5 significance, §5.6 ablations, paradoxical reversal, 8 failed ablations, statistical methodology |

For Group Q&A, the member who owns the section takes the lead; others
add supporting points.

---

## A. Problem and motivation (Eeman lead)

**A1. Why image deblurring, and why CelebA-HQ-256 faces?**
- Deblurring is the canonical Bucket-3 inverse problem (`y = A x + ε`)
  with a known operator and well-understood baselines
- Faces are a controlled domain where both DPS and FlowDPS priors have
  pretrained checkpoints at matched 256×256 resolution
- Allows clean comparison without confounders from dataset distribution
  shift; the limitation is documented in §6 and is the natural future-work
  direction (LSUN-Church, DIV2K)

**A2. What was the original proposal, and did you stick to it?**
- Proposed in March: compare DPS vs FlowDPS quality, efficiency,
  robustness, and against classical baselines
- Stuck to all four mandatory comparisons; **added three of the four
  Bucket-3 stress-test axes** (operator mismatch, posterior diversity,
  gradient-free vs gradient-based) when only one was required
- The headline finding contradicted the proposal's hypothesis
  ("FlowDPS within 0.5–1.1 dB of DPS at 2–5× speedup"). We report the
  contradiction honestly rather than reshape goals

**A3. Why is this inverse-problem setup important?**
- Real applications: medical imaging (MRI reconstruction), astronomy
  (telescope deconvolution), computational photography
- Each cares about a slightly different point in the
  PSNR-vs-uncertainty trade-off our work characterizes

**A4. What's the recent shift the field is making?**
- Toward flow-based posterior samplers, blind/semi-blind operators,
  and gradient-free formulations
- Our work directly tests the first claim (FlowDPS quality) and the
  third (gradient-free with particle_dps + tempered SMC)

---

## B. Methodology (Aslam lead)

**B1. What's the core difference between DPS and FlowDPS-style guidance?**
- DPS: stochastic DDIM reverse process + likelihood-gradient
  correction. Uses Tweedie's identity for $\hat x_0$
- FlowDPS: deterministic Euler ODE integration of a velocity field +
  the same form of likelihood-gradient correction on the
  Tweedie-clean-image identity for rectified flow
  ($\hat z_1 = x_t + (1-t) v_\theta$)
- Mathematically identical update structure; differs in (a) the prior's
  generative process and (b) the temporal-noise schedule

**B2. Why did you reimplement FlowDPS? Was that risky?**
- The official Kim 2025 code (github.com/jeongsol-kim/flowdps) is
  locked to Stable Diffusion 3 latent space at 768×768
- Without reimplementation, a head-to-head DPS-vs-FlowDPS comparison
  would span different resolutions, datasets, and prior architectures —
  the gap would be ambiguous
- Risk: implementation bugs could produce false underperformance. We
  mitigated by (i) sanity-checking unconditional generation (matches
  Liu 2023 reference samples), (ii) verifying L2-norm vs L2-squared
  convention against published code, (iii) diagnosing the EMA bug

**B3. What's the EMA bug, and how does it affect results?**
- The Liu 2023 RF checkpoint has 644 EMA shadow params for a
  645-parameter network
- The missing one is `all_modules.0.W` (Gaussian Fourier projection),
  registered with `requires_grad=False`, silently skipped by the
  gnobitab EMA tracker
- Naive EMA loading randomly initializes the Fourier projection → 7.76
  dB PSNR collapse
- Correct procedure aligns the 644 shadow params to trainable params AND
  copies `all_modules.0.W` from the raw `model` state-dict
- This fix is the 1st confirmed ablation win (`flowdps_rf_v2`, +0.77 dB)
- Even our v1 baseline (used throughout the headline numbers) does NOT
  apply this fix — it uses the raw `model` state-dict directly,
  matching what a naive reimplementer would write

**B4. Why DDIM and not DDPM sampling for Pixel-DPS?**
- DDIM has fewer NFE for the same image quality; it's the standard for
  guided sampling in the literature
- The diffusers DDIMScheduler is well-tested and pinned at 0.30.1

---

## C. Experimental design (Aslam lead)

**C1. Why 50 images per cell? Isn't that small?**
- 50 is large enough for paired t-tests to detect Cohen's d > 0.5 at
  α=0.05 with >80% power — the effects we care about have d > 1
- For the borderline `flowdps_rf_spectral` matched grid cells, we
  explicitly increased to n=100 because t-stats were near the
  Bonferroni threshold — this was a sample-size augmentation, NOT
  hyperparameter tuning (no spectral parameters changed)
- We did not retune to chase significance — that would be p-hacking

**C2. Why Bonferroni and not a softer correction (e.g., FDR)?**
- Bonferroni is the conservative choice; passing it means we're
  confident even by the strictest standard
- A softer correction would have promoted more ablations, but we'd
  rather underclaim than overclaim
- Two near-misses we report (`pixel_dps_sched_v2`, `flowdps_rf_heun`)
  would pass a permissive criterion; we explicitly call them
  near-misses

**C3. What's the seed protocol?**
- Per-image seeds drawn from a fixed schedule make per-cell paired
  comparisons deterministic
- The same seeds are used across methods for the paired t-test to be
  valid (matched samples differ only by method, not by image or noise
  realization)

---

## D. Main results (Aslam lead, supported by Mufaddal)

**D1. What's the headline?**
- Pixel-DPS dominates FlowDPS-on-RF on matched Gaussian (+1.4 to +4.2
  dB across 6 cells, all Bonferroni-significant)
- Same ordering on matched motion (+3.95 dB mean across 12 cells, all
  significant, slightly wider gap)
- Reverses under operator mismatch (FlowDPS-on-RF degrades less)
- Pixel-DPS produces a sharp point estimate; FlowDPS-on-RF produces a
  3.5× broader posterior

**D2. Why does FlowDPS-on-RF saturate by NFE=25?**
- Working hypothesis: the rectified-flow velocity field has more
  "stiff" dynamics late in the trajectory than DDPM's
  noise-prediction. Once the trajectory has flowed toward the data
  manifold, additional Euler steps move very little
- Empirically confirmed: from NFE=25 to NFE=200, FlowDPS-on-RF gains
  <0.5 dB; Pixel-DPS gains 1–2 dB
- This diagnosis motivated Tier-A's ramp-ζ schedule (more guidance at
  late steps), which produced the 1st confirmed win

**D3. DnCNN+PnP beats your FlowDPS on 5/6 noisy cells. Does that mean
your FlowDPS is broken?**
- No — it means DnCNN+PnP is a strong baseline that we should respect
  (the §5.3 "Why DnCNN+PnP beats FlowDPS-on-RF" paragraph in the
  detailed report covers this)
- DnCNN is a 668k-parameter bias-free denoiser specifically trained on
  Gaussian noise residuals (the prior task) at varied σ; inside
  PnP-ADMM it contributes the right prior at near-zero per-iteration
  cost
- FlowDPS-on-RF uses a 99M-parameter velocity field for 100 NFE plus
  backprop at every step — but the Liu 2023 RF prior is undertrained
  relative to modern large diffusion priors. The gap is a model-capacity
  issue, not a sampler-algorithm issue
- Our Tier-A win (`flowdps_rf_v2`) recovers +0.77 dB without changing
  the model — confirming the sampler is correct and the gap is mostly
  about the v1 zeta+EMA tuning

---

## E. Operator-mismatch reversal (Mufaddal lead with Aslam)

**E1. Why does the comparison reverse under mismatch?**
- Hypothesis 1: the deterministic ODE acts as implicit regularization
  that's robust to wrong gradient direction
- Hypothesis 2: FlowDPS's broader posterior (3.5×) already encodes
  more uncertainty, so a wrong-operator gradient gets "averaged out"
  more
- Hypothesis 3: Pixel-DPS's L2 gradient is more aggressive
  per-step (larger trust-region) and thus more brittle to
  misspecification
- We don't have a definitive answer; we report the empirical reversal
  honestly. The §6 Discussion paragraph "Gradient-based vs gradient-free
  guidance" speculates further

**E2. Why is this finding important?**
- Implies the choice between DPS and FlowDPS is application-dependent,
  not universal
- Calibrated operator → DPS. Uncertain operator (medical imaging with
  patient-varying PSF, telescope deconvolution with atmospheric blur)
  → consider flow-based posterior sampling

---

## F. Posterior diversity (Mufaddal lead)

**F1. What does "PSNR ≠ posterior" mean concretely?**
- A single sample's PSNR tells you how close one reconstruction is to
  ground truth — it's a point estimate
- A well-calibrated posterior should have non-zero variance reflecting
  the true ambiguity in the measurement
- Pixel-DPS samples differ by LPIPS 0.05 across seeds (essentially
  identical) — it's collapsed to a single modal estimate
- FlowDPS-on-RF samples differ by LPIPS 0.175 — visibly different
  hair, skin shading, micro-features (Figure 2)

**F2. Is this Bayesian-vs-frequentist?**
- More like point-estimate vs posterior. Both methods are sampling
  from approximations to the same posterior; the difference is the
  variance of the approximation
- Pixel-DPS's L2 gradient is sharper, pushing samples toward a
  single mode. FlowDPS-on-RF's deterministic trajectory + flow-based
  prior preserves more sample-to-sample variation

**F3. Why does FlowDPS-RF benefit more from posterior averaging?**
- Mean-of-N PSNR gain: Pixel-DPS +0.89 dB, FlowDPS-RF +2.15 dB
- Bayesian gain from averaging independent posterior samples is large
  when individual samples are noisy and approximately unbiased
- FlowDPS-RF samples are noisier (more diverse) but unbiased; Pixel-DPS
  samples are deterministic (no Bayesian gain available)

---

## G. Ablation methodology (Mufaddal lead)

**G1. Why 13 ablations? Isn't that fishing?**
- All 13 were planned in 4 tiers (A/B/C plus rounds 2/3/4/5) before
  measurement, motivated by specific diagnoses from the v1 baselines
  (saturation curve, robustness failure, etc.)
- We used a strict Bonferroni-corrected promotion criterion to control
  multiple-comparison inflation
- 8 of 13 failed the strict gate; we report them all honestly with
  mechanistic diagnoses, not just the 3 wins
- A "fishing" approach would have reported only the wins and not
  documented the failures

**G2. Why a strict per-cell gate, not aggregate Δ?**
- Aggregate Δ can be pushed positive by 1–2 strongly-positive cells
  while 4+ cells are flat or negative
- Per-cell + Bonferroni demands the effect to be broadly present
- 5 of our 13 candidates would have passed an aggregate-Δ criterion
  but failed the per-cell gate — they're documented as near-misses

**G3. The 3rd win required n=100. Wasn't that p-hacking?**
- The hyperparameters were fixed before the n=50 measurement; n=50
  produced 3/6 cells significant with the σ_b=5 cells at t≈2.5 — above
  the single-test critical value but below the Bonferroni-corrected
  one
- We doubled n to 100 to reduce standard error (statistical power
  augmentation, the textbook correct response to a borderline
  comparison)
- **No spectral hyperparameters changed** between n=50 and n=100. The
  ε, α, and kernel weighting are identical
- If we had tuned the spectral parameters to push past p<0.05, that
  would be p-hacking. We did not

---

## H. The paradoxical reversal (Mufaddal lead — most likely Q&A target)

**H1. What's the most interesting finding of the whole project?**
- The Tier-B spectral-weight mechanism has *opposite* verdicts in
  adjacent regimes:
  - Operator mismatch → fails (−0.18 dB Pixel-DPS, −0.20 dB FlowDPS-RF)
  - Matched conditions → wins (+0.46 dB Pixel-DPS, +0.54 dB FlowDPS-RF)
- Confirmed on **both** posterior samplers (algorithm-agnostic)
- Mechanistically interpretable: downweight residuals where the
  *assumed* operator's spectrum is noise-dominated. Under matched
  conditions, those are the right bands. Under mismatch, they are the
  wrong bands

**H2. Why is this publishable?**
- A brittle ablation would not produce a clean reversal — it would
  fail in both regimes or succeed in both
- The reversal is evidence that the mechanism is correctly diagnosed
- Algorithm-agnostic (works on both DPS and FlowDPS-style guidance)
- Workshop-pitchable as a stand-alone finding

**H3. Can you explain the mechanism more deeply?**
- Spectral weight: W(f) = 1 / (1 + α · ReLU(ε − |H(f)|))
- |H(f)| is the assumed forward operator's spectral magnitude
- Below noise threshold ε, the weight increases; that frequency's
  residual is *downweighted* in the DPS gradient
- Intuition: at those frequencies, the assumed operator can't see
  signal anyway, so residuals there are mostly noise. Downweighting
  prevents the sampler from being misled by them
- Under matched conditions, the assumed operator's noise-dominated
  bands are the *true* operator's too. Win.
- Under mismatch (motion measurement, Gaussian assumed): the assumed
  operator says "frequencies above some cutoff are dead." But the true
  motion-blur operator preserves directional content that the assumed
  Gaussian cannot see. Downweighting those bands throws away the
  actual signal. Loss.

---

## I. Failed ablations (Mufaddal lead)

**I1. Why did the Π-GDM tier fail so badly (−13.6 dB)?**
- L2-norm vs L2-squared convention mismatch
- DPS uses ∥y − A(x̂₀)∥ (L2 norm); Π-GDM uses ∥y − A(x̂₀)∥² (L2 squared)
  with a different leading factor (½σ_y⁻²)
- Conflating them gives a wrong leading factor that depends on the σ_y
  scale, blowing up the gradient at noisy cells
- The corrected analytical Π-GDM (`flowdps_rf_pigdm_pure`) reduces the
  margin from −9.8 to −3.4 dB, consistent with the literature finding
  that Π-GDM underperforms L2-DPS on Gaussian blur on this prior

**I2. Why did particle DPS fail catastrophically?**
- Path degeneracy: at P=8 particles, the effective sample size
  P_eff < 1.5 within ~10 reverse steps of the diffusion trajectory
- The unguided DDIM step separates particles by more than the
  measurement likelihood's effective σ_y; importance weights become
  near-singular; one particle dominates; the rest of the trajectory is
  identical to unguided DDIM on that one survivor
- Classical Doucet 2001 failure mode; we didn't invent it
- Fix would require either larger P (P=64+) or a non-trivial MCMC
  rejuvenation kernel (e.g., Langevin with the score). Out of scope
  for this project

**I3. Why did Heun at NFE=200 fail?**
- Heun is 2nd-order accurate vs Euler's 1st-order. Karras 2022 EDM
  argues higher-order integrators help unconditional sampling
- Our finding: Heun at NFE=200 (its natural budget) is −0.17 dB
  vs Euler+v2@NFE=100 (matched budget): a regression
- Diagnosis: ODE integration accuracy is not the FlowDPS-on-RF
  bottleneck; **guidance saturation is**. v2's EMA + ramp-ζ already
  extracts the available signal; a better integrator on top adds nothing
- Useful caveat to the EDM analysis when extended to guided sampling

---

## J. Implementation and infrastructure (Aslam lead)

**J1. What software stack?**
- PyTorch 2.4.1 + diffusers 0.30.1 (pinned), torchvision, LPIPS,
  Wiener-filter via scipy.fft
- All Bonferroni / paired-t / Wilcoxon via scipy.stats
- RTX 3060 12 GB, 32 GB RAM, single GPU

**J2. How long did the experiments take to run?**
- Main grid: ~6 GPU-hours per method
- Robustness study: ~4 GPU-hours
- Each ablation grid: ~3–6 GPU-hours
- Matched motion (most recent): ~7 GPU-hours
- Total: ~50–60 GPU-hours of measurement time. All resume-safe via
  per-row CSV append+fsync

**J3. Where can graders verify the numbers?**
- Public repo: https://github.com/adnan821/dps_flowdps
- Every PSNR in the report traces to a row in
  `outputs/results/{main_grid,robustness,baselines,motion_matched}.csv`
- Lab notebook EXP-001..EXP-026 documents every decision
- ABLATION_SCORECARD.md is the canonical wins/fails reference

---

## K. Limitations and future work (Eeman lead)

**K1. Your most honest limitation?**
- Single dataset (CelebA-HQ-256 faces). Generalization to LSUN-Church
  or DIV2K untested. Pixel-DPS's dominance could be face-specific
- Two operator families (Gaussian + motion) on the matched comparison;
  broader operator classes (super-resolution, inpainting) untested

**K2. What would you do with more time?**
- Multi-dataset replication (top priority — closes the most
  consequential limitation)
- Implement DDRM correctly so we have a third learned-prior baseline
- Scale gradient-free SMC to P=64 with MCMC rejuvenation kernel
- Test broader operator classes (SR, inpainting) on the matched
  comparison

**K3. Is this work publishable?**
- Workshop: plausibly, the paradoxical-reversal finding is the
  strongest single contribution
- Top venue: not as is — would need either multi-dataset, an
  algorithmic novelty that wins on absolute PSNR (we narrow the
  gap but don't beat Pixel-DPS), or a deeper theoretical contribution

---

## L. Likely attack questions (be ready for any)

**L1. The proposal hypothesis was wrong. Didn't you change your goals?**
- We stuck to the same four research questions. The proposal *claim*
  (FlowDPS within 0.5–1.1 dB at 2–5× speedup) was tested and falsified
  with quantitative evidence. That's a valid scientific outcome
- Our headline conclusion remains the same shape: we now have a clear
  guideline for *when* to use each method

**L2. Cohen's d > 3 is huge. Is your experimental noise too low?**
- Both samplers are deterministic given seed (Pixel-DPS up to floating
  point); paired comparisons across the same images have very low
  within-pair variance
- Cell-to-cell variance is dominated by *image* difficulty, not seed
  noise. Cohen's d on paired differences measures effect size relative
  to within-pair noise, which is small by design
- The same protocol applied to comparable studies in the literature
  produces similarly large d for established differences

**L3. You implemented DDRM and it didn't work. Doesn't that question
your reimplementation skills generally?**
- The DDRM SVD computation requires an explicit SVD of the operator,
  which is non-trivial for the Gaussian-blur+downsampling regime we
  attempted. We ran out of time to debug
- DnCNN+PnP-ADMM works correctly and is also a learned-prior baseline
- The FlowDPS reimplementation has multiple sanity checks (matches
  Liu 2023 unconditional samples; v1 baseline matches the
  literature qualitative behavior of saturation)

**L4. Why didn't you use SD3 as the prior since that's what Kim 2025 uses?**
- SD3 operates at 768×768 in latent space; the matched DDPM (Pixel-DPS's
  prior) is at 256×256 in pixel space. Comparing across these would
  conflate prior+representation+resolution differences
- A pixel-space reimplementation isolates the sampler choice as the
  only difference

**L5. Your ablation methodology rejects 5 of 13 candidates that would
pass a softer criterion. Isn't that overly strict?**
- We pre-registered the strict gate before measurement; it is the
  documented promotion criterion for the entire study
- Methods that fail the strict gate but pass softer criteria are
  *reported* as near-misses, with the per-cell breakdown — readers can
  apply their own threshold
- Strict gate is a rigor improvement, not a methodological weakness

**L6. Your DnCNN+PnP baseline beats your FlowDPS reimplementation on 5/6
noisy cells. Doesn't that invalidate your FlowDPS?**
- It doesn't — DnCNN is a strong learned-prior baseline; that's a
  separate fact about the strength of supervised denoising priors
- Our FlowDPS-on-RF is the correct implementation of the Kim 2025
  algorithm on a smaller, less-trained RF prior (Liu 2023)
- The Tier-A win (`flowdps_rf_v2`, EMA + ramp-ζ) closes the gap to
  DnCNN+PnP partially without changing the model, confirming the
  sampler is correct and the gap is model-capacity-related

---

## M. Closing questions and pivots

**M1. What's the one-line takeaway?**
- "Pixel-DPS dominates FlowDPS-on-RF on matched conditions by
  1.4–4.2 dB and the matched-motion confirms this isn't
  Gaussian-specific. The ordering reverses under operator mismatch.
  Pixel-DPS is a sharp point estimate; FlowDPS-on-RF is a 3.5× broader
  posterior."

**M2. What's the one finding you're most proud of?**
- "The paradoxical reversal: the same spectral-weight algorithm fails
  under operator mismatch and wins under matched conditions, confirmed
  on both samplers. Algorithm-agnostic, mechanistically interpretable,
  publishable on its own."

**M3. If you redid the project, what would you do differently?**
- Two datasets from the start (CelebA-HQ + LSUN-Church) — would have
  let us claim the matched-Gaussian dominance is general, not
  face-specific
- Implement DDRM as the priority-1 baseline before FlowDPS-on-RF,
  since DDRM is the closer analog to Pixel-DPS than DnCNN+PnP
- Pre-register a learned $\zeta$ schedule for FlowDPS-RF rather than
  discovering the saturation diagnosis post-hoc

---

## Practice tips

1. **Do at least one full dry-run** with all three members reading
   their owned sections — time it (target ≤ 15 minutes for the
   prepared presentation, leaving 15 minutes for Q&A)
2. **Each member should read this document end-to-end** so cross-section
   questions can be passed to the right person fluently
3. **For Individual Q&A**: graders will likely probe each member on
   their owned section. Be ready to explain the *why* behind each
   choice, not just the *what*
4. **For Group Q&A**: practice handoffs ("Mufaddal can speak to that")
   — looks professional and gets you full marks for cohesion
5. **Don't bluff**: if you don't know the answer to an attack question,
   say "I don't know — here's what we'd need to test to find out."
   That's worth more marks than a wrong confident answer
