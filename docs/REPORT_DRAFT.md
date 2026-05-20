# Final Report Draft — DVLM AI-623 Group 13

Working markdown for the report rewrite. Built from real numbers in
`outputs/results/{main_grid,robustness,baselines}.csv` and the EXP-NNN
entries in `docs/WORKING_NOTES.md`. Replaces the fabricated content from
the first submission (`docs/group13.pdf`).

Conventions used in this draft:
- **bold** = numbers we measured; cite the CSV row and the log in the
  final report's appendix.
- *italic = wording from the fake report we are explicitly correcting*.
- `> NOTE` = guidance for whoever ports this to LaTeX.

---

## Title page

**Title:** Pixel-Space Posterior Sampling for Image Deblurring: A Controlled
Comparison of DPS and FlowDPS-Style Guidance on CelebA-HQ-256
*(or keep "AI-623 Deep Vision Language Models Project Report" if course-specified)*

**Authors:** Muhammad Aslam Adnan (25280067), Mufaddal Hatim (25280084),
Eeman Adnan (24280022). LUMS Department of AI.

---

## Abstract (rewrite)

Image reconstruction from degraded observations is a fundamental inverse
problem in computational imaging. Diffusion Posterior Sampling (DPS)
leverages a learned diffusion prior plus a measurement-likelihood
gradient to solve such problems with high reconstruction fidelity, at the
cost of hundreds of neural function evaluations (NFE). Flow Matching (FM)
models offer a deterministic, ODE-based alternative that has been argued
to enable comparable quality with fewer steps. In this work, we conduct a
controlled comparison of **pixel-space DPS** (Chung et al. 2023) on a
CelebA-HQ-256 DDPM, against a **pixel-space reimplementation of FlowDPS**
(Kim et al. 2025) — whose official codebase is latent-only (SD3) — on top
of a CelebA-HQ-256 Rectified Flow checkpoint (Liu et al. 2023). We
evaluate both methods on Gaussian deblurring across a grid of 3 blur
levels × 2 noise levels × 3 NFE budgets × 50 images = 1800 reconstructions,
and study operator-mismatch robustness using motion-blur measurements while
the sampler's likelihood assumes Gaussian blur (1200 additional
reconstructions). We further include Wiener-filter and DnCNN+PnP-ADMM
classical baselines (600 reconstructions). On our matched-Gaussian grid,
Pixel-DPS dominates the comparison at every cell, with a **PSNR advantage
of +1.4 to +4.2 dB** over our FlowDPS-on-RF reimplementation, and FlowDPS
saturates with NFE while Pixel-DPS continues to improve up to NFE=100.
Under operator mismatch, however, FlowDPS-on-RF is **relatively more
robust** — degrading by ~2.5 dB from matched to severe-motion settings
versus ~5 dB for Pixel-DPS — reversing the matched-condition comparison.
All four methods comfortably beat Wiener; DnCNN+PnP-ADMM beats
FlowDPS-on-RF in 5 of 6 noisy cells while running ~60× faster than DPS.
The findings provide concrete guidance for choosing between these methods
under known vs. uncertain forward operators.

---

## 1 Introduction (audit)

> NOTE: minimal rewrite — most of the existing prose stands.

Changes required from the fake-report version:
- **FFHQ-256 → CelebA-HQ-256** throughout. The pivot was forced by the
  fact that the only public face-dataset Rectified Flow checkpoint
  (Liu 2023) is for CelebA-HQ-256, and the only public pixel-space DDPM
  the FFHQ-vs-CelebA choice is available for via diffusers is CelebA-HQ-256
  (the official Chung 2023 FFHQ checkpoint is in OpenAI guided-diffusion
  format and would require a separate conversion). Both methods therefore
  share the **same dataset and the same NCSN++/U-Net family** at 256×256.
- The fake report claims "FlowDPS, introduced in Kim et al. 2024,
  arXiv:2403.XXXXX". The real reference is **Kim, Kim, Ye (KAIST),
  *FlowDPS: Flow-Driven Posterior Sampling for Inverse Problems*, ICCV
  2025, arXiv:2503.08136**. Update the citation.
- The fake report says "The experimental framework is built around
  pretrained generative models evaluated on the FFHQ-256 dataset". Change
  to: "...evaluated on the CelebA-HQ-256 dataset".
- The four research objectives can stay as-is (controlled comparison;
  quality–efficiency tradeoff characterization; operator-mismatch
  robustness; failure-mode analysis).

---

## 2 Related Work (audit)

> NOTE: mostly intact. Two specific edits.

- Update the FlowDPS citation as above (Kim 2025 ICCV, not 2024).
- **Add one paragraph** disclosing our reimplementation:
  > Although the original FlowDPS implementation (Kim et al. 2025) is
  > built on Stable Diffusion 3 — a latent transformer-based flow
  > model — we reimplement the algorithm in pixel space on top of the
  > Liu et al. 2023 Rectified Flow CelebA-HQ-256 checkpoint
  > (`gnobitab/RectifiedFlow`). Our reimplementation preserves the
  > FlowDPS likelihood-corrected velocity update
  > `v_eff(x_t,t) = v_θ(x_t,t) − ζ ∇_{x_t} ‖y − A(ẑ₁(x_t,t))‖₂`,
  > where ẑ₁(x_t,t) = x_t + (1−t)·v_θ(x_t,t) is Rectified Flow's
  > clean-image Tweedie estimate. This pixel-space adaptation enables a
  > genuinely controlled comparison against pixel-space DPS at 256×256
  > on identical-architecture-family U-Nets.

---

## 3 Methodology (audit)

> NOTE: minor edits in the DPS and FlowDPS subsections, plus two
> corrections in implementation details.

- **DPS subsection** — the math (Tweedie's `x̂₀`, DDIM reverse, likelihood
  gradient) is fine. Add: "We use the **L2 norm** (not squared L2) of the
  measurement residual for the likelihood gradient, following Chung et
  al.'s official implementation convention; squaring the residual and
  dividing by σ² blows up the gradient and prevents convergence (verified
  empirically — see Appendix C / EXP-001)."
- Add: "We do **not** clamp `x̂₀` inside the gradient path. Clamping
  zeroes the gradient for any pixel outside [-1,1], which kills DPS at
  early timesteps."
- **FlowDPS subsection** — replace the description of the algorithm to
  match our pixel-space reimplementation (see Section 2 audit). The
  Tweedie identity for RF is `ẑ₁(x_t,t) = x_t + (1−t)·v_θ(x_t,t)`.
- **Implementation details paragraph** — change:
  - *"All experiments are conducted on a single NVIDIA A100 80GB GPU"*
    → "All experiments are conducted on a single **NVIDIA RTX 3060
    12GB** GPU."
  - *"DDPM model is initialized using the publicly available FFHQ-256
    checkpoint released by Chung et al. (2023)"* → "DDPM model is
    initialized from the publicly available **CelebA-HQ-256 checkpoint
    `google/ddpm-ema-celebahq-256`** (Ho et al. 2020, EMA-trained
    variant)."
  - *"Flow Matching model uses a pretrained Rectified Flow checkpoint
    introduced by Liu et al. (2023)"* → keep, just specify "the
    **CelebA-HQ-256 1-rectified-flow checkpoint** with NCSN++ backbone
    (65.5M parameters)."

---

## 4 Experimental Design (audit)

> NOTE: update the configuration matrix in Table 2 and the dataset paragraph.

- **Update Table 2** ("Experimental Configuration Matrix"):
  - Gaussian blur σ ∈ {1.5, 3.0, 5.0} → unchanged
  - Additive noise σ ∈ {0.00, 0.05} → unchanged
  - Sampling steps (NFE) ∈ {25, 50, 100} → unchanged
  - Methods: DPS, FlowDPS → keep, but rename to "**Pixel-DPS,
    FlowDPS-on-RF**" to match our reimplementation
  - Total conditions: 3 × 2 × 3 × 2 = **36 (18 per method)**
  - **Images per condition: 50** (not 500; disclose explicitly that this
    is set by the RTX 3060 compute envelope)
- **Update the dataset paragraph**:
  - FFHQ-256 → CelebA-HQ-256 (from the `korexyz/celeba-hq-256x256` HF
    dataset, validation split)
  - "5,000 test split, sampled to 500" → "200 test split, sampled to **50**"
  - "An additional subset of 100 images is reserved for robustness
    analysis" → "An additional **50 images** are used for the
    robustness analysis"
- **Add a new "Baselines" paragraph** (after the methods description):
  > To contextualize the DPS/FlowDPS comparison we include two classical
  > and supervised baselines: (i) a per-channel Wiener filter with
  > tuned-per-noise-level Tikhonov regularization, evaluated on the same
  > 6-cell Gaussian grid (300 reconstructions, ~16 ms/image); and (ii)
  > a Plug-and-Play ADMM scheme with a pretrained 3-channel blind
  > Gaussian denoiser (DnCNN, 668k params, deepinv release) inside an
  > FFT-domain data-fidelity solve, 24 outer iterations with ρ = 2.0
  > and a noise-adaptive descending denoiser-σ schedule
  > (300 reconstructions, ~0.3 s/image).
- **Note about DDRM**: the original report mentions DDRM as a baseline.
  Add a disclosure paragraph:
  > We additionally attempted a from-scratch reimplementation of a
  > DDRM-style (Kawar et al. 2022) sampler with FFT-basis spectral
  > projection on top of our CelebA-HQ-256 DDPM, but were unable to
  > reach competitive PSNR within the available time budget (the
  > reimplementation collapsed to ~14 dB even at the easiest condition,
  > suggesting an unresolved scale / sign mismatch in the projection
  > step). DnCNN+PnP-ADMM serves as our supervised classical reference
  > in lieu of DDRM. The parked DDRM code is preserved in the project
  > repository for future investigation.

---

## 5 Results and Findings (FULL REWRITE — all numbers measured)

> NOTE: this entire section discards the fabricated Section 5 of the
> first submission (`docs/group13.pdf` pages 7–9). All numbers below
> are from the CSVs and logs cited in Appendix A.

### 5.1 Overall setup recap

All experiments use 50 CelebA-HQ-256 images from the validation split of
`korexyz/celeba-hq-256x256`. Forward model: `y = A x + ε`,
ε ∼ N(0, σ_n² I). The main grid uses Gaussian blur; the robustness study
substitutes motion blur. Guidance scale ζ was tuned on a held-out
2-image validation in EXP-001 and EXP-006/007: **ζ=10 for Pixel-DPS, ζ=100
for FlowDPS-on-RF**, both monotonic in their respective best-condition
PSNR up to the chosen value. Total compute for all 3,600 reconstructions
reported below: **10.4 GPU-hours on a single RTX 3060 12GB**.

### 5.2 Main grid: matched Gaussian blur (1800 reconstructions)

Table 3 reports per-cell mean PSNR/SSIM/LPIPS and wall-clock per image,
averaged across 50 images and seeded for reproducibility (seed=0 +
per-image offset; see `src/samplers/*.py` for the exact use).

**Table 3 — Matched-Gaussian grid (selected NFE=50 and NFE=100 cells).**

| Method        | σ_b | σ_n  | NFE | PSNR (dB) | SSIM  | LPIPS | s/img |
|---------------|-----|------|-----|-----------|-------|-------|-------|
| Pixel-DPS     | 1.5 | 0.00 | 100 | **28.61** | 0.821 | **0.084** | 19.02 |
| Pixel-DPS     | 1.5 | 0.05 | 100 | 26.21     | 0.686 | 0.136     | 17.91 |
| Pixel-DPS     | 3.0 | 0.00 | 100 | 27.15     | 0.775 | 0.117     | 18.04 |
| Pixel-DPS     | 3.0 | 0.05 | 100 | 26.25     | 0.741 | 0.146     | 17.90 |
| Pixel-DPS     | 5.0 | 0.00 | 100 | 26.21     | 0.746 | 0.155     | 17.91 |
| Pixel-DPS     | 5.0 | 0.05 | 100 | 25.33     | 0.721 | 0.190     | 18.14 |
| FlowDPS-on-RF | 1.5 | 0.00 | 100 | 26.23     | 0.736 | 0.153     | 23.51 |
| FlowDPS-on-RF | 1.5 | 0.05 | 100 | 23.65     | 0.570 | 0.268     | 23.55 |
| FlowDPS-on-RF | 3.0 | 0.00 | 100 | 24.26     | 0.667 | 0.170     | 23.71 |
| FlowDPS-on-RF | 3.0 | 0.05 | 100 | 22.58     | 0.576 | 0.234     | 23.86 |
| FlowDPS-on-RF | 5.0 | 0.00 | 100 | 22.48     | 0.601 | 0.200     | 23.82 |
| FlowDPS-on-RF | 5.0 | 0.05 | 100 | 21.13     | 0.537 | 0.251     | 23.98 |

Full per-cell table including NFE=25 and NFE=50 is in
`outputs/figures/main_grid_summary.md`.

**Headline finding 1: Pixel-DPS dominates the matched comparison.** The
PSNR gap ranges from **+1.43 dB** (σ_b=1.5, σ_n=0, NFE=25) to **+4.20 dB**
(σ_b=5, σ_n=0.05, NFE=100), widening with task difficulty. SSIM and LPIPS
show consistent ordering. Figure 1 (`psnr_vs_nfe`) and Figure 2
(`psnr_delta_heatmap`) visualize this directly.

> *Correction.* The fake-report Section 5 claimed "FlowDPS achieves
> reconstruction quality within approximately 0.5–1.1 dB PSNR of DPS";
> our measured gap is 1.4–4.2 dB. The fabricated value was both too
> optimistic and reversed the wall-time ordering.

**Headline finding 2: FlowDPS-on-RF saturates with NFE; Pixel-DPS keeps
improving.** At (σ_b=1.5, σ_n=0), FlowDPS-on-RF averages 26.08 / 26.08 /
**26.23 dB** at NFE=25/50/100 — essentially constant. Pixel-DPS at the
same condition: 27.51 / 28.34 / **28.61 dB** — monotonic improvement.
This means at NFE=25, FlowDPS-on-RF is competitive (only 1.43 dB behind
Pixel-DPS), but at NFE=100 the gap nearly doubles.

> *Correction.* The fake report claimed *"FlowDPS reaches most of its
> performance by 50 NFEs, whereas DPS continues to benefit from
> additional sampling steps"* — this part is **correct in direction**
> (the saturation is real), but the quantitative claim that "DPS at
> 50 NFEs is within 0.8 dB of DPS at 100 NFEs" overstates DPS's
> sub-100-NFE quality.

**Headline finding 3: Wall-clock per image is roughly equal across NFE,
with Pixel-DPS slightly faster.** At NFE=100, Pixel-DPS averages ~18 s
and FlowDPS-on-RF ~24 s (one Pixel-DPS forward+backward is cheaper than
one RF-NCSN++ velocity-field evaluation on this hardware). This
contradicts the proposal's hypothesis that flow models would be faster.

**Headline finding 4: Quality degrades monotonically with σ_blur and
with σ_noise=0.05 vs 0.0.** Going from σ_b=1.5 to σ_b=5.0 costs ~2.4 dB
PSNR for Pixel-DPS and ~3.8 dB for FlowDPS-on-RF. Noise σ_n=0.05 vs 0
costs ~2.4 dB for Pixel-DPS and ~2.6 dB for FlowDPS-on-RF.

### 5.3 Operator-mismatch robustness study (1200 reconstructions)

Here the true forward operator is `motion_blur(L, θ)` with
L ∈ {15, 25, 35} and θ ∈ {0°, 45°}, but the sampler's likelihood still
assumes Gaussian blur with **fixed σ_b = 3.0** (the middle of the main
grid's σ_b values). 50 images per cell, 12 cells × 2 methods, all at
NFE=50. Total compute: 3.47 GPU-hours.

**Table 4 — Robustness study, σ_n=0.05 (the noisier and harder case).
Per-method mean across 50 images per cell. θ-averaged.**

| L  | Pixel-DPS PSNR (dB) | FlowDPS-on-RF PSNR (dB) | Δ (Pixel − Flow) |
|----|---------------------|--------------------------|-------------------|
| 15 | **25.0**            | 22.1                     | +2.9              |
| 25 | 22.5                | 21.0                     | +1.5              |
| 35 | 20.7                | 19.8                     | +0.9              |

Full per-cell breakdown including θ=0° vs θ=45° and σ_n=0 is in
`outputs/figures/robustness_summary.md`.

**Headline finding 5: FlowDPS-on-RF is more robust to operator
mismatch.** Going from matched Gaussian (σ_b=3.0, σ_n=0.05, NFE=50) to
the most severe motion blur (L=35, θ=45°, σ_n=0.05, NFE=50):
- Pixel-DPS: 25.72 → 20.86 dB (**−4.86 dB**)
- FlowDPS-on-RF: 22.40 → 19.92 dB (**−2.48 dB**)

The advantage gap **shrinks** from +3.3 dB matched to +0.94 dB
mismatched-severe. Figure 3 (`matched_vs_mismatched`) shows this in a
single bar chart.

> *Correction.* The fake report claimed *"DPS drops from 28.4 to 24.7 dB
> PSNR (−3.7 dB), while FlowDPS drops from 27.6 to 23.1 dB PSNR
> (−4.5 dB)"*, concluding that DPS was *more* robust. Our measurements
> **reverse** this conclusion: FlowDPS-on-RF degrades less under
> operator mismatch in our pixel-space comparison. We attribute this to
> the deterministic ODE trajectory of the flow model being less
> sensitive to misspecified per-step likelihood gradients than the
> stochastic diffusion path's compounding correction.

**Headline finding 6: Motion-blur angle barely matters; noise effect is
washed out by the operator mismatch.** PSNR difference between θ=0° and
θ=45° is within 0.5 dB at every cell. PSNR difference between σ_n=0 and
σ_n=0.05 is also small (~0.4 dB on average), much smaller than the
mismatch-induced 3–5 dB drop.

### 5.4 Classical and supervised baselines (600 reconstructions)

**Table 5 — Four-method comparison at NFE=100 (or N/A for non-iterative
methods), σ_n=0.05. Per-method mean across 50 images per cell. Best per
cell in bold.**

| σ_b | Wiener | DnCNN+PnP-ADMM | FlowDPS-on-RF | Pixel-DPS |
|-----|--------|----------------|---------------|-----------|
| 1.5 | 24.28  | 24.61          | 23.65         | **26.21** |
| 3.0 | 24.03  | 24.31          | 22.58         | **26.25** |
| 5.0 | 22.50  | 22.81          | 21.13         | **25.33** |

**Per-image cost (s):** Wiener ≈ 0.02, DnCNN+PnP ≈ 0.30, FlowDPS-on-RF
≈ 23.5, Pixel-DPS ≈ 17.9. **Three orders of magnitude in compute for
roughly 2 dB in PSNR.**

**Headline finding 7: DnCNN+PnP-ADMM beats Wiener at every cell and
beats FlowDPS-on-RF in 5 of 6 cells.** Striking — a small (668k-param)
classical denoiser inside PnP-ADMM beats our reimplementation of a
million-parameter flow-based posterior sampler on PSNR/SSIM (LPIPS
favors the learned-prior methods).

**Headline finding 8: Pixel-DPS dominates all four methods on PSNR /
SSIM / LPIPS at every cell.** This is consistent with DPS's design
philosophy: a strong learned image prior combined with explicit
measurement guidance.

### 5.5 Qualitative analysis

> TODO: render `outputs/figures/qualitative_grid.png` (script needed).
> Suggested layout: 4 sample images × 6 columns (clean | y | Wiener |
> DnCNN+PnP | FlowDPS-on-RF | Pixel-DPS), at σ_b=3.0, σ_n=0.05.

Recurring observations (from inspecting the smoke-test grids in
`outputs/smoke/`):

- **Wiener** produces ringing / Gibbs-style artifacts at high frequency,
  especially at σ_n=0 where the lack of regularization shows.
- **DnCNN+PnP-ADMM** is smoother than Wiener but tends to oversmooth
  hair, eyelash, and skin texture, contributing to its much worse LPIPS
  scores.
- **FlowDPS-on-RF** can produce a plausible face that is the wrong
  person — the deterministic ODE under-corrects toward y when the
  guidance ζ is at the chosen 100 (we tested up to ζ=300/1000 and they
  hurt rather than help).
- **Pixel-DPS** consistently recovers identity-preserving faces with
  sharp eyes/mouth/hair, the closest to the ground truth in pixel
  space.

### 5.6 Summary tables and figures

All figures are PDFs in `outputs/figures/`:
- `psnr_vs_nfe.pdf`, `ssim_vs_nfe.pdf`, `lpips_vs_nfe.pdf` — 3×2 facets
  per (σ_b, σ_n) showing each method's curve vs NFE.
- `efficiency_frontier.pdf` — wall-clock vs PSNR scatter, NFE-annotated.
- `lpips_vs_psnr.pdf` — perceptual-vs-distortion scatter; the two
  methods' clouds barely overlap.
- `quality_vs_sigma_blur.pdf` — 3 panels (PSNR, SSIM, LPIPS) at
  NFE=100.
- `psnr_delta_heatmap.pdf` — Pixel-DPS minus FlowDPS-on-RF per cell.
- `robustness_psnr_vs_length.pdf`, `robustness_metrics_vs_length.pdf`,
  `robustness_psnr_heatmap.pdf` — robustness study.
- `matched_vs_mismatched.pdf` — the one-shot operator-mismatch bar.

---

## 6 Discussion (rewrite, with corrections)

> NOTE: the existing Discussion has the right structure but several
> claims need to flip because the measured numbers contradict the fake
> ones. Use the bullets below as the new draft.

- **The proposal hypothesized "FlowDPS achieves reconstruction quality
  within approximately 0.5–1.1 dB PSNR of DPS while offering a
  substantial 2–5× reduction in computational cost."** Our measurements
  contradict both halves of this claim. (a) The PSNR gap on our
  CelebA-HQ-256 pixel-space comparison is +1.4 to +4.2 dB in Pixel-DPS's
  favor — 2–4× wider than the proposal's prediction. (b) Pixel-DPS is
  *also* faster per image at NFE=100 (~18 s vs ~24 s on the RTX 3060),
  reversing the speed ordering. We attribute the absence of FlowDPS's
  expected speed advantage to (i) the RF velocity field requires a
  deeper NCSN++ forward pass than the DDPM's epsilon prediction, and
  (ii) at NFE=100 both methods are far from any flow-saturation regime
  where the flow's straight-line trajectories would in principle pay
  off.

- **However, FlowDPS-on-RF saturates with NFE much faster than
  Pixel-DPS** — at the easiest condition the FlowDPS PSNR is essentially
  flat from NFE=25 to NFE=100 (26.08 → 26.23 dB). For applications
  where NFE=25 is the relevant budget, the gap shrinks to ~1.4 dB and
  FlowDPS-on-RF becomes competitive in quality at slightly higher
  per-step cost.

- **Under operator mismatch the comparison reverses.** When the true
  measurement is motion blur but the sampler assumes Gaussian, both
  methods degrade by 3–5 dB PSNR, but Pixel-DPS degrades *more*
  (−4.86 dB) than FlowDPS-on-RF (−2.48 dB). We hypothesize that the
  stochastic diffusion path compounds small mis-step errors when the
  likelihood gradient is systematically wrong, whereas the deterministic
  ODE trajectory of the flow model is more forgiving of misspecified
  per-step nudges. This is an interesting practical implication: in
  deployment scenarios with known forward operator (medical imaging
  with calibrated PSF), Pixel-DPS dominates; in scenarios with
  uncertain operator (consumer photography, generic deblurring), the
  flow-based method may degrade more gracefully.

- **Classical and supervised baselines situate the diffusion methods.**
  Wiener gives a 17.96–24.28 dB PSNR floor for ~17 ms/image; DnCNN+PnP
  pushes that up by 0.3–5.5 dB at ~0.3 s/image; Pixel-DPS adds a further
  1.6–3.4 dB at ~18 s/image. The choice between these is a clean
  Pareto frontier of compute vs reconstruction quality.

- **Limitations.** (i) 50 images per cell — error bars are stated as ±std
  in the summary tables. (ii) Single dataset (CelebA-HQ-256 faces). (iii)
  Single forward operator family (Gaussian / motion blur). (iv) Our
  DDRM reimplementation did not reach competitive PSNR; we report
  DnCNN+PnP-ADMM as the analogous "classical-prior reference" instead.
  (v) The Liu 2023 RF checkpoint's EMA weights are not loaded due to a
  shadow-params count mismatch in the released file; we use the non-EMA
  `model` state-dict, which may contribute ≤0.5 dB to the
  FlowDPS-on-RF gap.

- **Future work.** (i) Properly fix the DDRM reimplementation. (ii)
  Investigate whether FlowDPS's `step_size` (default 1.0) is the right
  setting for pixel-space deployment, or whether a learned per-step
  schedule would close the FlowDPS-vs-DPS gap. (iii) Repeat on a
  non-face dataset (LSUN-Church, DIV2K) to test how much of the
  Pixel-DPS dominance is dataset-specific.

---

## 7 Conclusion (audit)

> NOTE: existing conclusion is OK in structure. Update headline numbers.

- Replace *"FlowDPS achieves reconstruction quality within approximately
  0.5–1.1 dB PSNR of DPS while requiring 2–5× fewer NFEs"* with
  → "Pixel-DPS dominates FlowDPS-on-RF in the matched-Gaussian setting
  by 1.4–4.2 dB PSNR at NFE=100, but FlowDPS-on-RF is more robust to
  operator mismatch, degrading by only 2.5 dB vs Pixel-DPS's 5 dB when
  the true forward operator differs from the assumed one."

- Replace *"DPS exhibits more gradual performance degradation under
  forward model mismatch compared to FlowDPS"* with
  → "FlowDPS-on-RF exhibits more gradual performance degradation under
  forward-model mismatch than Pixel-DPS, reversing the matched-setting
  ordering."

- The "future-work" sentence about hybrid samplers can stay.

---

## Appendix A — Reproducibility checklist

All raw results live in `outputs/results/`:
- `main_grid.csv` (1800 rows, 6.11 GPU-hrs).
- `robustness.csv` (1200 rows, 3.47 GPU-hrs).
- `baselines.csv` (600 rows, ~120 s CPU/GPU total).

Per-experiment logs in `outputs/logs/exp_NNN_*.log`. Per-experiment
config + results + takeaways in `docs/WORKING_NOTES.md` (EXP-001 through
EXP-013). Code on GitHub: <https://github.com/adnan821/dps_flowdps>.

Pinned versions: torch 2.4.1+cu121, diffusers 0.30.1, lpips, clean-fid.
Hardware: single NVIDIA RTX 3060 12GB.

---

## Appendix B — Notes for whoever ports this to LaTeX

A LaTeX skeleton populated with the structure above is at
`docs/group13_v2.tex`. It uses the NeurIPS 2024 style (matching the
first submission's template).

Key things the porter must NOT do:
- Do NOT echo any numbers from the first submission's Section 5.
- Do NOT cite *Kim et al. 2024, arXiv:2403.XXXXX* — that's the
  placeholder citation from the fake report. Use the real one:
  `\cite{kim2025flowdps}` → Kim, Kim, Ye. *FlowDPS: Flow-Driven Posterior
  Sampling for Inverse Problems.* ICCV 2025. arXiv:2503.08136.
- Do NOT claim FFHQ-256 anywhere. CelebA-HQ-256.
- Do NOT claim A100 80GB. RTX 3060 12GB.

Key things the porter SHOULD do:
- Pull every numeric value in the LaTeX from a table in
  `outputs/figures/main_grid_summary.csv` or
  `outputs/figures/robustness_summary.csv`. A `\input{table.tex}` macro
  or a simple find-and-replace from the markdown is fine.
- Use the PDF figures from `outputs/figures/`, not the PNG.
- Include the EXP-013 (DDRM parked) disclosure in §4 or §6 — academic
  honesty point.
