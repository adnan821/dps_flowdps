# Reading List — `dps_flowdps` Project

Curated list of the literature behind every method, baseline, and
analytical observation in this project. Organized by reading priority
for someone preparing to defend or extend the work.

For each entry: **citation**, **what it provides for this project**,
and (if in the report) the BibTeX key in `docs/group13_v2.tex`.

---

## Tier 1 — Must read

These six form the technical backbone of the project. Without them
the report's §3 (Methodology) and §5 (Results) cannot be read.

### 1. Diffusion Posterior Sampling (DPS) — Chung et al. 2023

- **Citation.** H. Chung, J. Kim, M. T. McCann, M. L. Klasky, and J. C. Ye.
  *Diffusion Posterior Sampling for General Noisy Inverse Problems.*
  International Conference on Learning Representations (ICLR), 2023.
- **Why for us.** The DPS algorithm is what `src/samplers/pixel_dps.py`
  implements. Equation (7) of the paper is the gradient-normalized
  step size we discussed for the "Chung 2023 restoration fix"
  candidate (held back in our audit).
- **BibTeX key.** `chung2023dps` (already in our bib).
- **arXiv.** 2209.14687.

### 2. FlowDPS — Kim, Kim & Ye 2025

- **Citation.** J. Kim, B. S. Kim, and J. C. Ye. *FlowDPS: Flow-Driven
  Posterior Sampling for Inverse Problems.* International Conference on
  Computer Vision (ICCV), 2025.
- **Why for us.** The exact method `src/samplers/flowdps_rf.py`
  reimplements on the Liu 2023 Rectified Flow backbone (the official
  release is SD3-latent-only). Defines the velocity-correction form
  `v_eff = v_θ − ζ ∇_{x_t} ‖y − A(z₁̂)‖`.
- **BibTeX key.** `kim2025flowdps` (already in our bib).
- **arXiv.** 2503.08136.

### 3. Rectified Flow — Liu, Gong & Liu 2023

- **Citation.** X. Liu, C. Gong, and Q. Liu. *Flow Straight and Fast:
  Learning to Generate and Transfer Data with Rectified Flow.*
  International Conference on Learning Representations (ICLR), 2023.
- **Why for us.** Provides the pretrained CelebA-HQ-256 1-RF
  checkpoint that `flowdps_rf` runs on. Defines the velocity
  convention `v = z_1 − z_0`, Tweedie identity for RF
  (`z_1_hat = x_t + (1-t)·v`), and the Euler ODE sampling we use.
- **BibTeX key.** `liu2023rectified` (already in our bib).

### 4. DDPM — Ho, Jain & Abbeel 2020

- **Citation.** J. Ho, A. Jain, and P. Abbeel. *Denoising Diffusion
  Probabilistic Models.* Neural Information Processing Systems (NeurIPS),
  2020.
- **Why for us.** The DDPM prior our Pixel-DPS uses
  (`google/ddpm-ema-celebahq-256`). The forward-process / Tweedie
  identity that DPS leans on is from here.
- **BibTeX key.** `ho2020ddpm` (already in our bib).

### 5. DDIM — Song, Meng & Ermon 2021 *(missing from our bib — to add)*

- **Citation.** J. Song, C. Meng, and S. Ermon. *Denoising Diffusion
  Implicit Models.* International Conference on Learning
  Representations (ICLR), 2021.
- **Why for us.** Our sampler uses `DDIMScheduler` from `diffusers`;
  this paper defines the deterministic-DDIM update we override in
  the Π-GDM-pure variant ("corrected x̂₀ in data term, original eps
  in noise direction" — DDRM-style step is on this template).
- **BibTeX key (to add).** `song2021ddim`.
- **arXiv.** 2010.02502.

### 6. Π-GDM — Song, Sun & Saremi 2023 *(missing from our bib — to add)*

- **Citation.** J. Song, A. Sun, and S. Saremi. *Pseudoinverse-Guided
  Diffusion Models for Inverse Problems.* International Conference on
  Learning Representations (ICLR), 2023.
- **Why for us.** Eq. 6 of this paper is the Bayesian Kalman update
  for `x̂₀ | x_t, y` that `flowdps_rf_pigdm_pure` implements
  analytically. The leading `r_t` factor in the posterior mean is the
  one I missed during the math iterations (commit history records
  three implementation passes before getting this right).
- **BibTeX key (to add).** `song2023pigdm`.
- **arXiv.** 2212.00490.

---

## Tier 2 — High priority

Read if you have a half-day. These directly support §5.4.1
*Gradient-free posterior sampling* and the discussion sections.

### 7. DDRM — Kawar, Elad, Ermon & Song 2022

- **Citation.** B. Kawar, M. Elad, S. Ermon, and J. Song. *Denoising
  Diffusion Restoration Models.* NeurIPS 2022.
- **Why for us.** Eq. 8 / Eq. 9 of DDRM is the precise template our
  `pigdm_pure` reverse-step uses (corrected `x̂₀` in the data term,
  ORIGINAL eps in the noise direction). Also the spectral-projection
  template we attempted as a baseline (parked at ~14 dB).
- **BibTeX key.** `kawar2022ddrm` (already in our bib).

### 8. Constrained Particle Seeking — Dou et al. 2026

- **Citation.** H. Dou et al. *Constrained Particle Seeking: Solving
  Diffusion Inverse Problems with Just Forward Passes.*
  arXiv:2603.01837, 2026.
- **Why for us.** Listed in our brief as a starter ref. The method
  that **looks** gradient-free but uses a score-Langevin rejuvenation
  move — exactly what our particle-DPS failure analysis says is
  necessary. §5.4.1 cites this for "what the literature does to
  avoid path-degeneracy".
- **BibTeX key.** `dou2026particle` (already in our bib).

### 9. SMC Samplers — Del Moral, Doucet & Jasra 2006 *(missing from our bib — to add)*

- **Citation.** P. Del Moral, A. Doucet, and A. Jasra. *Sequential
  Monte Carlo samplers.* Journal of the Royal Statistical Society,
  Series B, 68(3):411–436, 2006.
- **Why for us.** The annealed-likelihood-tempering construction
  `π_i(x) ∝ p(x) · L(y|x)^(1/β_i^2)` that our
  `particle_dps_tempered` implements. The math note
  `docs/TEMPERED_PARTICLE_DPS.md` §4 cites this for the
  derivation.
- **BibTeX key (to add).** `delmoral2006smc`.

### 10. EDM — Karras, Aittala, Aila & Laine 2022 *(missing from our bib — to add)*

- **Citation.** T. Karras, M. Aittala, T. Aila, and S. Laine.
  *Elucidating the Design Space of Diffusion-Based Generative
  Models.* NeurIPS 2022.
- **Why for us.** Practical-tuning paper that covers EMA-weight
  loading (relevant to our EXP-014 fix in `rf_celebahq.py`) and
  Heun-vs-Euler integrator trade-offs. Their finding that
  "2nd-order helps unconditional but is task-dependent for guided
  sampling" is the textbook citation behind our Heun NFE=200 result.
- **BibTeX key (to add).** `karras2022edm`.
- **arXiv.** 2206.00364.

---

## Tier 3 — Medium priority

Skim for context. Not essential for understanding the project's core
results, but cited in passing or relevant to extensions.

### 11. Diffusion Distribution Matching — Meanti et al. 2025

- **Citation.** G. Meanti et al. *Unsupervised Imaging Inverse Problems
  with Diffusion Distribution Matching.* ICCV 2025.
- **Why for us.** Brief starter ref. Alternative posterior-sampling
  framing (distribution matching instead of gradient-based guidance).
  One Related Work sentence.
- **BibTeX key.** `meanti2025dim` (already in our bib).

### 12. Physics-Constrained CFM — Dasgupta et al. 2026

- **Citation.** A. Dasgupta et al. *Solving Physics-Constrained Inverse
  Problems with Conditional Flow Matching.* arXiv:2603.14135, 2026.
- **Why for us.** Brief starter ref. Covers the "physics axis" of
  Bucket 3 that we did NOT pursue (we chose faces, not PDEs).
- **BibTeX key.** `dasgupta2026pcfm` (already in our bib).

### 13. Flow Matching — Lipman, Chen, Ben-Hamu, Nickel & Le 2023

- **Citation.** Y. Lipman, R. T. Q. Chen, H. Ben-Hamu, M. Nickel, and
  M. Le. *Flow Matching for Generative Modeling.* ICLR 2023.
- **Why for us.** The general CFM training objective behind Rectified
  Flow. Needed to defend "why Liu 2023 RF is the appropriate flow
  prior for our pixel-space comparison".
- **BibTeX key.** `lipman2023flow` (already in our bib).

### 14. PnP-ADMM — Chan, Wang & Elgendy 2017

- **Citation.** S. H. Chan, X. Wang, and O. A. Elgendy. *Plug-and-Play
  ADMM for Image Restoration: Fixed-Point Convergence and
  Applications.* IEEE Transactions on Computational Imaging, 2017.
- **Why for us.** Theoretical foundation of our DnCNN+PnP-ADMM
  baseline. Establishes the convergence framework that makes
  PnP-ADMM a legitimate posterior-style optimizer.
- **BibTeX key.** `chan2017pnp` (already in our bib).

### 15. DnCNN — Zhang, Zuo, Chen, Meng & Zhang 2017

- **Citation.** K. Zhang, W. Zuo, Y. Chen, D. Meng, and L. Zhang.
  *Beyond a Gaussian Denoiser: Residual Learning of Deep CNN for
  Image Denoising.* IEEE Transactions on Image Processing, 2017.
- **Why for us.** The denoiser inside our PnP-ADMM baseline (668 k
  params, blind Gaussian, via `deepinv`). Needed for the
  "Why DnCNN+PnP beats FlowDPS-RF" paragraph we plan to add to §5.3.
- **BibTeX key.** `zhang2017dncnn` (already in our bib).

### 16. MCG-Diff — Cardoso et al. 2024 *(optional — to add if cited)*

- **Citation.** G. Cardoso et al. *Monte Carlo guided Diffusion for
  Bayesian linear inverse problems.* ICML 2024.
- **Why for us.** Modern SMC-for-diffusion paper that includes the
  MCMC rejuvenation our `particle_dps_tempered` was missing. Closest
  comparable to what we attempted.
- **BibTeX key (to add if cited).** `cardoso2024mcgdiff`.

---

## Tier 4 — Background

You should be able to recognize these and answer questions like
"what is LPIPS" or "what is NCSN++". Not required reading.

### 17. Score-Based Generative Modeling through SDEs — Song et al. 2021

- **Citation.** Y. Song, J. Sohl-Dickstein, D. P. Kingma, A. Kumar,
  S. Ermon, and B. Poole. *Score-Based Generative Modeling through
  Stochastic Differential Equations.* ICLR 2021.
- **Why for us.** Defines the **NCSN++** U-Net architecture that the
  Liu 2023 RF checkpoint actually uses (`external/RectifiedFlow/`).
  Foundational unifying view of diffusion as an SDE / score-matching.

### 18. SMC in Practice — Doucet, de Freitas & Gordon 2001

- **Citation.** A. Doucet, N. de Freitas, and N. J. Gordon, editors.
  *Sequential Monte Carlo Methods in Practice.* Springer, 2001.
- **Why for us.** Chapter 12 contains the **path-degeneracy** theorem
  that our `particle_dps_tempered` analysis directly applies (vanilla
  importance resampling cannot create new particle states on a
  deterministic trajectory). Cited in `docs/TEMPERED_PARTICLE_DPS.md`.

### 19. LPIPS — Zhang, Isola, Efros, Shechtman & Wang 2018

- **Citation.** R. Zhang, P. Isola, A. A. Efros, E. Shechtman, and
  O. Wang. *The Unreasonable Effectiveness of Deep Features as a
  Perceptual Metric.* CVPR 2018.
- **Why for us.** One of our three quality metrics. Need to be able
  to explain what LPIPS measures and why it's perceptual.

### 20. SSIM — Wang, Bovik, Sheikh & Simoncelli 2004

- **Citation.** Z. Wang, A. C. Bovik, H. R. Sheikh, and E. P. Simoncelli.
  *Image Quality Assessment: From Error Visibility to Structural
  Similarity.* IEEE TIP, 2004.
- **Why for us.** Another of our three metrics. Need to be able to
  explain the luminance/contrast/structure decomposition.

### 21. Tweedie's Formula — Efron 2011

- **Citation.** B. Efron. *Tweedie's Formula and Selection Bias.*
  Journal of the American Statistical Association, 106(496):1602–1614,
  2011.
- **Why for us.** The clean-image-estimate-from-noisy-state identity
  underlying DPS, Π-GDM, and FlowDPS. Modern derivation; the
  original Robbins 1955 empirical Bayes paper is the foundational
  reference but unnecessary to read directly.

---

## Suggested reading order (1-day plan)

If you have 1 working day before the defense:

| Block | Time | Read |
|---|---|---|
| Morning 1 | 1.5 h | Chung 2023 DPS (Tier 1.1). |
| Morning 2 | 1 h | Kim 2025 FlowDPS (Tier 1.2). |
| Lunch | 0.5 h | Skim Liu 2023 RF intro + sampling section (Tier 1.3). |
| Afternoon 1 | 1 h | Song 2023 Π-GDM (Tier 1.6). |
| Afternoon 2 | 1 h | Kawar 2022 DDRM, focus on Eq. 8–9 (Tier 2.7). |
| Evening | 1 h | Skim Karras 2022 EDM §4 (EMA), §5 (integrators) (Tier 2.10). |

That covers ≈ 80 % of what's defensible about the project.

## Citations to add to `docs/group13_v2.tex` before submission

Five missing entries that the report's new sections (§5.4.1, the
Π-GDM-pure discussion, the Heun NFE=200 paragraph) need:

1. `song2021ddim` — DDIM (ICLR 2021).
2. `song2023pigdm` — Π-GDM (ICLR 2023).
3. `karras2022edm` — EDM (NeurIPS 2022).
4. `delmoral2006smc` — SMC samplers (JRSSB 2006).
5. `doucet2001smc` — SMC in Practice (Springer book, Ch. 12). *(Cited in `docs/TEMPERED_PARTICLE_DPS.md`; the report should match.)*

These will be added during the report-writing pass.
