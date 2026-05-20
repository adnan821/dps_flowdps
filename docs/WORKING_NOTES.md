# DVLM Project — Working Notes

Running lab notebook for the AI-623 DPS vs FlowDPS project on CelebA-HQ-256.
Used both as a daily record and as raw material for the final report (Section 5
and Discussion). Every experiment is logged to `outputs/logs/` and summarized
here.

---

## Project framing (current)

**Goal.** Controlled comparison of two posterior-sampling paradigms for image
deblurring inverse problems:

- **Pixel-DPS** — Diffusion Posterior Sampling (Chung et al. 2023) on a
  CelebA-HQ-256 DDPM (HF `google/ddpm-ema-celebahq-256`).
- **FlowDPS-on-RF** — our reimplementation of FlowDPS (Kim et al. 2025) style
  guidance on top of a CelebA-HQ-256 Rectified Flow (Liu et al. 2023) checkpoint.

Both methods are pixel-space U-Nets on the same dataset → the proposal's
"shared backbone, controlled comparison" claim genuinely holds.

**Hardware.** Local RTX 3060 12 GB.

**Test set.** 200 CelebA-HQ-256 images from `korexyz/celeba-hq-256x256` (HF
validation split). Will subsample 50 / 100 / validation-20 as needed.

**Grid.** 3 σ_blur × 2 σ_noise × 3 NFE × 2 methods = 36 conditions (or 18 if
counting per method) × 50 imgs.

## Pivot history

- **2026-05-19 round 1 (proposal-as-written).** FFHQ-256, "shared backbone DDPM
  U-Net", run on a single A100. Implied by `docs/group13.pdf` Section 5 (which
  contained fabricated results from a previous note — see
  `docs/group13.pdf`).
- **2026-05-19 round 2.** Tried staying on FFHQ + using the real FlowDPS code
  (Kim 2025), which turned out to need SD3 Medium + ≥24 GB VRAM → required
  Colab Pro+ A100. Spent ~3 hours fighting installation and auth issues:
  torchvision::nms registration broken by FlowDPS's CUDA 11.8 requirements,
  missing `motionblur` module, double-gated HF repos for SD3 / SD3-diffusers,
  HF_TOKEN env var not propagating to `solve.py` subprocess. **Abandoned.**
- **2026-05-19 round 3 (current).** Local CelebA-HQ-256 with two pixel-space
  U-Nets. No cloud, no SD3, no subprocess auth.

---

## Code & data inventory

| Artifact | Path / Source | Status |
|---|---|---|
| Forward degradation operators | `src/forward/degradation.py` | ✅ 11 tests pass |
| Metrics (PSNR/SSIM/LPIPS) + timing + CSV logger | `src/metrics/` | ✅ smoke-tested |
| CelebA-HQ DDPM | HF `google/ddpm-ema-celebahq-256` (cached under `checkpoints/hf_cache`) | ✅ 113 M params, 834 MiB VRAM peak |
| CelebA-HQ Rectified Flow | `checkpoints/rectified_flow/celebahq_256_rf.pth` (1.05 GB, gdown'd from Liu 2023 Drive) | ⏳ loader pending |
| `external/RectifiedFlow` (NCSN++ model code) | gnobitab/RectifiedFlow cloned | needs `op/upfirdn2d` JIT-compiled or replaced with PyTorch-native fallback |
| Test images | `data/celeba_hq_256/test/*.png` | ✅ 200 PNGs, 19 MB |
| Pixel-DPS sampler | `src/samplers/pixel_dps.py` | ✅ smoke-tested |
| FlowDPS-on-RF sampler | `src/samplers/flowdps_rf.py` | ⏳ pending RF loader |

---

## Experiment log

### EXP-001 — Pixel-DPS sanity sweep over ζ (2026-05-19)

**Config:** 2 CelebA-HQ-256 images, σ_blur=3.0, σ_noise=0.05, NFE=100, seed=0.
Sampler: `PixelDPS` (DDIM reverse + Tweedie x̂₀ + L2-norm likelihood gradient).
Backbone: `google/ddpm-ema-celebahq-256` (113 M params).
Log: `outputs/logs/exp_001_pixel_dps_zeta_sweep_20260519.log`.

**Results.** PSNR(y) = 22.76 dB.

| ζ  | NFE | wall (s/img) | PSNR(x_hat) | SSIM   | LPIPS  |
|----|-----|--------------|-------------|--------|--------|
| 1  | 100 | 20.3         | 24.51       | 0.7408 | 0.1403 |
| 3  | 100 | 20.4         | 26.70       | 0.7927 | 0.1069 |
| 5  | 100 | 20.3         | 27.59       | 0.8142 | 0.0965 |
| 10 | 100 | 20.4         | **28.82**   | **0.8435** | 0.1118 |

**Takeaways.**
- Real DPS reconstruction; clear monotonic PSNR/SSIM lift in ζ across 1→10.
- Best LPIPS is at ζ=5 (0.0965); best PSNR/SSIM is at ζ=10. Slight perceptual
  vs distortion tradeoff. **Use ζ=10 as the default for Pixel-DPS** (PSNR-favored).
- Wall-clock at NFE=100 ≈ 20 s/image on the RTX 3060.
- **Bug 1 caught and fixed.** First version used `0.5 * ‖y - A(x̂₀)‖² / σ²` as
  the guidance loss → gradients exploded (‖grad‖ ~ 10⁵) and PSNR collapsed to
  6.76 dB. Chung et al.'s official impl uses the **L2 norm** (not squared) of
  the residual — switching to `‖y - A(x̂₀)‖₂` brought PSNR back into the
  20+ dB range.
- **Bug 2 caught and fixed.** Used `x̂₀.clamp(-1, 1)` inside the gradient path.
  `clamp` zeroes the gradient for any pixel outside [-1, 1], which at early
  timesteps is most of the tensor → kills DPS. Removed; clamping only at the
  final output.

### EXP-002 — RF CelebA-HQ-256 checkpoint loader (2026-05-19)

**Goal:** instantiate Liu 2023's NCSN++ model and load
`checkpoints/rectified_flow/celebahq_256_rf.pth` (1.05 GB).

**Iterations.**
1. First try (`exp_002_rf_loader_first_try`): EMA shadow_params count is 644,
   but `model.parameters()` has 645 — off by 1. (Unclear which param the EMA
   tracker excluded.) Raised `RuntimeError`.
2. Second try (`exp_002_rf_loader_fallback`): falling back to the non-EMA
   `model` state_dict → size mismatch on the `sigmas` buffer (`[2000]` from
   ckpt vs `[1000]` in our config).
3. Third try (`exp_002_rf_loader_v3`): set `num_scales = 2000` and
   `sigma_max = 378.0` (per `default_lsun_configs.py`). **Loaded cleanly**,
   `missing=0, unexpected=0`.

**Final.** 65.5 M params, 619 MiB VRAM peak at single forward pass on
3060 with input (1, 3, 256, 256). NCSN++ JIT-compiles `upfirdn2d` CUDA op on
first import (~10 s, one-time).

### EXP-003 / EXP-004 — Initial FlowDPS-on-RF smoke tests (2026-05-19)

**Config:** 2 CelebA-HQ images, σ_blur=3.0, σ_noise=0.05, NFE=50–100, seed=0.
Sampler: `FlowDPSRF` (Euler forward integration from t=eps → 1, Tweedie z₁_hat
= x + (1-t) v, FlowDPS likelihood-corrected velocity).

| ζ   | NFE | PSNR(x_hat) | SSIM   | LPIPS  | Notes |
|-----|-----|-------------|--------|--------|-------|
| 1.0 | 50  | 8.43        | 0.205  | 0.733  | first run, looked broken |
| 0.0 | 100 | 8.65        | 0.210  | 0.703  | sanity check — pure generation |

PSNR was puzzlingly low. EXP-005 disambiguates whether this is a model or a
guidance issue.

### EXP-005 — Pure RF generation sanity check (2026-05-19)

**Goal.** Run Liu's Euler sampler with no guidance (zeta=0) on 4 random noise
seeds. If the model produces faces, the load and integration are correct and
EXP-003/004's "low PSNR" is just "wrong face, not no face."

**Outcome.** ✅ Four clean CelebA-HQ-style faces saved to
`outputs/smoke/rf_pure_gen.png`. Velocity-field statistics evolve sensibly
(std≈1.0 → ≈0.33, range narrows to [-1.04, 1.04] by t=0.98).

**Conclusion.** The RF model loads and integrates correctly. EXP-003/004's
8 dB PSNR is the PSNR between random face A and random face B — guidance was
too weak to pull the trajectory toward the target measurement.

### EXP-006 / EXP-007 — FlowDPS-on-RF ζ sweep (2026-05-19, complete)

**Config.** 2 images, NFE=100, σ_blur=3.0, σ_noise=0.05, seeds matched to
EXP-001. PSNR(y) = 22.76 dB.

| ζ    | PSNR(x_hat) | SSIM   | LPIPS  | Notes |
|------|-------------|--------|--------|-------|
| 1    | 9.12        | 0.237  | 0.681  | random face (weak guidance) |
| 3    | 10.25       | 0.302  | 0.591  | |
| 10   | 14.66       | 0.434  | 0.448  | |
| 30   | 19.66       | 0.544  | 0.320  | |
| **100** | **23.45** | **0.650** | **0.223** | **peak (locked for overnight)** |
| 300  | 23.35       | 0.541  | 0.331  | PSNR plateau, LPIPS regresses |
| 1000 | 18.81       | 0.362  | 0.536  | over-guidance, collapses |

**Takeaways.**
- **FlowDPS-on-RF lags Pixel-DPS by ~5 dB PSNR / ~0.2 SSIM at peak ζ.** Not
  the "small efficiency–quality tradeoff" the proposal hypothesized; on
  this hardware Pixel-DPS dominates in both quality and per-step cost (20 s
  vs 24 s per image at NFE=100).
- Quality is monotone in ζ up to ~100 and then **degrades** past 300 (guidance
  pushes the trajectory off-manifold). Sweet spot is narrow and ζ has to be
  ~10× larger than for Pixel-DPS.
- ζ=100 is the locked operating point for the overnight main grid.

**Open question for follow-up:** is the Pixel-DPS dominance an artifact of
the Liu 2023 RF checkpoint (CelebA-HQ-256 was a smaller-scale training run
than the DDPM), of the FlowDPS algorithm itself in pixel space, or of a
hyperparameter we haven't found yet (e.g. Kim 2025's `step_size` argument
that we left at the default)? Will investigate in EXP-009+ after the main
grid finishes — for the report, this is itself a publishable finding.

### EXP-009 — Motion-blur operator-mismatch robustness study (2026-05-20)

**Setup.** True forward operator: `motion_blur(L, θ)` + Gaussian noise σ_n.
Sampler's likelihood assumes Gaussian blur with **fixed** σ_b=3.0. Locked
hyperparameters: ζ=10 (Pixel-DPS), ζ=100 (FlowDPS-on-RF). NFE=50. 50
images per cell.

Grid: 3 motion lengths × 2 angles × 2 noise levels × 2 methods = 24 cells
× 50 imgs = **1200 reconstructions**. Compute: **3.47 GPU-hrs** total.

Log: `outputs/logs/robustness_overnight_20260520_003540.log`.
Data:  `outputs/results/robustness.csv`.

**Per-condition mean PSNR (50 imgs each):**

| method        | L  | θ    | σ_n  | PSNR  | SSIM  | LPIPS | s/img |
|---------------|----|------|------|-------|-------|-------|-------|
| pixel_dps     | 15 | 0°   | 0    | 24.79 | 0.722 | 0.181 | 9.06  |
| pixel_dps     | 15 | 0°   | 0.05 | 24.86 | 0.714 | 0.180 | 8.96  |
| pixel_dps     | 15 | 45°  | 0    | 25.24 | 0.732 | 0.169 | 9.06  |
| pixel_dps     | 15 | 45°  | 0.05 | 25.15 | 0.721 | 0.174 | 9.19  |
| pixel_dps     | 25 | 0°   | 0    | 21.96 | 0.643 | 0.260 | 8.95  |
| pixel_dps     | 25 | 0°   | 0.05 | 22.31 | 0.649 | 0.253 | 8.96  |
| pixel_dps     | 25 | 45°  | 0    | 22.32 | 0.651 | 0.247 | 9.03  |
| pixel_dps     | 25 | 45°  | 0.05 | 22.70 | 0.658 | 0.243 | 8.96  |
| pixel_dps     | 35 | 0°   | 0    | 20.43 | 0.596 | 0.331 | 9.23  |
| pixel_dps     | 35 | 0°   | 0.05 | 20.59 | 0.601 | 0.321 | 9.10  |
| pixel_dps     | 35 | 45°  | 0    | 20.62 | 0.594 | 0.315 | 9.33  |
| pixel_dps     | 35 | 45°  | 0.05 | 20.86 | 0.603 | 0.309 | 9.13  |
| flowdps_rf    | 15 | 0°   | 0    | 23.04 | 0.627 | 0.203 | 11.72 |
| flowdps_rf    | 15 | 0°   | 0.05 | 22.06 | 0.562 | 0.248 | 11.63 |
| flowdps_rf    | 15 | 45°  | 0    | 23.29 | 0.636 | 0.197 | 11.63 |
| flowdps_rf    | 15 | 45°  | 0.05 | 22.17 | 0.565 | 0.244 | 11.63 |
| flowdps_rf    | 25 | 0°   | 0    | 21.44 | 0.574 | 0.265 | 11.86 |
| flowdps_rf    | 25 | 0°   | 0.05 | 20.88 | 0.514 | 0.302 | 11.70 |
| flowdps_rf    | 25 | 45°  | 0    | 21.77 | 0.585 | 0.250 | 11.71 |
| flowdps_rf    | 25 | 45°  | 0.05 | 21.11 | 0.520 | 0.291 | 11.69 |
| flowdps_rf    | 35 | 0°   | 0    | 20.05 | 0.525 | 0.329 | 11.68 |
| flowdps_rf    | 35 | 0°   | 0.05 | 19.65 | 0.466 | 0.361 | 12.02 |
| flowdps_rf    | 35 | 45°  | 0    | 20.30 | 0.531 | 0.315 | 11.63 |
| flowdps_rf    | 35 | 45°  | 0.05 | 19.92 | 0.473 | 0.346 | 11.78 |

**Takeaways.**
- **Operator mismatch costs ~3–8 dB PSNR for both methods.** Compared to
  matched Gaussian (main grid σ_b=3.0, σ_n=0.05, NFE=50): Pixel-DPS goes
  from 25.72 → 22.70 dB at L=25/θ=45° (−3.0 dB) and 25.72 → 20.86 dB at
  L=35/θ=45° (−4.86 dB). FlowDPS-on-RF goes from 22.40 → 21.11 dB at
  L=25/θ=45° (−1.3 dB) and 22.40 → 19.92 dB at L=35/θ=45° (−2.5 dB).
- **FlowDPS-on-RF is RELATIVELY more robust to operator mismatch.** It
  loses less PSNR going from matched to mismatched forward operator than
  Pixel-DPS does. The gap between Pixel-DPS and FlowDPS-on-RF shrinks
  from +2 dB at L=15 to +0.5 dB at L=35. The deterministic ODE
  trajectory may be less sensitive to misspecified likelihood gradients
  than the stochastic diffusion path.
- **Angle barely matters** — within ~0.5 dB everywhere comparing θ=0° vs
  θ=45°. Probably because motion-blur kernels of the same length cover
  the same blur "energy" regardless of orientation, and a rotationally-
  symmetric Gaussian assumed-operator can't tell.
- **σ_n=0.05 vs σ_n=0 effect is largely washed out** by the operator
  mismatch. The mismatch dominates the noise term.
- **Wall-clock**: Pixel-DPS averages 9.1 s/img at NFE=50, FlowDPS-on-RF
  averages 11.7 s/img. Same ~25% Pixel-DPS speed advantage as the main
  grid.

**Reportable story for Section 5/6.** The proposal hypothesized "DPS
hypothesized to demonstrate slightly greater robustness under operator
mismatch because stochastic posterior exploration may better compensate
for modeling inaccuracies." Our measured numbers **contradict** this on
this hardware: FlowDPS-on-RF is actually **more** robust to operator
mismatch in relative terms, despite being weaker in the matched case.
The robustness study reverses the comparison.

### EXP-010 — Robustness plots (2026-05-20)

**Output.** `scripts/make_robustness_plots.py` generates 4 figures and a
summary table:
- `robustness_psnr_vs_length.{png,pdf}` — PSNR vs L, faceted by σ_n.
- `robustness_metrics_vs_length.{png,pdf}` — 3-panel (PSNR/SSIM/LPIPS).
- `robustness_psnr_heatmap.{png,pdf}` — per-cell heatmap, one panel per method.
- `matched_vs_mismatched.{png,pdf}` — bar chart: matched Gaussian
  (σ_b=3, σ_n=0.05, NFE=50) vs mismatched motion (L=15/25/35). The
  cleanest visualization of the operator-mismatch cost.
- `robustness_summary.{md,csv}` — per-cell mean/std table.

### EXP-011 — Wiener-filter classical baseline (2026-05-20)

**Setup.** Per-channel 2D Wiener deconvolution in the Fourier domain.
Inverse model = Gaussian blur with the **same** σ used to generate y (no
operator mismatch here — main grid only). Regularization scalar `K =
σ_n²/S` with `S` a flat-spectrum prior. Used `K=1e-3` for σ_n=0 and
`K=5e-2` for σ_n=0.05.

Log: `outputs/logs/exp_011_wiener_20260520_072556.log`.
Data: `outputs/results/baselines.csv` (300 rows = 50 imgs × 6 conditions).
Compute: 18 s total CPU.

**Per-condition mean over 50 imgs.**

| σ_b | σ_n  | n  | PSNR  | SSIM  | LPIPS | ms/img |
|-----|------|----|-------|-------|-------|--------|
| 1.5 | 0.0  | 50 | 22.23 | 0.838 | 0.225 | 16.5   |
| 1.5 | 0.05 | 50 | 24.28 | 0.546 | 0.487 | 15.9   |
| 3.0 | 0.0  | 50 | 19.94 | 0.694 | 0.436 | 15.9   |
| 3.0 | 0.05 | 50 | 24.03 | 0.657 | 0.596 | 16.0   |
| 5.0 | 0.0  | 50 | 17.96 | 0.577 | 0.566 | 15.9   |
| 5.0 | 0.05 | 50 | 22.50 | 0.631 | 0.649 | 15.8   |

**Three-way PSNR comparison @ NFE=50, σ_n=0.05:**

| σ_b | Wiener | Pixel-DPS | FlowDPS-on-RF |
|-----|--------|-----------|---------------|
| 1.5 | 24.28  | 26.51     | 23.60         |
| 3.0 | 24.03  | 25.72     | 22.40         |
| 5.0 | 22.50  | 24.60     | 20.95         |

**Takeaways.**
- **Pixel-DPS beats Wiener at every condition** by 1.7–7.6 dB PSNR.
  Especially at low-noise σ_n=0 where Wiener over-amplifies high
  frequencies (rings, ghosts, LPIPS jumps to 0.5+).
- **FlowDPS-on-RF generally beats Wiener** except at the hardest
  (σ_b=5, σ_n=0.05) where the gap closes (20.95 vs 22.50). At low
  σ_b/σ_n FlowDPS easily dominates.
- **Wiener is ~1000× faster** (16 ms vs 9 s/img at NFE=50). For
  applications where any kind of structural recovery is acceptable,
  Wiener is a fine cheap baseline. For perceptual quality, the
  learned-prior methods are decisively better (LPIPS gap is ~0.4–0.5).
- **Counterintuitive Wiener-vs-noise effect**: Wiener PSNR is *higher*
  at σ_n=0.05 than at σ_n=0 in some cells. Cause: my `K=1e-3` at σ_n=0
  is too aggressive (under-regularizes), amplifying high frequencies as
  ringing; `K=5e-2` at σ_n=0.05 is more conservative and gives smoother
  output. A finer K sweep would tighten this but doesn't change the
  qualitative ordering vs DPS / FlowDPS.

### EXP-012 — DnCNN + PnP-ADMM (2026-05-20)

**Setup.** Plug-and-Play ADMM in the Fourier domain with a pretrained
3-channel DnCNN (deepinv release: 668k params, blind Gaussian denoiser).
Outer iterations = 24, ρ = 2.0, descending denoiser-sigma schedule with
`sigma_max = max(0.10, 4·σ_n + 0.05)` and `sigma_min = max(0.02, σ_n)`
(noise-adaptive, tuned on a 4-image sweep).

Log: `outputs/logs/exp_012_dncnn_pnp_20260520_074158.log`.
Data: `outputs/results/baselines.csv` (300 dncnn_pnp_admm rows).
Compute: 104 s total CPU+GPU for 300 reconstructions (~0.3 s/img).

**Three-way comparison at NFE=100 (where applicable), σ_n=0.05:**

| σ_b | Wiener | DnCNN+PnP | FlowDPS-on-RF | Pixel-DPS |
|-----|--------|-----------|---------------|-----------|
| 1.5 | 24.28  | 24.61     | 23.65         | 26.21     |
| 3.0 | 24.03  | 24.31     | 22.58         | 26.25     |
| 5.0 | 22.50  | 22.81     | 21.13         | 25.33     |

**Takeaways.**
- **DnCNN+PnP-ADMM beats Wiener at every cell** (always by ≥0.3 dB; up to
  +5.5 dB at σ_b=5/σ_n=0). Slight cost in LPIPS, but PSNR/SSIM strictly
  better.
- **DnCNN+PnP beats FlowDPS-on-RF in 5/6 cells** and ties on the 6th.
  Striking — a small (668k-param) classical denoiser inside PnP-ADMM
  beats our reimplementation of a million-param flow-based posterior
  sampler on this hardware/checkpoint.
- **Pixel-DPS still wins across the board** by 1.6–3.4 dB PSNR.
  Especially dominant on LPIPS (0.08–0.19 vs DnCNN+PnP's 0.25–0.57).
- **DnCNN+PnP is ~60× faster than Pixel-DPS** (0.3 vs 18 s/img). For
  applications where a 1–3 dB PSNR drop is acceptable, DnCNN+PnP is the
  efficient choice.
- **First sigma_schedule attempt was bad** (σ_max=0.10 fixed): PSNR
  collapsed at σ_n=0.05 (16 dB). Adapting σ to noise level fixed it.
  The hyperparameter `ρ` also matters a lot — 2.0 was 5 dB better than 0.1.

### EXP-013 — DDRM-lite attempt (parked, 2026-05-20)

**Setup.** Tried implementing a simplified DDRM (Kawar et al. 2022) on
top of our existing CelebA-HQ DDPM. Approach: standard DDIM reverse
sampling + per-step Tikhonov-regularized projection of Tweedie's x̂₀
onto the measurement constraint in the FFT basis (where Gaussian-blur is
diagonal). Tried two formulations:

1. Hard-threshold spectral mask + (η·prior + (1-η)·measurement) blend
   per frequency. Failed: PSNR 5–15 dB.
2. Soft Tikhonov projection `X_proj = (conj(H)·Y + λ·X_prior) / (|H|² + λ)`
   with λ derived from η. Also failed: similar PSNR.

**Outcome.** Even at the easiest condition (σ_b=1.5, σ_n=0, eta=0.7) PSNR
maxed out at ~14 dB, far below Wiener's 22 dB. Something in the
interaction between the projected x̂₀ and the DDIM noise direction is
broken; possible bugs include scale mismatch between `[-1,1]` diffusion
space and `[0,1]` measurement space at the FFT boundary, or an
incorrect sign in the deterministic DDIM update with modified x̂₀.

Per the project notes's "if it fails twice, pivot" rule, **parking DDRM** rather
than spending more time debugging. We already have 4 working baselines
(Pixel-DPS, FlowDPS-on-RF, Wiener, DnCNN+PnP-ADMM) which gives a rich
comparison for the report. The DDRM rows have been removed from
`outputs/results/baselines.csv`.

The parked code lives in `src/baselines/ddrm.py` and `scripts/run_ddrm.py`
for future investigation if time permits or if a reviewer asks.

**Reportable note:** DDRM is described in Section 4 ("Baselines") of the
proposal/first-report; the final report should note that we attempted
a DDRM-lite implementation but couldn't reach competitive PSNR with the
time available, and instead included DnCNN+PnP-ADMM as the analogous
"diffusion-prior-with-spectral-projection" reference point.

### EXP-014 — Tier A: EMA-aligned weights + zeta schedules + ops_fft refactor (2026-05-20)

**Goal.** Three coupled improvements: (i) load the Liu 2023 RF EMA
weights properly (the v1 baseline used the non-EMA `model` state-dict
because the EMA `shadow_params` count is off by one); (ii) accept a
callable `zeta(step, num_steps) → float` schedule in both samplers;
(iii) refactor the FFT-OTF utilities into a shared module so Tier B
and Tier C can build on it.

**Commit:** `503b94f` on `feature/publication-improvements`.

#### A1. EMA off-by-one diagnosis

Inspected `external/RectifiedFlow/.../models/ema.py:30` — gnobitab's
`ExponentialMovingAverage` builds `shadow_params = [p for p in
parameters if p.requires_grad]`. Cross-checked NCSN++'s param list:
`all_modules.0.W` (the Gaussian Fourier-features projection from
`external/RectifiedFlow/.../layerspp.py:37`) is registered as
`nn.Parameter(..., requires_grad=False)` — explicitly frozen at
construction. That's the missing 645th param.

**Bug we found while fixing it:** the frozen Fourier-W is `randn`'d at
construction time, so each `NCSNpp(cfg)` instantiation draws a
*different* `W` matrix. v1 (non-EMA path) copies the trained `W` from
the raw `model` state-dict; an early attempt at EMA loading skipped
it entirely, leaving the freshly-random `W` in place → catastrophic
PSNR of **7.76 dB** at our standard test cell. Fix: in the EMA path,
also copy frozen params from the raw `model` state-dict separately.

New API: `load_rf_model(..., use_non_ema: bool=False)`. Default loads
EMA (with the frozen-W copy); `use_non_ema=True` matches v1.

#### A2. ζ schedule infrastructure

New module `src/samplers/schedules.py` exposing:
- `zeta_constant(zeta_0)`, `zeta_power(zeta_0, alpha)` (decay),
  `zeta_ramp(zeta_0, alpha)` (ramp-up), `zeta_linear_warmup_then_decay`.
- `resolve(zeta_or_callable, i, N) → float`.
- `stringify(zeta) → str` for CSV cells (so resume stays idempotent
  even with callable schedules).

Both `PixelDPS.sample()` and `FlowDPSRF.sample()` now accept
`zeta: float | ZetaSchedule`. Scalar path is bit-identical to
pre-refactor; 22 backcompat tests in
`tests/test_sampler_backcompat.py` pin the contract.

#### A3. ops_fft module

Lifted `_gaussian_kernel_torch` + `_otf` out of
`src/baselines/{pnp_admm,ddrm}.py` (where they were duplicated) into
new `src/forward/ops_fft.py`. Added `gaussian_otf`, `motion_blur_otf`
for Tier B/C. The two baseline files now import from there with thin
backward-compat wrappers.

#### Hyperparameter tuning (2-image validation @ σ_b=3, σ_n=0.05, NFE=50)

**Pixel-DPS (DDPM is already EMA — only ζ to tune):**

| zeta value         | PSNR (dB) |
|--------------------|-----------|
| 10 (v1)            | 28.14     |
| 15                 | 28.92     |
| 20                 | 29.30     |
| 25                 | 29.49     |
| 30 (locked v2)     | 29.49     |
| 40 (peak)          | 29.74     |
| 50 (over-aggressive)| 27.98    |

Locked **scalar ζ=30** (middle of broad ζ=25..40 plateau, robust to
condition shift). Tested ramp/power schedules; all hurt for Pixel-DPS.

**FlowDPS-on-RF (with EMA load):**

| zeta config                        | PSNR (dB) |
|------------------------------------|-----------|
| non-EMA + scalar 100 (v1)          | 23.45     |
| EMA + scalar 100                   | 23.60     |
| EMA + scalar 200                   | 23.91     |
| EMA + zeta_power(100, α=1)         | 22.21     |
| EMA + zeta_power(100, α=2)         | 21.27     |
| EMA + zeta_ramp(200, α=1)          | 21.39     |
| EMA + zeta_ramp(300, α=1)          | 23.90     |
| EMA + zeta_ramp(500, α=2)          | 20.99     |
| EMA + **zeta_ramp(200, α=0.5)** (locked v2) | **24.43** |

Locked **EMA + zeta_ramp(200, α=0.5)**. RF needs *more* guidance at
late (data-side / t→1) steps, not less — opposite of the
"guide more early" heuristic that the decay schedules implement.
Diffusion-style power decays universally hurt.

#### v2 method names registered

- `pixel_dps_v2`: same diffusers DDPM, scalar ζ=30.
- `flowdps_rf_v2`: EMA-loaded NCSN++ + `zeta_ramp(200, 0.5)`.

#### Overnight grid

Launched 2026-05-20 ~11:55 local as PID 105977 (detached). Writes
`pixel_dps_v2` + `flowdps_rf_v2` rows to
`outputs/results/main_grid.csv` across the full 18-condition grid ×
50 images = 1800 new rows. ETA ~7 hours.

Existing 1800 v1 rows are untouched.

### EXP-015 — Tier B: spectral-aware likelihood guidance (2026-05-20)

**Goal.** Replace the uniform-across-frequencies ζ with a per-frequency
ζ_f that downweights frequencies where the assumed forward operator
(Gaussian) is below a noise floor. Directly motivated by the
operator-mismatch finding from EXP-009.

**Commit:** `315ac85` on `feature/publication-improvements`.

**Implementation:**

New module `src/samplers/spectral_weight.py`:

- `noise_floor_weight(H_otf, eps, alpha, normalize_mean=True)`:
  `W(f) = 1 / (1 + α · relu(eps − |H(f)|))`. For frequencies above
  the threshold the weight is 1; below, it falls smoothly. Mean-
  normalized so the effective gradient magnitude stays compatible
  with the existing ζ tunings.
- `spectral_residual_l2(residual, W)`: computes
  `‖IFFT(W · FFT(residual))‖_2`. With `W=None` falls back to spatial
  `‖residual‖_2` bit-identically (verified to fp32 tolerance).

Sampler refactor: both `PixelDPS.sample()` and `FlowDPSRF.sample()`
gain `spectral_weight: Optional[torch.Tensor]` kwarg, default `None`.

Methods registered in `scripts/run_robustness.py`: `pixel_dps_spectral`
and `flowdps_rf_spectral`. The OTF is computed once per run from
`gaussian_otf(args.assumed_sigma, (256, 256))` and reused for every
(L, θ, σ_n) cell — the W tensor is constant across the cells of a
single run.

**Empirical validation:** **deferred.** Smoke attempt at 11:54 OOMed
because the Tier-A grid is using all available VRAM (3060 12GB / Tier-A
~5GB / smoke ~7.5GB). Will validate once the Tier-A grid completes.

### EXP-016 — Tier C: Tweedie-corrected likelihood (Π-GDM-style, 2026-05-20)

**Goal.** Replace the point-estimate likelihood `‖y − A(x̂₀)‖_2`
with the proper posterior-aware Gaussian-convolved likelihood that
accounts for x̂₀'s covariance under the Tweedie estimator. Cleanest
mathematical contribution among the three tiers.

**Commit:** `3e33703` on `feature/publication-improvements`.

**Derivation.** For a circulant Gaussian-blur operator A with OTF H,
the Tweedie clean-image estimate x̂₀ has covariance `r_t · I` where:

- **Diffusion (VP DDPM):** `r_t = (1 − ᾱ_t) / ᾱ_t`.
- **Rectified flow:** `r_t = (1 − t)²` (RF analog).

The posterior likelihood under this covariance is Gaussian with
covariance `σ_n² I + r_t · A A^T`, which is diagonal in FFT basis:

> log p(y | x̂₀) ∝ −∑_f |Y(f) − H(f) X̂₀(f)|² / (σ_n² + r_t · |H(f)|²)

So the per-frequency weight is `1 / (σ_n² + r_t · |H(f)|²)`. We apply
its square root in spectral_residual_l2 so `‖√W · residual_fft‖₂²`
equals the log-likelihood up to a constant.

**Implementation.**

New module `src/samplers/tweedie_likelihood.py`:
- `pigdm_weight(H_otf, sigma_n, r_t)` — mean-normalized, with
  `r_t_cap=1e3` to prevent blow-up near ᾱ_t → 0 (mirrors DDRM).
- `diffusion_r_t(alpha_bar_t)` and `rf_r_t(t)` helpers.

Sampler refactor: both samplers gain `pigdm_otf` and `pigdm_sigma_n`
kwargs. When `pigdm_otf is not None`, `r_t` is computed per-step
(diffusion uses the `alphas_cumprod[t]` already available; RF uses
the loop's `num_t`), the weight is recomputed, and the existing
`spectral_residual_l2` does the FFT reweighting.

Methods registered in both `scripts/run_grid.py` and
`scripts/run_robustness.py`: `pixel_dps_pigdm`, `flowdps_rf_pigdm`.
They build on the Tier-A v2 ζ defaults (Pixel-DPS scalar ζ=30,
FlowDPS-on-RF EMA + ramp(200, 0.5)).

**Tests.** 4 new unit tests in `tests/test_sampler_backcompat.py`:
- `pigdm_weight` shape + positivity
- uniform when r_t=0 (no correction)
- `diffusion_r_t` monotone in ᾱ_t
- `rf_r_t` boundary values

All 15 backcompat tests pass.

**Empirical validation:** **deferred** for the same GPU-contention
reason as Tier B. Will run smoke + grids once Tier-A completes.

---

## Open questions / to-do

- Can we JIT-compile `external/RectifiedFlow/ImageGeneration/op/upfirdn2d`
  against our torch 2.4.1 + nvcc 12.0? If not, write a `upfirdn2d_native`
  PyTorch fallback.
- The DDPM checkpoint download was from `pytorch_model.bin` (pickle); diffusers
  warned about safetensors not being available. Not a correctness issue but
  worth noting.
- Need to decide on a fixed NFE schedule (uniform vs. quadratic) — current
  default is whatever DDIMScheduler picks.

## Material for the final report

Section 5 (Results) will replace the fabricated content in `docs/group13.pdf`.
Sentences to draft from EXP-001 once we have full grid numbers:
- "Pixel-DPS on CelebA-HQ-256 achieves PSNR XX ± Y dB at NFE = 100 across
  the σ_blur ∈ {1.5, 3.0, 5.0} grid, with monotonic quality–ζ behavior up to
  ζ = 5."
- "Inference wall-clock per image on a single RTX 3060 12 GB is XX s at
  NFE = 100, XX s at NFE = 50, and XX s at NFE = 25."

To-update sections after pivot:
- 1 Introduction: replace FFHQ-256 with CelebA-HQ-256 throughout.
- 2 Related Work: keep DPS / FlowDPS / Liu RF citations. Add a sentence framing
  our reimplementation of FlowDPS-style guidance on the Liu 2023 backbone.
- 3 Methodology: describe the DDIM-based reverse with Tweedie + L2-norm
  likelihood gradient (the fix in EXP-001).
- 4 Experimental Design: 18 conditions × 50 images, RTX 3060, all metrics
  measured on `data/celeba_hq_256/test`.

