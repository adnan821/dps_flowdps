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

### EXP-006 — FlowDPS-on-RF ζ sweep (2026-05-19, ongoing)

**Config.** 2 images, NFE=100, σ_blur=3.0, σ_noise=0.05, seeds matched to
EXP-001 for comparison.

| ζ   | PSNR(x_hat) | SSIM   | LPIPS  |
|-----|-------------|--------|--------|
| 1   | 9.12        | 0.237  | 0.681  |
| 3   | 10.25       | 0.302  | 0.591  |
| 10  | 14.66       | 0.434  | 0.448  |
| 30  | (pending)   | —      | —      |
| 100 | (pending)   | —      | —      |

PSNR(y) = 22.76 dB. Quality is monotonically improving in ζ but still well
below Pixel-DPS at the same NFE/ζ. **Hypothesis:** FlowDPS-on-RF needs much
larger ζ because the gradient path through `x_t + (1-t) v` has a `(1-t)`
factor that shrinks late-step contributions, and the velocity-field magnitudes
are different from the diffusion-noise magnitudes that drove Pixel-DPS's
gradient scale.

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

