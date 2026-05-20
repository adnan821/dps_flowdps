"""Load Liu et al. 2023 Rectified Flow CelebA-HQ-256 checkpoint into a clean
velocity-field interface.

The NCSN++ model architecture and config come from the gnobitab/RectifiedFlow
repository (cloned at `external/RectifiedFlow/ImageGeneration`). We:
1. Put their `ImageGeneration` directory on `sys.path` so their relative
   imports work.
2. Use their `configs/rectified_flow/celeba_hq_pytorch_rf_gaussian.py` config.
3. Instantiate `NCSNpp(config)` and load the EMA weights from the .pth.
4. Expose a clean `velocity(x_t, t)` interface used by our FlowDPS sampler.

Conventions:
- `x_t ∈ [-1, 1]^(B, C, H, W)` — same scale as DDPM checkpoints, since the
  config has `data.centered = True`.
- `t ∈ [0, 1]^B` — continuous-time scalar per batch element.
- `velocity(x_t, t)` returns the rectified-flow velocity field with the same
  shape as `x_t`.
"""
from __future__ import annotations

import os
import sys
from pathlib import Path
from typing import Optional

import torch
import torch.nn as nn

# --- Make the gnobitab/RectifiedFlow ImageGeneration package importable ---
_REPO_ROOT = Path(__file__).resolve().parents[2]
_RF_IG = _REPO_ROOT / "external" / "RectifiedFlow" / "ImageGeneration"
if not _RF_IG.exists():
    raise RuntimeError(
        f"gnobitab/RectifiedFlow not cloned at {_RF_IG}. "
        f"Run: git clone --depth 1 https://github.com/gnobitab/RectifiedFlow.git external/RectifiedFlow"
    )
sys.path.insert(0, str(_RF_IG))


# --- Build the config inline (port of the CelebA-HQ RF + lsun defaults) ---
def _build_config() -> "ml_collections.ConfigDict":
    """Mirror of configs/rectified_flow/celeba_hq_pytorch_rf_gaussian.py + the
    lsun defaults it inherits from, condensed to just what NCSNpp uses."""
    import ml_collections

    config = ml_collections.ConfigDict()

    # training (only the bits NCSNpp reads via sde_lib indirection)
    config.training = ml_collections.ConfigDict()
    config.training.sde = "rectified_flow"
    config.training.continuous = False
    config.training.reduce_mean = True
    config.training.likelihood_weighting = False
    config.training.beta_min = 0.1
    config.training.beta_max = 20.0
    config.training.num_scales = 2000  # matches gnobitab default_lsun_configs (sigmas buffer shape)

    # data
    config.data = ml_collections.ConfigDict()
    config.data.dataset = "CelebA-HQ-Pytorch"
    config.data.image_size = 256
    config.data.num_channels = 3
    config.data.centered = True
    config.data.uniform_dequantization = False
    config.data.random_flip = True

    # model — verbatim from the CelebA-HQ RF config
    config.model = m = ml_collections.ConfigDict()
    m.name = "ncsnpp"
    m.scale_by_sigma = True
    m.ema_rate = 0.999
    m.normalization = "GroupNorm"
    m.nonlinearity = "swish"
    m.nf = 128
    m.ch_mult = (1, 1, 2, 2, 2, 2, 2)
    m.num_res_blocks = 2
    m.attn_resolutions = (16,)
    m.resamp_with_conv = True
    m.conditional = True
    m.fir = True
    m.fir_kernel = (1, 3, 3, 1)
    m.skip_rescale = True
    m.resblock_type = "biggan"
    m.progressive = "output_skip"
    m.progressive_input = "input_skip"
    m.progressive_combine = "sum"
    m.attention_type = "ddpm"
    m.init_scale = 0.0
    m.fourier_scale = 16
    m.conv_size = 3
    m.dropout = 0.0
    # required by NCSNpp ctor for VESDE/RF setups
    m.sigma_min = 0.01
    m.sigma_max = 378.0  # matches default_lsun_configs (sigma_max for sigmas buffer)
    m.num_scales = 2000  # matches gnobitab default_lsun_configs (sigmas buffer shape)
    m.embedding_type = "fourier"

    # optim — not used at inference but the registration code touches it
    config.optim = ml_collections.ConfigDict()
    config.optim.optimizer = "Adam"
    config.optim.lr = 2e-4
    config.optim.beta1 = 0.9
    config.optim.eps = 1e-8
    config.optim.weight_decay = 0.0
    config.optim.warmup = 5000
    config.optim.grad_clip = 1.0

    config.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    config.seed = 42
    return config


def load_rf_model(
    ckpt_path: str = "checkpoints/rectified_flow/celebahq_256_rf.pth",
    device: str | torch.device = "cuda",
    use_non_ema: bool = False,
) -> "RFVelocityModel":
    """Instantiate NCSN++ and load weights. EMA by default (best quality);
    set `use_non_ema=True` to load the raw 'model' state_dict instead (used
    for the v1 baseline so the v2-vs-v1 comparison is apples-to-apples
    against the originally-shipped non-EMA numbers)."""
    # These imports must come AFTER sys.path is munged.
    from models import ncsnpp  # noqa: F401  (registers 'ncsnpp')
    from models.ncsnpp import NCSNpp

    cfg = _build_config()
    model = NCSNpp(cfg)
    state_dict_blob = torch.load(ckpt_path, map_location="cpu", weights_only=False)

    # Try EMA shadow_params. gnobitab/RectifiedFlow's EMA tracker
    # (`external/RectifiedFlow/ImageGeneration/models/ema.py:30`) builds
    # `shadow_params = [p.clone() for p in parameters if p.requires_grad]`
    # — it skips frozen parameters. NCSN++'s Gaussian Fourier-features
    # projection at `layerspp.py:37` is registered as
    # `nn.Parameter(..., requires_grad=False)` — that's the missing 645th
    # param. To align cleanly, walk `model.named_parameters()` in registration
    # order and skip those with `requires_grad=False`, mapping the remaining
    # in-order to the saved `shadow_params`.
    loaded = False
    ema = state_dict_blob.get("ema")
    if isinstance(ema, dict) and "shadow_params" in ema and not use_non_ema:
        shadow = ema["shadow_params"]
        named_params = list(model.named_parameters())
        trainable = [(n, p) for n, p in named_params if p.requires_grad]
        frozen = [(n, p) for n, p in named_params if not p.requires_grad]
        if len(shadow) == len(trainable):
            # Verify shape-by-shape before any copy.
            for i, ((n, p), sp) in enumerate(zip(trainable, shadow)):
                if p.shape != sp.shape:
                    raise RuntimeError(
                        f"[rf_celebahq] EMA shape mismatch at trainable idx {i} ({n}): "
                        f"model {tuple(p.shape)} vs shadow {tuple(sp.shape)}"
                    )
            with torch.no_grad():
                for (_, p), sp in zip(trainable, shadow):
                    p.copy_(sp.to(p.dtype))
            # CRITICAL: frozen params (e.g. `all_modules.0.W`, the Fourier-
            # features random projection) are excluded from EMA but ARE
            # initialized randomly each time we construct the model. They
            # must still be copied from the checkpoint's raw 'model'
            # state_dict, otherwise we use a fresh random projection that
            # doesn't match what the trained weights expect.
            raw = state_dict_blob.get("model", {})
            raw = {k.replace("module.", ""): v for k, v in raw.items()}
            n_frozen_copied = 0
            with torch.no_grad():
                for name, p in frozen:
                    if name in raw:
                        p.copy_(raw[name].to(p.dtype))
                        n_frozen_copied += 1
            print(
                f"[rf_celebahq] Loaded EMA shadow_params ({len(shadow)} tensors); "
                f"also copied {n_frozen_copied}/{len(frozen)} frozen param(s) "
                f"from the raw 'model' state_dict "
                f"({[n for n, _ in frozen][:3]})."
            )
            loaded = True
        else:
            print(
                f"[rf_celebahq] EMA shadow_params count ({len(shadow)}) does not match "
                f"trainable model params ({len(trainable)}); falling back to "
                f"'model' state_dict."
            )

    if not loaded:
        raw = state_dict_blob["model"] if "model" in state_dict_blob else state_dict_blob
        sd = {k.replace("module.", ""): v for k, v in raw.items()}
        missing, unexpected = model.load_state_dict(sd, strict=False)
        print(
            f"[rf_celebahq] Loaded raw 'model' state_dict "
            f"(missing={len(missing)} unexpected={len(unexpected)})."
        )
        # If there are missing keys, we have a real problem — dump the first few.
        if missing:
            print(f"[rf_celebahq] First missing keys: {list(missing)[:5]}")
        if unexpected:
            print(f"[rf_celebahq] First unexpected keys: {list(unexpected)[:5]}")

    model = model.to(device).eval()
    for p in model.parameters():
        p.requires_grad_(False)
    return RFVelocityModel(model, cfg, device=device)


class RFVelocityModel:
    """Wrap NCSN++ in a clean velocity-field interface used by FlowDPS-on-RF.

    The Liu 2023 rectified-flow training parameterization:
        x_t = (1 - t) * x_0 + t * x_1,  x_0 ~ data, x_1 ~ N(0, I)
        v(x_t, t) = x_1 - x_0          (the constant target velocity)
    At inference we integrate dx/dt = v_theta(x_t, t) BACKWARD from t=1 (noise)
    to t=0 (clean image).

    NCSN++ takes integer-ish "scaled" timesteps (`t * 999`) as input.
    """

    def __init__(
        self,
        model: nn.Module,
        config: "ml_collections.ConfigDict",
        device: str | torch.device = "cuda",
        time_scale: float = 999.0,
    ):
        self.model = model
        self.config = config
        self.device = torch.device(device)
        self.time_scale = time_scale

    def velocity(self, x_t: torch.Tensor, t: torch.Tensor | float) -> torch.Tensor:
        """Predict v_theta(x_t, t). Shapes: x_t (B,C,H,W); t scalar or (B,)."""
        if isinstance(t, (int, float)):
            t = torch.full((x_t.shape[0],), float(t), device=x_t.device, dtype=x_t.dtype)
        elif t.dim() == 0:
            t = t.expand(x_t.shape[0])
        # NCSN++ expects scaled integer-like timesteps. RF training used t*999.
        t_scaled = t.to(x_t.device, x_t.dtype) * self.time_scale
        return self.model(x_t, t_scaled)


if __name__ == "__main__":
    print("Loading RF CelebA-HQ-256 model...")
    rf = load_rf_model()
    print("Loaded.")
    print(f"Param count: {sum(p.numel() for p in rf.model.parameters()):,}")
    # Sanity forward pass
    x = torch.randn(1, 3, 256, 256, device=rf.device)
    t = torch.tensor([0.5], device=rf.device)
    with torch.no_grad():
        v = rf.velocity(x, t)
    print(f"v.shape: {v.shape}  dtype: {v.dtype}")
    print(f"CUDA peak mem: {torch.cuda.max_memory_allocated() / 2**20:.1f} MiB")
