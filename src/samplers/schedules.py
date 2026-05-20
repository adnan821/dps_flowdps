"""Step-wise guidance-scale (ζ) schedules for posterior samplers.

A `ZetaSchedule` is just `Callable[[int, int], float]` taking
`(step_idx, total_steps)` and returning the guidance scale to use at
that step. The same interface works for both Pixel-DPS (DDIM step idx)
and FlowDPS-on-RF (Euler step idx).

Convention: `step_idx` runs from 0 (first reverse / first ODE step) to
`total_steps - 1` (last step). For diffusion this means step_idx=0 is
the high-noise end; for flow this means step_idx=0 is the noise-side
of the trajectory. We expose the same convention so callers don't
have to care.
"""
from __future__ import annotations

from typing import Callable

ZetaSchedule = Callable[[int, int], float]


def zeta_constant(zeta_0: float) -> ZetaSchedule:
    """ζ(i) = ζ₀ for all i. Matches the pre-refactor scalar behavior."""
    def _f(i: int, N: int) -> float:  # noqa: ARG001
        return float(zeta_0)
    _f.__name__ = f"const({zeta_0})"
    return _f


def zeta_power(zeta_0: float, alpha: float = 1.0) -> ZetaSchedule:
    """ζ(i) = ζ₀ · (1 - i/N)^α.

    With α > 0 this puts MORE guidance at early (noisy) steps and LESS at
    late (clean) steps — the typical "guide more at the start" heuristic.
    α = 0 collapses to a constant; α = 1 is linear decay; α = 2 is quadratic.

    Equivalently, in flow time t ∈ [0, 1] going noise (t=0) → data (t=1),
    `1 - i/N ≈ 1 - t` is large near noise and small near data, so
    larger ζ near t=0 and smaller ζ near t=1.
    """
    def _f(i: int, N: int) -> float:
        if N <= 1:
            return float(zeta_0)
        return float(zeta_0) * ((1.0 - i / N) ** alpha)
    _f.__name__ = f"power({zeta_0},{alpha})"
    return _f


def zeta_ramp(zeta_0: float, alpha: float = 1.0) -> ZetaSchedule:
    """ζ(i) = ζ₀ · (i/N)^α.

    Mirror of `zeta_power`: more guidance at LATE (data-side / low-noise)
    steps. With α > 0 we start near 0 and ramp up toward `zeta_0`.
    Useful for FlowDPS-on-RF where the late-t (data-side) steps need
    more measurement consistency, not less.
    """
    def _f(i: int, N: int) -> float:
        if N <= 1:
            return float(zeta_0)
        return float(zeta_0) * ((i / N) ** alpha)
    _f.__name__ = f"ramp({zeta_0},{alpha})"
    return _f


def zeta_linear_warmup_then_decay(
    zeta_0: float, warmup_frac: float = 0.1, alpha: float = 1.0
) -> ZetaSchedule:
    """Linear ramp from 0 → ζ₀ over the first `warmup_frac` of steps, then
    power-law decay. Useful when the first few steps are unstable."""
    base = zeta_power(zeta_0, alpha)

    def _f(i: int, N: int) -> float:
        if N <= 1:
            return float(zeta_0)
        warmup_end = max(1, int(warmup_frac * N))
        if i < warmup_end:
            return float(zeta_0) * (i + 1) / warmup_end
        return base(i, N)
    _f.__name__ = f"warmup({zeta_0},{warmup_frac},{alpha})"
    return _f


def resolve(zeta: float | ZetaSchedule, i: int, N: int) -> float:
    """Resolve a possibly-callable ζ to a scalar for step `i` of `N`."""
    if callable(zeta):
        return float(zeta(i, N))
    return float(zeta)


def stringify(zeta: float | ZetaSchedule) -> str:
    """Stringify a ζ value or schedule so we can put it in a CSV cell
    without writing a callable's raw repr. Idempotent for resume."""
    if callable(zeta):
        name = getattr(zeta, "__name__", None) or repr(zeta)
        return name
    return f"{float(zeta)}"
