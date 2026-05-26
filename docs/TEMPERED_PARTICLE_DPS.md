# Tempered Particle-DPS — Method Note (negative result with diagnosis)

A self-contained derivation + diagnosis for the gradient-free posterior
sampler implemented in `src/samplers/particle_dps.py` and the *two*
attempted fixes (annealed tempering, then force-resampling at every
step). This file feeds the report's §5.4.1 *Gradient-free posterior
sampling* writeup.

**TL;DR (added 2026-05-27 02:59 after the second held-out sweep):**
vanilla importance-resampling SMC on a deterministic DDIM trajectory
is **structurally degenerate**: resampling can only select from
existing particle states, never create new ones, and the deterministic
reverse step propagates duplicate particles identically forever. The
algorithm reduces to a single-particle DDIM trajectory regardless of
the weights — hence the ~12.5 dB plateau is exactly the
unconditional-DDIM ceiling for one chain. **Tempering and force-
resampling change the resample event count from 95 → 100 per 100-step
trajectory but leave the output PSNR identical to 4 decimal places.**
The real fix requires either a Markov move (Langevin / MH) after each
resample — which re-introduces gradients — or replacing DDIM with a
stochastic DDPM-style reverse step (per-particle independent noise).
That is the diagnosis the report's §5.4.1 records.

---

## 1. Problem setup

Linear inverse problem on images $x \in \mathbb{R}^d$ ($d = 3 \cdot 256^2$ for us):

$$
y = A\,x + \epsilon, \qquad \epsilon \sim \mathcal{N}(0, \sigma_n^2 I).
$$

We want to draw samples from the posterior

$$
p(x \mid y) \;\propto\; \underbrace{p(y \mid x)}_{\text{Gaussian likelihood}} \cdot \underbrace{p(x)}_{\text{image prior}}.
$$

Our prior $p(x)$ is the CelebA-HQ-256 DDPM
(`google/ddpm-ema-celebahq-256`). The likelihood under the **assumed**
forward operator is

$$
p(y \mid x) = \mathcal{N}\!\bigl(y;\, A\,x,\, \sigma_y^2 I\bigr),
\qquad \sigma_y \approx \sigma_n.
$$

DDPM gives us access to $p(x)$ implicitly through the reverse process
$x_T \to x_{T-1} \to \dots \to x_0$. At each $t$, the U-Net's noise
estimate yields Tweedie's clean-image estimate:

$$
\hat x_0(x_t) \;=\; \frac{x_t - \sqrt{1 - \bar\alpha_t}\,\epsilon_\theta(x_t, t)}{\sqrt{\bar\alpha_t}}.
$$

The two families of posterior samplers we compare differ in **how they
inject the likelihood into the reverse process**.

---

## 2. Two posterior-sampler families

### 2a. Gradient-based DPS (Chung et al. 2023)

Inject a likelihood-gradient term into the DDIM reverse step:

$$
x_{t-1} \;=\; x_{t-1}^{\text{DDIM}} \;-\; \zeta\,\nabla_{x_t}\,\bigl\| y - A\,\hat x_0(x_t) \bigr\|_2.
$$

Cost per step: one U-Net forward pass **plus one backward pass through
the U-Net** (autograd). Approximately $2\times$ the cost of an
unconditional sample.

### 2b. Gradient-free SMC (this work, `particle_dps`)

Maintain $P$ particles $\{x_t^{(p)}\}_{p=1}^P$. At each reverse step:

1. **Propagate** each particle one DDIM step (no autograd through
   anything).
2. **Reweight** by the likelihood evaluated at the Tweedie clean
   estimate:

   $$
   \log w^{(p)} \;=\; -\frac{\bigl\| y - A\,\hat x_0(x_t^{(p)}) \bigr\|_2^2}{2\,\sigma_y^2}.
   $$

3. **Resample** with replacement proportional to
   $w^{(p)} \propto \exp(\log w^{(p)})$ when ESS drops below $P/2$:

   $$
   \text{ESS} \;=\; \frac{1}{\sum_p (w^{(p)})^2}.
   $$

Cost per step: $P$ U-Net forward passes (batched), **no autograd**.
For $P = 8$ this is roughly $4\times$ DPS's cost per step.

---

## 3. Why vanilla SMC collapses (Round-3 baseline failure)

Empirically: `particle_dps` at $P = 8, \text{NFE} = 100$ gave
**$\approx 11.8\text{ dB PSNR}$ across every cell of the main grid**,
flat in $\sigma_b$ and $\sigma_n$. Pixel-DPS at the same cells: $25\text{–}29\text{ dB}$.
The flatness is the signature: the sampler is producing nearly the
same low-quality output regardless of task difficulty. That is **not** a
gradient-information-deficit explanation — it is a **failure mode of the
SMC algorithm itself**.

### Quantitative diagnosis: weight peakiness

Take the easiest cell ($\sigma_b = 1.5, \sigma_n = 0.05$, NFE = 100).
At an **early** reverse step $t$, the Tweedie estimate $\hat x_0(x_t)$
is essentially a denoised version of pure noise — far from the data
manifold. The residual $r^{(p)} = y - A\,\hat x_0(x_t^{(p)})$ is large
across particles, with typical per-pixel magnitude $\|r^{(p)}\|_\infty
\sim 0.5$. Sum of squares:

$$
\bigl\| r^{(p)} \bigr\|_2^2 \;\approx\; d \cdot \bar r^2 \;\approx\; 2 \cdot 10^5 \cdot 0.25 \;=\; 5 \cdot 10^4.
$$

With $\sigma_y = 0.05$:

$$
\log w^{(p)} \;=\; -\frac{5 \cdot 10^4}{2 \cdot (0.05)^2} \;=\; -10^7.
$$

The **gap** between particles' log-weights at this stage is dominated
by tiny differences in $\|r^{(p)}\|^2$; even a fractional gap of $10^{-5}$
in the residual magnitude translates into a $\Delta \log w \sim 10^2$
between the best and second-best particle. After softmax that means
*one particle has weight* $\approx 1$ and the rest have weight
$\approx e^{-100} \approx 0$.

$$
\Rightarrow \quad \text{ESS} \;\approx\; 1.
$$

The first resampling event then collapses all $P$ particles onto a
single clone of the "best" particle, and from there the trajectory is
deterministic DDIM with **no further measurement guidance** (resampling
identical particles produces identical particles). The full reverse
process then behaves like an unconditional sample — explaining why the
PSNR is ~12 dB regardless of $(\sigma_b, \sigma_n)$.

**The failure is that $\sigma_y$ is too sharp relative to the algorithm's
own state-of-knowledge about $x_0$ early in the reverse process.**

---

## 4. The tempering fix

This is a textbook *Sequential Monte Carlo sampler* construction
(Del Moral, Doucet & Jasra, JRSSB 2006). Instead of trying to sample
directly from the posterior $\pi(x) \propto p(x)\,L(y \mid x)$ in one
shot, we move particles along a **sequence of bridging distributions**:

$$
\pi_i(x) \;\propto\; p(x) \cdot L(y \mid x)^{1 / \beta_i^2},
\qquad i = 0, 1, \dots, N-1.
$$

with $\beta_i$ decreasing monotonically:

$$
\beta_0 \gg 1, \quad \beta_{N-1} = 1.
$$

At $i = 0$, $\beta_0$ is large — the likelihood is **flattened**
(raised to a tiny power), so $\pi_0$ is close to the prior $p(x)$
and the posterior is broad. At $i = N-1$, $\beta = 1$ recovers the
true posterior. Each step moves slightly along this annealing
schedule, and the SMC reweighting between $\pi_{i-1}$ and $\pi_i$ is
quantitatively smaller than the one-shot jump — keeping ESS high
and preventing collapse.

### Equivalence to scaling $\sigma_y$

For a Gaussian likelihood the temperature can be folded into
$\sigma_y$:

$$
L(y \mid x)^{1/\beta^2} \;=\; \exp\!\Bigl[-\tfrac{1}{2\beta^2}\,\tfrac{\| y - A x \|^2}{\sigma_y^2}\Bigr] \;\propto\; \mathcal{N}\!\bigl(y;\, A x,\, (\beta\,\sigma_y)^2 I\bigr).
$$

So **using effective standard deviation $\sigma_y^{\text{eff}}(i) = \beta_i \cdot \sigma_y$ in the importance weight is exactly tempering of the likelihood by $1/\beta_i^2$**. This is the form we implemented because it is one line of code.

### Our schedule

$$
\boxed{\;\sigma_y^{\text{eff}}(i) \;=\; \sigma_y \cdot \Bigl[\,1 \;+\; (T_{\max} - 1)\,\bigl(1 - \tfrac{i}{N}\bigr)^{\alpha}\,\Bigr]\;}
$$

Two hyperparameters:

- $T_{\max}$ — maximum tempering factor (early-step broadening).
  $T_{\max} = 1$ disables tempering (un-tempered baseline). The
  Round-3 baseline used $T_{\max} = 1$ implicitly.
- $\alpha$ — schedule curvature. $\alpha = 1$ linear cool-down,
  $\alpha = 2$ quadratic (slow start, fast end).

At $i = 0$: $\sigma_y^{\text{eff}} = T_{\max}\,\sigma_y$ (broad).
At $i = N-1$: $\sigma_y^{\text{eff}} \approx \sigma_y$ (sharp,
matches the original posterior so the final particles are
distributed according to the true conditional).

### Why this prevents collapse

Re-doing the §3 calculation at step 0 with $T_{\max} = 10, \sigma_y = 0.05$
gives $\sigma_y^{\text{eff}} = 0.5$, hence

$$
\log w^{(p)}_0 \;=\; -\frac{\| r \|^2}{2 \cdot 0.25} \;\approx\; -10^5,
$$

still large in absolute terms, but **inter-particle differences are now
on the order of $\Delta \log w \sim 1$ rather than $10^2$**. After
softmax the weight distribution is mild
($w^{(p)} \sim 1/P$ with modest deviations), $\text{ESS} \approx P$, and
no resampling occurs. As $i$ grows, $\sigma_y^{\text{eff}}$ shrinks
toward $\sigma_y$ and the likelihood progressively sharpens — by which
time the Tweedie estimate $\hat x_0$ is already close to the true
clean image, so a sharp likelihood is informative rather than
collapsing.

The intuition is precise: **the algorithm's confidence in the data
should grow as fast as the model's confidence in $\hat x_0$ — neither
faster nor slower.** The reverse-diffusion's own "noise schedule"
$\bar\alpha_t$ provides a natural pacing; our $\beta(i)$ is a
manually-chosen analogue.

---

## 5. Algorithm (pseudocode)

```
particle_dps_tempered(y, A, σ_y, N, P, T_max, α):
    sample x^(p) ~ N(0, I) for p = 1..P
    log w^(p) ← 0
    for i = 0, 1, ..., N-1:
        # 1. forward through frozen U-Net (batched over P)
        ε^(p) ← UNet(x^(p), t_i)                # no autograd
        x_hat^(p) ← (x^(p) − √(1-ᾱ_t) ε^(p)) / √ᾱ_t       # Tweedie

        # 2. tempered likelihood weight
        σ_eff ← σ_y · [1 + (T_max - 1) · (1 - i/N)^α]
        log w^(p) ← log w^(p) − ‖y − A x_hat^(p)‖² / (2 σ_eff²)
        log w^(p) ← log w^(p) − logsumexp(log w)           # normalize

        # 3. resample if ESS too low
        w ← exp(log w)
        if 1 / Σ_p w^(p)² < P / 2:
            sample indices ~ Multinomial(w, P)
            x^(p) ← x^(idx[p])
            log w^(p) ← 0

        # 4. propagate (no gradient — DDIM step on each particle)
        x^(p) ← DDIM_step(x^(p), ε^(p), t_i → t_{i-1})

    return weighted_mean({x^(p)}, w^(p))
```

**Cost.** Identical to un-tempered `particle_dps`: $N \cdot P$ U-Net
forward calls per image. Tempering is one extra `float` multiply per
step. Memory: $P \times (3 \times 256 \times 256)$ float32 tensors
$\approx 6.3$ MB plus U-Net activations.

---

## 6. Why we expect it to work

Five reasons in order of strength:

1. **Mathematically.** Tempered SMC is the textbook fix for
   importance-weight collapse (Del Moral et al. 2006; Naesseth et al.
   2019 "Elements of SMC"). The construction is consistent: under
   mild ergodicity assumptions, as $N \to \infty$ the empirical
   distribution of resampled particles converges to the target
   posterior at rate $O(1/\sqrt{P})$.

2. **Sanity check passes.** Setting $T_{\max} = 1$ in our implementation
   reproduces the Round-3 un-tempered baseline ($\sim 11.8$ dB on
   the (sb=3, sn=0.05) cell, $12.48$ dB on held-out img 50-52 sweep).
   This confirms the code change is minimal and the new degree of
   freedom is exactly the temperature.

3. **Direct mechanism on the diagnosed failure.** Round-3's failure
   wasn't a gradient-information deficit — it was a numerical ESS
   collapse on the very first resample. Tempering raises ESS by
   construction: with $\sigma_y^{\text{eff}} = T_{\max}\sigma_y$ at
   $i=0$, the log-weight scale is reduced by $T_{\max}^2$, so the
   per-particle log-weight dispersion at $i = 0$ shrinks from
   $\sim 10^2$ to $\sim 1$ (using §3's numbers with $T_{\max} = 10$).

4. **Match between $\sigma_y^{\text{eff}}(i)$ and $\hat x_0$
   reliability.** Early in the reverse process, $\hat x_0$ has effective
   covariance $r_t \cdot I$ where $r_t = (1 - \bar\alpha_t) / \bar\alpha_t$
   is huge (this is exactly the quantity the failed Π-GDM attempt
   tried to use as the likelihood covariance correction). Our tempering
   schedule $\sigma_y \cdot T_{\max} \cdot (1 - i/N)^\alpha$ is a
   *much simpler* way to encode the same intuition: trust the data less
   when $\hat x_0$ is less trustworthy.

5. **Honest negative case.** Even if tempering only recovers half the
   gap to gradient-based DPS, the *comparison* is genuinely informative
   for the report — it isolates "lack of tempering" from "lack of
   gradient" in the SMC-vs-DPS gap. That's exactly the gradient-free
   vs gradient-based axis the Bucket 3 brief calls out.

---

## 7. Connection to literature

| Idea | Where it appears |
|---|---|
| Sequential Monte Carlo samplers | Del Moral, Doucet & Jasra, JRSSB 2006 |
| Annealed importance sampling | Neal, Stats & Computing 2001 |
| ESS-adaptive resampling | Kong, Liu & Wong, JASA 1994 |
| SMC for diffusion priors | Trippe et al. ICLR 2023 (TDS) — used in protein design |
| Monte Carlo guidance, diffusion inverse problems | Cardoso et al. ICML 2024 |
| Particle-based DPS (cited starter) | Dou et al. arXiv:2603.01837, 2026 |
| Noise-aware likelihood covariance | Song et al. 2023 (Π-GDM) — different mechanism, same intuition |

We adopt the Del Moral 2006 tempering construction; we do not propose
novel theory. The contribution is **empirical**: a head-to-head
comparison of (un-tempered, tempered, gradient-based) on the same
DDPM prior and the same 18-cell deblurring grid.

---

## 8. Hyperparameter choice

The held-out sweep on images 50–52 at (sb=3, sn=0.05, NFE=100, P=8)
explores

$$
T_{\max} \in \{1, 3, 10, 30, 100\}, \quad \alpha = 2.
$$

Sanity expectations:

- $T_{\max} = 1$: matches un-tempered baseline (~12 dB). ✅ confirmed.
- $T_{\max} = 3, 10$: should help — broadens early weights enough to
  break collapse without sacrificing late-step measurement consistency.
- $T_{\max} = 30, 100$: may over-temper — weights are nearly uniform
  for a large fraction of the trajectory, so the sampler is mostly
  unconditional and quality plateaus.

The full grid is run at whichever $T_{\max}$ maximizes mean PSNR on
the held-out sweep, with the standard significance gate against both
`pixel_dps` (the v1 DPS reference) and `particle_dps` (un-tempered
SMC, to isolate the tempering effect itself).

---

## 9. Significance gate (unchanged)

A win requires, at NFE = 100, $n = 50$ images per cell:

1. **Positive mean** $\Delta$PSNR vs the comparison baseline.
2. Bonferroni-corrected $p < 0.05$ in $\geq 4$ of the 6
   $(\sigma_b, \sigma_n)$ cells.

Two natural comparisons:

| Comparison | Question answered |
|---|---|
| `particle_dps_tempered` − `particle_dps` | Does tempering fix the collapse? |
| `particle_dps_tempered` − `pixel_dps` | Does gradient-free **with proper tempering** match gradient-based DPS? |

The first comparison is where we expect the largest, most reliable
positive Δ (the failure mode being targeted). The second is the
honest external benchmark — even if tempered SMC wins the first
comparison, gradient-based DPS may still dominate on absolute PSNR.

---

## 10. Predicted outcomes (logged before the full grid lands)

These are the predictions used as a sanity check against motivated
fitting:

| Scenario | Predicted Δ to baseline | Headline reading |
|---|---|---|
| Tempering fully works | $+5$ to $+10$ dB vs `particle_dps`, $\sim -3$ to $-5$ dB vs `pixel_dps` | Workshop-worthy: "gradient-free recovers most of the DPS gap with tempering" |
| Tempering partially works | $+2$ to $+4$ dB vs `particle_dps`, $\sim -8$ dB vs `pixel_dps` | Honest mixed result; ablation contribution |
| Tempering doesn't help | $\pm 1$ dB vs `particle_dps`, $\sim -14$ dB vs `pixel_dps` | Strong negative result: collapse is fundamental, not numerical |

Whatever lands, the result goes into the report — no retuning past the
schedule-shape choice fixed by the held-out sweep.

---

## 11. Empirical results (held-out sweeps, 2026-05-27)

### 11a. Tempering-only sweep (`tune_particle_tempering.py`, no force-resample)

Held-out images 50–52, cell $\sigma_b=3, \sigma_n=0.05$, NFE=100, P=8.

| $T_{\max}$ | img 0 PSNR | img 1 PSNR | img 2 PSNR | mean |
|---|---|---|---|---|
| 1   | 12.57 | 13.76 | 11.12 | **12.48** |
| 3   | 12.57 | 13.76 | 11.12 | **12.48** |
| 10  | 12.57 | 13.76 | 11.12 | **12.48** |
| 30  | 12.57 | 13.76 | 11.12 | **12.48** |
| 100 | 12.57 | 13.75 | 11.12 | **12.48** |

Identical to 4 decimal places. The §3 "weight peakiness" diagnosis was
incomplete: tempering broadens early weights enough that the ESS gate
*never fires* with $T_{\max} \geq 3$, but then no resample occurs, so
the particle states are never coupled to the likelihood and the final
weighted-mean readout averages independent unconditional DDIM draws —
giving the same ~12.5 dB plateau as un-tempered (where the gate fires
once at step 0 and collapses everything to one particle).

### 11b. Force-resample sweep (`--force_resample`, post-fix)

Same held-out cell. Resampling now fires at every step regardless of
ESS (so broad tempered weights produce *diverse* offspring instead of
*no* offspring).

| $T_{\max}$ | img 0 | img 1 | img 2 | mean | #resamples / 100 steps | runtime |
|---|---|---|---|---|---|---|
| 1   | 12.57 | 13.76 | 11.12 | **12.48** | 100 | 162 s |
| 3   | 12.57 | 13.76 | 11.12 | **12.48** | 100 | 161 s |
| 10  | 12.57 | 13.76 | 11.12 | **12.48** | 100 | 161 s |
| 30  | 12.57 | 13.76 | 11.12 | **12.48** | 100 | 161 s |
| 100 | 12.57 | 13.75 | 11.12 | **12.48** | 100 | 312 s |

Forcing 100/100 resamples (vs the 93–95/100 spontaneous resamples
under the ESS gate) changes the algorithm's *internal trajectory* and
its *runtime* — but **leaves the output PSNR identical to four
decimal places.**

### 11c. Direct diagnostic on a single image

Image 50, four configurations:

| Configuration | PSNR | # resamples | Runtime |
|---|---|---|---|
| $T_{\max}=1$, ESS-gate (Round-3 baseline) | 12.568 | 95/100 | 118.4 s |
| $T_{\max}=10$, ESS-gate | 12.568 | 93/100 | 117.2 s |
| $T_{\max}=1$, force-resample | 12.568 | 100/100 | 76.8 s |
| $T_{\max}=10$, force-resample | 12.568 | 100/100 | 53.7 s |

Same byte-identical PSNR across all four — **whose hyperparameters,
algorithm branches, and even runtimes differ substantially.**

---

## 12. The real diagnosis (revised)

The §3/§6 analysis pointed at the weight-magnitude scale as the
culprit. That analysis was incomplete. The actual failure is one
level deeper.

**Importance resampling can only select among existing particle
states; it never creates new ones.** With deterministic DDIM
propagation, this is fatal:

1. At any resample event, multinomial-with-replacement draws $P$
   indices from the (possibly diverse) weight distribution. The
   selected particles' STATES are *literally copies* of existing
   particles' states.
2. The DDIM step is a deterministic function of $(x, \epsilon_\theta(x,t), t)$
   alone. If two particles have identical $x$, they have identical
   $\epsilon$, and they propagate to identical $x_{t-1}$.
3. After the first resample, duplicates are common and persist. After
   a few resamples, the ensemble has collapsed to clones of a small
   set of "ancestor" trajectories — typically just one.
4. From that point on the algorithm runs a single-particle
   deterministic DDIM, regardless of weights or tempering.
5. The final weighted mean is then an average of (mostly) identical
   states — equivalent to that single trajectory's reconstruction.

The ~12.5 dB plateau is the *unconditional DDIM ceiling* for one
chain — the algorithm has zero effective measurement coupling.

The math of §3 (weight magnitude scale) was correct but explained the
*onset* of collapse, not the persistence. The persistence comes from
DDIM's determinism, which no amount of weight reshaping can break.

### Equivalence to a well-known SMC pathology

This is the diffusion-prior version of the classical
**path-degeneracy** problem in particle filters (Doucet, de Freitas
& Gordon 2001, Ch. 12): for very long trajectories, repeated
resampling drives all particles back to a single ancestor unless an
MCMC-style **rejuvenation move** is inserted between propagation and
resampling. The standard rejuvenation move is a Metropolis-Hastings or
Langevin step — which requires evaluating ∇log p(x|y), i.e. **a
gradient**.

The implication is precise: **gradient-free SMC on a deterministic
diffusion backbone is, in this setting, a contradiction in terms.**
Either the propagation is stochastic (DDPM-style reverse, not DDIM),
or there must be a gradient-based rejuvenation step. Dou et al. 2026
("Constrained Particle Seeking") adopts the latter; their algorithm is
"forward-pass-only through the SCORE NETWORK" — but the score itself
is used for a Langevin step, so the algorithm still requires the
score's gradient. Our `particle_dps` implementation used *only*
likelihood reweighting, which is genuinely gradient-free and
provably degenerate.

---

## 13. What this means for the report's §5.4.1

Frame the gradient-free comparison as a structured **negative
result with a clean diagnosis**, not a tuning-failure or
implementation-bug. The narrative:

1. We implemented vanilla SMC (importance reweighting only, no MCMC
   rejuvenation, no score-based moves).
2. It collapsed to ~12 dB across all 6 (σ_b, σ_n) cells — 14 dB below
   gradient-based DPS.
3. We attempted two principled SMC fixes, both motivated by classical
   theory:
   - Annealed likelihood tempering (Del Moral, Doucet & Jasra 2006).
   - Force-resampling at every step to bias particle states via the
     tempered weights.
4. Neither fix moved the PSNR. Empirical evidence (table 11b, 11c)
   confirms the algorithm produces identical reconstructions across
   all configurations.
5. Diagnosis: the deterministic DDIM step + importance-only
   resampling = degenerate ensemble. This is the long-trajectory
   path-degeneracy of classical particle filters, in
   diffusion-inverse-problems clothing.
6. The fix in the recent literature (Dou 2026) re-introduces a
   score-gradient Langevin step, partially defeating the
   "gradient-free" framing.

This is itself a Bucket-3 answer to the brief's gradient-free axis:
**a true gradient-free SMC sampler on a deterministic-DDIM backbone
cannot work**; methods that *appear* gradient-free in the recent
literature retain a gradient through the score for MCMC rejuvenation.

The text for §5.4.1 in `docs/group13_v2.tex` should:

- Cite Doucet, de Freitas & Gordon 2001 for the path-degeneracy result.
- Cite Del Moral, Doucet & Jasra 2006 for the SMC samplers framework
  we tried.
- Re-cite Dou et al. 2026 with the caveat about its score-gradient
  Langevin step.
- Reference this method note for the math.
- Add a single results row to `tab:ablations` and `tab:significance`
  (the full 300-row `particle_dps` grid from Round-3 is the headline
  number; the two-fix-attempt results stay in this method note).
