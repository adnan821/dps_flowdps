"""Generate notebooks/flowdps_smoke.ipynb from a flat list of cells.

The notebook is a Day-1/Day-2 Colab Pro+ smoke test for the official
FlowDPS repository: clones the repo, downloads SD3 weights, and runs
both FlowDPS and PSLD on one sample image as Gaussian deblur.

Run: `python scripts/build_smoke_notebook.py`
"""
from __future__ import annotations

import json
from pathlib import Path


def md(text: str) -> dict:
    return {
        "cell_type": "markdown",
        "metadata": {},
        "source": text.splitlines(keepends=True),
    }


def code(text: str) -> dict:
    return {
        "cell_type": "code",
        "metadata": {},
        "execution_count": None,
        "outputs": [],
        "source": text.splitlines(keepends=True),
    }


CELLS: list[dict] = [
    md(
        """# FlowDPS smoke test on Colab Pro+ A100

Project: `dps_flowdps` — AI-623 DVLM final project.

This notebook clones the official FlowDPS repo
(<https://github.com/FlowDPS-Inverse/FlowDPS>), downloads SD3 Medium
weights, and runs both **FlowDPS** and **PSLD** (latent-DPS) on a single
sample image as our first end-to-end smoke test.

**Pre-reqs (do these once before running this notebook):**
1. Colab Pro+ subscription, A100 runtime selected.
2. Hugging Face account + read token stored as the Colab secret `HF_TOKEN`.
3. SD3 Medium license accepted at <https://huggingface.co/stabilityai/stable-diffusion-3-medium>.
4. Google Drive mountable (you will be prompted in cell 3).
"""
    ),
    md("## 1. Confirm we have an A100"),
    code(
        """!nvidia-smi | head -10
"""
    ),
    md("## 2. Mount Google Drive for persistent artifacts"),
    code(
        """from google.colab import drive
drive.mount('/content/drive')

import os
PROJECT_DIR = '/content/drive/MyDrive/dvlm_proj'
for sub in ['checkpoints', 'outputs', 'data', 'notebooks', 'logs']:
    os.makedirs(os.path.join(PROJECT_DIR, sub), exist_ok=True)
print('Project dir:', PROJECT_DIR)
!ls {PROJECT_DIR}
"""
    ),
    md(
        """## 2b. Redirect the Hugging Face cache to Drive

SD3 Medium weights are ~6 GB. By default `huggingface_hub` caches them
to `/content/` (the VM's local disk), which is wiped when the runtime
disconnects. Point the cache at Drive so the weights are downloaded
once and reused across sessions.

This must run **before** any `from_pretrained(...)` or `login(...)`
call — env vars are read at HF library init."""
    ),
    code(
        """import os
HF_CACHE = '/content/drive/MyDrive/dvlm_proj/checkpoints/hf_cache'
os.makedirs(HF_CACHE, exist_ok=True)
os.environ['HF_HOME'] = HF_CACHE
os.environ['HUGGINGFACE_HUB_CACHE'] = HF_CACHE
os.environ['TRANSFORMERS_CACHE'] = HF_CACHE
print('HF cache → Drive:', HF_CACHE)
!du -sh {HF_CACHE} 2>/dev/null || echo '(cache empty for now)'
"""
    ),
    md(
        """## 3. Authenticate to Hugging Face (for SD3 weights)

We do three things here:
1. Read the `HF_TOKEN` Colab secret.
2. `huggingface_hub.login(...)` — writes the token to
   `~/.cache/huggingface/token` so any HF library on this filesystem can
   find it.
3. **Also set `HF_TOKEN` as an environment variable** so subprocesses
   we launch later (like `!python solve.py ...`) inherit the auth
   without having to call login again."""
    ),
    code(
        """import os
from google.colab import userdata
from huggingface_hub import login
hf_token = userdata.get('HF_TOKEN')
assert hf_token and hf_token.startswith('hf_'), 'HF_TOKEN secret missing or malformed'
login(token=hf_token, add_to_git_credential=False)
os.environ['HF_TOKEN'] = hf_token
os.environ['HUGGING_FACE_HUB_TOKEN'] = hf_token  # legacy var some libs still check
print('HF login OK; HF_TOKEN exported to env for subprocesses.')

# Quick sanity: can we hit a gated SD3 repo right now?
from huggingface_hub import hf_hub_download
try:
    p = hf_hub_download(
        repo_id='stabilityai/stable-diffusion-3-medium-diffusers',
        filename='model_index.json',
    )
    print('SD3-diffusers access OK:', p)
except Exception as e:
    print('SD3-diffusers access FAILED — check license at:')
    print('  https://huggingface.co/stabilityai/stable-diffusion-3-medium-diffusers')
    raise
"""
    ),
    md(
        """## 4. Clone the FlowDPS repo and install its deps

We clone into `/content/FlowDPS` (the Colab VM's local disk, not Drive)
because pip installs and git operations are 10x faster on local SSD than
on Drive."""
    ),
    code(
        """%cd /content
![ -d FlowDPS ] || git clone https://github.com/FlowDPS-Inverse/FlowDPS.git
%cd /content/FlowDPS
!git log -1 --oneline

# FlowDPS imports `from motionblur.motionblur import Kernel` (LeviBorodenko's
# motion-blur kernel generator) but it's NOT listed in their requirements.txt.
# Clone it as a subdirectory of FlowDPS so Python finds it on import.
![ -d motionblur ] || git clone https://github.com/LeviBorodenko/motionblur.git
!ls motionblur/ 2>&1 | head -5
"""
    ),
    md(
        """### 4a. Install a known-good torch + torchvision pair, then restart

Colab's current base image has a torch / torchvision pair that doesn't
match (torchvision's C++ ops fail to register — you'd see
`RuntimeError: operator torchvision::nms does not exist`). FlowDPS
also pins torch 2.4.1 + torchvision 0.19.1 in its requirements, so
we use that exact pair from the CUDA 12.1 channel (matches Colab's
CUDA 12.x driver).

After this cell finishes, **the kernel will kill itself with
`os.kill`** so the new torch can be picked up on reconnect. Drive
mount and Hugging Face login both survive the restart — you can
resume from cell 4b directly without re-running cells 1–3."""
    ),
    code(
        """import subprocess, sys, os, time

print('Installing matched torch 2.4.1 + torchvision 0.19.1 (CUDA 12.1 wheels)...')
subprocess.run([
    sys.executable, '-m', 'pip', 'install', '--quiet', '--force-reinstall',
    'torch==2.4.1', 'torchvision==0.19.1',
    '--index-url', 'https://download.pytorch.org/whl/cu121',
], check=True)
print('Done. Killing the kernel in 3 s so the new torch loads...')
time.sleep(3)
os.kill(os.getpid(), 9)
"""
    ),
    md(
        """### 4b. Verify torch / torchvision are matched, then install FlowDPS deps

After the kernel restart above, Colab should auto-reconnect. The
notebook session state is gone (Python variables reset) but the
**Drive mount, HF login, and the cloned `/content/FlowDPS` checkout
all survive** because they're filesystem-level.

This cell verifies the new torch is healthy and installs the
remaining FlowDPS application-level dependencies."""
    ),
    code(
        """# Sanity-check torch + torchvision are now matched.
import torch, torchvision, torchvision.ops
print('torch:       ', torch.__version__,
      ' cuda:', torch.version.cuda,
      ' cudnn:', torch.backends.cudnn.version())
print('torchvision: ', torchvision.__version__)
assert hasattr(torchvision.ops, 'nms'), 'torchvision::nms still missing'
print('torchvision::nms OK')

# Filter FlowDPS requirements: skip torch/torchvision (already done),
# all nvidia-* CUDA 11.8 wheels, triton (bundled with torch), and
# numpy (ABI-coupled with torch's C extensions).
import os, re, subprocess
os.chdir('/content/FlowDPS')

with open('requirements.txt') as f:
    all_lines = [l.strip() for l in f if l.strip() and not l.startswith('#')]

SKIP_PATTERNS = (
    r'^torch(==|$)', r'^torchvision(==|$)', r'^torchaudio(==|$)',
    r'^nvidia-', r'^triton(==|$)',
    r'^numpy(==|$)',
)
keep_list = [l for l in all_lines
             if not any(re.match(p, l, re.IGNORECASE) for p in SKIP_PATTERNS)]
skip_list = [l for l in all_lines if l not in keep_list]
print(f'Installing {len(keep_list)} FlowDPS app deps; skipping {len(skip_list)}: {skip_list}')
subprocess.run(['pip', 'install', '--quiet', *keep_list], check=True)
print('Install complete.')

# Quick smoke imports to catch any remaining brokenness early.
import diffusers, transformers, accelerate, huggingface_hub, safetensors
print('diffusers:', diffusers.__version__,
      'transformers:', transformers.__version__,
      'accelerate:', accelerate.__version__)
"""
    ),
    md("## 5. Inspect the repo so we know the exact CLI we are about to use"),
    code(
        """!ls /content/FlowDPS
print('---')
!head -80 /content/FlowDPS/solve.py
"""
    ),
    code(
        """!python /content/FlowDPS/solve.py --help 2>&1 | tail -60
"""
    ),
    md(
        """## 6. Pick a sample image

The repo ships sample images under `samples/`. We'll use one of those
for the first smoke run — no FFHQ download needed yet."""
    ),
    code(
        """import os
samples_dir = '/content/FlowDPS/samples'
print(os.listdir(samples_dir) if os.path.isdir(samples_dir) else 'samples/ not found')
"""
    ),
    code(
        """# Pick the first JPEG / PNG we can find.
import glob
candidates = sorted(
    glob.glob('/content/FlowDPS/samples/**/*.jpg', recursive=True)
    + glob.glob('/content/FlowDPS/samples/**/*.png', recursive=True)
    + glob.glob('/content/FlowDPS/samples/**/*.jpeg', recursive=True)
)
print(f'Found {len(candidates)} candidate images')
SAMPLE_IMG = candidates[0] if candidates else None
print('Will use:', SAMPLE_IMG)

# Show it inline
from PIL import Image
from IPython.display import display
if SAMPLE_IMG:
    img = Image.open(SAMPLE_IMG)
    print('Size:', img.size)
    display(img.resize((256, 256)))
"""
    ),
    md(
        """## 7. Smoke run #1 — vanilla FlowDPS on Gaussian deblur

This is the moment of truth: does the canonical FlowDPS implementation
run end-to-end on Colab A100 for our image? We use the documented CLI:

```
python solve.py --img_size 768 --img_path <img> --prompt "..." \\
                --task deblur_gauss --deg_scale 61 \\
                --method flowdps --efficient_memory
```

The first run will also pull SD3 Medium weights from Hugging Face (~6 GB).
That download is one-time; subsequent runs reuse the cache."""
    ),
    code(
        """import os, subprocess, time

OUT_DIR = '/content/drive/MyDrive/dvlm_proj/outputs/smoke_flowdps'
os.makedirs(OUT_DIR, exist_ok=True)
PROMPT = 'a photo of a person'

# Guard: HF_TOKEN must be in parent env before we launch the subprocess.
# If this assert fires, re-run cell 3 (the HF auth cell) — the kernel
# restart in 4a wipes os.environ, so HF_TOKEN has to be re-exported.
assert os.environ.get('HF_TOKEN', '').startswith('hf_'), (
    'HF_TOKEN missing from os.environ. Re-run cell 3 (HF auth) first.'
)

# Build the subprocess env explicitly. The `!python` shell magic
# sometimes drops env vars when invoking via bash on Colab; using
# subprocess.run with env=os.environ.copy() is bulletproof.
env = os.environ.copy()
# Belt + suspenders: huggingface_hub checks several env-var names.
env['HF_TOKEN'] = env['HF_TOKEN']
env['HUGGING_FACE_HUB_TOKEN'] = env['HF_TOKEN']
env['HUGGINGFACE_HUB_TOKEN'] = env['HF_TOKEN']

cmd = [
    'python', '/content/FlowDPS/solve.py',
    '--img_size', '768',
    '--img_path', SAMPLE_IMG,
    '--prompt', PROMPT,
    '--task', 'deblur_gauss',
    '--deg_scale', '61',
    '--method', 'flowdps',
    '--efficient_memory',
]

print(f'Launching FlowDPS subprocess with HF_TOKEN (len={len(env["HF_TOKEN"])}) ...')
print('Command:', ' '.join(cmd))

t0 = time.time()
result = subprocess.run(
    cmd, env=env, cwd='/content/FlowDPS',
    capture_output=True, text=True,
)
elapsed = time.time() - t0

# Trim TF/protobuf noise: show only stdout's last 50 lines + any stderr.
stdout_lines = result.stdout.splitlines()
stderr_lines = result.stderr.splitlines()
print(f'\\n--- stdout (last 50 of {len(stdout_lines)} lines) ---')
for ln in stdout_lines[-50:]:
    print(ln)
if result.returncode != 0:
    print(f'\\n--- stderr (last 80 of {len(stderr_lines)} lines) ---')
    for ln in stderr_lines[-80:]:
        print(ln)
    print(f'\\nFlowDPS FAILED with exit {result.returncode} after {elapsed:.1f} s')
else:
    print(f'\\nFlowDPS run took {elapsed:.1f} s')
"""
    ),
    code(
        """# Find what the repo wrote and copy to Drive.
import shutil, glob
results = sorted(glob.glob('/content/FlowDPS/result*/**/*.png', recursive=True)) + \\
          sorted(glob.glob('/content/FlowDPS/*.png'))
print('Wrote files:')
for r in results[-10:]: print(' ', r)

for r in results:
    shutil.copy(r, OUT_DIR)
print(f'Copied {len(results)} files into {OUT_DIR}')
"""
    ),
    md("## 8. Smoke run #2 — PSLD on the same image (our Latent-DPS baseline)"),
    code(
        """import os, subprocess, time

OUT_DIR_PSLD = '/content/drive/MyDrive/dvlm_proj/outputs/smoke_psld'
os.makedirs(OUT_DIR_PSLD, exist_ok=True)

env = os.environ.copy()
env['HUGGING_FACE_HUB_TOKEN'] = env.get('HF_TOKEN', '')
env['HUGGINGFACE_HUB_TOKEN'] = env.get('HF_TOKEN', '')

cmd = [
    'python', '/content/FlowDPS/solve.py',
    '--img_size', '768',
    '--img_path', SAMPLE_IMG,
    '--prompt', PROMPT,
    '--task', 'deblur_gauss',
    '--deg_scale', '61',
    '--method', 'psld',
    '--efficient_memory',
]

t0 = time.time()
result = subprocess.run(
    cmd, env=env, cwd='/content/FlowDPS',
    capture_output=True, text=True,
)
elapsed = time.time() - t0

stdout_lines = result.stdout.splitlines()
stderr_lines = result.stderr.splitlines()
print(f'--- stdout (last 50 of {len(stdout_lines)} lines) ---')
for ln in stdout_lines[-50:]:
    print(ln)
if result.returncode != 0:
    print(f'\\n--- stderr (last 80 of {len(stderr_lines)} lines) ---')
    for ln in stderr_lines[-80:]:
        print(ln)
    print(f'\\nPSLD FAILED with exit {result.returncode} after {elapsed:.1f} s')
else:
    print(f'\\nPSLD run took {elapsed:.1f} s')
"""
    ),
    md(
        """## 9. What did we measure?

If both runs completed cleanly, we now know:

- A100 + SD3 Medium fits in 40 GB VRAM with `--efficient_memory`.
- Wall-clock cost per image for FlowDPS and PSLD at this resolution / NFE
  setting (printed above).
- The output file structure the FlowDPS repo writes — we'll script
  around this for the full grid.

Next steps (back in the local repo):
1. Wrap `solve.py` invocation as a Python function that returns a tensor
   so we can plug it into our shared metric/CSV pipeline.
2. Download a real FFHQ-256 test set (start with ~50 images) and store
   under Drive's `data/ffhq/`.
3. Sweep zeta / NFE on a small validation subset.
"""
    ),
]


def build_notebook(cells: list[dict]) -> dict:
    return {
        "cells": cells,
        "metadata": {
            "kernelspec": {
                "display_name": "Python 3",
                "language": "python",
                "name": "python3",
            },
            "language_info": {"name": "python", "version": "3.10"},
            "colab": {"provenance": [], "toc_visible": True, "gpuType": "A100"},
            "accelerator": "GPU",
        },
        "nbformat": 4,
        "nbformat_minor": 5,
    }


def main() -> None:
    out_path = Path(__file__).resolve().parent.parent / "notebooks" / "flowdps_smoke.ipynb"
    out_path.parent.mkdir(parents=True, exist_ok=True)
    with out_path.open("w") as f:
        json.dump(build_notebook(CELLS), f, indent=1)
        f.write("\n")
    print(f"Wrote {out_path}")


if __name__ == "__main__":
    main()
