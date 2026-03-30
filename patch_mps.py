"""
patch_mps.py
============
Monkey-patches LIBERO internals so that all device routing works correctly
on Apple Silicon (MPS) or CPU — without touching the upstream source.

Import this module BEFORE any libero.lifelong imports.

    import patch_mps          # must be first
    from libero.lifelong import ...

What it fixes
-------------
1. safe_device(x, device)
   Original only handles "cpu" and "cuda:*".  Passing "mps" silently
   returns None, crashing everything downstream.  We replace it to also
   handle "mps".

2. torch_load_model(path, map_location)
   Original passes map_location straight to torch.load.  On MPS you need
   map_location="cpu" first (weights are moved to MPS later), otherwise
   PyTorch raises errors loading CUDA-saved tensors onto MPS directly.

3. DummyVectorEnv preferred over SubprocVectorEnv
   MPS lives in the main process.  SubprocVectorEnv forks child processes
   that cannot share the MPS context.  Callers should pass use_mp=False
   (env_num=1) — this patch does NOT force that, but the evaluation
   scripts in this repo do.
"""

from __future__ import annotations

import torch
import libero.lifelong.utils as _lu
import libero.lifelong.metric as _lm


# ---------------------------------------------------------------------------
# 1. Patched safe_device
# ---------------------------------------------------------------------------

def _safe_device_mps(x, device: str = "cpu"):
    """
    Move tensor/module x to the requested device.

    Priority:
      "mps"    -> MPS if available, else CPU
      "cuda:*" -> CUDA if available, else CPU
      "cpu"    -> CPU
      anything else -> CPU (safe fallback)
    """
    if device == "cpu":
        return x.cpu()
    if device == "mps":
        if torch.backends.mps.is_available():
            return x.to("mps")
        return x.cpu()
    if "cuda" in device:
        if torch.cuda.is_available():
            return x.to(device)
        return x.cpu()
    # unknown device string — fall back to CPU
    return x.cpu()


# ---------------------------------------------------------------------------
# 2. Patched torch_load_model
# ---------------------------------------------------------------------------

def _torch_load_model_mps(model_path: str, map_location=None):
    """
    Load a LIBERO checkpoint safely on MPS/CPU.

    torch.load with map_location="mps" can fail for checkpoints that were
    saved on CUDA.  We always load to CPU first; the caller's safe_device
    will move tensors to the right device when building the model.
    """
    if map_location is None or map_location == "cpu":
        effective_loc = "cpu"
    elif map_location == "mps":
        effective_loc = "cpu"          # load to CPU, move later
    elif isinstance(map_location, str) and "cuda" in map_location:
        effective_loc = "cpu" if not torch.cuda.is_available() else map_location
    else:
        effective_loc = "cpu"

    model_dict = torch.load(model_path, map_location=effective_loc)
    cfg = model_dict.get("cfg", None)
    previous_masks = model_dict.get("previous_masks", None)
    return model_dict["state_dict"], cfg, previous_masks


# ---------------------------------------------------------------------------
# Apply patches
# ---------------------------------------------------------------------------

_lu.safe_device = _safe_device_mps
_lm.safe_device = _safe_device_mps          # metric.py imports it via *
_lu.torch_load_model = _torch_load_model_mps

# Also patch the reference that lifelong algos see via `from utils import *`
import libero.lifelong.algos as _algos_pkg
import importlib, pkgutil

for _importer, _modname, _ispkg in pkgutil.walk_packages(
    path=_algos_pkg.__path__,
    prefix=_algos_pkg.__name__ + ".",
):
    try:
        _mod = importlib.import_module(_modname)
        if hasattr(_mod, "safe_device"):
            _mod.safe_device = _safe_device_mps
    except Exception:
        pass


def get_device() -> str:
    """Return the best available device string for this machine."""
    if torch.cuda.is_available():
        return "cuda:0"
    if torch.backends.mps.is_available():
        return "mps"
    return "cpu"
