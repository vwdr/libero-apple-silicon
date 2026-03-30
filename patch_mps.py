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

3. SinusoidalPositionEncoding.forward
   Original uses .type(self.inv_freq.type()) to cast a CPU/CUDA tensor.
   On MPS, tensor.type() returns 'torch.mps.FloatTensor' which is not
   accepted by the .type() casting API, raising a ValueError.
   Fix: replace with .to(dtype=self.inv_freq.dtype).

4. DataLoader persistent_workers
   LIBERO hardcodes persistent_workers=True in three algo files
   (base.py, er.py, multitask.py).  PyTorch raises ValueError if
   persistent_workers=True while num_workers=0 (which we set for macOS
   fork safety).  We wrap DataLoader.__init__ to force
   persistent_workers=False whenever num_workers == 0.

5. compute_flops
   Uses thop.profile which runs a full forward pass under hook-based MAC
   counting.  thop does not support MPS and triggers the same
   SinusoidalPositionEncoding crash even after fix #3 (because thop
   temporarily moves the model to CPU mid-pass, causing a device mismatch).
   On non-CUDA devices we short-circuit this call and return (0.0, 0.0);
   GFLOPs/params are informational only and not needed for training.

5. DummyVectorEnv preferred over SubprocVectorEnv
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
# 3. Patched DataLoader — fix persistent_workers=True with num_workers=0
# ---------------------------------------------------------------------------

from torch.utils.data import DataLoader as _OrigDataLoader

class _DataLoaderMPS(_OrigDataLoader):
    """
    Thin subclass that silently corrects:
        persistent_workers=True + num_workers=0  →  persistent_workers=False

    PyTorch raises ValueError on this combo.  LIBERO hardcodes
    persistent_workers=True in base.py, er.py, and multitask.py;
    we set num_workers=0 for macOS safety.
    """
    def __init__(self, *args, **kwargs):
        if kwargs.get("num_workers", 0) == 0 and kwargs.get("persistent_workers", False):
            kwargs["persistent_workers"] = False
        super().__init__(*args, **kwargs)

# Patch the name in all algo modules so their `DataLoader(...)` calls hit our subclass
import torch.utils.data as _tud
_tud.DataLoader = _DataLoaderMPS          # affects new imports
import libero.lifelong.algos.base as _base_mod
_base_mod.DataLoader = _DataLoaderMPS
try:
    import libero.lifelong.algos.er as _er_mod
    _er_mod.DataLoader = _DataLoaderMPS
except Exception:
    pass
try:
    import libero.lifelong.algos.multitask as _mt_mod
    _mt_mod.DataLoader = _DataLoaderMPS
except Exception:
    pass


# ---------------------------------------------------------------------------
# 4. Patched SinusoidalPositionEncoding.forward
# ---------------------------------------------------------------------------

def _sinusoidal_forward_mps(self, x):
    """
    Drop-in replacement for SinusoidalPositionEncoding.forward.

    Original:  torch.arange(...).type(self.inv_freq.type())
    Problem:   On MPS, .type() returns 'torch.mps.FloatTensor' which the
               .type() casting API rejects with ValueError.
    Fix:       Use .to(dtype=self.inv_freq.dtype) — device-agnostic.
    """
    pos_x = torch.arange(x.shape[1], device=x.device).to(dtype=self.inv_freq.dtype)
    sin_inp_x = torch.einsum("i,j->ij", pos_x, self.inv_freq)
    emb_x = torch.cat((sin_inp_x.sin(), sin_inp_x.cos()), dim=-1)
    return emb_x * self.factor


# ---------------------------------------------------------------------------
# 4. Patched compute_flops — skip thop on non-CUDA
# ---------------------------------------------------------------------------

def _compute_flops_mps(algo, dataset, cfg):
    """
    On CUDA, delegate to the real thop-based compute_flops.
    On MPS/CPU, return (0.0, 0.0) — thop does not support MPS and triggers
    the SinusoidalPositionEncoding crash even with fix #3.
    """
    if "cuda" in str(cfg.device) and torch.cuda.is_available():
        import copy
        from torch.utils.data import DataLoader
        import robomimic.utils.tensor_utils as TensorUtils
        from thop import profile as thop_profile
        model = copy.deepcopy(algo.policy)
        tmp_loader = DataLoader(dataset, batch_size=1, num_workers=0, shuffle=True)
        data = next(iter(tmp_loader))
        data = TensorUtils.map_tensor(data, lambda x: _safe_device_mps(x, cfg.device))
        macs, params = thop_profile(model, inputs=(data,), verbose=False)
        del model
        return macs * 2 / 1e9, params / 1e6
    # Non-CUDA: skip thop, return zeros (informational only)
    return 0.0, 0.0


# ---------------------------------------------------------------------------
# Apply patches
# ---------------------------------------------------------------------------

_lu.safe_device = _safe_device_mps
_lm.safe_device = _safe_device_mps          # metric.py imports it via *
_lu.torch_load_model = _torch_load_model_mps
_lu.compute_flops = _compute_flops_mps

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
        if hasattr(_mod, "compute_flops"):
            _mod.compute_flops = _compute_flops_mps
    except Exception:
        pass

# Patch compute_flops in main.py's module namespace
import libero.lifelong.main as _lmain
_lmain.compute_flops = _compute_flops_mps

# Patch SinusoidalPositionEncoding.forward in the transformer module
from libero.lifelong.models.modules.transformer_modules import SinusoidalPositionEncoding
import types
SinusoidalPositionEncoding.forward = _sinusoidal_forward_mps


def get_device() -> str:
    """Return the best available device string for this machine."""
    if torch.cuda.is_available():
        return "cuda:0"
    if torch.backends.mps.is_available():
        return "mps"
    return "cpu"
