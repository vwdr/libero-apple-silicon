# LIBERO Inference on Apple Silicon

A reproducible evaluation starter for [LIBERO](https://github.com/Lifelong-Robot-Learning/LIBERO) pretrained policies on Apple Silicon (MPS backend).

Most LIBERO guides assume NVIDIA GPUs. This repo fills that gap: it patches LIBERO's device routing, runs the full rollout loop on MPS/CPU, and measures inference performance across task suites.

---

## What's in here

| File | Purpose |
|---|---|
| `patch_mps.py` | Monkey-patches `safe_device` + `torch_load_model` for MPS/CPU |
| `train.py` | Training wrapper — sets MPS-safe hydra overrides before launching |
| `run_inference.py` | Single-task rollout with optional video saving |
| `evaluate.py` | Evaluates a checkpoint across all tasks in a suite |
| `profile_device.py` | Benchmarks forward-pass latency on MPS vs CPU |
| `results/` | Output directory for stats, JSON, and videos |

---

## Setup

### 1. Clone LIBERO and create the conda env

```bash
git clone https://github.com/Lifelong-Robot-Learning/LIBERO.git
cd LIBERO
conda create -n libero python=3.8 -y
conda activate libero
pip install torch torchvision           # Apple Silicon: CPU+MPS, no CUDA needed
pip install -e .
pip install matplotlib                  # for profile_device.py plots
```

### 2. Download datasets

```bash
cd LIBERO
python benchmark_scripts/download_libero_datasets.py --datasets libero_spatial libero_object libero_goal
```

### 3. Clone this repo next to LIBERO

```
your-workspace/
├── LIBERO/          # upstream source
└── AppleSiliconLIBERO/   # this repo
```

```bash
cd your-workspace/AppleSiliconLIBERO
```

---

## Workflow

### Step 1 — Train a policy

```bash
# BC-Transformer, sequential algo, LIBERO-Spatial, MPS
python train.py \
    --benchmark libero_spatial \
    --policy bc_transformer_policy \
    --algo sequential \
    --seed 10000 \
    --n_epochs 50 \
    --device mps

# Faster option: disable rollout eval during training
python train.py \
    --benchmark libero_spatial \
    --no_eval_during_training \
    --device mps
```

Checkpoints land at:
```
experiments/LIBERO_SPATIAL/Sequential/BCTransformerPolicy_seed10000/run_001/
    task0_model.pth
    task1_model.pth
    ...
    task9_model.pth
```

### Step 2 — Single-task inference + video

```bash
python run_inference.py \
    --model_path experiments/LIBERO_SPATIAL/Sequential/BCTransformerPolicy_seed10000/run_001/task0_model.pth \
    --benchmark libero_spatial \
    --task_id 0 \
    --n_eval 20 \
    --device mps \
    --save_video
```

### Step 3 — Full suite evaluation

```bash
# Evaluate task9 checkpoint (trained on all 10 tasks sequentially) across all tasks
python evaluate.py \
    --model_path experiments/LIBERO_SPATIAL/Sequential/BCTransformerPolicy_seed10000/run_001/task9_model.pth \
    --benchmark libero_spatial \
    --n_eval 20 \
    --device mps
```

Output: `results/libero_spatial_eval_<timestamp>.json`

### Step 4 — MPS vs CPU profiling

```bash
python profile_device.py \
    --model_path experiments/LIBERO_SPATIAL/Sequential/BCTransformerPolicy_seed10000/run_001/task0_model.pth \
    --benchmark libero_spatial \
    --n_warmup 20 \
    --n_runs 200
```

Output: `results/profile_libero_spatial.json` + `results/profile_libero_spatial.png`

---

## Why does this need a patch?

LIBERO's `safe_device` utility (in `libero/lifelong/utils.py`) handles only `"cpu"` and `"cuda:*"`. Passing `"mps"` silently falls through and returns `None`, crashing any downstream `.to(device)` call.

`patch_mps.py` fixes this without touching the upstream source:

```python
# Before (LIBERO upstream)
def safe_device(x, device="cpu"):
    if device == "cpu":
        return x.cpu()
    elif "cuda" in device:
        if torch.cuda.is_available():
            return x.to(device)
        else:
            return x.cpu()
    # "mps" falls through here — returns None!

# After (our patch)
def safe_device(x, device="cpu"):
    if device == "cpu":  return x.cpu()
    if device == "mps":
        return x.to("mps") if torch.backends.mps.is_available() else x.cpu()
    if "cuda" in device:
        return x.to(device) if torch.cuda.is_available() else x.cpu()
    return x.cpu()
```

Additionally, `SubprocVectorEnv` forks child processes — MPS contexts cannot be inherited across `fork`. Our scripts use `DummyVectorEnv` (single process) for MPS/CPU and set `num_workers=0` for data loaders.

---

## Results

*(Fill in after running evaluations)*

### Success rates — LIBERO-Spatial (BC-Transformer, Sequential)

| Task | Description | Success rate |
|---|---|---|
| 0 | pick up the black bowl between the plate and the ramekin and place it on the plate | — |
| … | … | … |
| **Mean** | | — |

### Forward-pass latency

| Device | Mean (ms) | p95 (ms) |
|---|---|---|
| MPS (M4) | — | — |
| CPU (M4) | — | — |

---

## Notes on MPS limitations

- **MPS does not support all PyTorch ops.** Some ops fall back to CPU automatically (PyTorch will print a warning). This is normal and usually doesn't affect correctness.
- **No multi-process env.** `SubprocVectorEnv` (`use_mp=True`) is disabled. Evaluation runs one episode at a time, which is slower but correct.
- **BERT tokenization** runs on CPU regardless of device — this is expected and takes < 1s at startup.

---

## Citation

```bibtex
@inproceedings{liu2023libero,
  title={LIBERO: Benchmarking Knowledge Transfer for Lifelong Robot Learning},
  author={Liu, Bo and Zhu, Yifeng and Gao, Chongkai and Feng, Yihao and Liu, Qiang and Zhu, Yuke and Stone, Peter},
  booktitle={Advances in Neural Information Processing Systems},
  year={2023}
}
```
