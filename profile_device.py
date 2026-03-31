"""
profile_device.py
=================
Benchmark LIBERO policy forward-pass latency on MPS vs CPU.
Requires a trained checkpoint.

Usage
-----
    python profile_device.py \
        --model_path experiments/LIBERO_SPATIAL/Sequential/BCTransformerPolicy_seed10000/run_001/task0_model.pth \
        --benchmark libero_spatial \
        --n_warmup 20 \
        --n_runs 200

Outputs a summary table and saves a JSON + matplotlib plot to results/.
"""

from __future__ import annotations

import argparse
import json
import os
import sys
import time

os.environ["TOKENIZERS_PARALLELISM"] = "false"

import patch_mps
from patch_mps import get_device

import numpy as np
import torch

from libero.libero import get_libero_path
from libero.libero.benchmark import get_benchmark
from libero.lifelong.algos import get_algo_class
from libero.lifelong.models import get_policy_class
from libero.lifelong.utils import torch_load_model, get_task_embs, control_seed, NpEncoder


BENCHMARK_NAME_MAP = {
    "libero_spatial": "LIBERO_SPATIAL",
    "libero_object":  "LIBERO_OBJECT",
    "libero_goal":    "LIBERO_GOAL",
    "libero_10":      "LIBERO_10",
}


def parse_args():
    p = argparse.ArgumentParser(description="MPS vs CPU forward-pass latency benchmark")
    p.add_argument("--model_path", required=True)
    p.add_argument("--benchmark",  required=True, choices=list(BENCHMARK_NAME_MAP.keys()))
    p.add_argument("--task_id",    type=int, default=0)
    p.add_argument("--algo",       default="sequential",
                   choices=["sequential", "multitask", "singletask"])
    p.add_argument("--n_warmup",   type=int, default=20,
                   help="Warmup forward passes (not timed)")
    p.add_argument("--n_runs",     type=int, default=200,
                   help="Timed forward passes per device")
    p.add_argument("--devices",    nargs="+", default=None,
                   help="Devices to benchmark. Default: auto-detect available ones")
    p.add_argument("--seed",       type=int, default=10000)
    p.add_argument("--results_dir", default="results")
    p.add_argument("--no_plot",    action="store_true",
                   help="Skip matplotlib plot (useful in headless environments)")
    return p.parse_args()


ALGO_CLASS_MAP = {"sequential": "Sequential", "multitask": "Multitask", "singletask": "SingleTask"}


def make_dummy_obs(cfg, task_emb: torch.Tensor, device: str) -> dict:
    """
    Build a single-step observation dict that matches the model's expected input.
    Uses random tensors — we're measuring latency, not correctness.
    """
    seq_len = cfg.data.seq_len
    h, w    = cfg.data.img_h, cfg.data.img_w

    # Shapes must match raw_obs_to_tensor_obs output (no time dim).
    # base_policy.preprocess_input adds the time dim via unsqueeze(1) at inference time.
    data = {
        "obs": {
            "agentview_rgb":   torch.zeros(1, 3, h, w),
            "eye_in_hand_rgb": torch.zeros(1, 3, h, w),
            "gripper_states":  torch.zeros(1, 2),
            "joint_states":    torch.zeros(1, 7),
        },
        "task_emb": task_emb.unsqueeze(0),
    }

    def _to(x):
        if device == "mps" and torch.backends.mps.is_available():
            return x.to("mps")
        if "cuda" in device and torch.cuda.is_available():
            return x.to(device)
        return x.cpu()

    data["obs"]      = {k: _to(v) for k, v in data["obs"].items()}
    data["task_emb"] = _to(data["task_emb"])
    return data


def benchmark_device(algo, dummy_obs: dict, device: str, n_warmup: int, n_runs: int) -> np.ndarray:
    """
    Run n_warmup + n_runs forward passes.  Return array of n_runs latencies (ms).
    """
    algo.eval()

    # Warmup
    with torch.no_grad():
        for _ in range(n_warmup):
            algo.policy.get_action(dummy_obs)
            if device == "mps":
                torch.mps.synchronize()
            elif "cuda" in device:
                torch.cuda.synchronize()

    # Timed runs
    latencies = np.empty(n_runs)
    with torch.no_grad():
        for i in range(n_runs):
            if device == "mps":
                torch.mps.synchronize()
            elif "cuda" in device:
                torch.cuda.synchronize()

            t0 = time.perf_counter()
            algo.policy.get_action(dummy_obs)

            if device == "mps":
                torch.mps.synchronize()
            elif "cuda" in device:
                torch.cuda.synchronize()

            latencies[i] = (time.perf_counter() - t0) * 1000  # ms

    return latencies


def print_stats(device: str, lat: np.ndarray):
    print(f"  {device:<10}  mean={lat.mean():6.2f}ms  "
          f"p50={np.percentile(lat,50):6.2f}ms  "
          f"p95={np.percentile(lat,95):6.2f}ms  "
          f"p99={np.percentile(lat,99):6.2f}ms  "
          f"std={lat.std():5.2f}ms")


def main():
    args = parse_args()
    control_seed(args.seed)
    os.makedirs(args.results_dir, exist_ok=True)

    # Auto-detect devices to benchmark
    if args.devices:
        devices = args.devices
    else:
        devices = ["cpu"]
        if torch.backends.mps.is_available():
            devices.insert(0, "mps")
        if torch.cuda.is_available():
            devices.insert(0, "cuda:0")

    print(f"[profile] devices to benchmark: {devices}")
    print(f"[profile] loading: {args.model_path}")

    sd, ckpt_cfg, _ = torch_load_model(args.model_path, map_location="cpu")
    if ckpt_cfg is None:
        print("[error] checkpoint has no embedded cfg.")
        sys.exit(1)

    cfg = ckpt_cfg
    cfg.folder             = get_libero_path("datasets")
    cfg.bddl_folder        = get_libero_path("bddl_files")
    cfg.init_states_folder = get_libero_path("init_states")
    cfg.eval.use_mp        = False
    os.makedirs(getattr(cfg, "experiment_dir", "experiments/tmp"), exist_ok=True)

    benchmark_name = BENCHMARK_NAME_MAP[args.benchmark]
    benchmark      = get_benchmark(benchmark_name)(cfg.data.task_order_index)
    descriptions   = [benchmark.get_task(i).language for i in range(benchmark.n_tasks)]

    algo_name  = ALGO_CLASS_MAP[args.algo]
    results    = {}
    all_lats   = {}

    print(f"\n{'Device':<10}  {'mean':>8}  {'p50':>8}  {'p95':>8}  {'p99':>8}  {'std':>7}")
    print("-" * 60)

    for device in devices:
        # Rebuild task embeddings on correct device
        cfg.device = device
        task_embs  = get_task_embs(cfg, descriptions)
        benchmark.set_task_embs(task_embs)
        task_emb = benchmark.get_task_emb(args.task_id)

        # Rebuild model on this device
        algo = patch_mps._safe_device_mps(
            get_algo_class(algo_name)(benchmark.n_tasks, cfg), device
        )
        algo.policy.load_state_dict(sd)

        dummy_obs = make_dummy_obs(cfg, task_emb, device)
        lat       = benchmark_device(algo, dummy_obs, device, args.n_warmup, args.n_runs)
        all_lats[device] = lat

        print_stats(device, lat)
        results[device] = {
            "mean_ms":  float(lat.mean()),
            "std_ms":   float(lat.std()),
            "p50_ms":   float(np.percentile(lat, 50)),
            "p95_ms":   float(np.percentile(lat, 95)),
            "p99_ms":   float(np.percentile(lat, 99)),
            "n_runs":   args.n_runs,
            "n_warmup": args.n_warmup,
        }

        # Free memory before next device
        del algo
        if device == "mps":
            torch.mps.empty_cache()
        elif "cuda" in device:
            torch.cuda.empty_cache()

    print("-" * 60)
    if len(devices) >= 2:
        ref  = results[devices[-1]]["mean_ms"]   # CPU is always last / reference
        fast = results[devices[0]]["mean_ms"]
        print(f"  Speedup {devices[0]} vs {devices[-1]}: {ref/fast:.2f}x")

    # ---- Save JSON ----
    out_json = os.path.join(args.results_dir, f"profile_{args.benchmark}.json")
    with open(out_json, "w") as f:
        json.dump({"benchmark": args.benchmark, "model": args.model_path,
                   "results": results}, f, indent=2, cls=NpEncoder)
    print(f"\n[profile] results saved to {out_json}")

    # ---- Plot ----
    if not args.no_plot:
        try:
            import matplotlib
            matplotlib.use("Agg")
            import matplotlib.pyplot as plt

            fig, axes = plt.subplots(1, 2, figsize=(12, 4))

            # Box plot
            ax = axes[0]
            ax.boxplot([all_lats[d] for d in devices], labels=devices, showfliers=False)
            ax.set_ylabel("Latency (ms)")
            ax.set_title("Forward-pass latency distribution (no outliers)")
            ax.grid(axis="y", alpha=0.4)

            # Mean bar chart
            ax = axes[1]
            means = [results[d]["mean_ms"] for d in devices]
            stds  = [results[d]["std_ms"]  for d in devices]
            bars  = ax.bar(devices, means, yerr=stds, capsize=5,
                           color=["#4CAF50", "#2196F3", "#FF9800"][:len(devices)])
            ax.set_ylabel("Mean latency (ms)")
            ax.set_title("Mean forward-pass latency ± std")
            ax.grid(axis="y", alpha=0.4)
            for bar, mean in zip(bars, means):
                ax.text(bar.get_x() + bar.get_width() / 2,
                        bar.get_height() + max(stds) * 0.05,
                        f"{mean:.1f}ms", ha="center", va="bottom", fontsize=9)

            fig.suptitle(
                f"LIBERO {args.benchmark.upper()} — policy inference latency\n"
                f"({args.n_runs} runs, {args.n_warmup} warmup)",
                fontsize=11
            )
            plt.tight_layout()

            out_plot = os.path.join(args.results_dir, f"profile_{args.benchmark}.png")
            plt.savefig(out_plot, dpi=150, bbox_inches="tight")
            plt.close()
            print(f"[profile] plot saved to {out_plot}")
        except ImportError:
            print("[profile] matplotlib not found — skipping plot (pip install matplotlib)")


if __name__ == "__main__":
    main()
