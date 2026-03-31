"""
run_inference.py
================
Run rollout inference with a trained LIBERO policy on a single task.
Designed for Apple Silicon (MPS) — works on CPU and CUDA too.

Usage
-----
    python run_inference.py \
        --model_path experiments/LIBERO_SPATIAL/Sequential/BCTransformerPolicy_seed10000/run_001/task0_model.pth \
        --benchmark libero_spatial \
        --task_id 0 \
        --n_eval 10 \
        --device mps \
        --save_video

The script patches LIBERO internals for MPS before importing any lifelong
modules, so it must not be split or reordered.
"""

from __future__ import annotations

import argparse
import os
import sys
import time

os.environ["TOKENIZERS_PARALLELISM"] = "false"

# ---- MPS patch must come before any libero.lifelong import ----
import patch_mps
from patch_mps import get_device

import numpy as np
import torch
import yaml
from easydict import EasyDict
from omegaconf import OmegaConf

# LIBERO imports (after patch)
from libero.libero import get_libero_path
from libero.libero.benchmark import get_benchmark
from libero.libero.envs import OffScreenRenderEnv, DummyVectorEnv
from libero.libero.utils.video_utils import VideoWriter
from libero.lifelong.algos import get_algo_class
from libero.lifelong.metric import raw_obs_to_tensor_obs
from libero.lifelong.utils import torch_load_model, get_task_embs, control_seed


BENCHMARK_NAME_MAP = {
    "libero_spatial": "LIBERO_SPATIAL",
    "libero_object":  "LIBERO_OBJECT",
    "libero_goal":    "LIBERO_GOAL",
    "libero_10":      "LIBERO_10",
}

ALGO_CLASS_MAP = {
    "sequential": "Sequential",
    "multitask":  "Multitask",
    "singletask": "SingleTask",
}


def parse_args():
    p = argparse.ArgumentParser(description="LIBERO single-task inference on Apple Silicon")
    p.add_argument("--model_path",  required=True,
                   help="Path to .pth checkpoint (output of LIBERO training)")
    p.add_argument("--benchmark",   required=True,
                   choices=list(BENCHMARK_NAME_MAP.keys()),
                   help="Task suite the model was trained on")
    p.add_argument("--task_id",     type=int, required=True,
                   help="Task index within the benchmark (0-indexed)")
    p.add_argument("--algo",        default="sequential",
                   choices=list(ALGO_CLASS_MAP.keys()))
    p.add_argument("--n_eval",      type=int, default=10,
                   help="Number of rollout episodes")
    p.add_argument("--device",      default=None,
                   help="Device: mps | cpu | cuda:0  (auto-detected if omitted)")
    p.add_argument("--seed",        type=int, default=10000)
    p.add_argument("--save_video",  action="store_true",
                   help="Save a side-by-side video of all rollouts to results/")
    p.add_argument("--results_dir", default="results",
                   help="Directory to store success stats and videos")
    return p.parse_args()


def build_cfg_from_checkpoint(ckpt_cfg: EasyDict, device: str, args) -> EasyDict:
    """
    Fill any missing config fields that the evaluation path requires.
    Checkpoints saved by LIBERO training carry a cfg dict; we augment it.
    """
    cfg = ckpt_cfg

    # Paths (use LIBERO defaults if not set)
    cfg.folder              = get_libero_path("datasets")
    cfg.bddl_folder         = get_libero_path("bddl_files")
    cfg.init_states_folder  = get_libero_path("init_states")
    cfg.device              = device

    # Eval settings (single-process to stay MPS-compatible)
    cfg.eval.use_mp     = False
    cfg.eval.num_procs  = 1
    cfg.eval.n_eval     = args.n_eval
    cfg.eval.max_steps  = getattr(cfg.eval, "max_steps", 600)

    # Make sure the experiment dir exists (required by algo __init__)
    os.makedirs(getattr(cfg, "experiment_dir", "experiments/tmp"), exist_ok=True)
    if not hasattr(cfg, "experiment_dir"):
        cfg.experiment_dir = "experiments/tmp"

    return cfg


def run_rollouts(cfg, algo, task, task_emb, n_eval: int, save_video: bool,
                 video_path: str) -> dict:
    """
    Run n_eval episodes on a single task.  Returns a stats dict.
    """
    env_args = {
        "bddl_file_name":  os.path.join(cfg.bddl_folder, task.problem_folder, task.bddl_file),
        "camera_heights":  cfg.data.img_h,
        "camera_widths":   cfg.data.img_w,
    }

    init_states_path = os.path.join(
        cfg.init_states_folder, task.problem_folder, task.init_states_file
    )
    init_states = torch.load(init_states_path)

    num_success = 0
    episode_lengths = []
    step_times = []

    # One environment at a time (MPS-safe)
    env = DummyVectorEnv([lambda: OffScreenRenderEnv(**env_args)])

    with VideoWriter(video_path, save_video) as video_writer:
        for ep_idx in range(n_eval):
            env.reset()
            env.seed(cfg.seed + ep_idx)
            algo.reset()

            init_idx = ep_idx % init_states.shape[0]
            obs = env.set_init_state(init_states[init_idx : init_idx + 1])

            # Warm up physics
            for _ in range(5):
                obs, _, _, _ = env.step(np.zeros((1, 7)))

            done = False
            steps = 0

            while steps < cfg.eval.max_steps and not done:
                t0 = time.perf_counter()
                with torch.no_grad():
                    data    = raw_obs_to_tensor_obs(obs, task_emb.unsqueeze(0), cfg)
                    actions = algo.policy.get_action(data)
                step_times.append(time.perf_counter() - t0)

                obs, _, done_arr, _ = env.step(actions)
                video_writer.append_vector_obs(obs, [done], camera_name="agentview_image")
                done = bool(done_arr[0])
                steps += 1

            episode_lengths.append(steps)
            num_success += int(done)
            print(f"  Episode {ep_idx+1:3d}/{n_eval}: "
                  f"{'SUCCESS' if done else 'FAIL   '} "
                  f"({steps} steps)")

    env.close()

    return {
        "success_rate":       num_success / n_eval,
        "num_success":        num_success,
        "n_eval":             n_eval,
        "mean_episode_steps": float(np.mean(episode_lengths)),
        "mean_step_ms":       float(np.mean(step_times) * 1000),
        "std_step_ms":        float(np.std(step_times) * 1000),
    }


def main():
    args = parse_args()
    device = args.device or get_device()
    print(f"[run_inference] device = {device}")
    control_seed(args.seed)
    os.makedirs(args.results_dir, exist_ok=True)

    # ---- Load checkpoint ----
    print(f"[run_inference] loading checkpoint: {args.model_path}")
    sd, ckpt_cfg, previous_masks = torch_load_model(args.model_path, map_location="cpu")
    if ckpt_cfg is None:
        print("[error] checkpoint has no embedded cfg — cannot reconstruct model.")
        sys.exit(1)

    cfg = build_cfg_from_checkpoint(ckpt_cfg, device, args)

    # Initialize ObsUtils before any rollout (normally done inside get_dataset)
    import robomimic.utils.obs_utils as ObsUtils
    ObsUtils.initialize_obs_utils_with_obs_specs({"obs": cfg.data.obs.modality})

    benchmark_name = BENCHMARK_NAME_MAP[args.benchmark]

    # ---- Build algorithm / policy ----
    algo_class_name = ALGO_CLASS_MAP[args.algo]
    benchmark = get_benchmark(benchmark_name)(cfg.data.task_order_index)
    descriptions = [benchmark.get_task(i).language for i in range(benchmark.n_tasks)]
    task_embs = get_task_embs(cfg, descriptions)
    benchmark.set_task_embs(task_embs)

    algo = patch_mps._safe_device_mps(
        get_algo_class(algo_class_name)(benchmark.n_tasks, cfg),
        device
    )
    algo.policy.load_state_dict(sd)
    algo.eval()

    task     = benchmark.get_task(args.task_id)
    task_emb = benchmark.get_task_emb(args.task_id)

    print(f"\n[run_inference] task {args.task_id}: \"{task.language}\"")
    print(f"[run_inference] running {args.n_eval} episodes...\n")

    # ---- Video path ----
    safe_name  = task.name[:60]
    video_path = os.path.join(
        args.results_dir,
        f"{args.benchmark}_task{args.task_id}_{safe_name}_videos"
    )
    stats_path = os.path.join(
        args.results_dir,
        f"{args.benchmark}_task{args.task_id}_{safe_name}_stats.pt"
    )

    # ---- Run ----
    t_start = time.perf_counter()
    stats   = run_rollouts(cfg, algo, task, task_emb,
                           n_eval=args.n_eval,
                           save_video=args.save_video,
                           video_path=video_path)
    elapsed = time.perf_counter() - t_start

    # ---- Report ----
    print(f"\n{'='*55}")
    print(f"  Task:          {task.language}")
    print(f"  Device:        {device}")
    print(f"  Success rate:  {stats['success_rate']:.1%}  ({stats['num_success']}/{args.n_eval})")
    print(f"  Avg episode:   {stats['mean_episode_steps']:.1f} steps")
    print(f"  Inference lat: {stats['mean_step_ms']:.1f} ± {stats['std_step_ms']:.1f} ms/step")
    print(f"  Wall time:     {elapsed:.1f}s")
    print(f"{'='*55}\n")

    stats["task"]    = task.language
    stats["device"]  = device
    stats["elapsed"] = elapsed
    torch.save(stats, stats_path)
    print(f"[run_inference] stats saved to {stats_path}")
    if args.save_video:
        print(f"[run_inference] video saved to {video_path}/")


if __name__ == "__main__":
    main()
