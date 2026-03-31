"""
evaluate.py
===========
Evaluate a trained LIBERO policy across all tasks in one or more task suites.
Designed for Apple Silicon (MPS) but works on CPU and CUDA too.

Usage
-----
    # Evaluate a sequential policy on all 10 tasks in LIBERO-Spatial
    python evaluate.py \
        --model_path experiments/LIBERO_SPATIAL/Sequential/BCTransformerPolicy_seed10000/run_001/task9_model.pth \
        --benchmark libero_spatial \
        --n_eval 20 \
        --device mps

    # Evaluate a multitask checkpoint (saved at epoch 50)
    python evaluate.py \
        --model_path experiments/LIBERO_SPATIAL/Multitask/BCTransformerPolicy_seed10000/run_001/multitask_model_ep50.pth \
        --benchmark libero_spatial \
        --algo multitask \
        --n_eval 20

Results are written to results/<benchmark>_eval_<timestamp>.json
"""

from __future__ import annotations

import argparse
import json
import os
import sys
import time
from datetime import datetime

os.environ["TOKENIZERS_PARALLELISM"] = "false"

# ---- MPS patch must come before any libero.lifelong import ----
import patch_mps
from patch_mps import get_device

import numpy as np
import torch

from libero.libero import get_libero_path
from libero.libero.benchmark import get_benchmark
from libero.libero.envs import OffScreenRenderEnv, DummyVectorEnv
from libero.lifelong.algos import get_algo_class
from libero.lifelong.metric import raw_obs_to_tensor_obs
from libero.lifelong.utils import torch_load_model, get_task_embs, control_seed, NpEncoder


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
    p = argparse.ArgumentParser(
        description="Evaluate a LIBERO policy across all tasks in a suite"
    )
    p.add_argument("--model_path",  required=True,
                   help="Path to .pth checkpoint")
    p.add_argument("--benchmark",   required=True,
                   choices=list(BENCHMARK_NAME_MAP.keys()))
    p.add_argument("--algo",        default="sequential",
                   choices=list(ALGO_CLASS_MAP.keys()))
    p.add_argument("--task_ids",    type=int, nargs="+", default=None,
                   help="Subset of task ids to evaluate (default: all)")
    p.add_argument("--n_eval",      type=int, default=20,
                   help="Rollout episodes per task")
    p.add_argument("--max_steps",   type=int, default=600,
                   help="Max steps per episode")
    p.add_argument("--device",      default=None,
                   help="mps | cpu | cuda:0  (auto if omitted)")
    p.add_argument("--seed",        type=int, default=10000)
    p.add_argument("--results_dir", default="results")
    return p.parse_args()


def evaluate_task(cfg, algo, task, task_emb, task_id: int, n_eval: int) -> dict:
    """Run n_eval episodes on a single task and return stats."""
    env_args = {
        "bddl_file_name": os.path.join(cfg.bddl_folder, task.problem_folder, task.bddl_file),
        "camera_heights": cfg.data.img_h,
        "camera_widths":  cfg.data.img_w,
    }
    init_states = torch.load(
        os.path.join(cfg.init_states_folder, task.problem_folder, task.init_states_file)
    )

    env = DummyVectorEnv([lambda: OffScreenRenderEnv(**env_args)])

    num_success = 0
    episode_lengths = []
    step_times = []

    for ep_idx in range(n_eval):
        env.reset()
        env.seed(cfg.seed + ep_idx)
        algo.reset()

        init_idx = ep_idx % init_states.shape[0]
        obs = env.set_init_state(init_states[init_idx : init_idx + 1])

        for _ in range(5):
            obs, _, _, _ = env.step(np.zeros((1, 7)))

        done  = False
        steps = 0

        while steps < cfg.eval.max_steps and not done:
            t0 = time.perf_counter()
            with torch.no_grad():
                data    = raw_obs_to_tensor_obs(obs, task_emb.unsqueeze(0), cfg)
                actions = algo.policy.get_action(data)
            step_times.append(time.perf_counter() - t0)
            obs, _, done_arr, _ = env.step(actions)
            done   = bool(done_arr[0])
            steps += 1

        num_success    += int(done)
        episode_lengths.append(steps)

    env.close()

    return {
        "task_id":            task_id,
        "task":               task.language,
        "success_rate":       num_success / n_eval,
        "num_success":        num_success,
        "n_eval":             n_eval,
        "mean_episode_steps": float(np.mean(episode_lengths)),
        "mean_step_ms":       float(np.mean(step_times) * 1000) if step_times else 0.0,
        "std_step_ms":        float(np.std(step_times)  * 1000) if step_times else 0.0,
    }


def main():
    args   = parse_args()
    device = args.device or get_device()
    print(f"[evaluate] device = {device}")
    control_seed(args.seed)
    os.makedirs(args.results_dir, exist_ok=True)

    # ---- Load checkpoint ----
    print(f"[evaluate] loading: {args.model_path}")
    sd, ckpt_cfg, previous_masks = torch_load_model(args.model_path, map_location="cpu")
    if ckpt_cfg is None:
        print("[error] checkpoint has no embedded cfg.")
        sys.exit(1)

    cfg = ckpt_cfg
    cfg.folder             = get_libero_path("datasets")
    cfg.bddl_folder        = get_libero_path("bddl_files")
    cfg.init_states_folder = get_libero_path("init_states")
    cfg.device             = device
    cfg.eval.use_mp        = False
    cfg.eval.num_procs     = 1
    cfg.eval.n_eval        = args.n_eval
    cfg.eval.max_steps     = args.max_steps
    os.makedirs(getattr(cfg, "experiment_dir", "experiments/tmp"), exist_ok=True)

    # ---- Initialize ObsUtils (normally done inside get_dataset) ----
    # robomimic's ObsUtils.process_obs needs OBS_KEYS_TO_MODALITIES populated.
    # In the training pipeline this happens via get_dataset(initialize_obs_utils=True).
    # For standalone evaluation we initialize it directly from the config.
    import robomimic.utils.obs_utils as ObsUtils
    ObsUtils.initialize_obs_utils_with_obs_specs({"obs": cfg.data.obs.modality})

    benchmark_name = BENCHMARK_NAME_MAP[args.benchmark]
    benchmark      = get_benchmark(benchmark_name)(cfg.data.task_order_index)
    n_tasks        = benchmark.n_tasks
    task_ids       = args.task_ids if args.task_ids is not None else list(range(n_tasks))

    descriptions = [benchmark.get_task(i).language for i in range(n_tasks)]
    task_embs    = get_task_embs(cfg, descriptions)
    benchmark.set_task_embs(task_embs)

    # ---- Build model ----
    algo_class_name = ALGO_CLASS_MAP[args.algo]
    algo = patch_mps._safe_device_mps(
        get_algo_class(algo_class_name)(n_tasks, cfg), device
    )
    algo.policy.load_state_dict(sd)
    algo.eval()

    # ---- Evaluate ----
    print(f"\n[evaluate] {benchmark_name} — {len(task_ids)} task(s), {args.n_eval} ep each\n")
    print(f"{'Task':>4}  {'Description':<60}  {'Success':>7}")
    print("-" * 76)

    per_task = []
    wall_t0  = time.perf_counter()

    for task_id in task_ids:
        task     = benchmark.get_task(task_id)
        task_emb = benchmark.get_task_emb(task_id)
        stats    = evaluate_task(cfg, algo, task, task_emb, task_id, args.n_eval)
        per_task.append(stats)
        print(f"{task_id:>4}  {task.language[:60]:<60}  {stats['success_rate']:>6.1%}")

    total_elapsed = time.perf_counter() - wall_t0
    mean_success  = float(np.mean([s["success_rate"] for s in per_task]))

    print("-" * 76)
    print(f"{'':>4}  {'MEAN':60}  {mean_success:>6.1%}")
    print(f"\n[evaluate] total wall time: {total_elapsed:.1f}s")

    # ---- Save results ----
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    out_path  = os.path.join(args.results_dir,
                             f"{args.benchmark}_eval_{timestamp}.json")
    results = {
        "benchmark":     args.benchmark,
        "model_path":    args.model_path,
        "algo":          args.algo,
        "device":        device,
        "n_eval":        args.n_eval,
        "seed":          args.seed,
        "mean_success":  mean_success,
        "elapsed_s":     total_elapsed,
        "per_task":      per_task,
    }
    with open(out_path, "w") as f:
        json.dump(results, f, indent=2, cls=NpEncoder)

    print(f"[evaluate] results saved to {out_path}")


if __name__ == "__main__":
    main()
