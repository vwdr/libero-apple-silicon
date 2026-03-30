"""
train.py
========
Train a LIBERO policy with Apple Silicon (MPS) support.

This is a thin wrapper around LIBERO's hydra-based training pipeline that:
  1. Patches safe_device / torch_load_model for MPS before hydra launches
  2. Overrides the device config to mps/cpu when CUDA is absent
  3. Disables multiprocessing evaluation (SubprocVectorEnv breaks on MPS)
  4. Adjusts num_workers=0 (macOS fork safety)

Usage
-----
    # Train BC-Transformer on LIBERO-Spatial, sequential algo
    python train.py \
        --benchmark libero_spatial \
        --policy bc_transformer_policy \
        --algo sequential \
        --seed 10000 \
        --n_epochs 50 \
        --device mps

    # Multitask BC-RNN on LIBERO-Object, CPU only
    python train.py \
        --benchmark libero_object \
        --policy bc_rnn_policy \
        --algo multitask \
        --device cpu

All standard hydra overrides still work via --hydra-overrides:
    python train.py --benchmark libero_spatial --hydra-overrides train.batch_size=64

Checkpoints are saved under:
    experiments/<BENCHMARK>/<ALGO>/<POLICY>_seed<SEED>/run_XXX/
"""

from __future__ import annotations

import argparse
import multiprocessing
import os
import sys

os.environ["TOKENIZERS_PARALLELISM"] = "false"

# ---- MPS patch must come first ----
import patch_mps
from patch_mps import get_device


BENCHMARK_NAME_MAP = {
    "libero_spatial": "LIBERO_SPATIAL",
    "libero_object":  "LIBERO_OBJECT",
    "libero_goal":    "LIBERO_GOAL",
    "libero_10":      "LIBERO_10",
    "libero_90":      "LIBERO_90",
}

POLICY_LIST = ["bc_transformer_policy", "bc_rnn_policy", "bc_vilt_policy"]
ALGO_LIST   = ["sequential", "multitask", "singletask", "er", "ewc", "packnet"]


def parse_args():
    p = argparse.ArgumentParser(
        description="Train LIBERO policy on Apple Silicon",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog=__doc__,
    )
    p.add_argument("--benchmark", required=True, choices=list(BENCHMARK_NAME_MAP.keys()))
    p.add_argument("--policy",    default="bc_transformer_policy", choices=POLICY_LIST)
    p.add_argument("--algo",      default="sequential",            choices=ALGO_LIST)
    p.add_argument("--seed",      type=int, default=10000)
    p.add_argument("--n_epochs",  type=int, default=50,
                   help="Training epochs per task")
    p.add_argument("--batch_size", type=int, default=32)
    p.add_argument("--device",    default=None,
                   help="mps | cpu | cuda:0  (auto if omitted)")
    p.add_argument("--task_order_index", type=int, default=0,
                   help="Task permutation index (0 = default order)")
    p.add_argument("--embedding",  default="bert",
                   choices=["bert", "gpt2", "clip", "roberta", "one-hot"],
                   help="Task language embedding encoder")
    p.add_argument("--pretrain_path", default="",
                   help="Optional path to a pretrained checkpoint to finetune from")
    p.add_argument("--no_eval_during_training", action="store_true",
                   help="Disable rollout evaluation during training (faster on slow hardware)")
    p.add_argument("--hydra-overrides", nargs="*", default=[],
                   dest="hydra_overrides",
                   help="Extra hydra config overrides, e.g. train.lr=1e-4")
    return p.parse_args()


def build_hydra_overrides(args, device: str) -> list[str]:
    """Convert CLI args to hydra override strings."""
    benchmark_name = BENCHMARK_NAME_MAP[args.benchmark]

    # lifelong.algo needs the class name (e.g. "Sequential"), not the CLI key
    algo_class_name = {
        "sequential": "Sequential",
        "multitask":  "Multitask",
        "singletask": "SingleTask",
        "er":         "ER",
        "ewc":        "EWC",
        "packnet":    "PackNet",
    }[args.algo]

    overrides = [
        f"benchmark_name={benchmark_name}",
        f"policy={args.policy}",
        f"lifelong.algo={algo_class_name}",    # dot-notation value override
        f"seed={args.seed}",
        f"device={device}",
        f"data.task_order_index={args.task_order_index}",
        f"task_embedding_format={args.embedding}",
        f"train.n_epochs={args.n_epochs}",
        f"train.batch_size={args.batch_size}",
        # MPS/CPU safety: disable subprocess eval & set workers=0
        "eval.use_mp=false",
        "eval.num_procs=1",
        "train.num_workers=0",
        "eval.num_workers=0",
    ]

    if args.pretrain_path:
        overrides.append(f"pretrain_model_path={args.pretrain_path}")
        overrides.append("load_previous_model=true")

    if args.no_eval_during_training:
        overrides.append("eval.eval=false")

    overrides += args.hydra_overrides
    return overrides


def main():
    if multiprocessing.get_start_method(allow_none=True) != "spawn":
        multiprocessing.set_start_method("spawn", force=True)

    args   = parse_args()
    device = args.device or get_device()
    print(f"[train] device = {device}")

    overrides = build_hydra_overrides(args, device)
    print(f"[train] hydra overrides: {overrides}\n")

    # Hydra reads overrides from sys.argv; inject ours before importing main.
    # The import must happen here (after patch_mps) so the patched safe_device
    # and torch_load_model are already live when hydra instantiates the config.
    sys.argv = [sys.argv[0]] + overrides

    from libero.lifelong.main import main as libero_main
    libero_main()


if __name__ == "__main__":
    main()
