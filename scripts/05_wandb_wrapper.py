#!/usr/bin/env python3
"""
Lightweight W&B wrapper: initializes wandb with sync_tensorboard=True,
then launches fairseq_hydra_train as a subprocess.

Usage:
    python scripts/05_wandb_wrapper.py \
        --wandb_project my_project \
        --wandb_entity my_entity \
        --wandb_run_name "run1_hard_10k" \
        -- \
        fairseq-hydra-train \
        --config-dir examples/hubert/config/pretrain \
        --config-name hubert_student_distill_layer6_k500_tc100 \
        task.data=/path/to/tsv ...

Everything after '--' is passed directly as the training command.
"""

import argparse
import os
import subprocess
import sys


def main():
    parser = argparse.ArgumentParser(
        description="W&B + TensorBoard sync wrapper for fairseq training"
    )
    parser.add_argument(
        "--wandb_project",
        type=str,
        default=os.environ.get("WANDB_PROJECT", "hubert-distill"),
    )
    parser.add_argument(
        "--wandb_entity",
        type=str,
        default=os.environ.get("WANDB_ENTITY", None),
    )
    parser.add_argument(
        "--wandb_run_name",
        type=str,
        default=os.environ.get("WANDB_RUN_NAME", None),
    )
    parser.add_argument(
        "--tb_logdir",
        type=str,
        default=None,
        help="TensorBoard log dir (default: inferred from hydra run dir + /tblog)",
    )

    # Split at '--'
    argv = sys.argv[1:]
    if "--" in argv:
        split_idx = argv.index("--")
        wrapper_args = argv[:split_idx]
        train_cmd = argv[split_idx + 1 :]
    else:
        wrapper_args = argv
        train_cmd = []

    args = parser.parse_args(wrapper_args)

    if not train_cmd:
        parser.error("No training command provided after '--'")

    # --- Initialize W&B ---
    try:
        import wandb
    except ImportError:
        print("[WARN] wandb not installed. Running training without W&B sync.")
        print(f"[CMD] {' '.join(train_cmd)}")
        sys.exit(subprocess.call(train_cmd))

    init_kwargs = {
        "project": args.wandb_project,
        "sync_tensorboard": True,
    }
    if args.wandb_entity:
        init_kwargs["entity"] = args.wandb_entity
    if args.wandb_run_name:
        init_kwargs["name"] = args.wandb_run_name

    wandb.init(**init_kwargs)
    print(f"[wandb] Initialized: project={args.wandb_project}, run={wandb.run.name}")

    # --- Run training ---
    print(f"[CMD] {' '.join(train_cmd)}")
    ret = subprocess.call(train_cmd)

    wandb.finish()
    sys.exit(ret)


if __name__ == "__main__":
    main()
