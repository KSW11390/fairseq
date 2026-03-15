#!/usr/bin/env python3
"""
SUPERB downstream evaluation — single CLI.

Takes a student checkpoint and runs s3prl downstream training + evaluation
for one or more SUPERB tasks.  Results are printed as a summary table.

Each task is defined by a YAML config in scripts/superb_tasks/<task>.yaml.
New tasks can be added by adding a YAML file there — no code changes needed.

Usage:
    python scripts/superb_eval.py \\
        --ckpt /path/to/student_checkpoint.pt \\
        --tasks pr,ks \\
        --s3prl /path/to/s3prl \\
        --upstream_dir /path/to/fairseq/examples/hubert/s3prl_upstream \\
        --data_root /path/to/superb_data \\
        --out_dir /path/to/results \\
        --gpu 0

    # Multiple GPUs: tasks run sequentially, cycling through GPUs
    python scripts/superb_eval.py --tasks pr,ks,asr --gpus 0,1 ...

Result:
    Prints per-task test metric (PER / Acc / WER) at dev-best step.
    Saves log and args YAML to <out_dir>/<task>/

Notes:
    - Uses customized_upstream from examples/hubert/s3prl_upstream/hubconf.py.
    - s3prl is invoked as a subprocess → environment isolation (avoids
      fairseq / s3prl dependency conflicts).
    - dev-best step is found by parsing the s3prl training log
      ("New best on dev at step X: Y").
    - CRITICAL: always check that the eval log shows
      "[Runner] - Resume from .../dev-best.ckpt"
      If it shows "Start a new experiment", the result is INVALID.
"""

import argparse
import logging
import os
import re
import subprocess
import sys
import time
from pathlib import Path

import yaml

logging.basicConfig(
    format="%(asctime)s | %(levelname)s | %(name)s | %(message)s",
    level=logging.INFO,
    datefmt="%Y-%m-%d %H:%M:%S",
)
logger = logging.getLogger(__name__)

TASKS_DIR = Path(__file__).resolve().parent / "superb_tasks"


# ---------------------------------------------------------------------------
# CLI
# ---------------------------------------------------------------------------

def parse_args():
    p = argparse.ArgumentParser(formatter_class=argparse.ArgumentDefaultsHelpFormatter)
    p.add_argument("--ckpt", required=True,
                   help="Path to student fairseq checkpoint (.pt)")
    p.add_argument("--tasks", required=True,
                   help="Comma-separated task names, e.g. 'pr,ks,asr'. "
                        f"Available: {[f.stem for f in TASKS_DIR.glob('*.yaml')]}")
    p.add_argument("--s3prl", required=True,
                   help="Path to s3prl repository root")
    p.add_argument("--upstream_dir", required=True,
                   help="Path to directory containing hubconf.py + expert.py "
                        "(examples/hubert/s3prl_upstream/)")
    p.add_argument("--data_root", required=True,
                   help="Root directory for SUPERB task data "
                        "(s3prl expects task data under this root)")
    p.add_argument("--out_dir", required=True,
                   help="Output root; results saved to <out_dir>/<task>/")
    p.add_argument("--gpus", default="0",
                   help="Comma-separated GPU indices, e.g. '0,1'. "
                        "Tasks are assigned sequentially cycling through GPUs.")
    p.add_argument("--python", default=sys.executable,
                   help="Python interpreter for s3prl subprocess "
                        "(use this to point at a different venv if needed)")
    p.add_argument("--extra_overrides", default="",
                   help="Additional s3prl override string appended to each task run, "
                        "e.g. 'config.runner.total_step=50000'")
    return p.parse_args()


# ---------------------------------------------------------------------------
# Task config loading
# ---------------------------------------------------------------------------

def load_task_config(task_name: str) -> dict:
    path = TASKS_DIR / f"{task_name}.yaml"
    if not path.exists():
        available = [f.stem for f in TASKS_DIR.glob("*.yaml")]
        raise FileNotFoundError(
            f"Task '{task_name}' not found in {TASKS_DIR}. "
            f"Available: {available}"
        )
    with open(path) as f:
        return yaml.safe_load(f)


# ---------------------------------------------------------------------------
# s3prl invocation
# ---------------------------------------------------------------------------

def build_s3prl_cmd(task_cfg: dict, args, task_name: str, gpu: int, exp_dir: str):
    """Build the s3prl run_downstream.py command for training."""
    run_downstream = os.path.join(args.s3prl, "s3prl", "run_downstream.py")

    cmd = [
        args.python, run_downstream,
        "-m", "train",
        "-u", "customized_upstream",
        "-k", args.ckpt,
        "-d", task_cfg["downstream"],
        "-p", exp_dir,
    ]

    # s3prl run_downstream.py defaults config to ./downstream/<task>/config.yaml,
    # but the actual file may be named differently (e.g. libriphone.yaml).
    # Explicitly pass -c to point to the correct yaml.
    if "downstream_config" in task_cfg:
        cfg_path = f"./downstream/{task_cfg['downstream']}/{task_cfg['downstream_config']}"
        cmd += ["-c", cfg_path]

    # s3prl -o accepts multiple overrides joined by ',,' in a single flag.
    # Passing two separate -o flags causes the second to overwrite the first.
    overrides = f"config.runner.total_steps={task_cfg['train_steps']}"
    if args.extra_overrides:
        overrides += f",,{args.extra_overrides}"
    cmd += ["-o", overrides]

    return cmd, {"CUDA_VISIBLE_DEVICES": str(gpu)}


def build_eval_cmd(task_cfg: dict, args, task_name: str, gpu: int,
                   exp_dir: str, best_ckpt: str):
    """Build the s3prl run_downstream.py command for evaluation."""
    run_downstream = os.path.join(args.s3prl, "s3prl", "run_downstream.py")

    cmd = [
        args.python, run_downstream,
        "-m", "evaluate",
        "-u", "customized_upstream",
        "-k", args.ckpt,
        "-d", task_cfg["downstream"],
        "-p", f"{exp_dir}_eval",
        "-e", best_ckpt,
    ]

    if "downstream_config" in task_cfg:
        cfg_path = f"./downstream/{task_cfg['downstream']}/{task_cfg['downstream_config']}"
        cmd += ["-c", cfg_path]

    return cmd, {"CUDA_VISIBLE_DEVICES": str(gpu)}


# ---------------------------------------------------------------------------
# Log parsing
# ---------------------------------------------------------------------------

def find_dev_best(log_path: str):
    """
    Parse s3prl training log to find dev-best step and dev score.
    Returns (step: int, dev_score: float) or (None, None) if not found.
    """
    best_step, best_score = None, None
    pattern = re.compile(r"New best on dev at step (\d+):\s+([\d.]+)")
    try:
        with open(log_path, encoding="utf-8", errors="replace") as f:
            for line in f:
                m = pattern.search(line)
                if m:
                    best_step = int(m.group(1))
                    best_score = float(m.group(2))
    except FileNotFoundError:
        pass
    return best_step, best_score


def find_test_score_at_step(log_path: str, step: int):
    """
    Parse s3prl training log to find test score reported at a specific step.
    Returns float or None.
    """
    pattern = re.compile(rf"test at step {step}:\s+([\d.]+)")
    try:
        with open(log_path, encoding="utf-8", errors="replace") as f:
            for line in f:
                m = pattern.search(line)
                if m:
                    return float(m.group(1))
    except FileNotFoundError:
        pass
    return None


def find_eval_metric(eval_log_path: str, metric: str):
    """
    Parse s3prl evaluate-mode log for the test metric.
    Also verifies that "Resume from" appears (not "Start a new experiment").
    Returns (score: float or None, valid: bool).
    """
    if not os.path.exists(eval_log_path):
        return None, False

    resumed = False
    score = None
    metric_patterns = {
        "per": re.compile(r"test per:\s+([\d.]+)"),
        "wer": re.compile(r"test wer:\s+([\d.]+)"),
        "acc": re.compile(r"test at step \d+:\s+([\d.]+)"),
        "si_sdr": re.compile(r"Average si_sdr of \d+ utts:\s+([\d.eE+\-]+)"),
    }
    pat = metric_patterns.get(metric, re.compile(r"test.*?:\s+([\d.]+)"))

    with open(eval_log_path, encoding="utf-8", errors="replace") as f:
        for line in f:
            if "Resume from" in line:
                resumed = True
            m = pat.search(line)
            if m:
                score = float(m.group(1))

    if not resumed:
        logger.warning(
            f"INVALID EVAL: '{eval_log_path}' does NOT contain 'Resume from'. "
            "The downstream head was not restored from dev-best.ckpt — result is wrong."
        )
    return score, resumed


# ---------------------------------------------------------------------------
# Task runner
# ---------------------------------------------------------------------------

def run_task(task_name: str, args, gpu: int) -> dict:
    """Train and evaluate one SUPERB task. Returns result dict."""
    task_cfg = load_task_config(task_name)
    exp_dir = os.path.join(args.out_dir, task_name)
    os.makedirs(exp_dir, exist_ok=True)

    log_path = os.path.join(exp_dir, "log.log")
    result = {"task": task_name, "gpu": gpu, "metric": task_cfg["metric"]}

    # ---- Training --------------------------------------------------------
    logger.info(f"[{task_name}] Starting downstream training on GPU {gpu} ...")
    train_cmd, train_env = build_s3prl_cmd(task_cfg, args, task_name, gpu, exp_dir)
    env = {**os.environ, **train_env}

    # s3prl run_downstream.py resolves configs relative to cwd,
    # so it must be run from within the s3prl/s3prl/ directory.
    s3prl_run_dir = os.path.join(args.s3prl, "s3prl")

    t0 = time.time()
    try:
        with open(log_path, "w") as log_f:
            ret = subprocess.run(
                train_cmd, env=env,
                stdout=log_f, stderr=subprocess.STDOUT,
                cwd=s3prl_run_dir,
            )
        elapsed = time.time() - t0
        if ret.returncode != 0:
            logger.error(
                f"[{task_name}] Training FAILED (returncode={ret.returncode}). "
                f"See {log_path}"
            )
            result["status"] = "train_failed"
            result["log"] = log_path
            return result
    except Exception as e:
        logger.error(f"[{task_name}] Training exception: {e}")
        result["status"] = "train_exception"
        return result

    logger.info(f"[{task_name}] Training done in {elapsed:.0f}s.")

    # ---- Find dev-best step ----------------------------------------------
    best_step, best_dev = find_dev_best(log_path)
    if best_step is None:
        logger.warning(f"[{task_name}] Could not find dev-best step in {log_path}.")
        # Fallback: use training-time test score at the last logged test step
        result["status"] = "no_dev_best"
        result["log"] = log_path
        return result

    logger.info(
        f"[{task_name}] Dev-best step: {best_step} (dev {task_cfg['metric']}={best_dev:.4f})"
    )
    result["dev_best_step"] = best_step
    result["dev_best_score"] = best_dev

    # ---- Training-time test score at dev-best ----------------------------
    train_test_score = find_test_score_at_step(log_path, best_step)
    if train_test_score is not None:
        result["test_score_from_train"] = train_test_score
        logger.info(
            f"[{task_name}] Test score at dev-best step (from training log): "
            f"{task_cfg['metric']}={train_test_score:.4f}"
        )

    # ---- Evaluate mode (explicit eval on best checkpoint) ---------------
    best_ckpt_name = task_cfg.get("best_ckpt_name", "dev-best.ckpt")
    best_ckpt = os.path.join(exp_dir, best_ckpt_name)
    if not os.path.exists(best_ckpt):
        logger.warning(
            f"[{task_name}] {best_ckpt_name} not found at {best_ckpt}. "
            "Skipping evaluate mode — using training-time test score."
        )
        result["test_score"] = train_test_score
        result["test_score_source"] = "training_log"
        result["status"] = "ok"
        return result

    eval_log_dir = os.path.join(args.out_dir, f"{task_name}_eval")
    os.makedirs(eval_log_dir, exist_ok=True)
    eval_log_path = os.path.join(eval_log_dir, "log.log")

    logger.info(f"[{task_name}] Running evaluate mode ...")
    eval_cmd, eval_env = build_eval_cmd(
        task_cfg, args, task_name, gpu, exp_dir, best_ckpt
    )
    env_eval = {**os.environ, **eval_env}

    try:
        with open(eval_log_path, "w") as ef:
            ret = subprocess.run(
                eval_cmd, env=env_eval,
                stdout=ef, stderr=subprocess.STDOUT,
                cwd=s3prl_run_dir,
            )
    except Exception as e:
        logger.error(f"[{task_name}] Evaluate exception: {e}")
        result["test_score"] = train_test_score
        result["test_score_source"] = "training_log_fallback"
        result["status"] = "eval_exception"
        return result

    eval_score, valid = find_eval_metric(eval_log_path, task_cfg["metric"])

    if valid and eval_score is not None:
        result["test_score"] = eval_score
        result["test_score_source"] = "evaluate_mode"
        logger.info(
            f"[{task_name}] Evaluate-mode test {task_cfg['metric']}={eval_score:.4f} ✓"
        )
    else:
        logger.warning(
            f"[{task_name}] Evaluate-mode result invalid or missing. "
            "Falling back to training-time test score."
        )
        result["test_score"] = train_test_score
        result["test_score_source"] = "training_log_fallback"

    result["status"] = "ok"
    result["eval_log"] = eval_log_path
    return result


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------

def main():
    args = parse_args()
    task_names = [t.strip() for t in args.tasks.split(",")]
    gpus = [int(g.strip()) for g in args.gpus.split(",")]
    os.makedirs(args.out_dir, exist_ok=True)

    logger.info(f"Student checkpoint: {args.ckpt}")
    logger.info(f"Tasks: {task_names}")
    logger.info(f"GPUs: {gpus}")

    results = []
    for i, task_name in enumerate(task_names):
        gpu = gpus[i % len(gpus)]
        logger.info(f"\n{'='*60}")
        logger.info(f"Task {i+1}/{len(task_names)}: {task_name} on GPU {gpu}")
        logger.info(f"{'='*60}")
        res = run_task(task_name, args, gpu)
        results.append(res)

    # ---- Summary table ---------------------------------------------------
    print("\n" + "=" * 60)
    print("SUPERB EVALUATION SUMMARY")
    print("=" * 60)
    print(f"Checkpoint: {args.ckpt}")
    print()
    print(f"{'Task':<8} {'Metric':<8} {'Score':<10} {'Dev-best step':<16} {'Source'}")
    print("-" * 60)
    for res in results:
        task = res["task"]
        metric = res["metric"].upper()
        score = res.get("test_score")
        step = res.get("dev_best_step", "?")
        source = res.get("test_score_source", res.get("status", "?"))
        score_str = f"{score:.4f}" if score is not None else "N/A"
        print(f"{task:<8} {metric:<8} {score_str:<10} {str(step):<16} {source}")
    print("=" * 60)

    # Save results YAML
    results_path = os.path.join(args.out_dir, "results.yaml")
    with open(results_path, "w") as f:
        yaml.dump(
            {"ckpt": args.ckpt, "results": results},
            f, default_flow_style=False,
        )
    logger.info(f"Results saved → {results_path}")


if __name__ == "__main__":
    main()
