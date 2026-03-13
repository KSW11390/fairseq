#!/usr/bin/env bash
# =============================================================================
# 04_superb_eval.sh — Run SUPERB benchmark evaluation via s3prl
#
# Features:
#   - nohup-safe: SSH 끊어져도 계속 실행
#   - Resume: 중단된 지점부터 자동 재개 (s3prl --past_exp)
#   - Logging: 타임스탬프 포함 로그 파일 자동 저장
#   - W&B/TensorBoard: 시각화 연동
#   - Multi-task: 여러 태스크를 순차 실행하는 batch 모드
#
# Prerequisites:
#   pip install s3prl wandb tensorboard
#
# Required environment variables:
#   CKPT_PATH       Path to DICEHuBERT checkpoint (checkpoint_best.pt)
#   TASK            SUPERB downstream task name(s), comma-separated
#                   e.g. "KS" or "KS,SID,IC,ER"
#
# Optional environment variables:
#   S3PRL_ROOT      Path to s3prl repo (default: uses installed package)
#   FAIRSEQ_ROOT    Path to fairseq repo (default: parent of this script)
#   DATA_ROOT       Root dir for downstream task data
#   EXP_ROOT        Root experiment directory (default: fairseq/exp)
#   GPUS            Number of GPUs (default: 1)
#   LR              Learning rate override
#   EPOCHS          Max training epochs override
#   RESUME          Set to "1" to resume from last checkpoint (default: 1)
#   WANDB_PROJECT   W&B project name (default: dicehubert-superb)
#   WANDB_ENTITY    W&B entity/team name
#   RUN_NAME        Run name prefix for logging/W&B
#   BACKGROUND      Set to "1" to auto-detach with nohup (default: 0)
#
# Supported SUPERB tasks:
#   PR  ASR  KS  QbE  SID  ASV  IC  SF  ER  SE  SS  ST
#
# Usage:
#   # Single task
#   CKPT_PATH=/path/to/ckpt.pt TASK=KS bash scripts/04_superb_eval.sh
#
#   # Multi-task batch
#   CKPT_PATH=/path/to/ckpt.pt TASK=KS,SID,IC,ER bash scripts/04_superb_eval.sh
#
#   # Background (SSH-safe)
#   CKPT_PATH=/path/to/ckpt.pt TASK=KS,SID,IC,ER BACKGROUND=1 bash scripts/04_superb_eval.sh
#
#   # Resume interrupted run
#   CKPT_PATH=/path/to/ckpt.pt TASK=KS RESUME=1 bash scripts/04_superb_eval.sh
# =============================================================================
set -euo pipefail

# ---------- resolve paths ----------
SCRIPT_DIR="$(cd "$(dirname "$0")" && pwd)"
FAIRSEQ_ROOT="${FAIRSEQ_ROOT:-$(dirname "$SCRIPT_DIR")}"
UPSTREAM_DIR="${FAIRSEQ_ROOT}/examples/hubert/s3prl_upstream"

# ---------- background mode: re-launch under nohup ----------
BACKGROUND="${BACKGROUND:-0}"
if [ "$BACKGROUND" = "1" ] && [ "${_SUPERB_NOHUP_CHILD:-0}" != "1" ]; then
    # Re-exec this script under nohup so it survives SSH disconnect
    export _SUPERB_NOHUP_CHILD=1
    EXP_ROOT="${EXP_ROOT:-${FAIRSEQ_ROOT}/exp}"
    mkdir -p "$EXP_ROOT"
    NOHUP_LOG="${EXP_ROOT}/superb_nohup_$(date +%Y%m%d_%H%M%S).log"
    echo "Launching in background. Logs: $NOHUP_LOG"
    echo "Monitor: tail -f $NOHUP_LOG"
    nohup bash "$0" > "$NOHUP_LOG" 2>&1 &
    BGPID=$!
    echo "PID: $BGPID"
    echo "$BGPID" > "${EXP_ROOT}/superb_eval.pid"
    exit 0
fi

# ---------- validate required vars ----------
if [ -z "${CKPT_PATH:-}" ]; then
    echo "ERROR: CKPT_PATH is not set. Set it to the DICEHuBERT checkpoint path."
    exit 1
fi
if [ -z "${TASK:-}" ]; then
    echo "ERROR: TASK is not set. Choose from: PR ASR KS QbE SID ASV IC SF ER SE SS ST"
    echo "       Multiple tasks: TASK=KS,SID,IC,ER"
    exit 1
fi
if [ ! -f "$CKPT_PATH" ]; then
    echo "ERROR: Checkpoint not found: $CKPT_PATH"
    exit 1
fi

# ---------- task mapping ----------
declare -A TASK_MAP=(
    [PR]=ctc
    [ASR]=asr
    [KS]=speech_commands
    [QbE]=qbe
    [SID]=voxceleb1
    [ASV]=sv_voxceleb1
    [IC]=fluent_commands
    [SF]=snips
    [ER]=emotion
    [SE]=enhancement_stft
    [SS]=separation_stft
    [ST]=speech_translation
)

# Tasks with non-default config files (relative to s3prl CWD)
declare -A CONFIG_MAP=(
    [PR]="downstream/ctc/libriphone.yaml"
)

# ---------- defaults ----------
EXP_ROOT="${EXP_ROOT:-${FAIRSEQ_ROOT}/exp}"
GPUS="${GPUS:-1}"
RESUME="${RESUME:-1}"
RUN_NAME="${RUN_NAME:-dicehubert}"
WANDB_PROJECT="${WANDB_PROJECT:-dicehubert-superb}"

# ---------- find s3prl runner ----------
if [ -n "${S3PRL_ROOT:-}" ]; then
    S3PRL_DIR="${S3PRL_ROOT}/s3prl"
else
    S3PRL_DIR="$(python3 -c 'import s3prl; import os; print(os.path.dirname(s3prl.__file__))')"
fi
RUNNER="${S3PRL_DIR}/run_downstream.py"
if [ ! -f "$RUNNER" ]; then
    echo "ERROR: s3prl runner not found at $RUNNER"
    echo "Install s3prl or set S3PRL_ROOT to the cloned s3prl repo"
    exit 1
fi
# s3prl resolves downstream configs relative to CWD; cd to s3prl dir
cd "$S3PRL_DIR"

# ---------- logging helper ----------
log() {
    echo "[$(date '+%Y-%m-%d %H:%M:%S')] $*"
}

# ---------- run single task ----------
run_task() {
    local task_name="$1"
    local downstream="${TASK_MAP[$task_name]:-}"

    if [ -z "$downstream" ]; then
        log "ERROR: Unknown task '$task_name'. Choose from: ${!TASK_MAP[*]}"
        return 1
    fi

    local exp_dir="${EXP_ROOT}/${RUN_NAME}_${task_name}"
    local log_dir="${exp_dir}/logs"
    mkdir -p "$log_dir"

    local train_log="${log_dir}/train_$(date +%Y%m%d_%H%M%S).log"
    local eval_log="${log_dir}/eval_$(date +%Y%m%d_%H%M%S).log"

    # build common args (s3prl 0.4.x API: -u, -k, -d, -p)
    # customized_upstream uses upstream/example/expert.py → patched with our DICEHuBERT expert
    local common_args=(
        -u customized_upstream
        -k "$CKPT_PATH"
        -d "$downstream"
        -p "$exp_dir"
    )

    # Non-default config file for certain tasks
    local task_config="${CONFIG_MAP[$task_name]:-}"
    if [ -n "$task_config" ]; then
        common_args+=(-c "$task_config")
    fi

    # --- resume logic (s3prl 0.4.x: -e for past_exp, -a for auto_resume) ---
    local resume_args=()
    if [ "$RESUME" = "1" ] && [ -d "$exp_dir" ]; then
        resume_args=(-a)   # auto-resume from latest checkpoint in expdir
    fi

    # --- check if already completed ---
    local result_file="${exp_dir}/dev-best.ckpt"
    if [ -f "$result_file" ] && [ "$RESUME" = "1" ]; then
        log "SKIP ${task_name}: appears already trained (dev-best.ckpt exists)"
        log "  Delete ${exp_dir} to re-run from scratch"
    fi

    log "============================================"
    log " SUPERB Task: ${task_name} (${downstream})"
    log " Checkpoint:  ${CKPT_PATH}"
    log " Output:      ${exp_dir}"
    log " Train log:   ${train_log}"
    log " Resume:      ${RESUME}"
    log "============================================"

    # --- Step 1: Train downstream ---
    log "[${task_name}] Step 1/2: Training downstream model..."
    python3 "$RUNNER" \
        -m train \
        "${common_args[@]}" \
        "${resume_args[@]}" \
        2>&1 | tee "$train_log"
    local train_exit=${PIPESTATUS[0]}

    if [ $train_exit -ne 0 ]; then
        log "ERROR: Training failed for ${task_name} (exit code: $train_exit)"
        log "  Check log: $train_log"
        return $train_exit
    fi

    # --- Step 2: Evaluate ---
    log "[${task_name}] Step 2/2: Evaluating..."
    python3 "$RUNNER" \
        -m evaluate \
        "${common_args[@]}" \
        2>&1 | tee "$eval_log"
    local eval_exit=${PIPESTATUS[0]}

    if [ $eval_exit -ne 0 ]; then
        log "ERROR: Evaluation failed for ${task_name} (exit code: $eval_exit)"
        log "  Check log: $eval_log"
        return $eval_exit
    fi

    log "[${task_name}] DONE. Results: ${exp_dir}"
    return 0
}

# ---------- main: parse comma-separated tasks ----------
IFS=',' read -ra TASKS <<< "$TASK"

log "============================================"
log " SUPERB Batch Evaluation"
log " Tasks: ${TASKS[*]}"
log " Checkpoint: ${CKPT_PATH}"
log " Exp root: ${EXP_ROOT}"
log " Resume: ${RESUME}"
log "============================================"

FAILED_TASKS=()
COMPLETED_TASKS=()
SKIPPED_TASKS=()

for t in "${TASKS[@]}"; do
    t=$(echo "$t" | xargs)  # trim whitespace
    if run_task "$t"; then
        # check if it was skipped or completed
        result_file="${EXP_ROOT}/${RUN_NAME}_${t}/result.json"
        if [ -f "$result_file" ]; then
            COMPLETED_TASKS+=("$t")
        else
            COMPLETED_TASKS+=("$t")
        fi
    else
        FAILED_TASKS+=("$t")
        log "WARNING: ${t} failed, continuing with next task..."
    fi
done

# ---------- summary ----------
log ""
log "============================================"
log " SUPERB Evaluation Summary"
log "============================================"
log " Completed: ${COMPLETED_TASKS[*]:-none}"
if [ ${#FAILED_TASKS[@]} -gt 0 ]; then
    log " FAILED:    ${FAILED_TASKS[*]}"
fi
log ""
log " Results directory: ${EXP_ROOT}"
log "============================================"

# collect all results into a summary
SUMMARY_FILE="${EXP_ROOT}/${RUN_NAME}_summary.txt"
{
    echo "SUPERB Evaluation Summary — $(date)"
    echo "Checkpoint: ${CKPT_PATH}"
    echo "---"
    for t in "${TASKS[@]}"; do
        t=$(echo "$t" | xargs)
        result_file="${EXP_ROOT}/${RUN_NAME}_${t}/result.json"
        if [ -f "$result_file" ]; then
            echo "[${t}] $(cat "$result_file")"
        else
            echo "[${t}] no result"
        fi
    done
} > "$SUMMARY_FILE"
log "Summary written to: ${SUMMARY_FILE}"

# exit with error if any task failed
if [ ${#FAILED_TASKS[@]} -gt 0 ]; then
    exit 1
fi
