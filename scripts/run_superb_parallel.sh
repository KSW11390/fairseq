#!/usr/bin/env bash
# =============================================================================
# run_superb_parallel.sh — 2-GPU 병렬 SUPERB evaluation
#
# 실험 구성:
#   - PR:  5 ablation students (DICEHuBERT, teacher 제외)
#   - ASR, IC, SF, SID, ASV, ER: 5 ablations + DICEHuBERT (teacher 제외)
#   - GPU 0, GPU 1에 교대로 배분하여 동시 2개 실험 실행
#
# Usage:
#   bash scripts/run_superb_parallel.sh [TASK]
#
#   TASK: PR | ASR | IC | SF | SID | ASV | ER | ALL (default: ALL)
#
# Required env vars (set before running):
#   DATA_LIBRISPEECH  — LibriSpeech root (for PR, ASR)
#   DATA_FLUENT       — Fluent Speech Commands root (for IC)
#   DATA_SNIPS        — AudioSLU/SNIPS root (for SF)
#   DATA_VOXCELEB1    — VoxCeleb1 root (for SID, ASV)
#   DATA_IEMOCAP      — IEMOCAP root (for ER)
#
# Optional:
#   EXP_ROOT          — experiment output root (default: /workspace/exp)
#   CKPT_ROOT         — checkpoint root (default: /workspace/checkpoints)
# =============================================================================
set -uo pipefail

SCRIPT_DIR="$(cd "$(dirname "$0")" && pwd)"
FAIRSEQ_ROOT="${FAIRSEQ_ROOT:-$(dirname "$SCRIPT_DIR")}"
S3PRL_DIR="$(python3 -c 'import s3prl; import os; print(os.path.dirname(s3prl.__file__))')"
RUNNER="${S3PRL_DIR}/run_downstream.py"

EXP_ROOT="${EXP_ROOT:-/workspace/exp}"
CKPT_ROOT="${CKPT_ROOT:-/workspace/checkpoints}"
LOG_ROOT="${LOG_ROOT:-/workspace/logs}"
TASK_ARG="${1:-ALL}"

mkdir -p "$EXP_ROOT" "$LOG_ROOT"

# -------- model registry --------
declare -A MODELS
MODELS[l9k4096_l9k32]="${CKPT_ROOT}/students/l9k4096_l9k32/checkpoint_last.pt"
MODELS[l8k4096_l9k32]="${CKPT_ROOT}/students/l8k4096_l9k32/checkpoint_last.pt"
MODELS[l8k4096_l8k32]="${CKPT_ROOT}/students/l8k4096_l8k32/checkpoint_last.pt"
MODELS[l9k32]="${CKPT_ROOT}/students/l9k32/checkpoint_last.pt"
MODELS[l9k4096]="${CKPT_ROOT}/students/l9k4096/checkpoint_last.pt"
MODELS[dicehubert]="${CKPT_ROOT}/dicehubert/checkpoint_best.pt"

# PR: only 5 ablations (no dicehubert, no teacher)
PR_MODELS=(l9k4096_l9k32 l8k4096_l9k32 l8k4096_l8k32 l9k32 l9k4096)
# Others: 5 ablations + dicehubert
ALL_MODELS=(l9k4096_l9k32 l8k4096_l9k32 l8k4096_l8k32 l9k32 l9k4096 dicehubert)

# -------- task → (downstream, config_flag, data_override) --------
get_task_info() {
    local task="$1"
    case "$task" in
        PR)
            DOWNSTREAM="ctc"
            CONFIG_FLAG="-c ${S3PRL_DIR}/downstream/ctc/libriphone.yaml"
            DATA_OVERRIDE="downstream_expert.corpus.path=${DATA_LIBRISPEECH:-/workspace/data/LibriSpeech}"
            MODELS_LIST=("${PR_MODELS[@]}")
            ;;
        ASR)
            DOWNSTREAM="asr"
            CONFIG_FLAG=""
            DATA_OVERRIDE="downstream_expert.datarc.libri_root=${DATA_LIBRISPEECH:-/workspace/data/LibriSpeech}"
            MODELS_LIST=("${ALL_MODELS[@]}")
            ;;
        IC)
            DOWNSTREAM="fluent_commands"
            CONFIG_FLAG=""
            DATA_OVERRIDE="downstream_expert.datarc.file_path=${DATA_FLUENT:-/workspace/data/fluent_speech_commands_dataset}"
            MODELS_LIST=("${ALL_MODELS[@]}")
            ;;
        SF)
            DOWNSTREAM="audio_snips"
            CONFIG_FLAG=""
            DATA_OVERRIDE="downstream_expert.datarc.file_path=${DATA_SNIPS:-/workspace/data/audio_slu}"
            MODELS_LIST=("${ALL_MODELS[@]}")
            ;;
        SID)
            DOWNSTREAM="voxceleb1"
            CONFIG_FLAG=""
            DATA_OVERRIDE="downstream_expert.datarc.file_path=${DATA_VOXCELEB1:-/workspace/data/VoxCeleb1}"
            MODELS_LIST=("${ALL_MODELS[@]}")
            ;;
        ASV)
            DOWNSTREAM="sv_voxceleb1"
            CONFIG_FLAG=""
            DATA_OVERRIDE="downstream_expert.datarc.file_path=${DATA_VOXCELEB1:-/workspace/data/VoxCeleb1}"
            MODELS_LIST=("${ALL_MODELS[@]}")
            ;;
        ER)
            DOWNSTREAM="emotion"
            CONFIG_FLAG=""
            DATA_OVERRIDE="downstream_expert.datarc.root=${DATA_IEMOCAP:-/workspace/data/IEMOCAP_full_release}"
            MODELS_LIST=("${ALL_MODELS[@]}")
            ;;
        *)
            echo "Unknown task: $task" >&2
            exit 1
            ;;
    esac
}

# -------- run one experiment --------
run_one() {
    local model_name="$1"
    local task="$2"
    local gpu_id="$3"
    local ckpt="${MODELS[$model_name]}"
    local expdir="${EXP_ROOT}/${task}_${model_name}"
    local logfile="${LOG_ROOT}/${task}_${model_name}.log"

    if [ ! -f "$ckpt" ]; then
        echo "[SKIP] Checkpoint not found: $ckpt" >&2
        return 1
    fi

    get_task_info "$task"

    echo "[$(date '+%H:%M:%S')] START ${task}/${model_name} GPU=${gpu_id}" | tee -a "$logfile"

    cd "$S3PRL_DIR"

    CUDA_VISIBLE_DEVICES=${gpu_id} \
    PYTHONPATH="${FAIRSEQ_ROOT}" \
    python3 "$RUNNER" \
        -m train \
        -u customized_upstream \
        -k "$ckpt" \
        -d "$DOWNSTREAM" \
        -p "$expdir" \
        ${CONFIG_FLAG:+${CONFIG_FLAG}} \
        -o "${DATA_OVERRIDE}" \
        -a \
        >> "$logfile" 2>&1

    local train_exit=$?
    if [ $train_exit -ne 0 ]; then
        echo "[$(date '+%H:%M:%S')] TRAIN FAILED ${task}/${model_name} (exit: $train_exit)" | tee -a "$logfile"
        return $train_exit
    fi
    echo "[$(date '+%H:%M:%S')] TRAIN DONE ${task}/${model_name}" | tee -a "$logfile"

    # Evaluate
    if [ ! -f "${expdir}/dev-best.ckpt" ]; then
        echo "[$(date '+%H:%M:%S')] WARN: no dev-best.ckpt for ${task}/${model_name}" | tee -a "$logfile"
        return 0
    fi

    cd "$S3PRL_DIR"

    CUDA_VISIBLE_DEVICES=${gpu_id} \
    PYTHONPATH="${FAIRSEQ_ROOT}" \
    python3 "$RUNNER" \
        -m evaluate \
        -u customized_upstream \
        -k "$ckpt" \
        -d "$DOWNSTREAM" \
        -p "$expdir" \
        ${CONFIG_FLAG:+${CONFIG_FLAG}} \
        -o "${DATA_OVERRIDE}" \
        -e "${expdir}/dev-best.ckpt" \
        >> "$logfile" 2>&1

    local eval_exit=$?
    if [ $eval_exit -ne 0 ]; then
        echo "[$(date '+%H:%M:%S')] EVAL FAILED ${task}/${model_name} (exit: $eval_exit)" | tee -a "$logfile"
        return $eval_exit
    fi

    echo "[$(date '+%H:%M:%S')] DONE ${task}/${model_name}" | tee -a "$logfile"
    return 0
}

# -------- schedule jobs across 2 GPUs --------
run_task_parallel() {
    local task="$1"
    get_task_info "$task"
    local models=("${MODELS_LIST[@]}")
    local n=${#models[@]}

    echo "=== Task: ${task} | ${n} models | 2 GPUs ==="

    local gpu=0
    local pids=()
    local names=()

    for model in "${models[@]}"; do
        run_one "$model" "$task" "$gpu" &
        pids+=($!)
        names+=("${task}/${model}[GPU${gpu}]")
        gpu=$(( (gpu + 1) % 2 ))

        # Wait when both GPUs are occupied (every 2 jobs)
        if [ ${#pids[@]} -eq 2 ]; then
            for i in 0 1; do
                if wait "${pids[$i]}"; then
                    echo "[OK] ${names[$i]}"
                else
                    echo "[FAIL] ${names[$i]}"
                fi
            done
            pids=()
            names=()
        fi
    done

    # Wait for remaining (odd number of models)
    for i in "${!pids[@]}"; do
        if wait "${pids[$i]}"; then
            echo "[OK] ${names[$i]}"
        else
            echo "[FAIL] ${names[$i]}"
        fi
    done

    echo "=== Task ${task} complete ==="
}

# -------- main --------
if [ "$TASK_ARG" = "ALL" ]; then
    TASKS=(PR ASR IC SF SID ASV ER)
else
    IFS=',' read -ra TASKS <<< "$TASK_ARG"
fi

for task in "${TASKS[@]}"; do
    run_task_parallel "$task"
done

echo ""
echo "=== All tasks complete. Results in ${EXP_ROOT} ==="
