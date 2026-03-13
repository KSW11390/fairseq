#!/usr/bin/env bash
# PR + ASR: 2-GPU parallel SUPERB evaluation
# Upload to /workspace/run_pr_asr.sh and run: nohup bash /workspace/run_pr_asr.sh > /workspace/logs/run_pr_asr.log 2>&1 &

S3PRL=/usr/local/lib/python3.11/dist-packages/s3prl
LIBRI=/workspace/data/LibriSpeech
EXP=/workspace/exp
CKPT=/workspace/checkpoints
LOG=/workspace/logs
mkdir -p $EXP $LOG

run_model() {
    local NAME=$1 TASK=$2 DOWNSTREAM=$3 GPU=$4 CKPT_PATH=$5 OVERRIDE=$6 CFG=$7
    local EXPDIR="${EXP}/${TASK}_${NAME}"
    local LOGF="${LOG}/${TASK}_${NAME}.log"
    cd $S3PRL
    export CUDA_VISIBLE_DEVICES=$GPU PYTHONPATH=/workspace/fairseq
    echo "[$(date +%H:%M:%S)] START ${TASK}/${NAME} GPU=${GPU}" | tee -a "$LOGF"

    if [ -n "$CFG" ]; then
        python3 run_downstream.py -m train \
            -u customized_upstream -k "$CKPT_PATH" \
            -d "$DOWNSTREAM" -p "$EXPDIR" \
            -c "$CFG" -o "$OVERRIDE" -a >> "$LOGF" 2>&1
    else
        python3 run_downstream.py -m train \
            -u customized_upstream -k "$CKPT_PATH" \
            -d "$DOWNSTREAM" -p "$EXPDIR" \
            -o "$OVERRIDE" -a >> "$LOGF" 2>&1
    fi
    local rc=$?
    if [ $rc -ne 0 ]; then
        echo "[$(date +%H:%M:%S)] TRAIN FAILED ${TASK}/${NAME} (rc=$rc)" | tee -a "$LOGF"
        return $rc
    fi
    echo "[$(date +%H:%M:%S)] TRAIN DONE ${TASK}/${NAME}" | tee -a "$LOGF"

    local BEST="${EXPDIR}/dev-best.ckpt"
    if [ ! -f "$BEST" ]; then
        echo "[$(date +%H:%M:%S)] WARN: no dev-best.ckpt for ${TASK}/${NAME}" | tee -a "$LOGF"
        return 0
    fi

    if [ -n "$CFG" ]; then
        python3 run_downstream.py -m evaluate \
            -u customized_upstream -k "$CKPT_PATH" \
            -d "$DOWNSTREAM" -p "${EXPDIR}_eval" \
            -c "$CFG" -o "$OVERRIDE" -e "$BEST" >> "$LOGF" 2>&1
    else
        python3 run_downstream.py -m evaluate \
            -u customized_upstream -k "$CKPT_PATH" \
            -d "$DOWNSTREAM" -p "${EXPDIR}_eval" \
            -o "$OVERRIDE" -e "$BEST" >> "$LOGF" 2>&1
    fi
    local eval_rc=$?
    if [ $eval_rc -ne 0 ]; then
        echo "[$(date +%H:%M:%S)] EVAL FAILED ${TASK}/${NAME} (rc=$eval_rc)" | tee -a "$LOGF"
        return $eval_rc
    fi
    echo "[$(date +%H:%M:%S)] EVAL DONE ${TASK}/${NAME}" | tee -a "$LOGF"
}

PR_CFG="${S3PRL}/downstream/ctc/libriphone.yaml"
PR_OVR="config.downstream_expert.corpus.path=${LIBRI}"
ASR_OVR="config.downstream_expert.datarc.libri_root=${LIBRI}"

echo "=== PR: 5 ablation students (2-GPU parallel) ==="

# Batch 1
run_model l9k4096_l9k32 PR ctc 0 "${CKPT}/students/l9k4096_l9k32/checkpoint_last.pt" "$PR_OVR" "$PR_CFG" &
JOB0=$!
run_model l8k4096_l9k32 PR ctc 1 "${CKPT}/students/l8k4096_l9k32/checkpoint_last.pt" "$PR_OVR" "$PR_CFG" &
JOB1=$!
wait $JOB0 && echo "[OK] PR/l9k4096_l9k32" || echo "[FAIL] PR/l9k4096_l9k32"
wait $JOB1 && echo "[OK] PR/l8k4096_l9k32" || echo "[FAIL] PR/l8k4096_l9k32"

# Batch 2
run_model l8k4096_l8k32 PR ctc 0 "${CKPT}/students/l8k4096_l8k32/checkpoint_last.pt" "$PR_OVR" "$PR_CFG" &
JOB0=$!
run_model l9k32         PR ctc 1 "${CKPT}/students/l9k32/checkpoint_last.pt"          "$PR_OVR" "$PR_CFG" &
JOB1=$!
wait $JOB0 && echo "[OK] PR/l8k4096_l8k32" || echo "[FAIL] PR/l8k4096_l8k32"
wait $JOB1 && echo "[OK] PR/l9k32"         || echo "[FAIL] PR/l9k32"

# Batch 3 (last 1)
run_model l9k4096 PR ctc 0 "${CKPT}/students/l9k4096/checkpoint_last.pt" "$PR_OVR" "$PR_CFG" \
    && echo "[OK] PR/l9k4096" || echo "[FAIL] PR/l9k4096"
echo "=== PR done ==="

echo "=== ASR: 5 ablations + dicehubert (2-GPU parallel) ==="

# Batch 1
run_model l9k4096_l9k32 ASR asr 0 "${CKPT}/students/l9k4096_l9k32/checkpoint_last.pt" "$ASR_OVR" "" &
JOB0=$!
run_model l8k4096_l9k32 ASR asr 1 "${CKPT}/students/l8k4096_l9k32/checkpoint_last.pt" "$ASR_OVR" "" &
JOB1=$!
wait $JOB0 && echo "[OK] ASR/l9k4096_l9k32" || echo "[FAIL] ASR/l9k4096_l9k32"
wait $JOB1 && echo "[OK] ASR/l8k4096_l9k32" || echo "[FAIL] ASR/l8k4096_l9k32"

# Batch 2
run_model l8k4096_l8k32 ASR asr 0 "${CKPT}/students/l8k4096_l8k32/checkpoint_last.pt" "$ASR_OVR" "" &
JOB0=$!
run_model l9k32         ASR asr 1 "${CKPT}/students/l9k32/checkpoint_last.pt"          "$ASR_OVR" "" &
JOB1=$!
wait $JOB0 && echo "[OK] ASR/l8k4096_l8k32" || echo "[FAIL] ASR/l8k4096_l8k32"
wait $JOB1 && echo "[OK] ASR/l9k32"          || echo "[FAIL] ASR/l9k32"

# Batch 3
run_model l9k4096    ASR asr 0 "${CKPT}/students/l9k4096/checkpoint_last.pt"   "$ASR_OVR" "" &
JOB0=$!
run_model dicehubert ASR asr 1 "${CKPT}/dicehubert/checkpoint_best.pt"          "$ASR_OVR" "" &
JOB1=$!
wait $JOB0 && echo "[OK] ASR/l9k4096"    || echo "[FAIL] ASR/l9k4096"
wait $JOB1 && echo "[OK] ASR/dicehubert" || echo "[FAIL] ASR/dicehubert"

echo "=== ASR done ==="
echo "=== ALL PR+ASR COMPLETE. Results: ${EXP} ==="
