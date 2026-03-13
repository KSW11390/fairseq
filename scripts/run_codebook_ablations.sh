#!/bin/bash
# =============================================================================
# Codebook Distillation Ablation Study
#
# Trains 4 EMA codebooks (L8-K32, L8-K4096, L9-K32, L9-K4096), generates
# label files, then runs 5 student experiments to compare distillation targets:
#
#   Exp 1 [vevo]       L9-K4096 + L9-K32   (VEVO-style: same layer, 2 scales)
#   Exp 2 [crosslayer] L8-K4096 + L9-K32   (cross-layer combination)
#   Exp 3 [l8same]     L8-K4096 + L8-K32   (same layer 8, 2 scales)
#   Exp 4 [l9k32]      L9-K32 only          (content bottleneck, single)
#   Exp 5 [l9k4096]    L9-K4096 only        (content+style, single)
#
# Usage:
#   bash scripts/run_codebook_ablations.sh [GPU_ID]
#
# Prerequisites:
#   - Codebook training already done OR set SKIP_CODEBOOK_TRAIN=0 to run it
#   - Label files already generated OR set SKIP_LABEL_GEN=0 to generate
# =============================================================================

set -euo pipefail

GPU=${1:-0}
export CUDA_VISIBLE_DEVICES=$GPU

# ---- Control flags ----------------------------------------------------------
SKIP_CODEBOOK_TRAIN=${SKIP_CODEBOOK_TRAIN:-0}   # Set to 1 to skip if codebooks exist
SKIP_LABEL_GEN=${SKIP_LABEL_GEN:-0}             # Set to 1 to skip if labels exist
RUN_EXPS=${RUN_EXPS:-"vevo crosslayer l8same l9k32 l9k4096"}  # Space-separated list

# ---- Paths (edit these) -----------------------------------------------------
FAIRSEQ_DIR="/root/fairseq"
TSV_DIR="/root/output/data/tc100"          # contains train.tsv, valid.tsv
TEACHER_CKPT="/root/output/ckpt/hubert_base_ls960.pt"
OUTPUT_DIR="/root/output/experiments/codebook_ablations"

CODEBOOK_DIR="$OUTPUT_DIR/codebooks"
LABEL_DIR="$OUTPUT_DIR/labels"
STUDENT_BASE_DIR="$OUTPUT_DIR/students"

# ---- Codebook config --------------------------------------------------------
CB_STEPS=50000
DECAY=0.99

# ---- Student config ---------------------------------------------------------
STUDENT_STEPS=10000   # Increase to 100k for full training

# ---- Env --------------------------------------------------------------------
export PATH=/root/dice_env/bin:$PATH
export PYTHONPATH=$FAIRSEQ_DIR:$PYTHONPATH

mkdir -p "$CODEBOOK_DIR" "$LABEL_DIR" "$STUDENT_BASE_DIR"

echo "============================================================"
echo "Codebook Distillation Ablation Study"
echo "  GPU: $GPU | Steps: CB=$CB_STEPS, Student=$STUDENT_STEPS"
echo "  Experiments: $RUN_EXPS"
echo "============================================================"


# =============================================================================
# Phase 1: Train 4 EMA codebooks (L8-K32, L8-K4096, L9-K32, L9-K4096)
# =============================================================================
CODEBOOK_CKPT="$CODEBOOK_DIR/codebooks_latest.pt"

if [ "$SKIP_CODEBOOK_TRAIN" -eq 1 ] && [ -f "$CODEBOOK_CKPT" ]; then
    echo ""
    echo "[Phase 1] Skipping codebook training (found: $CODEBOOK_CKPT)"
else
    echo ""
    echo "[Phase 1] Training 4 EMA codebooks (steps=$CB_STEPS)..."
    python "$FAIRSEQ_DIR/scripts/train_codebooks.py" \
        --teacher_ckpt "$TEACHER_CKPT" \
        --tsv "$TSV_DIR/train.tsv" \
        --codebooks "8:32,8:4096,9:32,9:4096" \
        --decay $DECAY \
        --steps $CB_STEPS \
        --out_dir "$CODEBOOK_DIR" \
        --device cuda \
        --log_interval 200 \
        --save_interval 10000
    echo "[Phase 1] Done. Codebooks saved at $CODEBOOK_CKPT"
fi


# =============================================================================
# Phase 2: Generate all 4 label files per split
# =============================================================================
NEED_LABELS=0
for NAME in l8k32 l8k4096 l9k32 l9k4096; do
    if [ ! -f "$LABEL_DIR/train.${NAME}" ]; then
        NEED_LABELS=1
        break
    fi
done

if [ "$SKIP_LABEL_GEN" -eq 1 ] && [ "$NEED_LABELS" -eq 0 ]; then
    echo ""
    echo "[Phase 2] Skipping label generation (all label files found)"
else
    echo ""
    echo "[Phase 2] Generating label files for all 4 codebooks..."
    for SPLIT in train valid; do
        if [ ! -f "$TSV_DIR/${SPLIT}.tsv" ]; then
            echo "  Skipping $SPLIT (TSV not found)"
            continue
        fi
        echo "  Processing split: $SPLIT"
        python "$FAIRSEQ_DIR/scripts/apply_codebooks.py" \
            --teacher_ckpt "$TEACHER_CKPT" \
            --codebook_ckpt "$CODEBOOK_CKPT" \
            --tsv "$TSV_DIR/${SPLIT}.tsv" \
            --split "$SPLIT" \
            --out_dir "$LABEL_DIR" \
            --device cuda
    done
    echo "[Phase 2] Done. Label files:"
    ls -lh "$LABEL_DIR/"
fi


# =============================================================================
# Phase 3: Train each student experiment
# =============================================================================
CONFIG_DIR="$FAIRSEQ_DIR/examples/hubert/config/pretrain"

run_student() {
    local EXP_NAME=$1
    local CONFIG_NAME=$2
    local EXP_DIR="$STUDENT_BASE_DIR/$EXP_NAME"

    echo ""
    echo "------------------------------------------------------------"
    echo "[Student: $EXP_NAME] config=$CONFIG_NAME | steps=$STUDENT_STEPS"
    echo "------------------------------------------------------------"

    mkdir -p "$EXP_DIR"

    python "$FAIRSEQ_DIR/fairseq_cli/hydra_train.py" \
        --config-dir "$CONFIG_DIR" \
        --config-name "$CONFIG_NAME" \
        task.data="$TSV_DIR" \
        task.label_dir="$LABEL_DIR" \
        optimization.max_update=$STUDENT_STEPS \
        distributed_training.distributed_port=$((29680 + RANDOM % 100)) \
        hydra.run.dir="$EXP_DIR"

    echo "[Student: $EXP_NAME] Done → $EXP_DIR"
}

echo ""
echo "[Phase 3] Training student models..."

for EXP in $RUN_EXPS; do
    case "$EXP" in
        vevo)
            run_student "vevo" "hubert_codebook_vevo"
            ;;
        crosslayer)
            run_student "crosslayer" "hubert_codebook_crosslayer"
            ;;
        l8same)
            run_student "l8same" "hubert_codebook_l8same"
            ;;
        l9k32)
            run_student "l9k32" "hubert_codebook_l9k32"
            ;;
        l9k4096)
            run_student "l9k4096" "hubert_codebook_l9k4096"
            ;;
        *)
            echo "  Unknown experiment: $EXP (skipping)"
            ;;
    esac
done


echo ""
echo "============================================================"
echo "Ablation study complete!"
echo "  Codebooks : $CODEBOOK_DIR"
echo "  Labels    : $LABEL_DIR"
echo "  Students  : $STUDENT_BASE_DIR"
echo ""
echo "Results summary:"
for EXP in $RUN_EXPS; do
    LAST_LOG="$STUDENT_BASE_DIR/$EXP/train.log"
    if [ -f "$LAST_LOG" ]; then
        LAST=$(grep -oP '"loss":\s*\K[\d.]+' "$LAST_LOG" | tail -1 || echo "N/A")
        echo "  $EXP: last_loss=$LAST"
    fi
done
echo "============================================================"
