#!/bin/bash
# =============================================================================
# Codebook Distillation — Full Pipeline (3 phases)
#
# Phase 1: Train EMA codebooks on teacher's layer 8 & 9 representations
# Phase 2: Apply codebooks to generate discrete label files
# Phase 3: Train student with codebook labels (standard HuBERT pretraining)
#
# Usage:
#   bash scripts/run_codebook_distill_pipeline.sh [GPU_ID]
# =============================================================================

set -euo pipefail

GPU=${1:-0}
export CUDA_VISIBLE_DEVICES=$GPU

# ---- Paths (edit these) -----------------------------------------------------
FAIRSEQ_DIR="/root/fairseq"
DATA_DIR="/root/data/LibriSpeech"
TSV_DIR="/root/output/data/tc100"          # contains train.tsv, valid.tsv
TEACHER_CKPT="/root/output/ckpt/hubert_base_ls960.pt"
OUTPUT_DIR="/root/output/experiments/codebook_distill"

CODEBOOK_DIR="$OUTPUT_DIR/codebooks"
LABEL_DIR="$OUTPUT_DIR/labels"
STUDENT_EXP_DIR="$OUTPUT_DIR/student"

# ---- Codebook config ---------------------------------------------------------
LAYER_A=8       # Teacher layer for codebook A
K_A=4096        # Codebook A size (fine-grained)
LAYER_B=9       # Teacher layer for codebook B
K_B=32          # Codebook B size (coarse)
DECAY=0.99
CB_STEPS=50000  # EMA update steps for codebook training

# ---- Student config ----------------------------------------------------------
STUDENT_STEPS=10000   # Increase to 100k for full training

# ---- Env ---------------------------------------------------------------------
export PATH=/root/dice_env/bin:$PATH
export PYTHONPATH=$FAIRSEQ_DIR:$PYTHONPATH

mkdir -p "$CODEBOOK_DIR" "$LABEL_DIR" "$STUDENT_EXP_DIR"

echo "============================================================"
echo "Codebook Distillation Pipeline"
echo "  GPU: $GPU | Layer A=$LAYER_A (K=$K_A) | Layer B=$LAYER_B (K=$K_B)"
echo "============================================================"


# =============================================================================
# Phase 1: Train EMA codebooks
# =============================================================================
echo ""
echo "[Phase 1] Training EMA codebooks (steps=$CB_STEPS)..."

python "$FAIRSEQ_DIR/scripts/train_codebooks.py" \
    --teacher_ckpt "$TEACHER_CKPT" \
    --tsv "$TSV_DIR/train.tsv" \
    --codebooks "${LAYER_A}:${K_A},${LAYER_B}:${K_B}" \
    --decay $DECAY \
    --steps $CB_STEPS \
    --out_dir "$CODEBOOK_DIR" \
    --device cuda \
    --log_interval 200 \
    --save_interval 10000

CODEBOOK_CKPT="$CODEBOOK_DIR/codebooks_latest.pt"
echo "[Phase 1] Done. Codebooks saved at $CODEBOOK_CKPT"


# =============================================================================
# Phase 2: Generate label files
# =============================================================================
echo ""
echo "[Phase 2] Generating label files..."

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

# apply_codebooks.py already writes files as <split>.<name> (e.g. train.l8k4096)
# No renaming needed.

echo "[Phase 2] Done. Label files:"
ls -lh "$LABEL_DIR/"


# =============================================================================
# Phase 3: Student pretraining with codebook labels
# =============================================================================
echo ""
echo "[Phase 3] Training student (steps=$STUDENT_STEPS)..."

python "$FAIRSEQ_DIR/fairseq_cli/hydra_train.py" \
    --config-dir "$FAIRSEQ_DIR/examples/hubert/config/pretrain" \
    --config-name hubert_student_codebook_distill \
    task.data="$TSV_DIR" \
    task.label_dir="$LABEL_DIR" \
    optimization.max_update=$STUDENT_STEPS \
    hydra.run.dir="$STUDENT_EXP_DIR"

echo ""
echo "============================================================"
echo "Pipeline complete!"
echo "  Codebooks: $CODEBOOK_DIR"
echo "  Labels:    $LABEL_DIR"
echo "  Student:   $STUDENT_EXP_DIR"
echo "============================================================"
