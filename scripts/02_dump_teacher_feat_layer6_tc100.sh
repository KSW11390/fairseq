#!/usr/bin/env bash
# Dump HuBERT teacher (iter-2) layer-6 features for train-clean-100.
#
# Usage:
#   bash scripts/02_dump_teacher_feat_layer6_tc100.sh
#
# Required env vars:
#   FAIRSEQ_ROOT       – fairseq repo root
#   TSV_DIR            – directory containing {train,valid}.tsv
#   FEAT_DIR           – output directory for features
#   TEACHER_CKPT_PATH  – path to HuBERT iter-2 checkpoint
#
# Optional env vars:
#   LAYER      – teacher layer to extract (default: 6, 1-based)
#   NSHARD     – number of shards (default: 1)
#   RANK       – shard rank (default: 0)
#   MAX_CHUNK  – max audio chunk in samples (default: 1600000, ~100s at 16kHz)

set -euo pipefail

: "${FAIRSEQ_ROOT:?Set FAIRSEQ_ROOT}"
: "${TSV_DIR:?Set TSV_DIR}"
: "${FEAT_DIR:?Set FEAT_DIR}"
: "${TEACHER_CKPT_PATH:?Set TEACHER_CKPT_PATH}"

LAYER="${LAYER:-6}"
NSHARD="${NSHARD:-1}"
RANK="${RANK:-0}"
MAX_CHUNK="${MAX_CHUNK:-1600000}"

SCRIPT_DIR="${FAIRSEQ_ROOT}/examples/hubert/simple_kmeans"

mkdir -p "${FEAT_DIR}"

echo "=== Dumping teacher features ==="
echo "  TSV_DIR          = ${TSV_DIR}"
echo "  FEAT_DIR         = ${FEAT_DIR}"
echo "  TEACHER_CKPT     = ${TEACHER_CKPT_PATH}"
echo "  LAYER            = ${LAYER} (1-based)"
echo "  NSHARD/RANK      = ${NSHARD}/${RANK}"
echo "  MAX_CHUNK        = ${MAX_CHUNK}"

for SPLIT in train valid; do
    echo "--- Dumping ${SPLIT} features ---"
    python "${SCRIPT_DIR}/dump_hubert_feature.py" \
        "${TSV_DIR}" \
        "${SPLIT}" \
        "${TEACHER_CKPT_PATH}" \
        "${LAYER}" \
        "${NSHARD}" \
        "${RANK}" \
        "${FEAT_DIR}" \
        --max_chunk "${MAX_CHUNK}"
    echo "--- ${SPLIT} done ---"
done

echo "=== Feature dump complete. Files in ${FEAT_DIR}: ==="
ls -lh "${FEAT_DIR}"/
