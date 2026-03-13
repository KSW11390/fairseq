#!/usr/bin/env bash
# Train k-means (k=500) on teacher layer-6 features, then apply to produce .km labels.
#
# Usage:
#   bash scripts/03_kmeans_k500_tc100_train_or_apply.sh
#
# Required env vars:
#   FAIRSEQ_ROOT   – fairseq repo root
#   FEAT_DIR       – directory with dumped features ({split}_{rank}_{nshard}.npy/.len)
#   LABEL_DIR      – output directory for labels ({train,valid}.km + dict.km.txt)
#
# Optional env vars:
#   KMEANS_PATH    – path to existing kmeans model; if set, skip training
#   K              – number of clusters (default: 500)
#   NSHARD         – number of shards (default: 1)
#   RANK           – shard rank (default: 0)

set -euo pipefail

: "${FAIRSEQ_ROOT:?Set FAIRSEQ_ROOT}"
: "${FEAT_DIR:?Set FEAT_DIR}"
: "${LABEL_DIR:?Set LABEL_DIR}"

K="${K:-500}"
NSHARD="${NSHARD:-1}"
RANK="${RANK:-0}"
KMEANS_PATH="${KMEANS_PATH:-${LABEL_DIR}/kmeans_k${K}.bin}"

SCRIPT_DIR="${FAIRSEQ_ROOT}/examples/hubert/simple_kmeans"

mkdir -p "${LABEL_DIR}"

# ---------- Step 1: Train k-means (or skip if model exists) ----------
if [ -f "${KMEANS_PATH}" ]; then
    echo "=== K-means model found at ${KMEANS_PATH}, skipping training ==="
else
    echo "=== Training k-means (k=${K}) on train split ==="
    python "${SCRIPT_DIR}/learn_kmeans.py" \
        "${FEAT_DIR}" \
        "train" \
        "${NSHARD}" \
        "${KMEANS_PATH}" \
        "${K}" \
        --seed 0 \
        --percent -1 \
        --max_iter 150 \
        --batch_size 10000
    echo "=== K-means training complete: ${KMEANS_PATH} ==="
fi

# ---------- Step 2: Apply k-means to generate .km labels ----------
for SPLIT in train valid; do
    echo "--- Applying k-means to ${SPLIT} ---"
    for R in $(seq 0 $((NSHARD - 1))); do
        python "${SCRIPT_DIR}/dump_km_label.py" \
            "${FEAT_DIR}" \
            "${SPLIT}" \
            "${KMEANS_PATH}" \
            "${NSHARD}" \
            "${R}" \
            "${LABEL_DIR}"
    done

    # Merge shards into single file
    echo "--- Merging ${SPLIT} label shards ---"
    > "${LABEL_DIR}/${SPLIT}.km"
    for R in $(seq 0 $((NSHARD - 1))); do
        cat "${LABEL_DIR}/${SPLIT}_${R}_${NSHARD}.km" >> "${LABEL_DIR}/${SPLIT}.km"
    done
    echo "  $(wc -l < "${LABEL_DIR}/${SPLIT}.km") utterances in ${SPLIT}.km"
done

# ---------- Step 3: Create dict.km.txt ----------
echo "=== Creating dict.km.txt ==="
python3 -c "
for i in range(${K}):
    print(f'{i} 1')
" > "${LABEL_DIR}/dict.km.txt"
echo "  dict.km.txt has $(wc -l < "${LABEL_DIR}/dict.km.txt") entries"

# ---------- Verify ----------
echo ""
echo "=== Verification ==="
echo "Label dir contents:"
ls -lh "${LABEL_DIR}"/*.km "${LABEL_DIR}"/dict.km.txt 2>/dev/null || true

echo ""
echo "Train .km line count vs train.tsv line count (should match, minus header):"
echo "  train.km lines:  $(wc -l < "${LABEL_DIR}/train.km")"
if [ -f "${LABEL_DIR}/../tsv/train.tsv" ]; then
    echo "  train.tsv lines: $(( $(wc -l < "${LABEL_DIR}/../tsv/train.tsv") - 1 ))"
fi

echo ""
echo "=== Done. Labels ready in ${LABEL_DIR} ==="
