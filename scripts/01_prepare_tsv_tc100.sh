#!/usr/bin/env bash
# Prepare LibriSpeech train-clean-100 TSV manifests for HuBERT.
# Produces {train,valid}.tsv in TSV_DIR.
#
# Usage:
#   bash scripts/01_prepare_tsv_tc100.sh
#
# Required env vars:
#   DATA_ROOT   – root of LibriSpeech (contains train-clean-100/, dev-clean/, etc.)
#   TSV_DIR     – output directory for TSV files
#
# The TSV format (fairseq-style):
#   Line 1: root directory
#   Lines 2+: relative_path<TAB>num_samples

set -euo pipefail

: "${DATA_ROOT:?Set DATA_ROOT to LibriSpeech root (e.g. /data/LibriSpeech)}"
: "${TSV_DIR:?Set TSV_DIR to output directory for TSV files}"

mkdir -p "${TSV_DIR}"

# ---------- helper ----------
make_tsv() {
    local audio_dir="$1"   # e.g. /data/LibriSpeech/train-clean-100
    local out_tsv="$2"     # e.g. /output/train.tsv
    local root_dir="$3"    # root written to first line of TSV

    echo "${root_dir}" > "${out_tsv}"

    find "${audio_dir}" -name "*.flac" | sort | while read -r fpath; do
        # Get relative path from root
        rel_path="${fpath#${root_dir}/}"
        # Get number of samples via soxi or python
        n_samples=$(python3 -c "
import soundfile as sf
info = sf.info('${fpath}')
print(info.frames)
")
        echo -e "${rel_path}\t${n_samples}" >> "${out_tsv}"
    done

    n_lines=$(( $(wc -l < "${out_tsv}") - 1 ))
    echo "[INFO] Wrote ${n_lines} entries to ${out_tsv}"
}

# ---------- train split: train-clean-100 ----------
echo "=== Preparing train.tsv from train-clean-100 ==="
make_tsv "${DATA_ROOT}/train-clean-100" "${TSV_DIR}/train.tsv" "${DATA_ROOT}"

# ---------- valid split: dev-clean ----------
echo "=== Preparing valid.tsv from dev-clean ==="
make_tsv "${DATA_ROOT}/dev-clean" "${TSV_DIR}/valid.tsv" "${DATA_ROOT}"

echo "=== Done. TSV files in ${TSV_DIR} ==="
ls -lh "${TSV_DIR}"/*.tsv
