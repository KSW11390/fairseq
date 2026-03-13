# DICEHuBERT Reproduction — train-clean-100 Scale

Teacher: HuBERT-base iter-2 (12L/768D, ~95M) | Student: 12L/384D (~27M) | Layer: 6 | K: 500 | T: 5 | GPU: 1×A100

**Goal**: DICEHuBERT 학습 재현 후 Teacher vs Student SUPERB 비교 분석

## Prerequisites

```bash
pip install soundfile joblib scikit-learn wandb tensorboard s3prl
# fairseq must be installed (pip install -e .)
```

Set environment variables (adjust paths):

```bash
export FAIRSEQ_ROOT=/path/to/fairseq
export DATA_ROOT=/path/to/LibriSpeech          # contains train-clean-100/, dev-clean/
export TSV_DIR=/path/to/output/tsv
export FEAT_DIR=/path/to/output/feat_layer6
export LABEL_DIR=/path/to/output/labels_k500
export TEACHER_CKPT_PATH=/path/to/hubert_base_iter2.pt
export EXP_DIR=/path/to/output/experiments
```

---

## Phase 1: Data Preparation

### Step 1: Prepare TSV manifests

```bash
bash ${FAIRSEQ_ROOT}/scripts/01_prepare_tsv_tc100.sh
```

Produces: `${TSV_DIR}/{train,valid}.tsv`

### Step 2: Dump teacher layer-6 features

```bash
bash ${FAIRSEQ_ROOT}/scripts/02_dump_teacher_feat_layer6_tc100.sh
```

Produces: `${FEAT_DIR}/{train,valid}_0_1.{npy,len}`

### Step 3: K-means (k=500) train + apply labels

```bash
bash ${FAIRSEQ_ROOT}/scripts/03_kmeans_k500_tc100_train_or_apply.sh
```

Produces:
- `${LABEL_DIR}/kmeans_k500.bin`
- `${LABEL_DIR}/{train,valid}.km`
- `${LABEL_DIR}/dict.km.txt`

---

## Phase 2: DICEHuBERT Training

```bash
python ${FAIRSEQ_ROOT}/scripts/05_wandb_wrapper.py \
    --wandb_project hubert-distill \
    --wandb_run_name "tc100_dicehubert_10k_T5" \
    -- \
    fairseq-hydra-train \
    --config-dir ${FAIRSEQ_ROOT}/examples/hubert/config/pretrain \
    --config-name hubert_student_distill_layer6_k500_tc100 \
    task.data=${TSV_DIR} \
    task.label_dir=${LABEL_DIR} \
    criterion.label_mode=soft_teacher_logits \
    criterion.teacher_path=${TEACHER_CKPT_PATH} \
    criterion.distill_temperature=5.0 \
    optimization.max_update=10000 \
    common.tensorboard_logdir=${EXP_DIR}/dicehubert/tblog \
    hydra.run.dir=${EXP_DIR}/dicehubert \
    hydra.sweep.dir=${EXP_DIR}/dicehubert
```

---

## Phase 3: SUPERB Evaluation — Teacher vs Student

두 모델을 동일한 SUPERB 태스크에서 평가하여 비교.

### Evaluated Tasks

| Category | Task | Downstream | Metric |
|----------|------|-----------|--------|
| Content | PR (Phoneme Recognition) | ctc | PER ↓ |
| Content | ASR (Speech Recognition) | asr | WER ↓ |
| Content | KS (Keyword Spotting) | speech_commands | Acc ↑ |
| Semantics | IC (Intent Classification) | fluent_commands | Acc ↑ |
| Semantics | SF (Slot Filling) | snips | F1 ↑ / CER ↓ |
| Speaker | SID (Speaker Identification) | voxceleb1 | Acc ↑ |
| Speaker | ASV (Speaker Verification) | sv_voxceleb1 | EER ↓ |
| Paralinguistics | ER (Emotion Recognition) | emotion | Acc ↑ |

### Run SUPERB: Teacher (HuBERT-base)

```bash
export CKPT_PATH=${TEACHER_CKPT_PATH}
export TASK=PR,ASR,KS,IC,SF,SID,ASV,ER
export RUN_NAME=teacher
export EXP_ROOT=${EXP_DIR}/superb
BACKGROUND=1 bash ${FAIRSEQ_ROOT}/scripts/04_superb_eval.sh
```

### Run SUPERB: Student (DICEHuBERT)

```bash
export CKPT_PATH=${EXP_DIR}/dicehubert/checkpoint_best.pt
export TASK=PR,ASR,KS,IC,SF,SID,ASV,ER
export RUN_NAME=dicehubert
export EXP_ROOT=${EXP_DIR}/superb
BACKGROUND=1 bash ${FAIRSEQ_ROOT}/scripts/04_superb_eval.sh
```

### Results Comparison

평가 완료 후 두 summary 파일 비교:

```bash
echo "=== Teacher ===" && cat ${EXP_DIR}/superb/teacher_summary.txt
echo "=== DICEHuBERT ===" && cat ${EXP_DIR}/superb/dicehubert_summary.txt
```

**Expected comparison table (to fill after eval):**

| Task | Metric | Teacher (95M) | DICEHuBERT (27M) | Gap |
|------|--------|:---:|:---:|:---:|
| PR | PER ↓ | | | |
| ASR | WER ↓ | | | |
| KS | Acc ↑ | | | |
| IC | Acc ↑ | | | |
| SF | F1 ↑ | | | |
| SID | Acc ↑ | | | |
| ASV | EER ↓ | | | |
| ER | Acc ↑ | | | |

---

## W&B Metrics (Training Phase)

| W&B metric name  | Source                   |
|-------------------|--------------------------|
| `train_loss`      | overall training loss    |
| `masked_kl`       | masked KL divergence     |
| `teacher_entropy` | H(teacher soft targets)  |
| `correct_m_0`     | masked prediction acc    |
| `loss_features_pen` | feature extractor penalty |
| `ppl`             | perplexity               |
| `lr`              | learning rate            |
| `grad_norm`       | gradient norm            |
| `wps` / `ups`     | words/updates per second |

---

## Common Failure Modes Checklist

### 1. label_rate mismatch
- HuBERT iter-2 features produce 50Hz frame rate (label_rate=50).
- MFCC features produce 100Hz (iter-1). Do NOT mix.
- Check: `model.label_rate` and `task.label_rate` must both be `50`.

### 2. Layer index confusion (0-index vs 1-index)
- `dump_hubert_feature.py` uses **1-based** layer index.
- `model.extract_features(output_layer=N)` internally converts to 0-based: `layer=N-1`.
- Layer 6 means `--layer 6` in the script (extracts 6th transformer layer output).

### 3. Time dimension mismatch
- Student and teacher must use the same conv feature extractor architecture
  (same strides → same temporal downsampling).
- The criterion asserts `T_student == T_teacher`. If this fires, the student
  has a different feature extractor stride.
- Labels are aligned via `feat2tar_ratio = label_rate * feature_ds_rate / sample_rate`.

### 4. fp16 instability with teacher forward pass
- By default, teacher runs in fp32 (`criterion.teacher_fp16=false`).
- If you enable `criterion.teacher_fp16=true` and see NaN losses, disable it.
- The student's own fp16 training (`common.fp16=true`) is separate and usually fine.

### 5. Shard merge ordering issues
- When using `nshard > 1`, labels must be merged in rank order (0, 1, ..., nshard-1).
- The script `03_*` handles this correctly with `seq 0 $((NSHARD-1))`.
- Verify: `wc -l train.km` should equal `wc -l train.tsv - 1` (minus header).

### 6. dict.km.txt format
- Must have entries `0 1`, `1 1`, ..., `499 1` (one per cluster).
- Total lines = K = 500.
- If wrong, fairseq will fail with dictionary size mismatch.

### 7. VRAM with teacher in memory
- soft_teacher_logits loads the teacher model (~360MB for base).
- Student (~27M) + teacher (~95M) fit comfortably on A100 (80GB).
- If tight on memory, reduce `dataset.max_tokens` or enable `criterion.teacher_fp16=true`.
