# HuBERT Student Distillation — Experiment Log

**Server**: 194.68.245.207:22051 (2×A40 46GB)
**Data**: LibriSpeech train-clean-100 (TC-100, ~100h)
**Student**: HuBERT-Small (12L / 384D / 6H, ~27M params)
**Teacher**: HuBERT Base (12L / 768D, 94.7M params)
**Codebooks**: RepCodec (EMA VQ), 24 total — layers 1-12, k=32 + k=512
**Codebook training**: 50k steps, completed on 2026-03-01

---

## Experiment Overview

| # | Name | Method | Codebook | Masking | Status |
|---|------|--------|----------|---------|--------|
| 1 | exp_hard_k32 | Hard CE, all 12 layers | k=32 | mask_prob=0.80 | ✅ DONE (2026-03-01 23:53 KST) |
| 2 | exp_hard_k512 | Hard CE, all 12 layers | k=512 | mask_prob=0.80 | ✅ DONE (2026-03-02 08:44 KST) |
| 3 | exp_hard_both | Hard CE, all 12 layers | k=32 + k=512 | mask_prob=0.80 | ✅ DONE (2026-03-02 23:31 KST) |
| 4 | exp_soft_k32 | Soft KL (probe), last layer | k=32 | mask_prob=0.0 (ALL frames) | ✅ DONE (2026-03-04 07:36 KST) |
| 5 | exp_soft_k512 | Soft KL (probe), last layer | k=512 | mask_prob=0.0 (ALL frames) | ✅ DONE (2026-03-04 07:20 KST) |
| 6 | exp_soft_both | Soft KL (probe), last layer | k=32 + k=512 | mask_prob=0.0 (ALL frames) | ✅ DONE (2026-03-04 11:25 KST) |
| 7 | exp_nomask_k32 | Hard CE, all 12 layers | k=32 | mask_prob=0.0 (ALL frames) | ✅ DONE (2026-03-03 09:38 KST) |
| 8 | exp_nomask_k512 | Hard CE, all 12 layers | k=512 | mask_prob=0.0 (ALL frames) | ✅ DONE (2026-03-03 14:40 KST) |
| 9 | exp_nomask_both | Hard CE, all 12 layers | k=32 + k=512 | mask_prob=0.0 (ALL frames) | ✅ DONE (2026-03-04 13:02 KST) |
| 10 | exp_nomask_k500 | Hard CE, layer 12 only | k-means k=500 (teacher L9) | mask_prob=0.0 (ALL frames) | ✅ DONE (2026-03-10 08:50 KST) |

---

## Research Questions

- Does masked token prediction (HuBERT-style 80% masking) help or hurt distillation?
- Does codebook resolution (k=32 fine vs k=512 coarse) affect downstream phonetic/semantic tasks?
- Does supervising all 12 student layers (layer-wise) outperform single-layer supervision?
- Hard CE (discrete targets) vs Soft KL (teacher probe soft distributions)?
- Does DICEHuBERT-style single-layer (L12→k-means k=500 teacher L9, no masking) match layer-wise RepCodec distillation?

**Ablation pair**: exp_hard_k32 vs exp_nomask_k32 / exp_hard_k512 vs exp_nomask_k512
→ Isolates the effect of masking while keeping everything else identical.

**New ablation (Exp 10)**: DICEHuBERT-style, no masking
→ `exp_nomask_k500`: student L12 → teacher L9 k-means k=500, all frames supervised (no masking).
→ Tests whether single-layer k-means supervision (DICEHuBERT objective) matches layer-wise RepCodec supervision.

---

## Experiment Details

### Common Configuration (all experiments)

| Param | Value |
|-------|-------|
| Student architecture | 12L / 384D / 6H |
| Max training steps | 100,000 |
| Optimizer | Adam (β=0.9/0.98, ε=1e-6, wd=0.01) |
| LR schedule | Polynomial decay, peak=5e-4, warmup=8k steps |
| Gradient clipping | 10.0 |
| fp16 | Enabled |
| encoder_layerdrop | 0.05 |
| feature_grad_mult | 0.1 |
| Checkpoint interval | every 2,000 steps |
| Label rate | 50 Hz (1 label per 20ms) |
| Sample rate | 16,000 Hz |

### Exp 1–3: Hard CE (Masked)

| Param | Value |
|-------|-------|
| Criterion | `hubert_layerwise` |
| Loss | Cross-Entropy at masked positions |
| mask_prob | 0.80 |
| max_tokens (k32) | 1,000,000 (tokens/batch) |
| max_tokens (k512) | 300,000 (reduced for OOM) |
| max_tokens (both) | 300,000 |

### Exp 4–6: Soft KL (No Mask)

| Param | Value |
|-------|-------|
| Criterion | `hubert_probe_distill` |
| Loss | KL divergence: student log-softmax vs teacher probe soft labels |
| mask_prob | 0.0 (no masking) |
| loss_on_all_frames | true |
| temperature | 2.0 (Hinton 2015 — softer distributions) |
| max_tokens | 250,000 |
| Student heads | Linear(384→K) at layer 12 only (l12k32, l12k512) |
| Teacher probes | Frozen Linear(768→K) trained on HuBERT-Base layer-12 output to predict k-means labels |

**Method**: Teacher's last-layer hidden states → frozen probe → soft distribution over K codebook entries → KL divergence against student's last-layer head. Unlike Hard CE, student learns a soft distribution (probabilities) instead of hard one-hot targets.

### Exp 7–9: Hard CE (No Mask)

| Param | Value |
|-------|-------|
| Criterion | `hubert_layerwise` + `loss_on_all_frames: true` |
| Loss | Cross-Entropy at ALL non-padding frames |
| mask_prob | 0.0 (no masking) |
| max_tokens (nomask_k32) | 500,000 |
| max_tokens (nomask_k512) | 250,000 (conservative for full-frame 512-class logits) |
| max_tokens (nomask_both) | 250,000 |

**Key difference**: With masking, loss is computed on ~80% of frames. Without masking, loss is on 100% of frames — model sees full clean context (no [MASK] tokens), similar to a standard frame-level classification task.

### Exp 10: exp_nomask_k500

| Param | Value |
|-------|-------|
| Criterion | `hubert_layerwise` + `loss_on_all_frames: true` |
| Loss | CE at ALL non-padding frames (student layer 12 only) |
| mask_prob | 0.0 (no masking) |
| Supervised layers | Layer 12 only (student) → k-means k=500 from teacher layer 9 |
| Codebook | k-means k=500, trained on teacher HuBERT-Base layer 9 features from TC-100 |
| Label files | `train.l9k500`, `valid.l9k500`, `dict.l9k500.txt` |
| layerwise_cls_specs | `"9:500"` → creates `l9k500` classifier |
| layer_specs | `"12:l9k500"` → criterion hooks student layer 12 |
| skip_masked | true |
| skip_nomask | true |
| max_tokens | 1,000,000 |
| warmup_updates | 8,000 |

**Method**: DICEHuBERT-style single-layer objective but without masking. Student layer 12 predicts teacher layer-9 k-means cluster IDs for ALL frames. Contrasts with exp_nomask_* (all-layer RepCodec) to isolate the effect of single-layer vs layer-wise supervision, and codebook type (k-means vs RepCodec).

---

## Criterion: `hubert_layerwise`

**File**: `fairseq/criterions/hubert_layerwise_criterion.py`

- Each student transformer layer L has a Linear(384, K) head predicting layer L's codebook label
- loss = Σ_layers CE(student_layer_out[positions], target[positions]) / N_positions
- positions = masked & non-padding frames (default) OR all non-padding frames (loss_on_all_frames=True)
- Logging: `loss_l{L}k{K}`, `acc_l{L}k{K}`, `ent_l{L}k{K}` per layer
- sample_size=1 in logging_output to prevent double-normalization

**Important**: fairseq Dictionary adds 4 special tokens (bos/pad/eos/unk) before real tokens.
criterion subtracts `nspecial=4` from targets to recover 0-based class indices.

---

## Bugs Fixed During Training

| Date (KST) | Bug | Fix |
|---|---|---|
| 2026-03-01 | `ValueError: the entire sequence is masked. sz=6` (recurring crash) | Patched `data_utils.py:compute_mask_indices` to truncate mask to `sz-1` instead of raising |
| 2026-03-01 | OOM on exp_hard_k512 (VRAM 97%) | Reduced max_tokens 1M→300k; clean restart |
| 2026-03-01 | 28 fairseq processes running simultaneously | Fixed wrong dir names in run script (hard_k32 → exp_hard_k32); added skip logic |
| 2026-03-01 | Pipeline log stuck at step 300 | fairseq-hydra writes to `{hydra.run.dir}/hydra_train.log`, not pipeline.log |
| 2026-03-03 | `TypeError: logical_and(): argument 'other' must be Tensor, not NoneType` at hubert.py:534 | `mask_prob=0.0` → `apply_mask` returns `mask_indices=None` → logical_and crashes. Fixed by `skip_masked: true`, `skip_nomask: true` in nomask YAMLs to bypass that code path entirely |
| 2026-03-03 | `RuntimeError: Cannot determine frame dimensions` during validation in exp_nomask_k32 | Validation batches with uniform-length sequences have `padding_mask=None`; combined with `mask_indices=None` (no masking) both are simultaneously None. Fixed criterion to fall back to `layer_outputs` shape to build all-True `eff_mask` |
| 2026-03-04 | `hubert_probe_distill_criterion.py` crashed with `RuntimeError: mask_indices is None` when `mask_prob=0.0` | Added `loss_on_all_frames` config flag; when True (or mask_indices is None), computes KL on all non-padding frames instead of masked positions only. Also fixed `mask_holder` capture via `apply_mask` method patching. |
| 2026-03-04 | exp_soft_both watchdog script never started due to bash syntax error in `run_soft_sequential.sh` | `log "$name COMPLETED (process exited cleanly)"` — `(` is a special character. Manually started exp_soft_both. |

---

## Results

### Exp 1: exp_hard_k32 — FINAL RESULTS

**Completed**: 2026-03-01 23:53 KST
**Total steps**: 100,001
**Config**: `hubert_layerwise_k32.yaml`

#### Metrics at step 100,001 — FINAL

| Layer | Train Acc (avg) | Val Acc |
|-------|-----------------|---------|
| l1k32 | 29.42% | 31.44% |
| l2k32 | 30.15% | 32.80% |
| l3k32 | 29.96% | 32.32% |
| l4k32 | 31.61% | 33.56% |
| l5k32 | 33.87% | 36.34% |
| l6k32 | 35.63% | 38.85% |
| l7k32 | 36.52% | **39.39%** |
| l8k32 | 36.00% | 38.95% |
| l9k32 | 36.05% | 38.79% |
| l10k32 | 36.95% | 39.48% |
| l11k32 | 37.47% | **39.95%** |
| l12k32 | 37.16% | 39.74% |

- Random baseline = 3.1% (1/32)
- Best val acc: l11k32=39.95%, l10k32=39.48%
- Val acc > Train acc throughout: masked training is harder than clean validation (expected)
- Train acc = running average over all 100k steps (from checkpoint AverageMeter)

---

### Exp 2: exp_hard_k512 — DONE

**Started**: 2026-03-02 00:53 KST (after OOM fixes)
**Completed**: 2026-03-02 08:44 KST
**Total steps**: 100,000
**Config**: `hubert_layerwise_k512.yaml` (max_tokens=300k)
**Training time**: ~29,609s (~8.2h)

#### Metrics at step 100,000 — FINAL

| Layer | Train Acc (avg) | Val Acc |
|-------|-----------------|---------|
| l1k512 | 7.82% | 11.36% |
| l2k512 | 8.03% | 13.04% |
| l3k512 | 8.14% | 13.24% |
| l4k512 | 8.25% | 12.55% |
| l5k512 | 8.42% | 12.18% |
| l6k512 | 9.11% | 13.58% |
| l7k512 | 9.68% | 14.54% |
| l8k512 | 9.82% | **14.91%** |
| l9k512 | 9.73% | 14.66% |
| l10k512 | 9.37% | 14.53% |
| l11k512 | 9.34% | 14.79% |
| l12k512 | 9.29% | 14.84% |

- Random baseline = 0.2% (1/512); val ~14% ≈ 70× above chance
- Best val acc: l8k512=14.91%
- GPU: 87%, ~40GB/46GB, OOM=0, loss_scale=256 (fp16 stable)
- Train acc = running average over all 100k steps

---

### Exp 3: exp_hard_both — DONE

**Started**: 2026-03-02 ~09:00 KST (after exp_hard_k512)
**Completed**: 2026-03-02 23:31 KST
**Total steps**: 100,000
**Config**: `hubert_layerwise_distill.yaml` (k=32 + k=512, 24 heads, max_tokens=300k)

#### Metrics at step 100,000 — FINAL

| Layer | Train Acc k32 (avg) | Train Acc k512 (avg) | Val Acc k32 | Val Acc k512 |
|-------|---------------------|----------------------|-------------|--------------|
| l1 | 19.89% | 7.28% | 25.65% | 11.34% |
| l2 | 19.92% | 7.52% | 26.28% | 13.25% |
| l3 | 19.20% | 7.63% | 25.25% | 13.28% |
| l4 | 19.81% | 7.77% | 25.69% | 12.68% |
| l5 | 20.81% | 7.95% | 27.17% | 12.40% |
| l6 | 21.29% | 8.59% | 28.77% | 13.64% |
| l7 | 21.88% | 9.15% | 28.75% | 14.47% |
| l8 | 21.91% | 9.30% | 28.80% | **14.96%** |
| l9 | 21.42% | 9.21% | 28.58% | 14.83% |
| l10 | 22.13% | 8.85% | 28.90% | 14.58% |
| l11 | 22.39% | 8.83% | **29.62%** | 14.88% |
| l12 | 21.60% | 8.76% | 29.04% | 15.00% |

- 24 classification heads (12×k32 + 12×k512)
- Best val acc: l11k32=29.62%, l12k512=15.00% (upper layers best)
- Train acc = running average over all 100k steps (train log 소실; checkpoint AverageMeter에서 복원)

---

### Exp 4: exp_soft_k32 — DONE

**Started**: 2026-03-04 (new server 194.68.245.207)
**Completed**: 2026-03-04 07:36 KST
**Total steps**: 100,000
**Config**: `hubert_probe_distill_k32_nomask.yaml` (mask_prob=0.0, loss_on_all_frames=true, temperature=2.0)
**Method**: Teacher HuBERT-Base layer-12 → frozen probe Linear(768→32) → soft labels; Student layer-12 → Linear(384→32) → KL div

#### Metrics at step 100,000 — FINAL

| Metric | Train | Val |
|--------|-------|-----|
| kl_k32 | 0.1665 | 0.1563 |
| acc_k32 | **73.66%** | **74.87%** |
| ent_k32 | 1.6995 | 1.681 |

- Random baseline = 3.1% (1/32); val 74.87% ≈ 24× above chance
- Note: Masking config changed from originally planned mask_prob=0.80 → mask_prob=0.0 (no masking) for fair comparison with exp_nomask_* series

---

### Exp 5: exp_soft_k512 — DONE

**Started**: 2026-03-04 (same server, parallel with k32)
**Completed**: 2026-03-04 07:20 KST
**Total steps**: 100,000
**Config**: `hubert_probe_distill_k512_nomask.yaml` (mask_prob=0.0, loss_on_all_frames=true, temperature=2.0)
**Method**: Teacher HuBERT-Base layer-12 → frozen probe Linear(768→512) → soft labels; Student layer-12 → Linear(384→512) → KL div

#### Metrics at step 100,000 — FINAL

| Metric | Train | Val |
|--------|-------|-----|
| kl_k512 | 0.3105 | 0.2822 |
| acc_k512 | **58.95%** | **60.52%** |
| ent_k512 | 2.800 | 2.8195 |

- Random baseline = 0.2% (1/512); val 60.52% ≈ 302× above chance
- k512 acc lower than k32 (60% vs 75%) — larger codebook is harder to predict

---

### Exp 6: exp_soft_both — DONE

**Started**: 2026-03-04 08:00 KST
**Completed**: 2026-03-04 11:25 KST
**Total steps**: 100,000
**Config**: `hubert_probe_distill_k32_k512_nomask.yaml` (mask_prob=0.0, loss_on_all_frames=true, temperature=2.0)
**Method**: Both k32 + k512 probes simultaneously; student has two last-layer heads (l12k32 + l12k512)

#### Metrics at step 100,000 — FINAL

| Metric | Train | Val |
|--------|-------|-----|
| kl_k32 | 0.1077 | 0.0974 |
| acc_k32 | **79.03%** | **80.49%** |
| ent_k32 | 1.6991 | 1.6783 |
| kl_k512 | 0.2891 | 0.2621 |
| acc_k512 | **60.10%** | **61.79%** |
| ent_k512 | 2.7999 | 2.8194 |

- Joint training (k32+k512) improves acc vs single-probe: k32 80.49% vs 74.87% (+5.6%), k512 61.79% vs 60.52% (+1.3%)
- Joint training acts as implicit regularization — shared encoder learns features useful for both granularities

---

### Exp 7: exp_nomask_k32 — DONE

**Started**: 2026-03-03 (resumed from step 30k after crash at step 32k)
**Completed**: 2026-03-03 09:38 KST
**Total steps**: 100,000
**Config**: `hubert_layerwise_k32_nomask.yaml` (mask_prob=0.0, loss_on_all_frames=true, max_tokens=500k)

#### Metrics at step 100,000 — FINAL

| Layer | Train Acc (avg) | Val Acc |
|-------|-----------------|---------|
| l1k32 | 71.59% | 70.34% |
| l2k32 | 71.79% | 71.22% |
| l3k32 | 71.00% | 70.46% |
| l4k32 | 71.75% | 70.95% |
| l5k32 | 74.30% | 73.96% |
| l6k32 | 76.61% | 76.70% |
| l7k32 | 76.52% | 76.40% |
| l8k32 | 74.77% | 74.89% |
| l9k32 | 75.05% | 74.85% |
| l10k32 | 76.46% | 76.25% |
| l11k32 | **78.32%** | **77.97%** |
| l12k32 | 77.58% | 77.22% |

- Random baseline = 3.1% (1/32); val ~77% ≈ 25× above chance
- Best val acc: l11k32=77.97%, l12k32=77.22%
- Train acc ≈ Val acc (unlike other experiments) — train avg includes early steps with lower acc; final val is snapshot. No systematic gap because no-mask removes the masking difficulty asymmetry.
- vs exp_hard_k32 (masked): val acc 77.97% vs 39.95% — masking dramatically reduces accuracy (~2× harder)

#### Crash History
- Crashed at step 32k during first validation run (criterion bug: `padding_mask=None` AND `mask_indices=None` simultaneously)
- Criterion patched; resumed from checkpoint_last.pt (step 30k)

---

### Exp 8: exp_nomask_k512 — DONE

**Started**: 2026-03-03 (after exp_hard_both)
**Completed**: 2026-03-03 14:40 KST
**Total steps**: 100,000
**Config**: `hubert_layerwise_k512_nomask.yaml` (mask_prob=0.0, loss_on_all_frames=true, max_tokens=250k)
**Training time**: 16,368s (~4.5h)

#### Metrics at step 100,000 — FINAL

| Layer | Train Acc (avg) | Val Acc |
|-------|-----------------|---------|
| l1k512 | 49.92% | 58.00% |
| l2k512 | 49.19% | 57.86% |
| l3k512 | 48.33% | 56.92% |
| l4k512 | 48.06% | 56.53% |
| l5k512 | 48.12% | 56.95% |
| l6k512 | 49.99% | 59.35% |
| l7k512 | 52.58% | 62.32% |
| l8k512 | 52.29% | **63.15%** |
| l9k512 | 50.57% | 61.55% |
| l10k512 | 50.69% | 61.26% |
| l11k512 | 51.90% | 62.92% |
| l12k512 | 52.20% | **63.30%** |

- Val acc consistently higher than train (no-mask mode: val uses full clean context, same as train — gap may reflect val set being easier on average)
- Best val acc: l12k512=63.30%, l8k512=63.15%
- No-masking dramatically increases accuracy vs masked (val ~63% vs ~15% for k512)
- Train acc = running average over all 100k steps

---

### Exp 9: exp_nomask_both — DONE

**Started**: 2026-03-04 08:04 KST
**Completed**: 2026-03-04 13:02 KST
**Total steps**: 100,000
**Training time**: ~4.89h (17,592s)
**Config**: `hubert_layerwise_both_nomask.yaml` (mask_prob=0.0, loss_on_all_frames=true, max_tokens=250k)
**Method**: Hard CE on all 12 layers × (k=32 + k=512) = 24 heads simultaneously; no masking

#### Metrics at step 95,000 — BEST (valid_best_loss=33.296)

| Layer | Val Acc k32 | Val Acc k512 |
|-------|-------------|--------------|
| l1 | 70.94% | 57.44% |
| l2 | 71.02% | 56.31% |
| l3 | 70.87% | 56.51% |
| l4 | 71.37% | 55.97% |
| l5 | 74.08% | 56.47% |
| l6 | 77.52% | 59.68% |
| l7 | 77.35% | 62.66% |
| l8 | 76.42% | 63.50% |
| l9 | 76.61% | 61.36% |
| l10 | 77.94% | 61.12% |
| l11 | **79.48%** | 62.87% |
| l12 | 78.75% | **63.36%** |

- Best valid_loss: 33.296 at 95k steps (100k slightly worse at 33.500)
- Best val acc: l11k32=79.48%, l12k512=63.36%
- Joint k32+k512 training: l12k32=78.75% (vs exp_nomask_k32 77.22% single), l12k512=63.36% (vs exp_nomask_k512 63.30% single)
- Slight improvement from joint training (consistent with exp_soft_both trend)

---


### Exp 10: exp_nomask_k500 — DONE

**Started**: 2026-03-10 05:27 KST
**Completed**: 2026-03-10 08:50 KST
**Total steps**: 100,000
**Training time**: 12,153s (~3.4h)
**Config**: `hubert_nomask_k500.yaml` (mask_prob=0.0, loss_on_all_frames=true, max_tokens=1M, encoder_layerdrop=0.0)
**Method**: DICEHuBERT-style single-layer supervision. Student layer 12 → k-means k=500 labels from teacher HuBERT-Base layer 9, all frames supervised (no masking).

#### Metrics at step 100,000 — FINAL

| Metric | Train Acc (final epoch avg) | Val Acc |
|--------|----------------------------|---------|
| l12→l9k500 | **72.87%** | **68.78%** |

- Random baseline = 0.2% (1/500); val 68.78% ≈ 344× above chance
- best_loss=1.35 at step 100k (best checkpoint = last checkpoint)
- Validation accuracy trend: 15.3% (step 2k) → 51.0% (10k) → 60.0% (20k) → 65.0% (40k) → 66.9% (60k) → 68.78% (100k)
- `encoder_layerdrop: 0.0` (required: single supervised layer — if L12 dropped, no grad_fn on loss)
- vs exp_nomask_k32 L12 val acc: 77.22% — k=500 harder (500 vs 32 classes), as expected
- vs exp_nomask_k512 L12 val acc: 63.30% — k=500 slightly harder than k=512 on same-layer single-head

---

## SUPERB Evaluation

### Setup (2026-03-04)
- s3prl cloned to `/workspace/s3prl` and installed (`pip install -e .`)
- LibriSpeech test-clean downloaded to `/workspace/data/librispeech/LibriSpeech`
- `upstream/example/expert.py` replaced with our `UpstreamExpert` (wraps fairseq HuBERT checkpoint, exposes 13 hidden states)
- Patched `runner.py` and `run_downstream.py` for huggingface_hub 1.5.0 (`HfFolder` removed)

### Eval command
```bash
python scripts/superb_eval.py \
    --ckpt /workspace/exp_NAME/checkpoints/checkpoint_last.pt \
    --tasks pr \
    --s3prl /workspace/s3prl \
    --upstream_dir examples/hubert/s3prl_upstream \
    --data_root /workspace/data/librispeech/LibriSpeech \
    --out_dir /workspace/results/pr/exp_NAME \
    --extra_overrides "config.downstream_expert.corpus.path=/workspace/data/librispeech/LibriSpeech"
```

**CRITICAL**: Always use `checkpoint_last.pt` or specify `-e dev-best.ckpt` during eval.
A fresh eval run without a trained head gives random results (wrong).

### PR Eval Status (2026-03-04 ~13:30 KST)

All 9 experiments' PR evals queued. 4 running in parallel on GPU 0 (A40 46GB, ~43GB used).

| Exp | Status | Progress | Notes |
|-----|--------|----------|-------|
| exp_hard_k32 | 🔄 RUNNING | ~10% (step 10k/100k) | Started separately, batch 0 |
| exp_hard_k512 | 🔄 RUNNING | ~1% (step 900/100k) | Batch 1 |
| exp_soft_k32 | 🔄 RUNNING | ~1% (step 750/100k) | Batch 1 |
| exp_hard_both | 🔄 RUNNING | ~1% (step 790/100k) | Batch 1 |
| exp_soft_k512 | ⏳ WAITING | — | GPU watcher (starts when >12GB free) |
| exp_soft_both | ⏳ QUEUED | — | Batch 2 (after batch 1 finishes) |
| exp_nomask_k32 | ⏳ QUEUED | — | Batch 2 |
| exp_nomask_k512 | ⏳ QUEUED | — | Batch 2 |
| exp_nomask_both | ⏳ QUEUED | — | Batch 2 |

ETA: ~22h for batch 1 to complete. Results will be logged here as they arrive.

### PR Eval Results

| Exp | Method | Codebook | Mask | Distill Best Step | PR Best Step | PR Test PER ↓ |
|-----|--------|----------|------|:-----------------:|:------------:|:-------------:|
| HuBERT Base | — | — | — | — | — | 5.70% |
| DICEHuBERT (10k) | — | — | — | — | — | 51.44% |
| exp_hard_k32 | Hard CE, 12L | k=32 | 80% | 100,000 | 46,000 | 36.60% ✅ |
| exp_hard_k512 | Hard CE, 12L | k=512 | 80% | — | — | 50.99% ✅ |
| exp_hard_both | Hard CE, 12L | k=32+512 | 80% | ~98,000 † | 78,000 | 49.74% ✅ |
| exp_soft_k32 | Soft KL probe | k=32 | 0% | 100,000 | 26,000 | 36.71% ✅ |
| exp_soft_k512 | Soft KL probe | k=512 | 0% | — | — | 30.86% ✅ |
| exp_soft_both | Soft KL probe | k=32+512 | 0% | — | — | 28.58% ✅ |
| exp_nomask_k32 | Hard CE, 12L | k=32 | 0% | — | — | **23.12%** ✅ |
| exp_nomask_k512 | Hard CE, 12L | k=512 | 0% | — | — | **19.45%** ✅ 🔥 |
| exp_nomask_both | Hard CE, 12L | k=32+512 | 0% | — | 24,000 | **18.38%** ✅ 🔥🔥 |
| exp_nomask_k500 | Hard CE, L12 only | k-means k=500 | 0% | 100,000 | — | **32.76%** ✅ |

### SID Eval Status (2026-03-11)

- 새 서버 (194.68.245.146:22095, A40 1×46GB) 셋업 완료
- VoxCeleb1 (~72GB wav/) 다운로드 완료 (`/workspace/data/voxceleb1/wav/` 1251 speakers)
- `file_path=/workspace/data` (glob 패턴 `*/wav/id{XXXX}/...` 기준)
- 100k steps (`config.runner.total_steps=100000` override via `,,` 구분자로 합산)
- `scripts/superb_tasks/sid.yaml`: `train_steps: 100000`, `metric: acc`
- 종속 패키지 추가 설치: `tensorboardX`, `librosa`

**SID eval 진행 순서**: exp_nomask_both → exp_nomask_k512 → exp_nomask_k500 → exp_nomask_k32 (순차)

### SID Eval Results

| Exp | Method | Codebook | Mask | SID Best Step | SID Test Acc ↑ |
|-----|--------|----------|------|:-------------:|:--------------:|
| HuBERT Base | — | — | — | — | — |
| exp_nomask_both | Hard CE, 12L | k=32+512 | 0% | 100,000 | **45.51%** ✅ |
| exp_nomask_k512 | Hard CE, 12L | k=512 | 0% | 100,000 | **48.32%** ✅ |
| exp_nomask_k500 | Hard CE, L12 only | k-means k=500 | 0% | 100,000 | **34.01%** ✅ |
| exp_nomask_k32 | Hard CE, 12L | k=32 | 0% | 100,000 | **40.02%** ✅ |

> **Distill Best Step**: step at which `checkpoint_best.pt` (lowest val loss) was saved during distillation training.
> Uses `checkpoint_last.pt` (step 100k) for SUPERB eval in all cases.
>
> **PR Best Step**: step at which `dev-best.ckpt` was saved during PR CTC downstream training (100k steps total).
> Test PER is evaluated with this checkpoint via `-m evaluate -e dev-best.ckpt`.
>
> † exp_hard_both distillation training log 소실 (partial log up to step 20k only).
> `checkpoint_best.pt` state: epoch=4, iter_in_epoch=23,732 → estimated step ≈ 98,732.

### Reference Results (from previous ablation)

| Model | PR PER ↓ | KS Acc ↑ |
|-------|---------|---------|
| HuBERT Base (teacher) | 5.70% | 96.36% |
| DICEHuBERT (10k steps) | 51.44% | 87.96% |
| l9k4096_l9k32 (prev) | 50.90% | 88.19% |

---

## File Reference

| File | Purpose |
|------|---------|
| `fairseq/criterions/hubert_layerwise_criterion.py` | Layer-wise CE criterion (hard distillation) |
| `fairseq/criterions/hubert_probe_distill_criterion.py` | Soft KL criterion (probe distillation) |
| `fairseq/models/hubert/hubert.py` | Student model with `layerwise_cls_specs` |
| `fairseq/data/data_utils.py` | Patched `compute_mask_indices` (no crash on short seqs) |
| `scripts/train_codebooks.py` | RepCodec codebook training |
| `scripts/apply_codebooks.py` | Label generation from codebooks |
| `scripts/superb_eval.py` | SUPERB evaluation CLI |
| `examples/hubert/config/pretrain/hubert_layerwise_k32.yaml` | Exp 1 config |
| `examples/hubert/config/pretrain/hubert_layerwise_k512.yaml` | Exp 2 config |
| `examples/hubert/config/pretrain/hubert_layerwise_distill.yaml` | Exp 3 config |
| `examples/hubert/config/pretrain/hubert_layerwise_k32_nomask.yaml` | Exp 7 config |
| `examples/hubert/config/pretrain/hubert_layerwise_k512_nomask.yaml` | Exp 8 config |
| `examples/hubert/config/pretrain/hubert_layerwise_both_nomask.yaml` | Exp 9 config |
| `examples/hubert/config/pretrain/hubert_nomask_k500.yaml` | Exp 10 config |
