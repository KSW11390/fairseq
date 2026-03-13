# Copyright (c) Facebook, Inc. and its affiliates.
#
# This source code is licensed under the MIT license found in the
# LICENSE file in the root directory of this source tree.

"""
Layer-wise classification distillation criterion for HuBERT student training.

Each student transformer layer L is supervised by the corresponding teacher
layer L's RepCodec codebook indices (pre-computed via train_codebooks.py +
apply_codebooks.py).

For each (student_layer, codebook_size) spec:
  - Hook encoder to capture layer_results (all intermediate outputs).
  - Index masked positions: student_layer_out[mask_indices] → [N_masked, D].
  - Apply model.layer_classifiers[f"l{layer}k{K}"] → logits [N_masked, K].
  - Cross-entropy loss against pre-computed label target [N_masked].

Per-layer logging:
  loss_l{layer}k{K}  — CE loss
  acc_l{layer}k{K}   — top-1 accuracy (fraction of masked positions correct)
  ent_l{layer}k{K}   — entropy of model predictions (nats)

Total loss = weighted sum of all per-layer CEs (default: uniform weights).

Note on layer indexing:
  - HuBERT encoder layer_results is a list of length encoder_layers.
  - layer_results[i] corresponds to transformer layer (i+1) (1-based to match
    train_codebooks.py convention).
  - layer_results[i][0] has shape (T, B, D) — time-first.
"""

import logging
import math
import re
from dataclasses import dataclass, field
from typing import List, Optional

import torch
import torch.nn.functional as F

from fairseq import utils
from fairseq.criterions import FairseqCriterion, register_criterion
from fairseq.dataclass import FairseqDataclass
from fairseq.logging import metrics

logger = logging.getLogger(__name__)


@dataclass
class HubertLayerwiseCriterionConfig(FairseqDataclass):
    layer_specs: str = field(
        default="",
        metadata={
            "help": "Mapping from student layer → label key, comma-separated. "
            "Format: 'layer:label_key,...'. "
            "layer must be 1-based (matches train_codebooks.py). "
            "label_key must match a key in sample['target_list'] "
            "(i.e. one of the task.labels entries). "
            "Example: '1:l1k32,1:l1k512,2:l2k32,...,12:l12k512'"
        },
    )
    layer_weights: Optional[str] = field(
        default=None,
        metadata={
            "help": "Per-spec loss weights, comma-separated floats. "
            "Must match the number of specs in layer_specs. "
            "Default None = uniform weights (1.0 each)."
        },
    )
    pred_masked_weight: float = field(
        default=1.0,
        metadata={"help": "Weight applied to the masked-position CE loss sum."},
    )
    loss_on_all_frames: bool = field(
        default=False,
        metadata={
            "help": "If True, compute CE loss on ALL non-padding frames instead of "
            "only masked frames. Use with model.mask_prob=0.0 for no-mask experiments. "
            "When False (default), only masked positions contribute to the loss."
        },
    )


def _parse_layer_specs(spec_str: str):
    """
    '1:l1k32,1:l1k512,...,12:l12k512'
    → [(1, 'l1k32'), (1, 'l1k512'), ..., (12, 'l12k512')]
    """
    result = []
    for s in spec_str.strip().split(","):
        layer_str, label_key = s.strip().split(":", 1)
        result.append((int(layer_str), label_key))
    return result


@register_criterion("hubert_layerwise", dataclass=HubertLayerwiseCriterionConfig)
class HubertLayerwiseCriterion(FairseqCriterion):
    """
    Layer-wise classification distillation loss.

    For each (layer, label_key) spec:
      CE(model.layer_classifiers[label_key](student_layer_out[mask]), target[mask])

    All per-layer losses are summed with configurable weights.
    """

    def __init__(self, task, layer_specs, layer_weights=None, pred_masked_weight=1.0, loss_on_all_frames=False):
        super().__init__(task)
        self.pred_masked_weight = pred_masked_weight
        self.loss_on_all_frames = loss_on_all_frames
        self.specs = _parse_layer_specs(layer_specs)  # [(layer_idx, label_key), ...]

        # Build label_key → target_list_position mapping using task.dictionaries order
        # (same order as task.labels / sample["target_list"])
        self.label_keys = [s[1] for s in self.specs]
        task_labels = list(getattr(task.cfg, "labels", []))
        self.label_to_target_idx = {lk: task_labels.index(lk) for lk in set(self.label_keys)}

        # fairseq Dictionary adds special tokens (bos/pad/eos/unk) before real tokens,
        # so label "0" maps to dictionary index nspecial (typically 4).
        # Subtract nspecial to recover the original 0-based class index.
        self.label_to_nspecial = {
            lk: task.dictionaries[task_labels.index(lk)].nspecial
            for lk in set(self.label_keys)
        }

        # Per-spec weights (default: uniform 1.0)
        if layer_weights is not None:
            ws = [float(w) for w in layer_weights.split(",")]
            assert len(ws) == len(self.specs), (
                f"layer_weights count ({len(ws)}) ≠ layer_specs count ({len(self.specs)})"
            )
            self.weights = ws
        else:
            self.weights = [1.0] * len(self.specs)

        logger.info(
            f"HubertLayerwiseCriterion: {len(self.specs)} specs "
            f"(layers: {sorted(set(l for l,_ in self.specs))})"
        )
        for (layer, lk), w in zip(self.specs, self.weights):
            logger.info(f"  layer={layer} label={lk} weight={w}")

    # ------------------------------------------------------------------
    # Hook helpers
    # ------------------------------------------------------------------

    @staticmethod
    def _install_hooks(model):
        """
        Capture:
          - mask_indices: [B, T] bool tensor from apply_mask
          - layer_outputs: dict {0-based layer idx → (T,B,D) tensor}

        Per-layer forward hooks are used instead of patching encoder.forward,
        so that layerdrop (random layer skipping during training) is handled
        correctly — dropped layers simply won't appear in layer_outputs.
        """
        holder = {"mask_indices": None, "layer_outputs": {}}

        orig_apply_mask = model.apply_mask

        def _patched_apply_mask(features, padding_mask, target_list):
            x, mi = orig_apply_mask(features, padding_mask, target_list)
            holder["mask_indices"] = mi
            return x, mi

        model.apply_mask = _patched_apply_mask

        # Register a forward hook on each encoder layer to capture its output.
        # Each TransformerSentenceEncoderLayer returns (x, (z, lr));
        # output[0] is the hidden state x with shape (T, B, D).
        fwd_hooks = []
        for i, enc_layer in enumerate(model.encoder.layers):
            def make_hook(idx):
                def hook(module, inputs, output):
                    holder["layer_outputs"][idx] = output[0]  # (T, B, D)
                return hook
            fwd_hooks.append(enc_layer.register_forward_hook(make_hook(i)))

        def cleanup():
            model.apply_mask = orig_apply_mask
            for h in fwd_hooks:
                h.remove()

        return holder, cleanup

    # ------------------------------------------------------------------
    # Forward
    # ------------------------------------------------------------------

    def forward(self, model, sample, reduce=True):
        assert model.layer_classifiers is not None, (
            "model.layer_classifiers is None. "
            "Set model.layerwise_cls_specs in HubertConfig to enable "
            "layer-wise classification heads."
        )

        holder, cleanup = self._install_hooks(model)
        try:
            net_output = model(
                target_list=sample["target_list"], **sample["net_input"]
            )
        finally:
            cleanup()

        mask_indices = holder["mask_indices"]       # [B, T] bool or None
        layer_outputs = holder["layer_outputs"]     # dict: {0-based idx → (T,B,D)}
        padding_mask = net_output.get("padding_mask", None)

        if self.loss_on_all_frames:
            # No-mask mode: supervise ALL non-padding frames
            if padding_mask is not None:
                eff_mask = ~padding_mask  # [B, T] — all non-padding frames
            else:
                # padding_mask is None → all sequences same length, no padding.
                # Get (B, T) from captured encoder layer outputs.
                if layer_outputs:
                    _first = next(iter(layer_outputs.values()))  # (T, B, D)
                    T_enc, B_enc = _first.shape[0], _first.shape[1]
                    dev = _first.device
                elif mask_indices is not None:
                    B_enc, T_enc = mask_indices.shape
                    dev = mask_indices.device
                else:
                    raise RuntimeError(
                        "Cannot determine frame dimensions: padding_mask, mask_indices, "
                        "and layer_outputs are all unavailable."
                    )
                eff_mask = torch.ones(B_enc, T_enc, dtype=torch.bool, device=dev)
        else:
            if mask_indices is None:
                raise RuntimeError(
                    "mask_indices is None — ensure mask_prob > 0 in the model config, "
                    "or set criterion.loss_on_all_frames=true for no-mask training."
                )
            # Standard mode: masked AND not padding
            if padding_mask is not None:
                eff_mask = mask_indices & ~padding_mask  # [B, T]
            else:
                eff_mask = mask_indices

        reduction = "sum" if reduce else "none"
        total_loss = torch.tensor(0.0, device=eff_mask.device)
        sample_size = int(eff_mask.sum())
        logging_output = {}

        for (layer_idx, label_key), weight in zip(self.specs, self.weights):
            # layer_idx is 1-based; layer_outputs is keyed by 0-based index
            enc_idx = layer_idx - 1
            if enc_idx not in layer_outputs:
                # Layer was dropped by layerdrop — skip this spec for this step
                continue
            lr_x = layer_outputs[enc_idx]        # (T, B, D) — time-first
            student_out = lr_x.permute(1, 0, 2)  # (B, T, D)

            # Align time dimension in case of minor mismatch
            T_s = student_out.shape[1]
            T_m = eff_mask.shape[1]
            T_min = min(T_s, T_m)
            eff_mask_i = eff_mask[:, :T_min]
            student_masked = student_out[:, :T_min][eff_mask_i]  # [N_masked, D]

            # Classification head
            classifier_key = label_key  # e.g. "l1k32"
            logits = model.layer_classifiers[classifier_key](student_masked)  # [N, K]

            # Target labels
            target_idx = self.label_to_target_idx[label_key]
            target = sample["target_list"][target_idx]  # [B, T']
            T_t = target.shape[1]
            T_min_t = min(T_min, T_t)
            eff_mask_it = eff_mask[:, :T_min_t]
            target_masked = target[:, :T_min_t][eff_mask_it]  # [N_masked]

            # Ensure N_masked matches (can differ by ≤1 frame due to sub/seq rounding)
            N_min = min(logits.shape[0], target_masked.shape[0])
            logits_t = logits[:N_min]
            nspecial = self.label_to_nspecial[label_key]
            target_t = (target_masked[:N_min] - nspecial).long()

            # Filter out special tokens (EOS/PAD/BOS) that may appear in masked
            # positions for short sequences.  After nspecial subtraction they are
            # negative (e.g. EOS index 2 – nspecial 4 = -2), which triggers a
            # CUDA device-side assert in nll_loss.  Simply remove those positions.
            K = logits_t.shape[-1]
            valid = (target_t >= 0) & (target_t < K)
            if not valid.all():
                logits_t = logits_t[valid]
                target_t = target_t[valid]

            if logits_t.shape[0] == 0:
                continue  # skip spec if no valid positions remain

            ce = F.cross_entropy(logits_t, target_t, reduction=reduction)
            total_loss = total_loss + weight * ce

            # Per-layer logging
            tag = label_key  # e.g. "l1k32"
            with torch.no_grad():
                logging_output[f"loss_{tag}"] = ce.detach().item() if reduce else ce.detach().mean().item()

                preds = logits_t.argmax(-1)
                acc = (preds == target_t).float().mean().item()
                logging_output[f"acc_{tag}"] = acc

                probs = F.softmax(logits_t.float(), dim=-1)
                ent = -(probs * torch.log(probs + 1e-10)).sum(-1).mean().item()
                logging_output[f"ent_{tag}"] = ent

        # Normalize by sample_size (number of masked tokens) to keep loss ~O(num_layers × log(K))
        # This prevents fp16 overflow: without normalization, loss ~ N_masked × 12 × log(K) > 60,000
        if sample_size > 0:
            total_loss = total_loss / sample_size
        loss = self.pred_masked_weight * total_loss

        logging_output = {
            "loss": loss.item() if reduce else loss,
            "ntokens": sample_size,
            "nsentences": sample["id"].numel(),
            "sample_size": 1,  # loss is already per-token normalized; prevent double-normalization
            **logging_output,
        }

        return loss, sample_size, logging_output

    # ------------------------------------------------------------------
    # Metric aggregation
    # ------------------------------------------------------------------

    @staticmethod
    def reduce_metrics(logging_outputs) -> None:
        loss_sum = sum(log.get("loss", 0) for log in logging_outputs)
        sample_size = sum(log.get("sample_size", 0) for log in logging_outputs)
        if sample_size == 0:
            return

        # sample_size is now 1 per step (loss already normalized per-token in forward)
        # loss_sum / sample_size = mean per-step loss; divide by log(2) to get bits
        metrics.log_scalar(
            "loss", loss_sum / sample_size / math.log(2), sample_size, round=3
        )
        # Also log ntokens for reference
        ntokens = sum(log.get("ntokens", 0) for log in logging_outputs)
        metrics.log_scalar("ntokens", ntokens / len(logging_outputs), 1, round=0)

        # Collect all per-layer keys from the first non-empty log
        example_log = logging_outputs[0] if logging_outputs else {}
        per_layer_keys = [k for k in example_log if k.startswith(("loss_", "acc_", "ent_"))]

        n = len(logging_outputs)
        for key in per_layer_keys:
            val = sum(log.get(key, 0.0) for log in logging_outputs)
            if key.startswith("loss_"):
                metrics.log_scalar(key, val / n / math.log(2), n, round=4)
            else:
                metrics.log_scalar(key, val / n, n, round=4)

    @staticmethod
    def logging_outputs_can_be_summed() -> bool:
        return False
