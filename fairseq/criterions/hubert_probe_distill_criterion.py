# Copyright (c) Facebook, Inc. and its affiliates.
#
# This source code is licensed under the MIT license found in the
# LICENSE file in the root directory of this source tree.

"""
Soft KL distillation criterion for HuBERT student training.

Pipeline:
  1. Student forward (with masking) → last-layer hidden states [B, T, D_student]
  2. Teacher forward (no masking, no grad) → last-layer hidden states [B, T, D_teacher]
  3. Trained linear probe: teacher_last → logits over K codebook entries → softmax = soft labels
  4. Student last-layer → model.layer_classifiers["l12k{K}"] → KL divergence against soft labels

Teacher probe is a Linear(768 → K) head trained on teacher's layer-12 output to predict
layer-6 RepCodec codebook indices (see scripts/train_teacher_probe.py).

Supports training with one or both codebook sizes (K=32, K=512) simultaneously.

Logging per probe:
  kl_k{K}  — KL divergence loss
  acc_k{K} — student top-1 accuracy vs teacher's argmax (soft-label argmax as pseudo-target)
  ent_k{K} — entropy of teacher soft distribution (nats)
"""

import logging
import math
from dataclasses import dataclass, field
from typing import List, Optional

import torch
import torch.nn as nn
import torch.nn.functional as F

from fairseq import utils
from fairseq.criterions import FairseqCriterion, register_criterion
from fairseq.dataclass import FairseqDataclass
from fairseq.logging import metrics

logger = logging.getLogger(__name__)


@dataclass
class HubertProbeDistillCriterionConfig(FairseqDataclass):
    teacher_path: str = field(
        default="???",
        metadata={"help": "Path to teacher HuBERT checkpoint (.pt)."},
    )
    probe_k32_path: Optional[str] = field(
        default=None,
        metadata={"help": "Path to trained probe checkpoint for K=32 (from train_teacher_probe.py)."},
    )
    probe_k512_path: Optional[str] = field(
        default=None,
        metadata={"help": "Path to trained probe checkpoint for K=512."},
    )
    probe_specs: str = field(
        default="32,512",
        metadata={"help": "Comma-separated codebook sizes to distill, e.g. '32' or '32,512'."},
    )
    temperature: float = field(
        default=1.0,
        metadata={"help": "Temperature for teacher soft label softmax (higher = softer)."},
    )
    teacher_fp16: bool = field(
        default=False,
        metadata={"help": "Run teacher in fp16 to save VRAM (default: fp32 for stability)."},
    )
    pred_masked_weight: float = field(
        default=1.0,
        metadata={"help": "Multiplier applied to the total KL loss."},
    )
    loss_on_all_frames: bool = field(
        default=False,
        metadata={"help": "If True (or mask_indices is None), compute KL on all non-padding frames instead of masked positions only."},
    )


@register_criterion("hubert_probe_distill", dataclass=HubertProbeDistillCriterionConfig)
class HubertProbeDistillCriterion(FairseqCriterion):
    """
    Soft KL distillation: student last layer predicts teacher probe's soft distribution
    over layer-6 codebook entries.
    """

    def __init__(
        self,
        task,
        teacher_path,
        probe_k32_path=None,
        probe_k512_path=None,
        probe_specs="32,512",
        temperature=1.0,
        teacher_fp16=False,
        pred_masked_weight=1.0,
        loss_on_all_frames=False,
    ):
        super().__init__(task)
        self.teacher_path = teacher_path
        self.probe_paths = {}
        if probe_k32_path:
            self.probe_paths[32] = probe_k32_path
        if probe_k512_path:
            self.probe_paths[512] = probe_k512_path

        self.probe_sizes = [int(k) for k in probe_specs.strip().split(",")]
        self.temperature = temperature
        self.teacher_fp16 = teacher_fp16
        self.pred_masked_weight = pred_masked_weight
        self.loss_on_all_frames = loss_on_all_frames

        for k in self.probe_sizes:
            assert k in self.probe_paths, (
                f"probe_specs includes K={k} but probe_k{k}_path is not set."
            )

        # Lazy-loaded — stored in a plain Python dict so PyTorch does NOT
        # register teacher/probes as submodules.  If stored as self._teacher = nn.Module,
        # PyTorch's __setattr__ would add them to self._modules, making them appear in
        # criterion.state_dict() and ballooning checkpoints by ~400 MB (teacher weight).
        # Using a plain dict bypasses submodule registration entirely.
        self._lazy: dict = {"teacher": None, "probes": {}}

        logger.info(
            f"HubertProbeDistillCriterion: probe_specs={self.probe_sizes}, "
            f"temperature={temperature}, teacher_fp16={teacher_fp16}, "
            f"loss_on_all_frames={loss_on_all_frames}"
        )

    # ------------------------------------------------------------------
    # Lazy loading
    # ------------------------------------------------------------------

    def _load_teacher_and_probes(self, device: torch.device):
        """Load teacher and probes on first forward call."""
        import fairseq

        logger.info(f"Loading teacher from {self.teacher_path} ...")
        models, _cfg, _task = fairseq.checkpoint_utils.load_model_ensemble_and_task(
            [self.teacher_path]
        )
        teacher = models[0]
        teacher.eval()
        for p in teacher.parameters():
            p.requires_grad_(False)
        if self.teacher_fp16:
            teacher = teacher.half()
        # Store in plain dict (not as self.attr) to avoid PyTorch submodule registration
        self._lazy["teacher"] = teacher.to(device)
        logger.info(
            f"Teacher loaded: {sum(p.numel() for p in teacher.parameters()) / 1e6:.1f}M params"
        )

        for k, path in self.probe_paths.items():
            logger.info(f"Loading probe K={k} from {path} ...")
            ckpt = torch.load(path, map_location="cpu")
            in_features = ckpt["state_dict"]["weight"].shape[1]
            probe = nn.Linear(in_features, k, bias=True)
            probe.load_state_dict(ckpt["state_dict"])
            probe.eval()
            for p in probe.parameters():
                p.requires_grad_(False)
            self._lazy["probes"][k] = probe.to(device)
            logger.info(f"Probe K={k} loaded.")

    # ------------------------------------------------------------------
    # Forward
    # ------------------------------------------------------------------

    def forward(self, model, sample, reduce=True):
        assert model.layer_classifiers is not None, (
            "model.layer_classifiers is None. "
            "Set layerwise_cls_specs (e.g. '12:32,12:512') in HubertConfig."
        )

        device = next(model.parameters()).device

        # Lazy-load teacher + probes on first call
        if self._lazy["teacher"] is None:
            self._load_teacher_and_probes(device)

        source = sample["net_input"]["source"]
        padding_mask = sample["net_input"].get("padding_mask", None)

        # ---- Student forward via extract_features (no encoder hook) ----
        # extract_features(mask=True) calls forward(features_only=True), which runs the
        # encoder and returns the last-layer hidden states directly as (B, T, D).
        # This is reliable across train/eval modes and across validation steps —
        # unlike register_forward_hook on the encoder which silently fails on the
        # second validation call.
        #
        # For mask_prob > 0 experiments we also need mask_indices.  We capture it by
        # temporarily patching apply_mask (a plain Python method, not nn.Module, so
        # method patching is always safe and does not interact with PyTorch internals).
        mask_holder = {"mask_indices": None}
        orig_apply_mask = model.apply_mask

        def _patched_apply_mask(features, pm, target_list):
            x, mi = orig_apply_mask(features, pm, target_list)
            mask_holder["mask_indices"] = mi
            return x, mi

        model.apply_mask = _patched_apply_mask
        try:
            # Returns (x [B,T,D_student], updated_padding_mask [B,T] or None).
            # apply_mask is called inside, so mask_holder["mask_indices"] gets set.
            student_last, out_padding = model.extract_features(
                source=source,
                padding_mask=padding_mask,
                mask=True,
            )
        finally:
            model.apply_mask = orig_apply_mask

        mask_indices = mask_holder["mask_indices"]  # [B, T] bool or None

        # Effective mask: masked positions (or all non-padding frames in no-mask mode)
        use_all_frames = self.loss_on_all_frames or (mask_indices is None)
        if use_all_frames:
            if out_padding is not None:
                eff_mask = ~out_padding  # [B, T] — all non-padding frames
            else:
                B, T, _ = student_last.shape
                eff_mask = torch.ones(B, T, dtype=torch.bool, device=student_last.device)
        else:
            # mask_prob > 0 path: loss on masked positions only
            T = min(mask_indices.shape[1], out_padding.shape[1]) if out_padding is not None else mask_indices.shape[1]
            if out_padding is not None:
                eff_mask = mask_indices[:, :T] & ~out_padding[:, :T]
            else:
                eff_mask = mask_indices

        # ---- Teacher forward (no masking, no grad) ----
        with torch.no_grad():
            src = source.float()
            if self.teacher_fp16:
                src = src.half()

            teacher_last, _ = self._lazy["teacher"].extract_features(
                source=src,
                padding_mask=padding_mask,
                mask=False,
            )  # [B, T, D_teacher]
            teacher_last = teacher_last.float()

        # ---- KL loss per probe ----
        total_loss = torch.tensor(0.0, device=device)
        sample_size = int(eff_mask.sum())
        logging_output: dict = {}

        for k in self.probe_sizes:
            cls_key = f"l12k{k}"

            # Align time dimensions
            T_s = student_last.shape[1]
            T_t = teacher_last.shape[1]
            T_m = eff_mask.shape[1]
            T_min = min(T_s, T_t, T_m)

            eff_k = eff_mask[:, :T_min]
            N_masked = int(eff_k.sum())
            if N_masked == 0:
                continue

            # Student: masked positions [N, D_student]
            s_masked = student_last[:, :T_min][eff_k]  # [N, D_s]

            # Teacher: masked positions [N, 768]
            t_masked = teacher_last[:, :T_min][eff_k].to(device)  # [N, 768]

            # Probe → soft target
            with torch.no_grad():
                teacher_logits = self._lazy["probes"][k](t_masked)  # [N, K]
                soft_target = F.softmax(
                    teacher_logits / self.temperature, dim=-1
                )  # [N, K]

            # Student classifier → logits
            student_logits = model.layer_classifiers[cls_key](s_masked)  # [N, K]
            student_log_probs = F.log_softmax(student_logits.float(), dim=-1)

            # KL divergence: sum(soft_target * (log soft_target - student_log_probs))
            # F.kl_div expects (input=log_prob, target=prob)
            kl = F.kl_div(
                student_log_probs,
                soft_target,
                reduction="batchmean",
            )
            total_loss = total_loss + kl

            # Logging
            with torch.no_grad():
                tag = f"k{k}"
                logging_output[f"kl_{tag}"] = kl.item()

                # Pseudo-accuracy: student argmax vs teacher argmax
                pseudo_tgt = soft_target.argmax(-1)
                acc = (student_logits.argmax(-1) == pseudo_tgt).float().mean().item()
                logging_output[f"acc_{tag}"] = acc

                # Teacher soft distribution entropy
                ent = -(soft_target * torch.log(soft_target + 1e-10)).sum(-1).mean().item()
                logging_output[f"ent_{tag}"] = ent

        loss = self.pred_masked_weight * total_loss

        logging_output = {
            "loss": loss.item(),
            "ntokens": sample_size,
            "nsentences": sample["id"].numel(),
            "sample_size": sample_size,
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

        metrics.log_scalar(
            "loss", loss_sum / sample_size / math.log(2), sample_size, round=3
        )

        example_log = logging_outputs[0] if logging_outputs else {}
        per_keys = [k for k in example_log if k.startswith(("kl_", "acc_", "ent_"))]
        n = len(logging_outputs)
        for key in per_keys:
            val = sum(log.get(key, 0.0) for log in logging_outputs)
            metrics.log_scalar(key, val / n, n, round=4)

    @staticmethod
    def logging_outputs_can_be_summed() -> bool:
        return False
