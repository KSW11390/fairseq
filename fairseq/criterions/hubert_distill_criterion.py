# Copyright (c) Facebook, Inc. and its affiliates.
#
# This source code is licensed under the MIT license found in the
# LICENSE file in the root directory of this source tree.

"""
HuBERT distillation criterion supporting three label modes:
  - hard:               standard HuBERT CE on discrete cluster IDs (default)
  - soft_teacher_logits: KL to teacher prediction-head logits at masked positions
  - soft_kmeans_dist:   KL to teacher-rep distances to k-means centroids at masked positions

Note on logit format:
  HuBERT uses NCE-style logits (K+1 dims, target=0) for the hard CE loss.
  For soft-mode KL divergence we need standard K-way logits (cosine similarity
  to each of the K label embeddings).  We compute these separately for both
  teacher and student so that class indices are aligned (class i = label_emb i).
"""

import logging
import math
import re
from dataclasses import dataclass, field
from typing import List, Optional

import joblib
import numpy as np
import torch
import torch.nn.functional as F

import fairseq
from fairseq import utils
from fairseq.criterions import FairseqCriterion, register_criterion
from fairseq.dataclass import FairseqDataclass
from fairseq.logging import metrics

logger = logging.getLogger(__name__)


@dataclass
class HubertDistillCriterionConfig(FairseqDataclass):
    # --- original HuBERT criterion fields ---
    pred_masked_weight: float = field(
        default=1.0,
        metadata={"help": "weight for predictive loss for masked frames"},
    )
    pred_nomask_weight: float = field(
        default=0.0,
        metadata={"help": "weight for predictive loss for unmasked frames"},
    )
    loss_weights: Optional[List[float]] = field(
        default=None,
        metadata={"help": "weights for additional loss terms (not first one)"},
    )
    log_keys: List[str] = field(
        default_factory=lambda: [],
        metadata={"help": "output keys to log"},
    )

    # --- distillation-specific fields ---
    label_mode: str = field(
        default="hard",
        metadata={
            "help": "Label mode: hard | soft_teacher_logits | soft_kmeans_dist"
        },
    )
    teacher_path: Optional[str] = field(
        default=None,
        metadata={"help": "Path to teacher checkpoint (required for soft modes)"},
    )
    kmeans_path: Optional[str] = field(
        default=None,
        metadata={"help": "Path to k-means model (required for soft_kmeans_dist)"},
    )
    teacher_layer: int = field(
        default=6,
        metadata={"help": "Teacher layer to extract reps from (1-based)"},
    )
    distill_temperature: float = field(
        default=5.0,
        metadata={"help": "Temperature T for soft target distributions"},
    )
    distill_alpha: float = field(
        default=1.0,
        metadata={
            "help": "Weight for KL (soft) loss. CE weight = 1 - distill_alpha. "
            "Default 1.0 = pure soft loss."
        },
    )
    teacher_fp16: bool = field(
        default=False,
        metadata={"help": "Run teacher forward pass in fp16 for speed"},
    )


@register_criterion("hubert_distill", dataclass=HubertDistillCriterionConfig)
class HubertDistillCriterion(FairseqCriterion):
    def __init__(
        self,
        task,
        pred_masked_weight,
        pred_nomask_weight,
        loss_weights=None,
        log_keys=None,
        label_mode="hard",
        teacher_path=None,
        kmeans_path=None,
        teacher_layer=6,
        distill_temperature=5.0,
        distill_alpha=1.0,
        teacher_fp16=False,
    ):
        super().__init__(task)
        self.pred_masked_weight = pred_masked_weight
        self.pred_nomask_weight = pred_nomask_weight
        self.loss_weights = loss_weights
        self.log_keys = [] if log_keys is None else log_keys
        self.label_mode = label_mode
        self.teacher_layer = teacher_layer
        self.distill_temperature = distill_temperature
        self.distill_alpha = distill_alpha
        self.teacher_fp16 = teacher_fp16

        assert label_mode in (
            "hard",
            "soft_teacher_logits",
            "soft_kmeans_dist",
        ), f"Unknown label_mode: {label_mode}"

        # ---- Teacher model (lazy-loaded on first forward) ----
        self._teacher = None
        self._teacher_path = teacher_path
        self._teacher_loaded = False

        # ---- K-means centroids (lazy-loaded on first forward) ----
        self._kmeans_C = None  # (D, K) on GPU
        self._kmeans_Cnorm = None  # (1, K) on GPU
        self._kmeans_path = kmeans_path

        if label_mode != "hard":
            assert teacher_path is not None, (
                f"teacher_path is required for label_mode={label_mode}"
            )
        if label_mode == "soft_kmeans_dist":
            assert kmeans_path is not None, (
                "kmeans_path is required for label_mode=soft_kmeans_dist"
            )

        logger.info(
            f"HubertDistillCriterion: label_mode={label_mode}, "
            f"T={distill_temperature}, alpha={distill_alpha}, "
            f"teacher_layer={teacher_layer}"
        )

    # ------------------------------------------------------------------
    # Lazy loading
    # ------------------------------------------------------------------
    def _load_teacher(self, device):
        if self._teacher_loaded:
            return
        logger.info(f"Loading teacher from {self._teacher_path}")
        models, _, _ = fairseq.checkpoint_utils.load_model_ensemble_and_task(
            [self._teacher_path]
        )
        teacher = models[0].eval().to(device)
        for p in teacher.parameters():
            p.requires_grad = False
        self._teacher = teacher
        self._teacher_loaded = True
        n_params = sum(p.numel() for p in teacher.parameters()) / 1e6
        logger.info(f"Teacher loaded: {n_params:.1f}M params on {device}")

    def _load_kmeans(self, device):
        if self._kmeans_C is not None:
            return
        logger.info(f"Loading k-means from {self._kmeans_path}")
        km = joblib.load(self._kmeans_path)
        C = km.cluster_centers_  # (K, D)
        self._kmeans_C = torch.from_numpy(C.T.copy()).float().to(device)  # (D, K)
        self._kmeans_Cnorm = (
            torch.from_numpy((C ** 2).sum(1, keepdims=True).T.copy())
            .float()
            .to(device)
        )  # (1, K)
        logger.info(f"K-means loaded: K={C.shape[0]}, D={C.shape[1]}")

    # ------------------------------------------------------------------
    # Hooks: capture mask_indices and encoder output from student forward
    # ------------------------------------------------------------------
    @staticmethod
    def _install_hooks(model):
        """
        Monkey-patch model.apply_mask and model.encoder.forward to capture
        mask_indices and encoder output.  Returns (holder, cleanup_fn).
        The encoder output is NOT detached so gradients flow through
        final_proj → encoder for the KL loss.
        """
        holder = {"mask_indices": None, "encoder_out": None}

        orig_apply_mask = model.apply_mask

        def _patched_apply_mask(features, padding_mask, target_list):
            x, mi = orig_apply_mask(features, padding_mask, target_list)
            holder["mask_indices"] = mi
            return x, mi

        model.apply_mask = _patched_apply_mask

        orig_encoder_fwd = model.encoder.forward

        def _patched_encoder_fwd(*args, **kwargs):
            out = orig_encoder_fwd(*args, **kwargs)
            # Keep gradient! This is the same tensor the model uses internally.
            holder["encoder_out"] = out[0]
            return out

        model.encoder.forward = _patched_encoder_fwd

        def cleanup():
            model.apply_mask = orig_apply_mask
            model.encoder.forward = orig_encoder_fwd

        return holder, cleanup

    # ------------------------------------------------------------------
    # Standard K-way cosine logits (class i = label_emb[i])
    # ------------------------------------------------------------------
    @staticmethod
    def _kway_cosine_logits(proj_x, label_embs, logit_temp):
        """(N, D), (K, D) → (N, K) cosine similarity / temp."""
        proj_x_n = F.normalize(proj_x.float(), dim=-1)
        label_n = F.normalize(label_embs.float(), dim=-1)
        return torch.matmul(proj_x_n, label_n.t()) / logit_temp

    # ------------------------------------------------------------------
    # Teacher soft targets at masked positions
    # ------------------------------------------------------------------
    def _teacher_soft_targets(self, source, padding_mask, masked_indices, T):
        """
        Returns (N_masked, K) probability distribution from teacher.
        """
        device = source.device
        self._load_teacher(device)
        teacher = self._teacher

        with torch.no_grad():
            # Teacher is fp32 by default; cast input to match
            t_source = source.float() if not self.teacher_fp16 else source
            t_pad = padding_mask

            ctx = (
                torch.cuda.amp.autocast
                if (self.teacher_fp16 and source.is_cuda)
                else _nullcontext
            )
            with ctx():
                if self.label_mode == "soft_teacher_logits":
                    # Need final-layer output for prediction head
                    t_feat, _ = teacher.extract_features(
                        source=t_source, padding_mask=t_pad,
                        mask=False, output_layer=None,
                    )
                else:  # soft_kmeans_dist
                    t_feat, _ = teacher.extract_features(
                        source=t_source, padding_mask=t_pad,
                        mask=False, output_layer=self.teacher_layer,
                    )

            # Time alignment — truncate to shorter (±1 frame from
            # required_seq_len_multiple differences between student/teacher)
            T_mask = masked_indices.size(1)
            T_teach = t_feat.size(1)
            if T_mask != T_teach:
                T_min = min(T_mask, T_teach)
                logger.debug(
                    f"Time dim mismatch: mask={T_mask}, teacher={T_teach}, "
                    f"truncating to {T_min}"
                )
                masked_indices = masked_indices[:, :T_min]
                t_feat = t_feat[:, :T_min, :]

            if self.label_mode == "soft_teacher_logits":
                proj = teacher.final_proj(t_feat[masked_indices])
                if teacher.untie_final_proj:
                    proj = proj.chunk(len(teacher.num_classes), dim=-1)[0]
                lbl_embs = teacher.label_embs_concat
                if hasattr(teacher, "num_classes"):
                    lbl_embs = lbl_embs.split(teacher.num_classes, 0)[0]
                logits = self._kway_cosine_logits(proj, lbl_embs, teacher.logit_temp)
                return F.softmax(logits / T, dim=-1)

            else:  # soft_kmeans_dist
                self._load_kmeans(device)
                reps = t_feat[masked_indices].float()
                dist = (
                    reps.pow(2).sum(1, keepdim=True)
                    - 2 * torch.matmul(reps, self._kmeans_C)
                    + self._kmeans_Cnorm
                )
                return F.softmax(-dist / T, dim=-1)

    # ------------------------------------------------------------------
    # Forward
    # ------------------------------------------------------------------
    def forward(self, model, sample, reduce=True, log_pred=False):
        is_soft = self.label_mode != "hard"

        # Install hooks to capture mask_indices (always) and encoder_out (soft only)
        holder, cleanup = self._install_hooks(model)
        try:
            net_output = model(
                target_list=sample["target_list"], **sample["net_input"]
            )
        finally:
            cleanup()

        mask_indices = holder["mask_indices"]
        encoder_out = holder["encoder_out"]  # (B, T, D), with grad

        loss = 0.0
        sample_size = 0
        logging_output = {}
        reduction = "sum" if reduce else "none"

        # ================================================================
        # NCE-style logits for hard CE (original HuBERT behavior)
        # ================================================================
        do_hard = (not is_soft) or (self.distill_alpha < 1.0)

        logp_m_list = model.get_logits(net_output, True)   # NCE (N, K+1)
        targ_m_list = model.get_targets(net_output, True)   # all zeros

        loss_m_list = []
        if do_hard:
            assert self.pred_masked_weight == 0 or len(logp_m_list) > 0
            for i, (logp_m, targ_m) in enumerate(zip(logp_m_list, targ_m_list)):
                lm = F.cross_entropy(logp_m, targ_m, reduction=reduction)
                loss_m_list.append(lm)
                logging_output[f"loss_m_{i}"] = lm.detach().item()

        if do_hard and self.pred_masked_weight > 0:
            w = (1.0 - self.distill_alpha) if is_soft else 1.0
            loss += w * self.pred_masked_weight * sum(loss_m_list)

        if len(targ_m_list) > 0:
            sample_size += targ_m_list[0].numel()

        logp_u_list = model.get_logits(net_output, False)
        targ_u_list = model.get_targets(net_output, False)
        loss_u_list = []
        if do_hard:
            for i, (logp_u, targ_u) in enumerate(zip(logp_u_list, targ_u_list)):
                lu = F.cross_entropy(logp_u, targ_u, reduction=reduction)
                loss_u_list.append(lu)
                logging_output[f"loss_u_{i}"] = lu.detach().item()
            if self.pred_nomask_weight > 0:
                w = (1.0 - self.distill_alpha) if is_soft else 1.0
                loss += w * self.pred_nomask_weight * sum(loss_u_list)
                sample_size += targ_u_list[0].numel()

        # ================================================================
        # SOFT: KL divergence at masked positions with K-way logits
        # ================================================================
        if is_soft and self.distill_alpha > 0:
            assert mask_indices is not None, "mask_indices is None"
            assert encoder_out is not None, "encoder_out not captured"

            pad_mask = net_output["padding_mask"]
            if pad_mask is not None:
                masked_indices = torch.logical_and(~pad_mask, mask_indices)
            else:
                masked_indices = mask_indices

            T = self.distill_temperature

            # ---- Teacher soft targets (N_masked, K) ----
            teacher_probs = self._teacher_soft_targets(
                sample["net_input"]["source"],
                sample["net_input"].get("padding_mask", None),
                masked_indices,
                T,
            )

            # ---- Student K-way logits (N_masked, K) with grad ----
            # Compute through model's final_proj and label_embs (both have grad).
            s_proj = model.final_proj(encoder_out[masked_indices])
            if model.untie_final_proj:
                s_proj = s_proj.chunk(len(model.num_classes), dim=-1)[0]

            s_label_embs = model.label_embs_concat
            if hasattr(model, "num_classes"):
                s_label_embs = s_label_embs.split(model.num_classes, 0)[0]

            student_logits = self._kway_cosine_logits(
                s_proj, s_label_embs, model.logit_temp
            )

            # Alignment checks
            N_t, K_t = teacher_probs.shape
            N_s, K_s = student_logits.shape
            assert N_t == N_s, (
                f"Masked-position count mismatch: teacher={N_t}, student={N_s}"
            )
            assert K_t == K_s, (
                f"Class count mismatch: teacher K={K_t}, student K={K_s}. "
                f"Ensure same k-means K."
            )

            # KL(teacher_probs || student_probs)
            student_log_probs = F.log_softmax(student_logits / T, dim=-1)
            kl = F.kl_div(
                student_log_probs,
                teacher_probs.float(),
                reduction="sum" if reduce else "none",
                log_target=False,
            )
            kl = kl * (T ** 2)  # standard distillation scaling

            logging_output["masked_kl"] = kl.detach().item()

            with torch.no_grad():
                tp = teacher_probs.float()
                ent = -(tp * torch.log(tp + 1e-10)).sum(-1).mean()
                logging_output["teacher_entropy"] = ent.item()

            loss += self.distill_alpha * self.pred_masked_weight * kl

        # ================================================================
        # Extra losses (features_pen, etc.)
        # ================================================================
        if self.loss_weights is not None:
            assert hasattr(model, "get_extra_losses")
            extra_losses, names = model.get_extra_losses(net_output)
            if torch.is_tensor(extra_losses):
                extra_losses = [extra_losses]
                names = [names]
            if len(self.loss_weights) == 1 and len(extra_losses) != 1:
                self.loss_weights = [self.loss_weights[0]] * len(extra_losses)
            assert len(extra_losses) == len(self.loss_weights), (
                f"{len(extra_losses)}, {len(self.loss_weights)}"
            )
            for p, n, coef in zip(extra_losses, names, self.loss_weights):
                if coef != 0 and p is not None:
                    p = coef * p.float() * sample_size
                    loss += p
                    logging_output[f"loss_{n}"] = p.item()

        # ================================================================
        # Final logging
        # ================================================================
        logging_output = {
            "loss": loss.item() if reduce else loss,
            "ntokens": sample_size,
            "nsentences": sample["id"].numel(),
            "sample_size": sample_size,
            **logging_output,
        }

        for lk in self.log_keys:
            if lk in net_output:
                logging_output[lk] = float(net_output[lk])

        def compute_correct(logits):
            if logits.numel() == 0:
                return 0, 0
            assert logits.dim() > 1, logits.shape
            mx = logits.argmax(-1) == 0
            mn = logits.argmin(-1) == 0
            both = mx & mn
            return (mx.long().sum().item() - both.long().sum().item(), mx.numel())

        with torch.no_grad():
            for i, logp_m in enumerate(logp_m_list):
                c, n = compute_correct(logp_m)
                logging_output[f"correct_m_{i}"] = c
                logging_output[f"count_m_{i}"] = n
            for i, logp_u in enumerate(logp_u_list):
                c, n = compute_correct(logp_u)
                logging_output[f"correct_u_{i}"] = c
                logging_output[f"count_u_{i}"] = n

        return loss, sample_size, logging_output

    # ------------------------------------------------------------------
    # Metric aggregation
    # ------------------------------------------------------------------
    @staticmethod
    def reduce_metrics(logging_outputs) -> None:
        loss_sum = sum(log.get("loss", 0) for log in logging_outputs)
        ntokens = sum(log.get("ntokens", 0) for log in logging_outputs)
        sample_size = sum(log.get("sample_size", 0) for log in logging_outputs)

        metrics.log_scalar(
            "loss", loss_sum / sample_size / math.log(2), sample_size, round=3
        )
        if sample_size != ntokens:
            metrics.log_scalar(
                "nll_loss", loss_sum / ntokens / math.log(2), ntokens, round=3
            )
            metrics.log_derived(
                "ppl", lambda m: utils.get_perplexity(m["nll_loss"].avg)
            )
        else:
            metrics.log_derived(
                "ppl", lambda m: utils.get_perplexity(m["loss"].avg)
            )

        if "masked_kl" in logging_outputs[0]:
            kl_sum = sum(log.get("masked_kl", 0) for log in logging_outputs)
            metrics.log_scalar(
                "masked_kl", kl_sum / sample_size / math.log(2),
                sample_size, round=3,
            )

        if "teacher_entropy" in logging_outputs[0]:
            ent = sum(log.get("teacher_entropy", 0) for log in logging_outputs)
            metrics.log_scalar("teacher_entropy", ent / len(logging_outputs), round=3)

        counts = {}
        for lk in logging_outputs[0].keys():
            if lk.startswith("count_"):
                val = sum(log[lk] for log in logging_outputs)
                metrics.log_scalar(lk, val)
                counts[lk] = val

        for lk in logging_outputs[0].keys():
            if lk.startswith("loss_"):
                val = sum(log[lk] for log in logging_outputs)
                metrics.log_scalar(lk, val / sample_size / math.log(2), round=3)
            elif lk.startswith("correct_"):
                val = sum(log[lk] for log in logging_outputs)
                metrics.log_scalar(lk, val / counts[re.sub("correct", "count", lk)])

    @staticmethod
    def logging_outputs_can_be_summed() -> bool:
        return False


class _nullcontext:
    """Minimal no-op context manager."""

    def __enter__(self):
        return None

    def __exit__(self, *args):
        pass
