# Copyright (c) Facebook, Inc. and its affiliates.
#
# This source code is licensed under the MIT license found in the
# LICENSE file in the root directory of this source tree.

"""
s3prl-compatible upstream wrapper for DICEHuBERT (distilled HuBERT).

Exposes all transformer layer hidden states for weighted-sum evaluation
on SUPERB benchmark tasks.

Usage with s3prl:
    python run_downstream.py \
        --upstream custom \
        --upstream_expert /path/to/this/hubconf.py \
        --upstream_ckpt /path/to/dicehubert/checkpoint_best.pt \
        --downstream <TASK> ...
"""

import logging
import os
import sys
from typing import Dict, List, Optional

import torch
import torch.nn as nn
import torch.nn.functional as F

# Ensure fairseq is importable when loaded from s3prl context
_FAIRSEQ_ROOT = os.path.join(os.path.dirname(__file__), "..", "..", "..")
if _FAIRSEQ_ROOT not in sys.path:
    sys.path.insert(0, os.path.abspath(_FAIRSEQ_ROOT))

import fairseq
import fairseq.checkpoint_utils

logger = logging.getLogger(__name__)

SAMPLE_RATE = 16000
# CNN strides: [(512,10,5)] + [(512,3,2)]*4 + [(512,2,2)]*2
# total stride = 5 * 2^4 * 2^2 = 320
CNN_DOWNSAMPLE_RATE = 320


class UpstreamExpert(nn.Module):
    """
    s3prl-compatible upstream that wraps a fairseq HuBERT checkpoint
    and exposes all transformer layer hidden states.
    """

    def __init__(self, ckpt: str, model_config: Optional[str] = None, **kwargs):
        super().__init__()

        models, cfg, task = fairseq.checkpoint_utils.load_model_ensemble_and_task(
            [ckpt]
        )
        self.model = models[0]
        self.model.eval()
        self.task = task

        self.normalize = task.cfg.normalize
        # encoder_layers via direct inspection (model.cfg not always exposed)
        self.encoder_layers = len(self.model.encoder.layers)

        logger.info(
            f"Loaded HuBERT checkpoint: layers={self.encoder_layers}, "
            f"normalize={self.normalize}"
        )

    def train(self, mode: bool = True):
        # Keep HuBERT model in eval mode even when s3prl runner sets upstream.train()
        # (upstream is always frozen in SUPERB eval — dropout must be disabled)
        super().train(mode)
        self.model.eval()
        return self

    def get_downsample_rates(self, key: str = "hidden_states") -> int:
        return CNN_DOWNSAMPLE_RATE

    @torch.no_grad()
    def forward(
        self, wavs: List[torch.Tensor]
    ) -> Dict[str, List[torch.Tensor]]:
        """
        Args:
            wavs: list of 1-D float tensors, each (T_i,) at 16 kHz

        Returns:
            dict with:
                "hidden_states": list of (B, T_feat, D) tensors,
                    length = 1 (CNN) + encoder_layers (transformer)
                "last_hidden_state": (B, T_feat, D) tensor
        """
        device = next(self.model.parameters()).device

        # --- pad waveforms to same length ---
        wav_lengths = [len(w) for w in wavs]
        max_len = max(wav_lengths)
        padded = torch.zeros(len(wavs), max_len, device=device)
        for i, w in enumerate(wavs):
            padded[i, : len(w)] = w.to(device)

        # build padding mask: True where padded
        padding_mask = torch.ones(len(wavs), max_len, dtype=torch.bool, device=device)
        for i, l in enumerate(wav_lengths):
            padding_mask[i, :l] = False

        # normalize waveform if required
        if self.normalize:
            padded = F.layer_norm(padded, padded.shape[1:])

        # --- CNN feature extraction ---
        features = self.model.forward_features(padded)  # (B, C, T_feat)
        features = features.transpose(1, 2)  # (B, T_feat, C)
        features = self.model.layer_norm(features)

        # adjust padding mask to feature time dimension
        feat_padding_mask = self.model.forward_padding_mask(features, padding_mask)

        if self.model.post_extract_proj is not None:
            features = self.model.post_extract_proj(features)  # (B, T_feat, D)

        # layer 0: CNN output (post-projection)
        cnn_output = features.clone()

        # --- transformer encoder: single pass, capture all layer_results ---
        # model.encoder is TransformerEncoder from wav2vec2.py
        # encoder.extract_features returns: (x, layer_results)
        #   x: (B, T, D) — final layer output
        #   layer_results: list of (x_T_first, z, lr) per layer
        #     where x_T_first is (T, B, D)
        final_out, layer_results = self.model.encoder(
            features,
            padding_mask=feat_padding_mask,
            layer=None,  # run all layers
        )

        # --- build hidden_states list ---
        hidden_states = [cnn_output]  # layer 0 = CNN
        for lr_x, _, _ in layer_results:
            # lr_x: (T, B, D) → (B, T, D)
            hidden_states.append(lr_x.transpose(0, 1))

        # zero out padded positions
        if feat_padding_mask is not None and feat_padding_mask.any():
            mask = feat_padding_mask.unsqueeze(-1)  # (B, T, 1)
            hidden_states = [h.masked_fill(mask, 0.0) for h in hidden_states]

        return {
            "hidden_states": hidden_states,
            "last_hidden_state": hidden_states[-1],
        }
