"""
RepCodecLayer — Full RepCodec codec for a single (layer, K) codebook.

Mirrors the architecture of RepCodec (mct10/RepCodec, ByteDance / Chutong Meng):
    Encoder (1D-conv + residual) → Projector (linear) → EMA VQ → Decoder (1D-conv + residual)

Key differences from the original RepCodec:
  - No temporal downsampling (enc_strides = dec_strides = 1).
    RepCodec is designed for audio codec (stride > 1); we need frame-aligned indices.
  - VQ is our existing EMACodebook (same EMA equations as RepCodec VectorQuantize).
  - Loss = MSE(reconstructed, original) + commitment_loss.
    The Encoder/Decoder parameters are updated by gradient;
    the codebook (embed, embed_avg, cluster_size buffers) is updated by EMA only.
  - No residual-VQ stacking — one codebook per RepCodecLayer instance.
  - Input/output shape: [B, T, D] (batch-first, unlike the original 1D-conv convention).

References:
  - RepCodec: https://github.com/mct10/RepCodec
  - AudioDec: https://github.com/facebookresearch/AudioDec
  - van den Oord et al., "Neural Discrete Representation Learning" (VQ-VAE)
"""

import logging
from typing import List, Tuple

import torch
import torch.nn as nn
import torch.nn.functional as F

from fairseq.modules.ema_codebook import EMACodebook

logger = logging.getLogger(__name__)


# ---------------------------------------------------------------------------
# Building blocks (mirrors RepCodec's residual_unit.py)
# ---------------------------------------------------------------------------

class ResidualUnit(nn.Module):
    """
    1D-conv residual block used in both Encoder and Decoder.
    kernel_size=3 with same padding keeps temporal resolution.
    """

    def __init__(self, channels: int, dilation: int = 1):
        super().__init__()
        pad = dilation  # same-padding for k=3
        self.net = nn.Sequential(
            nn.ELU(),
            nn.Conv1d(channels, channels, kernel_size=3,
                      dilation=dilation, padding=pad),
            nn.ELU(),
            nn.Conv1d(channels, channels, kernel_size=1),
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return x + self.net(x)


class ConvBlock(nn.Module):
    """
    One stride-1 conv block: point-wise projection + stack of residual units.
    Matches RepCodec's EncoderBlock / DecoderBlock (with stride=1 specialisation).
    """

    def __init__(self, in_channels: int, out_channels: int,
                 num_residuals: int = 2, dilations: Tuple[int, ...] = (1, 3)):
        super().__init__()
        layers: List[nn.Module] = [nn.Conv1d(in_channels, out_channels, kernel_size=1)]
        for d in dilations[:num_residuals]:
            layers.append(ResidualUnit(out_channels, dilation=d))
        self.net = nn.Sequential(*layers)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.net(x)


# ---------------------------------------------------------------------------
# RepCodecLayer — the main codec module
# ---------------------------------------------------------------------------

class RepCodecLayer(nn.Module):
    """
    Single-codebook RepCodec layer.

    Args:
        input_dim:      D in teacher feature (768 for HuBERT Base).
        code_dim:       VQ codebook dimension.  Defaults to input_dim.
        codebook_size:  K — number of codebook entries.
        encode_dim:     Hidden channels in the 1D-conv encoder.
        decode_dim:     Hidden channels in the 1D-conv decoder.
        num_conv_layers: Number of ConvBlock stages in encoder and decoder.
        decay:          EMA decay for the VQ codebook.
        commitment:     Commitment loss weight λ (inside EMACodebook).
    """

    def __init__(
        self,
        input_dim: int,
        codebook_size: int,
        code_dim: int = None,
        encode_dim: int = 256,
        decode_dim: int = 256,
        num_conv_layers: int = 2,
        decay: float = 0.99,
        commitment: float = 1.0,
    ):
        super().__init__()
        if code_dim is None:
            code_dim = input_dim
        self.input_dim = input_dim
        self.code_dim = code_dim
        self.codebook_size = codebook_size
        self._encode_dim = encode_dim
        self._decode_dim = decode_dim
        self._num_conv_layers = num_conv_layers

        # ---- Encoder (input_dim → encode_dim → code_dim) -----------------
        enc_layers: List[nn.Module] = [
            nn.Conv1d(input_dim, encode_dim, kernel_size=3, padding=1)
        ]
        for i in range(num_conv_layers):
            enc_layers.append(ConvBlock(encode_dim, encode_dim))
        self.encoder = nn.Sequential(*enc_layers)

        # Projector: encode_dim → code_dim (linear, no bias, mirrors RepCodec)
        self.projector = nn.Conv1d(encode_dim, code_dim, kernel_size=1, bias=False)

        # ---- EMA VQ --------------------------------------------------------
        self.vq = EMACodebook(
            num_codes=codebook_size,
            dim=code_dim,
            decay=decay,
            commitment=commitment,
        )

        # ---- Decoder (code_dim → decode_dim → input_dim) ------------------
        dec_layers: List[nn.Module] = [
            nn.Conv1d(code_dim, decode_dim, kernel_size=3, padding=1)
        ]
        for i in range(num_conv_layers):
            dec_layers.append(ConvBlock(decode_dim, decode_dim))
        dec_layers.append(nn.Conv1d(decode_dim, input_dim, kernel_size=1))
        self.decoder = nn.Sequential(*dec_layers)

        logger.info(
            f"RepCodecLayer: D={input_dim}, code_dim={code_dim}, K={codebook_size}, "
            f"encode_dim={encode_dim}, decode_dim={decode_dim}, nlayers={num_conv_layers}"
        )

    # ------------------------------------------------------------------
    # Forward — training (EMA update active)
    # ------------------------------------------------------------------

    def forward(
        self, x: torch.Tensor
    ) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        """
        Args:
            x: [B, T, D]  teacher feature at one layer.

        Returns:
            reconstructed:  [B, T, D]  MSE target.
            vq_loss:        scalar   commitment loss (from EMACodebook).
            perplexity:     scalar   codebook utilisation metric.
        """
        # [B, T, D] → [B, D, T]  for Conv1d
        xt = x.transpose(1, 2)

        # Encode
        z = self.encoder(xt)         # [B, encode_dim, T]
        z = self.projector(z)        # [B, code_dim, T]

        # VQ: expects [B, T, code_dim] or [N, code_dim]
        z_bt = z.transpose(1, 2)                      # [B, T, code_dim]
        zq_bt, vq_loss, perplexity = self.vq(z_bt)   # straight-through
        zq = zq_bt.transpose(1, 2)                    # [B, code_dim, T]

        # Decode
        rec = self.decoder(zq)       # [B, D, T]
        rec = rec.transpose(1, 2)    # [B, T, D]

        return rec, vq_loss, perplexity

    # ------------------------------------------------------------------
    # Encode — inference (no EMA update, returns indices)
    # ------------------------------------------------------------------

    @torch.no_grad()
    def encode(self, x: torch.Tensor) -> torch.Tensor:
        """
        Assign each frame to the nearest codebook entry.

        Args:
            x: [B, T, D]

        Returns:
            indices: [B, T]  integer codebook assignments (0-based).
        """
        xt = x.transpose(1, 2)
        z = self.encoder(xt)
        z = self.projector(z)
        z_bt = z.transpose(1, 2)                      # [B, T, code_dim]
        _, indices = self.vq.encode(z_bt)             # [B, T]
        return indices

    # ------------------------------------------------------------------
    # Persistence — save / load full model (encoder+projector+VQ+decoder)
    # ------------------------------------------------------------------

    def save(self, path: str) -> None:
        torch.save(
            {
                "input_dim": self.input_dim,
                "code_dim": self.code_dim,
                "codebook_size": self.codebook_size,
                "encode_dim": self._encode_dim,
                "decode_dim": self._decode_dim,
                "num_conv_layers": self._num_conv_layers,
                "state_dict": self.state_dict(),
            },
            path,
        )
        logger.info(f"RepCodecLayer saved → {path}")

    @classmethod
    def load(cls, path: str, device: str = "cpu") -> "RepCodecLayer":
        ckpt = torch.load(path, map_location=device)
        model = cls(
            input_dim=ckpt["input_dim"],
            codebook_size=ckpt["codebook_size"],
            code_dim=ckpt["code_dim"],
            encode_dim=ckpt.get("encode_dim", 256),
            decode_dim=ckpt.get("decode_dim", 256),
            num_conv_layers=ckpt.get("num_conv_layers", 2),
        )
        model.load_state_dict(ckpt["state_dict"])
        model.eval()
        return model
