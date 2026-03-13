"""
EMA-updated Vector Quantization Codebook.

Follows RepCodec's VectorQuantize implementation (mct10/RepCodec, ByteDance).
  - Codebook updated via exponential moving average — no gradient flows back.
  - ema_inplace / laplace_smoothing helpers match RepCodec exactly.
  - Straight-through estimator in forward().
  - Perplexity as codebook utilisation metric.

Reference:
  - RepCodec: https://github.com/mct10/RepCodec
  - AudioDec: https://github.com/facebookresearch/AudioDec
  - van den Oord et al., "Neural Discrete Representation Learning" (VQ-VAE)
"""

import logging
from typing import Tuple

import torch
import torch.nn as nn
import torch.nn.functional as F

logger = logging.getLogger(__name__)


class EMACodebook(nn.Module):
    """
    EMA-updated vector quantization codebook (RepCodec-style).

    Buffers:
        embed        [K, D]  — codebook entries, learned via EMA
        cluster_size [K]     — EMA of per-entry assignment counts
        embed_avg    [K, D]  — EMA of per-entry embedding sums

    Args:
        num_codes:  Number of codebook entries (K).
        dim:        Feature dimension (D).
        decay:      EMA decay rate γ (default 0.99).
        commitment: Commitment loss weight λ (default 1.0).
        eps:        Laplace-smoothing ε (default 1e-5).
    """

    def __init__(
        self,
        num_codes: int,
        dim: int,
        decay: float = 0.99,
        commitment: float = 1.0,
        eps: float = 1e-5,
    ):
        super().__init__()
        self.num_codes = num_codes
        self.dim = dim
        self.decay = decay
        self.commitment = commitment
        self.eps = eps

        # Codebook buffers — randn init, no lazy init (follows RepCodec)
        embed = torch.randn(num_codes, dim)
        self.register_buffer("embed", embed)                          # [K, D]
        self.register_buffer("cluster_size", torch.zeros(num_codes)) # [K]
        self.register_buffer("embed_avg", embed.clone())              # [K, D]

    @property
    def codebook(self) -> torch.Tensor:
        """Alias: embed [K, D]."""
        return self.embed

    # ------------------------------------------------------------------
    # EMA helpers — identical to RepCodec VectorQuantize
    # ------------------------------------------------------------------

    def ema_inplace(self, moving_avg: torch.Tensor, new: torch.Tensor, decay: float) -> None:
        """In-place EMA:  moving_avg = γ · moving_avg + (1−γ) · new"""
        moving_avg.data.mul_(decay).add_(new, alpha=(1 - decay))

    def laplace_smoothing(self, x: torch.Tensor, n_categories: int, eps: float = 1e-5) -> torch.Tensor:
        """Laplace-smoothed cluster probabilities: (x + ε) / (Σx + K·ε)"""
        return (x + eps) / (x.sum() + n_categories * eps)

    # ------------------------------------------------------------------
    # Distance computation
    # ------------------------------------------------------------------

    @torch.no_grad()
    def _compute_distances(self, z_flat: torch.Tensor) -> torch.Tensor:
        """Squared L2 distances from z_flat [N, D] to all K entries. Returns [N, K]."""
        return (
            z_flat.pow(2).sum(1, keepdim=True)      # [N, 1]
            - 2 * (z_flat @ self.embed.T)            # [N, K]
            + self.embed.pow(2).sum(1).unsqueeze(0)  # [1, K]
        )

    # ------------------------------------------------------------------
    # Forward with EMA update — RepCodec VectorQuantize.forward()
    # ------------------------------------------------------------------

    def forward(
        self, z: torch.Tensor
    ) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        """
        Quantize z and update codebook via EMA during training.

        Mirrors RepCodec VectorQuantize.forward() exactly.

        EMA update equations:
            N̂_k  ← γ · N̂_k  + (1−γ) · n_k          (cluster-size EMA)
            m̂_k  ← γ · m̂_k  + (1−γ) · Σ_{j→k} z_j  (embedding-sum EMA)
            e_k  =  m̂_k / laplace_smooth(N̂_k)         (codebook entry)

        Args:
            z: [B, T, D] or [N, D]

        Returns:
            quantize:   straight-through quantized z (same shape as z).
            loss:       commitment loss  MSE(quantize.detach(), z) · λ.
            perplexity: codebook utilisation  exp(−Σ p·log p).
        """
        orig_shape = z.shape
        flatten = z.reshape(-1, self.dim)  # [N, D]
        dtype = flatten.dtype

        dist = self._compute_distances(flatten)          # [N, K]
        _, embed_ind = (-dist).max(1)                    # [N]  (argmin dist)
        embed_onehot = F.one_hot(embed_ind, self.num_codes).to(dtype)  # [N, K]

        # ---- EMA update (training only) ----------------------------------
        if self.training:
            # Cluster-size EMA
            self.ema_inplace(self.cluster_size, embed_onehot.sum(0), self.decay)

            # Embedding-sum EMA:  Σ_{j assigned to k} z_j
            embed_sum = embed_onehot.T @ flatten          # [K, D]
            self.ema_inplace(self.embed_avg, embed_sum, self.decay)

            # Normalize with Laplace smoothing
            cluster_size = (
                self.laplace_smoothing(self.cluster_size, self.num_codes, self.eps)
                * self.cluster_size.sum()
            )
            self.embed.data.copy_(self.embed_avg / cluster_size.unsqueeze(1))

        # ---- Quantize ----------------------------------------------------
        quantize = F.embedding(embed_ind, self.embed)    # [N, D]

        # Commitment loss (no gradient into codebook side)
        loss = F.mse_loss(quantize.detach(), flatten) * self.commitment

        # Straight-through estimator: gradient passes through to z
        quantize = flatten + (quantize - flatten).detach()
        quantize = quantize.reshape(orig_shape)

        # Perplexity (codebook utilisation metric)
        avg_probs = embed_onehot.mean(0)                 # [K]
        perplexity = torch.exp(-torch.sum(avg_probs * torch.log(avg_probs + 1e-10)))

        return quantize, loss, perplexity

    # ------------------------------------------------------------------
    # Encode (inference) — RepCodec VectorQuantize.forward_index()
    # ------------------------------------------------------------------

    def encode(self, z: torch.Tensor) -> Tuple[torch.Tensor, torch.Tensor]:
        """
        Assign z to nearest codebook entries without EMA update.

        Mirrors RepCodec VectorQuantize.forward_index().

        Args:
            z: [..., D]

        Returns:
            quantize: straight-through quantized z (same shape as z).
            indices:  integer assignment indices (shape z.shape[:-1]).
        """
        orig_shape = z.shape
        flatten = z.reshape(-1, self.dim)

        dist = self._compute_distances(flatten)
        _, indices = (-dist).max(1)                      # [N]
        indices = indices.reshape(orig_shape[:-1])        # [...]

        quantize = F.embedding(indices, self.embed)
        quantize = z + (quantize - z).detach()

        return quantize, indices

    # ------------------------------------------------------------------
    # Persistence
    # ------------------------------------------------------------------

    def save(self, path: str) -> None:
        torch.save(
            {
                "num_codes": self.num_codes,
                "dim": self.dim,
                "decay": self.decay,
                "commitment": self.commitment,
                "state_dict": self.state_dict(),
            },
            path,
        )
        logger.info(f"EMACodebook saved → {path}")

    @classmethod
    def load(cls, path: str, device: str = "cpu") -> "EMACodebook":
        ckpt = torch.load(path, map_location=device)
        cb = cls(
            ckpt["num_codes"],
            ckpt["dim"],
            ckpt.get("decay", 0.99),
            ckpt.get("commitment", 1.0),
        )
        cb.load_state_dict(ckpt["state_dict"])
        cb.eval()
        logger.info(f"EMACodebook loaded ← {path}  (K={cb.num_codes}, D={cb.dim})")
        return cb
