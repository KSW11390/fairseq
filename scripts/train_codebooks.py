#!/usr/bin/env python3
"""
Phase 1: Train N RepCodec codebooks on frozen teacher HuBERT representations.

Architecture: Encoder (1D-conv) → Projector → EMA VQ → Decoder (1D-conv).
Loss: MSE(reconstructed, original) + commitment_loss.
Codebook entries are updated via EMA — no gradient flows through VQ.

RepCodec reference: https://github.com/mct10/RepCodec

Key decisions vs RepCodec original:
  - No temporal downsampling (stride=1): we need one index per frame.
  - Single VQ per codec (no residual-VQ stacking).
  - Input: teacher hidden states (not raw audio); no audio encoder needed.
  - Step-based training (not epoch-based) to match fairseq convention.

Usage (A40, 24 codebooks for layers 1-12):
    python scripts/train_codebooks.py \\
        --teacher_ckpt /path/to/hubert_base_ls960.pt \\
        --tsv /path/to/train.tsv \\
        --codebooks "1:32,1:512,2:32,2:512,...,12:32,12:512" \\
        --steps 50000 \\
        --out_dir /path/to/codebooks/

Codebook name convention: l<layer>k<K>  (e.g., l8k32, l9k512)
"""

import argparse
import logging
import os
import random
import sys
import time
from collections import defaultdict
from pathlib import Path

import numpy as np
import torch
import torch.nn.functional as F
import torch.optim as optim

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

import fairseq
from fairseq.modules.repcodec_codec import RepCodecLayer

try:
    import torchaudio
except ImportError:
    raise ImportError("pip install torchaudio")

logging.basicConfig(
    format="%(asctime)s | %(levelname)s | %(name)s | %(message)s",
    level=logging.INFO,
    datefmt="%Y-%m-%d %H:%M:%S",
)
logger = logging.getLogger(__name__)


# ---------------------------------------------------------------------------
# CLI
# ---------------------------------------------------------------------------

def parse_args():
    p = argparse.ArgumentParser(formatter_class=argparse.ArgumentDefaultsHelpFormatter)
    p.add_argument("--teacher_ckpt", required=True)
    p.add_argument("--teacher_dim", type=int, default=768,
                   help="Teacher feature dimension (768 for HuBERT Base)")
    p.add_argument("--tsv", required=True,
                   help="Fairseq TSV manifest (train split)")
    p.add_argument(
        "--codebooks",
        default="1:32,1:512,2:32,2:512,3:32,3:512,4:32,4:512,5:32,5:512,"
                "6:32,6:512,7:32,7:512,8:32,8:512,9:32,9:512,10:32,10:512,"
                "11:32,11:512,12:32,12:512",
        help="Comma-separated layer:K pairs, e.g. '1:32,1:512,...,12:32,12:512'",
    )
    p.add_argument("--decay", type=float, default=0.99,
                   help="EMA decay rate for VQ codebook")
    p.add_argument("--steps", type=int, default=50000)
    p.add_argument("--lr", type=float, default=1e-4,
                   help="Adam LR for encoder and decoder parameters")
    p.add_argument("--encode_dim", type=int, default=256,
                   help="Hidden channels in 1D-conv encoder/decoder")
    p.add_argument("--num_conv_layers", type=int, default=2,
                   help="Number of ConvBlock stages in encoder and decoder")
    p.add_argument("--sample_rate", type=int, default=16000)
    p.add_argument("--max_sec", type=float, default=15.0)
    p.add_argument("--batch_sec", type=float, default=80.0)
    p.add_argument("--out_dir", required=True)
    p.add_argument("--device", default="cuda")
    p.add_argument("--seed", type=int, default=1337)
    p.add_argument("--log_interval", type=int, default=200)
    p.add_argument("--save_interval", type=int, default=10000)
    return p.parse_args()


def parse_codebook_specs(spec_str):
    """'1:32,1:512,...,12:512' → [('l1k32', 1, 32), ('l1k512', 1, 512), ...]"""
    specs = []
    for s in spec_str.strip().split(","):
        layer, k = s.strip().split(":")
        layer, k = int(layer), int(k)
        name = f"l{layer}k{k}"
        specs.append((name, layer, k))
    return specs


# ---------------------------------------------------------------------------
# Data helpers
# ---------------------------------------------------------------------------

def load_tsv(tsv_path):
    entries = []
    with open(tsv_path) as f:
        root = f.readline().strip()
        for idx, line in enumerate(f):
            parts = line.strip().split("\t")
            if len(parts) >= 2:
                entries.append((root, parts[0], int(parts[1]), idx))
    logger.info(f"Loaded {len(entries)} entries from {tsv_path}")
    return entries


def build_batches(entries, max_samples, batch_samples):
    shuffled = entries[:]
    random.shuffle(shuffled)
    batches, cur, cur_len = [], [], 0
    for root, path, n_frames, idx in shuffled:
        n = min(n_frames, max_samples)
        if cur_len + n > batch_samples and cur:
            batches.append(cur)
            cur, cur_len = [], 0
        cur.append((root, path, n_frames, idx))
        cur_len += n
    if cur:
        batches.append(cur)
    return batches


def load_batch(batch_entries, sample_rate, max_samples, device):
    wavs = []
    for root, path, *_ in batch_entries:
        wav, sr = torchaudio.load(os.path.join(root, path))
        if sr != sample_rate:
            wav = torchaudio.functional.resample(wav, sr, sample_rate)
        wavs.append(wav.mean(0)[:max_samples])
    lengths = torch.tensor([w.shape[0] for w in wavs])
    max_len = int(lengths.max())
    padded = torch.stack([F.pad(w, (0, max_len - w.shape[0])) for w in wavs])
    return padded.to(device), lengths.to(device)


def make_padding_mask(lengths, max_len, cnn_stride=320):
    feat_lens = (lengths // cnn_stride).long()
    T = max_len // cnn_stride
    return torch.arange(T, device=lengths.device).unsqueeze(0) >= feat_lens.unsqueeze(1)


# ---------------------------------------------------------------------------
# Teacher
# ---------------------------------------------------------------------------

def load_teacher(ckpt_path, device):
    logger.info(f"Loading teacher from {ckpt_path}")
    models, _, _ = fairseq.checkpoint_utils.load_model_ensemble_and_task([ckpt_path])
    teacher = models[0].eval().to(device)
    for p in teacher.parameters():
        p.requires_grad = False
    logger.info(f"Teacher: {sum(p.numel() for p in teacher.parameters())/1e6:.1f}M params")
    return teacher


@torch.no_grad()
def extract_all_layers(teacher, source, padding_mask, unique_layers, use_fp16):
    """
    Run teacher forward once, capture features at all requested layers.
    Returns dict[layer → Tensor[N_valid, D]] — valid (non-padded) frames only.
    """
    layer_feats_raw = {}
    with torch.cuda.amp.autocast(enabled=use_fp16):
        for layer in unique_layers:
            feats, _ = teacher.extract_features(
                source=source, padding_mask=padding_mask,
                mask=False, output_layer=layer,
            )  # [B, T_feat, D]
            layer_feats_raw[layer] = feats.float()

    T_actual = next(iter(layer_feats_raw.values())).shape[1]
    T_mask = min(padding_mask.shape[1], T_actual)
    valid_mask = ~padding_mask[:, :T_mask]          # [B, T], True = valid

    result = {}
    for layer, feats in layer_feats_raw.items():
        # Keep batch structure [B, T, D] with valid mask — needed for Conv1d
        result[layer] = feats[:, :T_mask, :]        # [B, T_valid_max, D]
    # Also return valid_mask for masking MSE loss
    return result, valid_mask


# ---------------------------------------------------------------------------
# Save / Load
# ---------------------------------------------------------------------------

def save_codebooks(codecs, specs, args, step):
    os.makedirs(args.out_dir, exist_ok=True)
    path = os.path.join(args.out_dir, f"codebooks_step{step:06d}.pt")
    payload = {
        "step": step,
        "args": vars(args),
        "codecs": {}
    }
    for (name, layer, k), codec in zip(specs, codecs):
        payload["codecs"][name] = {
            "state_dict": codec.state_dict(),
            "input_dim": args.teacher_dim,
            "code_dim": args.teacher_dim,
            "codebook_size": k,
            "encode_dim": args.encode_dim,
            "num_conv_layers": args.num_conv_layers,
            "layer": layer,
            "name": name,
        }
    torch.save(payload, path)
    latest = os.path.join(args.out_dir, "codebooks_latest.pt")
    if os.path.islink(latest) or os.path.exists(latest):
        os.remove(latest)
    os.symlink(os.path.basename(path), latest)
    logger.info(f"Saved checkpoint → {path}")


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------

def main():
    args = parse_args()
    random.seed(args.seed)
    torch.manual_seed(args.seed)
    device = torch.device(args.device if torch.cuda.is_available() else "cpu")
    use_fp16 = (device.type == "cuda")
    logger.info(f"Device: {device}, fp16={use_fp16}")

    specs = parse_codebook_specs(args.codebooks)
    logger.info(f"Codebooks: {[(n, f'L{l}', f'K={k}') for n,l,k in specs]}")

    # Group codebook indices by teacher layer (one forward pass per layer group)
    layer_to_cb_idx = defaultdict(list)
    for i, (name, layer, k) in enumerate(specs):
        layer_to_cb_idx[layer].append(i)
    unique_layers = sorted(layer_to_cb_idx.keys())
    logger.info(f"Unique teacher layers needed: {unique_layers}")

    teacher = load_teacher(args.teacher_ckpt, device)

    # One RepCodecLayer per spec — Encoder+VQ+Decoder
    codecs = [
        RepCodecLayer(
            input_dim=args.teacher_dim,
            codebook_size=k,
            code_dim=args.teacher_dim,
            encode_dim=args.encode_dim,
            num_conv_layers=args.num_conv_layers,
            decay=args.decay,
        ).to(device).train()
        for _, _, k in specs
    ]

    # Optimizer: Encoder and Decoder parameters only (VQ updated by EMA)
    all_enc_dec_params = []
    for codec in codecs:
        all_enc_dec_params += list(codec.encoder.parameters())
        all_enc_dec_params += list(codec.projector.parameters())
        all_enc_dec_params += list(codec.decoder.parameters())
    optimizer = optim.Adam(all_enc_dec_params, lr=args.lr)

    entries = load_tsv(args.tsv)
    max_samples = int(args.max_sec * args.sample_rate)
    batch_samples = int(args.batch_sec * args.sample_rate)

    # Exponential moving average metrics for logging
    mse_ema  = {name: 0.0 for name, _, _ in specs}
    ppl_ema  = {name: 0.0 for name, _, _ in specs}
    metric_alpha = 0.99

    step = 0
    t0 = time.time()

    while step < args.steps:
        for batch_entries in build_batches(entries, max_samples, batch_samples):
            if step >= args.steps:
                break

            # ---- Load batch -----------------------------------------------
            try:
                source, lengths = load_batch(
                    batch_entries, args.sample_rate, max_samples, device
                )
            except Exception as e:
                logger.warning(f"Batch load failed: {e}")
                continue

            B, T_wav = source.shape
            padding_mask = make_padding_mask(lengths, T_wav)
            if padding_mask.all():
                continue

            # ---- Teacher forward — all layers in one pass ------------------
            try:
                layer_feats, valid_mask = extract_all_layers(
                    teacher, source, padding_mask, unique_layers, use_fp16
                )
            except Exception as e:
                logger.warning(f"Teacher forward failed: {e}")
                continue

            # ---- RepCodec forward + MSE loss per codebook -----------------
            optimizer.zero_grad()
            total_loss = torch.tensor(0.0, device=device)

            for i, (name, layer, k) in enumerate(specs):
                x = layer_feats[layer]                  # [B, T_feat, D]
                vm = valid_mask                          # [B, T_feat]

                rec, vq_loss, perplexity = codecs[i](x)  # [B, T, D], scalar, scalar

                # MSE only on valid (non-padded) frames
                if vm.any():
                    mse = F.mse_loss(
                        rec[vm],   # [N_valid, D]
                        x[vm],     # [N_valid, D]
                    )
                else:
                    mse = F.mse_loss(rec, x)

                loss_i = mse + vq_loss
                total_loss = total_loss + loss_i

                mse_ema[name] = metric_alpha * mse_ema[name] + (1 - metric_alpha) * mse.item()
                ppl_ema[name]  = metric_alpha * ppl_ema[name]  + (1 - metric_alpha) * perplexity.item()

            total_loss.backward()
            optimizer.step()

            step += 1

            if step % args.log_interval == 0:
                metrics_str = " | ".join(
                    f"{name} mse={mse_ema[name]:.4f} ppl={ppl_ema[name]:.1f}"
                    for name, _, _ in specs
                )
                logger.info(
                    f"step={step:6d}/{args.steps} | {metrics_str} | "
                    f"elapsed={time.time()-t0:.0f}s"
                )

            if step % args.save_interval == 0:
                save_codebooks(codecs, specs, args, step)

    save_codebooks(codecs, specs, args, step)
    logger.info("Codebook training complete!")


if __name__ == "__main__":
    main()
