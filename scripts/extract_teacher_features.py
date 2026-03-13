#!/usr/bin/env python3
"""
Pre-extract teacher HuBERT features and cache to disk as float16 .npy files.

Run ONCE before train_codebooks.py to eliminate repeated teacher forward passes.
Codebook training then reads cached features directly — no GPU needed for teacher.

Output layout:
  <out_dir>/
    meta.json          — {"N", "dim", "layers", "sample_rate"}
    l8/
      000000.npy       — float16  [T_i, D]
      000001.npy
      ...
    l9/
      000000.npy
      ...

Storage estimate: ~21 GB per layer for TC-100 (float16).

Usage:
    python scripts/extract_teacher_features.py \\
        --teacher_ckpt /path/to/hubert_base_ls960.pt \\
        --tsv /path/to/train.tsv \\
        --layers "8,9" \\
        --out_dir /path/to/feature_cache/ \\
        --batch_size 16
"""

import argparse
import json
import logging
import os
import sys
import time
from pathlib import Path

import numpy as np
import torch
import torch.nn.functional as F

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

import fairseq

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
    p.add_argument("--tsv", required=True)
    p.add_argument("--layers", default="8,9",
                   help="Comma-separated layer indices to extract (e.g. '8,9')")
    p.add_argument("--out_dir", required=True)
    p.add_argument("--sample_rate", type=int, default=16000)
    p.add_argument("--max_sec", type=float, default=15.0)
    p.add_argument("--batch_size", type=int, default=16)
    p.add_argument("--device", default="cuda")
    p.add_argument("--log_interval", type=int, default=500,
                   help="Log every N samples")
    p.add_argument("--resume", action="store_true",
                   help="Skip samples whose .npy files already exist")
    return p.parse_args()


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

def load_tsv(tsv_path):
    entries = []
    with open(tsv_path) as f:
        root = f.readline().strip()
        for line in f:
            parts = line.strip().split("\t")
            if len(parts) >= 2:
                entries.append((root, parts[0], int(parts[1])))
    logger.info(f"Loaded {len(entries)} entries from {tsv_path}")
    return entries


def load_teacher(ckpt_path, device):
    logger.info(f"Loading teacher from {ckpt_path}")
    models, _, _ = fairseq.checkpoint_utils.load_model_ensemble_and_task([ckpt_path])
    teacher = models[0].eval().to(device)
    for p in teacher.parameters():
        p.requires_grad = False
    logger.info(f"Teacher loaded ({sum(p.numel() for p in teacher.parameters())/1e6:.1f}M params)")
    return teacher


def probe_feature_dim(teacher, entries, sample_rate, max_samples, layer, device, use_fp16):
    root, path, _ = entries[0]
    wav, sr = torchaudio.load(os.path.join(root, path))
    if sr != sample_rate:
        wav = torchaudio.functional.resample(wav, sr, sample_rate)
    wav = wav.mean(0)[:max_samples].to(device)
    with torch.no_grad(), torch.cuda.amp.autocast(enabled=use_fp16):
        feats, _ = teacher.extract_features(
            source=wav.unsqueeze(0), padding_mask=None,
            mask=False, output_layer=layer,
        )
    return feats.shape[-1]


# ---------------------------------------------------------------------------
# Core extraction
# ---------------------------------------------------------------------------

@torch.no_grad()
def extract_and_save_batch(
    teacher, batch_entries, batch_indices, layers, out_dirs,
    sample_rate, max_samples, device, use_fp16,
):
    """
    Load wavs, run teacher forward in fp16, save per-sample .npy (float16).
    Returns the number of successfully saved samples.
    """
    # Load wavs — track which succeeded
    wavs, valid_indices = [], []
    for (root, path, _), idx in zip(batch_entries, batch_indices):
        try:
            wav, sr = torchaudio.load(os.path.join(root, path))
            if sr != sample_rate:
                wav = torchaudio.functional.resample(wav, sr, sample_rate)
            wavs.append(wav.mean(0)[:max_samples])
            valid_indices.append(idx)
        except Exception as e:
            logger.warning(f"[{idx}] load failed: {e}")

    if not wavs:
        return 0

    # Pad batch and build feature-resolution padding mask
    lengths = torch.tensor([w.shape[0] for w in wavs])
    max_len = int(lengths.max())
    padded = torch.stack(
        [F.pad(w, (0, max_len - w.shape[0])) for w in wavs]
    ).to(device)                                          # [B, T_wav]

    feat_lengths = (lengths // 320).tolist()              # CNN stride = 320
    T_feat_max = max_len // 320
    padding_mask = (
        torch.arange(T_feat_max, device=device).unsqueeze(0)
        >= torch.tensor(feat_lengths, device=device).unsqueeze(1)
    )                                                     # [B, T_feat], True = pad

    # Teacher forward for all requested layers (fp16)
    layer_feats = {}
    with torch.cuda.amp.autocast(enabled=use_fp16):
        for layer in layers:
            feats, _ = teacher.extract_features(
                source=padded, padding_mask=padding_mask,
                mask=False, output_layer=layer,
            )                                             # [B, T_feat, D]
            layer_feats[layer] = feats.float().cpu()      # fp32 on CPU

    # Save per-sample .npy (float16, unpadded)
    for b, idx in enumerate(valid_indices):
        T = feat_lengths[b]
        for layer in layers:
            feat = layer_feats[layer][b, :T].numpy().astype(np.float16)  # [T, D]
            np.save(os.path.join(out_dirs[layer], f"{idx:06d}.npy"), feat)

    return len(valid_indices)


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------

def main():
    args = parse_args()
    os.makedirs(args.out_dir, exist_ok=True)
    device = torch.device(args.device if torch.cuda.is_available() else "cpu")
    use_fp16 = (device.type == "cuda")
    layers = [int(l.strip()) for l in args.layers.split(",")]

    # Create per-layer output dirs
    out_dirs = {}
    for layer in layers:
        d = os.path.join(args.out_dir, f"l{layer}")
        os.makedirs(d, exist_ok=True)
        out_dirs[layer] = d

    teacher = load_teacher(args.teacher_ckpt, device)
    entries = load_tsv(args.tsv)
    N = len(entries)
    max_samples = int(args.max_sec * args.sample_rate)

    # Probe feature dimension
    D = probe_feature_dim(teacher, entries, args.sample_rate, max_samples,
                          layers[0], device, use_fp16)
    logger.info(f"Feature dim: D={D} | layers={layers} | fp16={use_fp16}")

    # Write metadata
    meta = {"N": N, "dim": D, "layers": layers, "sample_rate": args.sample_rate}
    with open(os.path.join(args.out_dir, "meta.json"), "w") as f:
        json.dump(meta, f, indent=2)

    t0 = time.time()
    total_saved = 0
    total_skipped = 0

    for batch_start in range(0, N, args.batch_size):
        batch_end = min(batch_start + args.batch_size, N)
        batch_entries = entries[batch_start:batch_end]
        batch_indices = list(range(batch_start, batch_end))

        # --resume: skip batches where all .npy files already exist
        if args.resume:
            missing = [
                idx for idx in batch_indices
                if not os.path.exists(
                    os.path.join(out_dirs[layers[0]], f"{idx:06d}.npy")
                )
            ]
            if not missing:
                total_saved += len(batch_entries)
                continue
            # Partial resume: filter to only missing samples
            keep = [(e, i) for e, i in zip(batch_entries, batch_indices) if i in missing]
            batch_entries = [e for e, _ in keep]
            batch_indices = [i for _, i in keep]

        n_saved = extract_and_save_batch(
            teacher, batch_entries, batch_indices, layers, out_dirs,
            args.sample_rate, max_samples, device, use_fp16,
        )
        total_saved += n_saved
        total_skipped += len(batch_entries) - n_saved

        done = batch_end
        if done % args.log_interval < args.batch_size:
            elapsed = time.time() - t0
            eta = elapsed / done * (N - done) if done < N else 0
            logger.info(
                f"[{done}/{N}] saved={total_saved} skipped={total_skipped} | "
                f"elapsed={elapsed:.0f}s ETA={eta:.0f}s"
            )

    elapsed = time.time() - t0
    size_gb = total_saved * 500 * D * 2 / 1e9  # rough: 500 avg frames, float16
    logger.info(
        f"Done. {total_saved}/{N} samples saved in {elapsed:.0f}s "
        f"(~{size_gb:.0f} GB per layer, rough estimate)."
    )
    logger.info(f"Cache dir: {args.out_dir}")


if __name__ == "__main__":
    main()
