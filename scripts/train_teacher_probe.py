#!/usr/bin/env python3
"""
Train linear probes on teacher HuBERT's last (12th) layer output to predict
6th-layer RepCodec codebook indices.

Two probes are trained simultaneously:
  - probe_k32:  Linear(768 → 32)
  - probe_k512: Linear(768 → 512)

Inputs:
  - Teacher hidden states at layer 12 (no masking, all frames)
  - Hard label files: train.l6k32 and train.l6k512 (from apply_codebooks.py)

Output:
  - /workspace/teacher_probe/probe_k32.pt
  - /workspace/teacher_probe/probe_k512.pt

Usage:
    python scripts/train_teacher_probe.py \
        --teacher_ckpt /workspace/checkpoints/hubert_base_ls960.pt \
        --tsv /workspace/data/manifests/train.tsv \
        --label_dir /workspace/labels \
        --out_dir /workspace/teacher_probe \
        --steps 50000 \
        --device cuda
"""

import argparse
import logging
import os
import time

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
import torchaudio

logging.basicConfig(
    format="%(asctime)s | %(levelname)s | %(name)s | %(message)s",
    level=logging.INFO,
    datefmt="%Y-%m-%d %H:%M:%S",
)
logger = logging.getLogger(__name__)


# ---------------------------------------------------------------------------
# Teacher loader
# ---------------------------------------------------------------------------

def load_teacher(ckpt_path: str, device: str):
    """Load HuBERT teacher model (returns model in eval mode)."""
    import fairseq

    models, cfg, task = fairseq.checkpoint_utils.load_model_ensemble_and_task(
        [ckpt_path]
    )
    model = models[0].to(device)
    model.eval()
    for p in model.parameters():
        p.requires_grad_(False)
    logger.info(
        f"Teacher loaded: {sum(p.numel() for p in model.parameters()) / 1e6:.1f}M params"
    )
    return model


@torch.no_grad()
def extract_teacher_layer(model, wav: torch.Tensor, target_layer: int = 12):
    """
    Extract hidden states at `target_layer` from teacher model.
    wav: [1, T] float32, raw waveform (16 kHz).
    Returns: [T', D] float32 tensor (time × dim).
    """
    # extract_features(output_layer=N) stops at layer N and returns its output.
    # Returns (features [B, T, D], padding_mask).
    features, _ = model.extract_features(
        wav, padding_mask=None, mask=False, output_layer=target_layer
    )
    return features.squeeze(0)  # [T', D]


# ---------------------------------------------------------------------------
# Dataset
# ---------------------------------------------------------------------------

class TSVLabelDataset(torch.utils.data.Dataset):
    """
    Loads audio files from a TSV manifest and corresponding label files.
    Labels are stored as one integer per line (frame-level).
    """

    def __init__(self, tsv_path: str, label_dir: str, label_exts: list):
        self.root, self.entries = self._load_tsv(tsv_path)
        self.label_dir = label_dir
        self.label_exts = label_exts  # e.g. ["l6k32", "l6k512"]
        # Load all labels into memory (they're small)
        basename = os.path.splitext(os.path.basename(tsv_path))[0]  # "train"
        self.labels = {}
        for ext in label_exts:
            label_path = os.path.join(label_dir, f"{basename}.{ext}")
            with open(label_path) as f:
                lines = [line.strip() for line in f]
            self.labels[ext] = lines
        logger.info(
            f"Dataset: {len(self.entries)} utterances, labels: {label_exts}"
        )

    @staticmethod
    def _load_tsv(tsv_path):
        with open(tsv_path) as f:
            root = f.readline().strip()
            entries = [line.strip() for line in f if line.strip()]
        return root, entries

    def __len__(self):
        return len(self.entries)

    def __getitem__(self, idx):
        rel_path, _ = self.entries[idx].split("\t")
        wav_path = os.path.join(self.root, rel_path)
        wav, sr = torchaudio.load(wav_path)
        assert sr == 16000, f"Expected 16kHz, got {sr}"
        wav = wav.mean(0, keepdim=True)  # mono [1, T]

        labels = {}
        for ext in self.label_exts:
            raw = self.labels[ext][idx]
            labels[ext] = torch.tensor(
                [int(x) for x in raw.split()], dtype=torch.long
            )
        return wav, labels


# ---------------------------------------------------------------------------
# Training
# ---------------------------------------------------------------------------

def parse_args():
    p = argparse.ArgumentParser()
    p.add_argument("--teacher_ckpt", required=True)
    p.add_argument("--tsv", required=True)
    p.add_argument("--label_dir", required=True)
    p.add_argument("--out_dir", required=True)
    p.add_argument("--steps", type=int, default=50000)
    p.add_argument("--teacher_layer", type=int, default=12,
                   help="Which teacher layer to use as input to probe (default: 12)")
    p.add_argument("--codebook_layer", type=int, default=6,
                   help="Which codebook layer to predict (default: 6)")
    p.add_argument("--codebook_sizes", default="32,512",
                   help="Comma-separated codebook sizes to train probes for")
    p.add_argument("--lr", type=float, default=1e-3)
    p.add_argument("--log_interval", type=int, default=500)
    p.add_argument("--max_wav_samples", type=int, default=480000,
                   help="Skip utterances longer than this many samples (default: 30s @ 16kHz)")
    p.add_argument("--device", default="cuda")
    return p.parse_args()


def main():
    args = parse_args()
    os.makedirs(args.out_dir, exist_ok=True)
    device = torch.device(args.device)

    codebook_sizes = [int(k) for k in args.codebook_sizes.split(",")]
    label_exts = [f"l{args.codebook_layer}k{k}" for k in codebook_sizes]

    # Load teacher
    teacher = load_teacher(args.teacher_ckpt, device)

    # Dataset (no fixed DataLoader — iterate randomly)
    dataset = TSVLabelDataset(args.tsv, args.label_dir, label_exts)
    indices = list(range(len(dataset)))

    # Probes: one Linear per codebook size
    probes = nn.ModuleDict({
        f"k{k}": nn.Linear(768, k, bias=True).to(device)
        for k in codebook_sizes
    })
    optimizer = torch.optim.Adam(probes.parameters(), lr=args.lr)

    logger.info(f"Training probes: {[f'k{k}' for k in codebook_sizes]}")
    logger.info(f"Teacher layer: {args.teacher_layer}, Codebook layer: {args.codebook_layer}")
    logger.info(f"Steps: {args.steps}, lr: {args.lr}")

    step = 0
    loss_accum = {f"k{k}": 0.0 for k in codebook_sizes}
    acc_accum = {f"k{k}": 0.0 for k in codebook_sizes}
    t0 = time.time()

    rng = np.random.default_rng(42)

    while step < args.steps:
        # Pick a random utterance
        idx = int(rng.integers(len(dataset)))
        wav, labels = dataset[idx]

        # Skip utterances that are too long (avoid OOM / stuck on very long files)
        if wav.shape[-1] > args.max_wav_samples:
            continue

        wav = wav.to(device)  # [1, T]

        # Extract teacher features at target layer
        feats = extract_teacher_layer(
            teacher, wav, target_layer=args.teacher_layer
        )  # [T', 768]

        # For each codebook size, compute CE loss
        optimizer.zero_grad()
        total_loss = torch.tensor(0.0, device=device)

        for k in codebook_sizes:
            key = f"k{k}"
            ext = f"l{args.codebook_layer}k{k}"
            tgt = labels[ext].to(device)  # [T'] — may differ from feats.shape[0] by 1

            # Trim to minimum length (feature extractor may output T'±1)
            min_len = min(feats.shape[0], tgt.shape[0])
            f = feats[:min_len]
            t = tgt[:min_len]

            logits = probes[key](f)  # [T', k]
            loss = F.cross_entropy(logits, t)
            total_loss = total_loss + loss

            with torch.no_grad():
                acc = (logits.argmax(dim=-1) == t).float().mean().item()
            loss_accum[key] += loss.item()
            acc_accum[key] += acc

        total_loss.backward()
        optimizer.step()
        step += 1

        if step % args.log_interval == 0:
            elapsed = time.time() - t0
            parts = []
            for k in codebook_sizes:
                key = f"k{k}"
                avg_loss = loss_accum[key] / args.log_interval
                avg_acc = acc_accum[key] / args.log_interval
                parts.append(f"{key} loss={avg_loss:.4f} acc={avg_acc:.4f}")
                loss_accum[key] = 0.0
                acc_accum[key] = 0.0
            logger.info(
                f"step={step:6d}/{args.steps} | {' | '.join(parts)} | elapsed={elapsed:.0f}s"
            )

    # Save probes
    for k in codebook_sizes:
        save_path = os.path.join(args.out_dir, f"probe_k{k}.pt")
        torch.save(
            {
                "state_dict": probes[f"k{k}"].state_dict(),
                "teacher_layer": args.teacher_layer,
                "codebook_layer": args.codebook_layer,
                "codebook_size": k,
                "steps": args.steps,
            },
            save_path,
        )
        logger.info(f"Saved probe → {save_path}")

    logger.info("Done.")


if __name__ == "__main__":
    main()
