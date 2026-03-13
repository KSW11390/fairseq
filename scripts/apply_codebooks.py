#!/usr/bin/env python3
"""
Phase 2: Apply trained RepCodec codebooks to generate discrete label files.

Loads the RepCodecLayer models from train_codebooks.py output.
For each audio sample, runs: teacher → RepCodec encoder → VQ encode → indices.
Outputs one label file per codebook in fairseq format (integers, space-separated).

Backward-compatible with old EMA-only checkpoints (key "codebooks" in .pt).
New RepCodec checkpoints use key "codecs".

Output:
  <out_dir>/<split>.l1k32    — Layer 1, K=32
  <out_dir>/<split>.l1k512   — Layer 1, K=512
  ...
  <out_dir>/<split>.l12k512  — Layer 12, K=512

Usage:
    python scripts/apply_codebooks.py \\
        --teacher_ckpt /path/to/hubert_base_ls960.pt \\
        --codebook_ckpt /path/to/codebooks_latest.pt \\
        --tsv /path/to/train.tsv \\
        --split train \\
        --out_dir /path/to/label_dir/ \\
        --batch_size 8
"""

import argparse
import logging
import os
import sys
import time
from collections import defaultdict
from pathlib import Path

import torch
import torch.nn.functional as F

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

import fairseq
from fairseq.modules.repcodec_codec import RepCodecLayer
from fairseq.modules.ema_codebook import EMACodebook  # legacy compat

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
    p.add_argument("--codebook_ckpt", required=True,
                   help="Output of train_codebooks.py (codebooks_latest.pt)")
    p.add_argument("--tsv", required=True)
    p.add_argument("--split", default="train")
    p.add_argument("--out_dir", required=True)
    p.add_argument(
        "--codebook_names",
        default=None,
        help="Comma-separated subset of codebook names to apply (default: all). "
             "E.g. 'l9k32,l9k512'",
    )
    p.add_argument("--sample_rate", type=int, default=16000)
    p.add_argument("--max_sec", type=float, default=15.0)
    p.add_argument("--batch_size", type=int, default=8,
                   help="Samples per teacher forward pass")
    p.add_argument("--device", default="cuda")
    p.add_argument("--log_interval", type=int, default=500)
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
    return teacher


def load_codebooks(ckpt_path, device, name_filter=None):
    """
    Load codebooks from train_codebooks.py output.
    Supports both new (RepCodecLayer, key="codecs") and
    legacy (EMACodebook, key="codebooks") checkpoint formats.

    Returns list of (name, layer, encoder_fn) where
      encoder_fn(feat: [T, D]) → indices: [T]  (int64, CPU)
    """
    ckpt = torch.load(ckpt_path, map_location=device)
    result = []

    if "codecs" in ckpt:
        # New format: full RepCodecLayer
        for name, info in sorted(ckpt["codecs"].items()):
            if name_filter and name not in name_filter:
                continue
            codec = RepCodecLayer(
                input_dim=info["input_dim"],
                codebook_size=info["codebook_size"],
                code_dim=info.get("code_dim", info["input_dim"]),
                encode_dim=info.get("encode_dim", 256),
                num_conv_layers=info.get("num_conv_layers", 2),
            )
            codec.load_state_dict(info["state_dict"])
            codec.eval().to(device)

            def make_encoder(c):
                @torch.no_grad()
                def _encode(feat):
                    # feat: [T, D] → indices: [T]
                    return c.encode(feat.unsqueeze(0)).squeeze(0)
                return _encode

            result.append((name, info["layer"], make_encoder(codec)))
            logger.info(
                f"  Loaded RepCodecLayer {name}: K={info['codebook_size']}, "
                f"layer={info['layer']}"
            )

    elif "codebooks" in ckpt:
        # Legacy format: bare EMACodebook
        logger.warning(
            "Loading legacy EMA-only checkpoint (key='codebooks'). "
            "For new experiments, use checkpoints from the RepCodec train_codebooks.py."
        )
        for name, info in sorted(ckpt["codebooks"].items()):
            if name_filter and name not in name_filter:
                continue
            cb = EMACodebook(info["num_codes"], info["dim"])
            cb.load_state_dict(info["state_dict"])
            cb.eval().to(device)

            def make_encoder_legacy(c):
                @torch.no_grad()
                def _encode(feat):
                    # feat: [T, D] → indices: [T]
                    _, idx = c.encode(feat)
                    return idx
                return _encode

            result.append((name, info["layer"], make_encoder_legacy(cb)))
            logger.info(
                f"  Loaded EMACodebook {name}: K={info['num_codes']}, "
                f"layer={info['layer']}"
            )
    else:
        raise ValueError(
            f"Unrecognised checkpoint format in {ckpt_path}. "
            "Expected key 'codecs' (new) or 'codebooks' (legacy)."
        )

    return result


# ---------------------------------------------------------------------------
# Batched processing
# ---------------------------------------------------------------------------

@torch.no_grad()
def process_batch(teacher, codebooks_by_layer, wavs, device, use_fp16):
    """
    Run teacher forward + RepCodec encode for a batch of wavs.

    Returns list of dict[name → list[int]], one dict per sample.
    """
    lengths = torch.tensor([w.shape[0] for w in wavs])
    max_len = int(lengths.max())
    padded = torch.stack(
        [F.pad(w, (0, max_len - w.shape[0])) for w in wavs]
    ).to(device)

    feat_lengths = (lengths // 320).tolist()
    T_feat_max = max_len // 320
    padding_mask = (
        torch.arange(T_feat_max, device=device).unsqueeze(0)
        >= torch.tensor(feat_lengths, device=device).unsqueeze(1)
    )

    unique_layers = sorted(codebooks_by_layer.keys())
    layer_feats = {}
    with torch.cuda.amp.autocast(enabled=use_fp16):
        for layer in unique_layers:
            feats, _ = teacher.extract_features(
                source=padded, padding_mask=padding_mask,
                mask=False, output_layer=layer,
            )
            layer_feats[layer] = feats.float()        # [B, T_feat, D]

    results = []
    for b in range(len(wavs)):
        T = feat_lengths[b]
        labels = {}
        for layer, cb_list in codebooks_by_layer.items():
            feat = layer_feats[layer][b, :T]          # [T, D]
            for name, encode_fn in cb_list:
                idx = encode_fn(feat)                  # [T] int64
                labels[name] = idx.cpu().tolist()
        T_min = min(len(v) for v in labels.values())
        results.append({name: vals[:T_min] for name, vals in labels.items()})

    return results


@torch.no_grad()
def process_single(teacher, codebooks_by_layer, wav, device, use_fp16):
    """Fallback: process one sample at a time."""
    source = wav.unsqueeze(0).to(device)
    unique_layers = sorted(codebooks_by_layer.keys())
    layer_feats = {}
    with torch.cuda.amp.autocast(enabled=use_fp16):
        for layer in unique_layers:
            feats, _ = teacher.extract_features(
                source=source, padding_mask=None,
                mask=False, output_layer=layer,
            )
            layer_feats[layer] = feats[0].float()     # [T_feat, D]

    labels = {}
    for layer, cb_list in codebooks_by_layer.items():
        feat = layer_feats[layer]
        for name, encode_fn in cb_list:
            idx = encode_fn(feat)                      # [T] int64
            labels[name] = idx.cpu().tolist()

    T_min = min(len(v) for v in labels.values())
    return {name: vals[:T_min] for name, vals in labels.items()}


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------

def main():
    args = parse_args()
    os.makedirs(args.out_dir, exist_ok=True)
    device = torch.device(args.device if torch.cuda.is_available() else "cpu")
    use_fp16 = (device.type == "cuda")

    name_filter = (
        set(args.codebook_names.split(",")) if args.codebook_names else None
    )

    teacher = load_teacher(args.teacher_ckpt, device)
    codebook_list = load_codebooks(args.codebook_ckpt, device, name_filter)

    # Group by teacher layer for efficient batching
    codebooks_by_layer = defaultdict(list)
    for name, layer, encode_fn in codebook_list:
        codebooks_by_layer[layer].append((name, encode_fn))

    entries = load_tsv(args.tsv)
    max_samples = int(args.max_sec * args.sample_rate)
    N = len(entries)

    names = [name for name, _, _ in codebook_list]
    out_paths = {
        name: os.path.join(args.out_dir, f"{args.split}.{name}")
        for name in names
    }
    logger.info(
        f"Writing label files for split='{args.split}' "
        f"(batch_size={args.batch_size}, fp16={use_fp16}):"
    )
    for name, path in out_paths.items():
        logger.info(f"  {name} → {path}")

    t0 = time.time()
    skipped = 0

    handles = {name: open(path, "w") for name, path in out_paths.items()}
    try:
        for batch_start in range(0, N, args.batch_size):
            batch_end = min(batch_start + args.batch_size, N)
            batch_entries = entries[batch_start:batch_end]

            wavs, load_ok = [], []
            for root, path, _ in batch_entries:
                try:
                    wav, sr = torchaudio.load(os.path.join(root, path))
                    if sr != args.sample_rate:
                        wav = torchaudio.functional.resample(wav, sr, args.sample_rate)
                    wavs.append(wav.mean(0)[:max_samples])
                    load_ok.append(True)
                except Exception as e:
                    logger.warning(f"Load failed {path}: {e}")
                    wavs.append(None)
                    load_ok.append(False)

            valid_wavs = [w for w, ok in zip(wavs, load_ok) if ok]
            batch_labels = None

            if valid_wavs:
                try:
                    batch_labels = process_batch(
                        teacher, codebooks_by_layer, valid_wavs, device, use_fp16
                    )
                except Exception as e:
                    logger.warning(
                        f"Batch failed (samples {batch_start}-{batch_end-1}): {e}. "
                        "Falling back to per-sample processing."
                    )
                    batch_labels = []
                    for wav in valid_wavs:
                        try:
                            batch_labels.append(
                                process_single(
                                    teacher, codebooks_by_layer, wav, device, use_fp16
                                )
                            )
                        except Exception as e2:
                            logger.warning(f"Single-sample fallback failed: {e2}")
                            batch_labels.append(None)

            valid_iter = 0
            for ok in load_ok:
                if not ok:
                    for f in handles.values():
                        f.write("\n")
                    skipped += 1
                    continue

                lbl = batch_labels[valid_iter] if batch_labels else None
                valid_iter += 1

                if lbl is None:
                    for f in handles.values():
                        f.write("\n")
                    skipped += 1
                    continue

                for name in names:
                    handles[name].write(" ".join(map(str, lbl[name])) + "\n")

            if batch_end % args.log_interval < args.batch_size or batch_end == N:
                logger.info(
                    f"[{batch_end}/{N}] elapsed={time.time()-t0:.0f}s | skipped={skipped}"
                )

    finally:
        for f in handles.values():
            f.close()

    logger.info(
        f"Done. {N - skipped}/{N} samples written in {time.time()-t0:.0f}s."
    )


if __name__ == "__main__":
    main()
