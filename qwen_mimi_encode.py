"""Encode audio into Qwen3-Omni Talker labels (16 Mimi codebooks, 12.5 Hz) the way the Talker's training labels behave.

Recipe (see the report): the official kyutai Mimi encoder from the `moshi` package, weights cast to bf16, first 16
codebooks, input zero-padded at the tail so that the last (partial) frame is not encoded from a truncated window.

  pip install moshi==0.2.13 soundfile soxr
  python qwen_mimi_encode.py --out codes_dir a.wav b.wav ...        # or --list files.txt
Output: one int16 .npy per input, shape [16, ceil(n_samples_24k / 1920)].
"""
import argparse, os, sys
import numpy as np, torch, torch.nn.functional as F

SR, HOP = 24000, 1920


def load_mimi(weight=None, device="cuda"):
    from moshi.models import loaders
    if weight is None:
        from huggingface_hub import hf_hub_download
        weight = hf_hub_download(loaders.DEFAULT_REPO, loaders.MIMI_NAME)  # kyutai/moshiko-pytorch-bf16
    try:
        m = loaders.get_mimi(weight, device=device, num_codebooks=16)
    except TypeError:  # older moshi
        m = loaders.get_mimi(weight, device=device); m.set_num_codebooks(16)
    return m.to(torch.bfloat16).eval()


def load_wav(path):
    import soundfile as sf, soxr
    w, sr = sf.read(path, dtype="float32", always_2d=True)
    w = w.mean(1)
    return soxr.resample(w, sr, SR, quality="VHQ").astype(np.float32) if sr != SR else w


@torch.no_grad()
def encode(mimi, wav):
    """wav: float32 mono 24 kHz numpy or tensor [n]. Returns LongTensor [16, ceil(n/1920)]."""
    x = torch.as_tensor(wav, dtype=torch.float32).reshape(1, 1, -1)
    n = -(-x.shape[-1] // HOP)
    x = F.pad(x, (0, x.shape[-1])).to(next(mimi.parameters()).device, torch.bfloat16)  # tail zero-pad (2x length)
    return mimi.encode(x)[0, :16, :n].long().cpu()


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("wavs", nargs="*")
    ap.add_argument("--list")
    ap.add_argument("--out", required=True)
    ap.add_argument("--weight", default=None, help="local tokenizer-e351c8d8-checkpoint125.safetensors (default: download)")
    a = ap.parse_args()
    files = list(a.wavs) + ([l.strip() for l in open(a.list, encoding="utf-8") if l.strip()] if a.list else [])
    os.makedirs(a.out, exist_ok=True)
    mimi = load_mimi(a.weight, "cuda" if torch.cuda.is_available() else "cpu")
    for f in files:
        z = encode(mimi, load_wav(f))
        np.save(os.path.join(a.out, os.path.splitext(os.path.basename(f))[0] + ".npy"), z.numpy().astype(np.int16))
    print(f"encoded {len(files)} files -> {a.out}", file=sys.stderr)


if __name__ == "__main__":
    main()
