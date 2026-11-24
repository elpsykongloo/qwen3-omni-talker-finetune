"""Encode a manifest into Talker labels.

  python -m talker_tune.prepare manifest.jsonl --out data_dir [--backend moshi|hf] [--held 20]

manifest.jsonl, one utterance per line:
  {"audio": "a.wav", "text": "spoken text", "speaker": "myvoice", "lang": "en", "user": "optional user turn"}
`speaker` defaults to "myvoice", `lang` is guessed (zh/en) when absent, `user` defaults to a read-aloud instruction.

Labels follow the recipe in the report: kyutai Mimi, first 16 codebooks, input zero-padded at the tail.
  moshi  the `moshi` package encoder with weights cast to bf16 (closest to the Talker's own labels)
  hf     transformers MimiModel (kyutai/mimi) in fp32 with the same tail padding (no extra dependency)
Writes data_dir/items.jsonl and data_dir/codes.pt (list of int16 [16, T]).
"""
import argparse, json, os, random, sys
import numpy as np, torch, torch.nn.functional as F

SR, HOP = 24000, 1920


def load_wav(path):
    import soundfile as sf, soxr
    w, sr = sf.read(path, dtype="float32", always_2d=True)
    w = w.mean(1)
    return soxr.resample(w, sr, SR, quality="VHQ").astype(np.float32) if sr != SR else w


class Encoder:
    def __init__(self, backend="moshi", device="cuda", weight=None):
        self.backend, self.device = backend, device
        if backend == "moshi":
            from moshi.models import loaders
            if weight is None:
                from huggingface_hub import hf_hub_download
                weight = hf_hub_download(loaders.DEFAULT_REPO, loaders.MIMI_NAME)
            try:
                m = loaders.get_mimi(weight, device=device, num_codebooks=16)
            except TypeError:
                m = loaders.get_mimi(weight, device=device); m.set_num_codebooks(16)
            self.m, self.dt = m.to(torch.bfloat16).eval(), torch.bfloat16
        elif backend == "hf":
            from transformers import MimiModel
            self.m, self.dt = MimiModel.from_pretrained(weight or "kyutai/mimi", dtype=torch.float32).to(device).eval(), torch.float32
        else:
            raise ValueError(backend)

    @torch.no_grad()
    def __call__(self, wav):
        x = torch.as_tensor(wav, dtype=torch.float32).reshape(1, 1, -1)
        n = -(-x.shape[-1] // HOP)
        x = F.pad(x, (0, x.shape[-1])).to(self.device, self.dt)  # tail zero-pad (2x length), keep n frames
        if self.backend == "moshi":
            z = self.m.encode(x)[0, :16, :n]
        else:
            z = self.m.encode(x, num_quantizers=16).audio_codes[0, :, :n]
        assert z.shape == (16, n), z.shape
        return z.short().cpu()


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("manifest")
    ap.add_argument("--out", required=True)
    ap.add_argument("--backend", default="moshi", choices=["moshi", "hf"])
    ap.add_argument("--weight", default=None, help="local Mimi weights (moshi: tokenizer-e351c8d8-checkpoint125.safetensors; hf: model dir)")
    ap.add_argument("--held", type=int, default=0, help="hold out N utterances per speaker for evaluation")
    ap.add_argument("--min-sec", type=float, default=0.5)
    ap.add_argument("--max-sec", type=float, default=30.0)
    a = ap.parse_args()
    sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
    from talker_tune.core import guess_lang
    rows = [json.loads(l) for l in open(a.manifest, encoding="utf-8") if l.strip()]
    base = os.path.dirname(os.path.abspath(a.manifest))
    enc = Encoder(a.backend, "cuda" if torch.cuda.is_available() else "cpu", a.weight)
    items, codes, skipped = [], [], 0
    for r in rows:
        p = r["audio"] if os.path.isabs(r["audio"]) else os.path.join(base, r["audio"])
        w = load_wav(p)
        if not (a.min_sec <= len(w) / SR <= a.max_sec) or not r.get("text", "").strip():
            skipped += 1
            continue
        items.append({"audio": p, "text": r["text"].strip(), "speaker": r.get("speaker", "myvoice").lower(),
                      "lang": r.get("lang") or guess_lang(r["text"]), "user": r.get("user"), "sec": round(len(w) / SR, 3),
                      "split": r.get("split", "train")})
        codes.append(enc(w))
    if a.held:
        rng = random.Random(0)
        for spk in sorted({it["speaker"] for it in items}):
            idx = [i for i, it in enumerate(items) if it["speaker"] == spk and it["split"] == "train"]
            for i in rng.sample(idx, min(a.held, len(idx) // 5)):
                items[i]["split"] = "held"
    os.makedirs(a.out, exist_ok=True)
    with open(f"{a.out}/items.jsonl", "w", encoding="utf-8") as f:
        for it in items:
            f.write(json.dumps(it, ensure_ascii=False) + "\n")
    torch.save(codes, f"{a.out}/codes.pt")
    json.dump({"backend": a.backend, "n": len(items), "skipped": skipped}, open(f"{a.out}/prepare.json", "w"))
    for spk in sorted({it["speaker"] for it in items}):
        tr = [it for it in items if it["speaker"] == spk and it["split"] == "train"]
        print(f"{spk}: train {len(tr)} utts {sum(it['sec'] for it in tr) / 60:.1f} min, held "
              f"{sum(1 for it in items if it['speaker'] == spk and it['split'] == 'held')}", flush=True)
    print(f"encoded {len(items)} ({a.backend}), skipped {skipped} -> {a.out}", flush=True)


if __name__ == "__main__":
    main()
