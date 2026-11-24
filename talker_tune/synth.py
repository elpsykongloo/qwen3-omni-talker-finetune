"""Synthesize with a fine-tuned Talker (text in, 24 kHz wav out; no Thinker generation).

  python -m talker_tune.synth --omni Qwen/Qwen3-Omni-30B-A3B-Instruct --adapter run_dir/step_800 --speaker myvoice \
      --text "Hello there." --out out_dir [--reps 1] [--seed 0]
  python -m talker_tune.synth ... --data data_dir --split held --out out_dir      # every held-out sentence
Writes out_dir/{k}_r{rep}.wav and out_dir/synth.jsonl (text, frames, ended_with_eos).
"""
import argparse, json, os, sys
import numpy as np, torch

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from talker_tune.core import TalkerOnly


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--omni", default="Qwen/Qwen3-Omni-30B-A3B-Instruct")
    ap.add_argument("--adapter", default=None)
    ap.add_argument("--speaker", default=None, help="default: the adapter's trained speaker, else ethan")
    ap.add_argument("--text", action="append", default=[])
    ap.add_argument("--data", default=None)
    ap.add_argument("--split", default="held")
    ap.add_argument("--out", required=True)
    ap.add_argument("--reps", type=int, default=1)
    ap.add_argument("--seed", type=int, default=0)
    ap.add_argument("--temperature", type=float, default=0.9)
    ap.add_argument("--top-k", type=int, default=50)
    ap.add_argument("--experts", default="grouped_mm")
    a = ap.parse_args()
    import soundfile as sf
    T_ = TalkerOnly(a.omni, experts=a.experts)
    meta = T_.load_adapter(a.adapter) if a.adapter else {}
    jobs = [{"text": t, "speaker": a.speaker} for t in a.text]
    if a.data:
        items = [json.loads(l) for l in open(f"{a.data}/items.jsonl", encoding="utf-8")]
        jobs += [{"text": it["text"], "user": it.get("user"), "lang": it["lang"], "speaker": a.speaker or it["speaker"], "i": i}
                 for i, it in enumerate(items) if it["split"] == a.split]
    trained = [s for s in meta.get("speakers", {}) if s not in ("chelsie", "ethan", "aiden")]
    os.makedirs(a.out, exist_ok=True)
    log = open(f"{a.out}/synth.jsonl", "w", encoding="utf-8")
    for k, j in enumerate(jobs):
        spk = (j["speaker"] or (trained[0] if trained else "ethan")).lower()
        ids = T_.ids(j["text"], j.get("user"), j.get("lang"))
        for rep in range(a.reps):
            z, w, ended = T_.synthesize(ids, T_.speakers[spk], seed=a.seed + 100000 * rep + k, top_k=a.top_k,
                                        temperature=a.temperature)
            name = f"{k:04d}_r{rep}"
            if w is not None:
                sf.write(f"{a.out}/{name}.wav", w.numpy(), 24000)
            np.save(f"{a.out}/{name}.npy", z.numpy().astype(np.int16))
            log.write(json.dumps({"name": name, "text": j["text"], "speaker": spk, "frames": int(z.shape[-1]), "ended": bool(ended),
                                  "i": j.get("i"), "rep": rep, "wav": w is not None}, ensure_ascii=False) + "\n"); log.flush()
        print(f"[{k + 1}/{len(jobs)}] {j['text'][:60]}", flush=True)
    print("SYNTH_DONE", a.out, flush=True)


if __name__ == "__main__":
    main()
