"""Build a pool of spoken user turns, so that training also covers users who speak instead of type.

When the question is spoken, the Talker reads the Thinker's hidden states at the audio positions, which a LoRA trained
on read-aloud prompts never saw. This script runs the full Thinker once over each spoken turn and stores those hidden
states (layer accept_hidden_layer, audio positions only); `train --spoken-pool` gives a fraction of the training
sentences a spoken turn of the same language instead of the read-aloud prompt.

  python -m talker_tune.spoken --data data_dir --out pool_dir [--n 200] [--voices chelsie,aiden]   # synthesized turns
  python -m talker_tune.spoken --audio-list turns.tsv --out pool_dir      # your own recordings: "path<TAB>lang" per line
Synthesized turns are N training sentences per language (split train) read by the base Talker, alternating the preset
voices. Needs the full model once (about 70 GB of GPU memory; with less, transformers offloads to CPU through
device_map="auto").
Writes pool_dir/pool.pt (list of {"ids", "hid", "lang", "audio", ...}) and pool_dir/wav/.

In our runs (2026-10-29, 3 seeds) same-language spoken turns in half of the samples fixed the voice drift on spoken
questions with one female and one male speaker per language, but made it worse with one speaker per language; with
two speakers of each gender per language they were not needed.
"""
import argparse, json, os, random, sys
import torch

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
SR = 24000


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--omni", default="Qwen/Qwen3-Omni-30B-A3B-Instruct")
    ap.add_argument("--data", default=None, help="data_dir from prepare: synthesize turns from its training sentences")
    ap.add_argument("--audio-list", default=None, help="text file, one 'audio_path<TAB>lang' per line (used as they are)")
    ap.add_argument("--out", required=True)
    ap.add_argument("--n", type=int, default=200, help="synthesized turns per language")
    ap.add_argument("--voices", default="chelsie,aiden")
    ap.add_argument("--seed", type=int, default=0)
    a = ap.parse_args()
    assert a.data or a.audio_list, "give --data and/or --audio-list"
    import soundfile as sf, soxr
    os.makedirs(f"{a.out}/wav", exist_ok=True)
    rows = []
    if a.audio_list:
        for l in open(a.audio_list, encoding="utf-8"):
            if l.strip():
                path, lang = l.rstrip("\n").split("\t")
                rows.append({"audio": path, "lang": lang})
    if a.data:
        from talker_tune.core import TalkerOnly
        items = [json.loads(l) for l in open(f"{a.data}/items.jsonl", encoding="utf-8")]
        rng, sel = random.Random(a.seed), []
        for lang in sorted({it["lang"] for it in items}):
            c = [it for it in items if it["split"] == "train" and it["lang"] == lang]
            sel += rng.sample(c, min(a.n, len(c)))
        voices = a.voices.split(",")
        T_ = TalkerOnly(a.omni)
        for k, it in enumerate(sel):
            v, path = voices[k % len(voices)], f"{a.out}/wav/{k:04d}.wav"
            if not os.path.exists(path):
                for s in range(3):  # resample if it did not stop
                    z, w, ended = T_.synthesize(T_.ids(it["text"], None, it["lang"]), T_.speakers[v], seed=a.seed + k + 1000 * s)
                    if ended and w is not None:
                        break
                if w is None:
                    continue
                sf.write(path + ".tmp.wav", w.numpy(), SR); os.replace(path + ".tmp.wav", path)
            rows.append({"audio": path, "text": it["text"], "lang": it["lang"], "voice": v})
            if k % 20 == 0:
                print(f"synth {k}/{len(sel)}", flush=True)
        del T_; torch.cuda.empty_cache()
    from transformers import Qwen3OmniMoeForConditionalGeneration, Qwen3OmniMoeProcessor
    proc = Qwen3OmniMoeProcessor.from_pretrained(a.omni)
    m = Qwen3OmniMoeForConditionalGeneration.from_pretrained(a.omni, dtype=torch.bfloat16, device_map="auto",
                                                             attn_implementation="sdpa").eval()
    cfg = m.config
    layer = cfg.talker_config.accept_hidden_layer
    pool = []
    for k, r in enumerate(rows):
        w, sr = sf.read(r["audio"], dtype="float32", always_2d=True)
        w16 = soxr.resample(w.mean(1), sr, 16000)
        text = proc.apply_chat_template([{"role": "user", "content": [{"type": "audio", "audio": r["audio"]}]}],
                                        add_generation_prompt=False, tokenize=False)
        inp = proc(text=text, audio=[w16], return_tensors="pt", padding=True).to(m.thinker.device)
        with torch.no_grad():
            hs = m.thinker(**inp, output_hidden_states=True, use_cache=False).hidden_states[layer][0]
        ids = inp["input_ids"][0]
        st = torch.nonzero(ids == cfg.im_start_token_id).view(-1).tolist() + [len(ids)]
        u = [j for j in range(len(st) - 1) if ids[st[j] + 1] == cfg.user_token_id][0]
        a0, b0 = st[u], st[u + 1]
        seg = ids[a0:b0]
        mm = seg == cfg.thinker_config.audio_token_id
        pool.append(dict(r, ids=seg.cpu(), hid=hs[a0:b0][mm.to(hs.device)].to(torch.bfloat16).cpu()))
        if k % 20 == 0:
            print(f"thinker {k}/{len(rows)}: {int(mm.sum())} audio positions", flush=True)
    torch.save(pool, f"{a.out}/pool.pt")
    print("SPOKEN_DONE", len(pool), {l: sum(p["lang"] == l for p in pool) for l in sorted({p["lang"] for p in pool})},
          f"{a.out}/pool.pt", flush=True)


if __name__ == "__main__":
    main()
