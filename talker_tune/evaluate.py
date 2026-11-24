"""Score synthesized speech: intelligibility (CER with Paraformer for Chinese, WER with Whisper otherwise, as in
Seed-TTS-eval), speaker similarity to the reference recordings (WavLM-SV cosine to the speaker centroid),
end-of-utterance quality and stopping. The held-out reference recordings are scored the same way ("real") as a floor.

  python -m talker_tune.evaluate --synth out_dir --data data_dir [--asr openai/whisper-large-v3-turbo] [--asr-zh whisper]

Whisper writes Chinese in traditional characters and Arabic digits about as often as not, which the CER counts as
errors, hence Paraformer (sherpa-onnx, `pip install sherpa-onnx`) for Chinese by default.
tail_db: RMS of the last 80 ms relative to the loudest 80 ms window. Clean endings sit around -40 dB or lower; a
tail above -20 dB ("tail_loud") is the audible noise burst that unpadded labels teach the Talker.
Writes out_dir/eval.json (summary per speaker and per utterance).
"""
import argparse, json, os, re, sys
import numpy as np, torch, torch.nn.functional as F

SR = 24000
CJK = re.compile(r"[぀-ヿ㐀-䶿一-鿿가-힯]")


def tail_db(w):
    k = int(0.08 * SR)
    if len(w) < 2 * k:
        return float("nan")
    fr = w[: len(w) // k * k].reshape(-1, k)
    ref = np.sqrt((fr ** 2).mean(1)).max() + 1e-8
    return float(20 * np.log10(np.sqrt((w[-k:] ** 2).mean()) / ref + 1e-8))


def norm(s, cjk):
    if cjk:
        return "".join(CJK.findall(s))
    s = s.lower().replace("’", "'")
    return " ".join(re.sub(r"[^\w' ]+", " ", s).split())


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--synth", required=True)
    ap.add_argument("--data", required=True, help="prepared data dir (reference audio per speaker)")
    ap.add_argument("--asr", default="openai/whisper-large-v3-turbo")
    ap.add_argument("--asr-zh", default="csukuangfj/sherpa-onnx-paraformer-zh-2024-03-09",
                    help="sherpa-onnx Paraformer repo for Chinese, or 'whisper' to use --asr")
    ap.add_argument("--sv", default="microsoft/wavlm-base-plus-sv")
    a = ap.parse_args()
    import soundfile as sf, soxr, jiwer
    from transformers import pipeline, AutoFeatureExtractor, WavLMForXVector
    rows = [json.loads(l) for l in open(f"{a.synth}/synth.jsonl", encoding="utf-8")]
    items = [json.loads(l) for l in open(f"{a.data}/items.jsonl", encoding="utf-8")]
    whisper = pipeline("automatic-speech-recognition", model=a.asr, dtype=torch.float16, device="cuda")
    paraformer = None
    if a.asr_zh != "whisper":
        import sherpa_onnx
        from huggingface_hub import hf_hub_download
        get = lambda f: hf_hub_download(a.asr_zh, f)
        paraformer = sherpa_onnx.OfflineRecognizer.from_paraformer(paraformer=get("model.onnx"), tokens=get("tokens.txt"), num_threads=4)
    fe = AutoFeatureExtractor.from_pretrained(a.sv)
    sv = WavLMForXVector.from_pretrained(a.sv).cuda().eval()
    to16 = lambda w: soxr.resample(w.astype(np.float32), SR, 16000)

    def asr(w, lang):
        if lang == "zh" and paraformer is not None:
            st = paraformer.create_stream(); st.accept_waveform(16000, to16(w)); paraformer.decode_stream(st)
            return st.result.text
        return whisper(to16(w), generate_kwargs={"language": lang, "task": "transcribe"})["text"]

    def err(ref, hyp):
        cjk = bool(CJK.search(ref))
        ref_n, hyp_n = norm(ref, cjk), norm(hyp, cjk) or "_"
        return jiwer.cer(ref_n, hyp_n) if cjk else jiwer.wer(ref_n, hyp_n), cjk

    @torch.no_grad()
    def emb(w):
        return F.normalize(sv(**fe(to16(w), sampling_rate=16000, return_tensors="pt").to("cuda")).embeddings, dim=-1)[0].cpu()

    def load(p):
        w, sr = sf.read(p, dtype="float32", always_2d=True)
        w = w.mean(1)
        return soxr.resample(w, sr, SR) if sr != SR else w
    cen, real = {}, {}
    for spk in sorted({r["speaker"] for r in rows}):
        refs = [it for it in items if it["speaker"] == spk and it["split"] == "held"] or [it for it in items if it["speaker"] == spk][:40]
        if refs:
            ws = [load(it["audio"]) for it in refs]
            cen[spk] = F.normalize(torch.stack([emb(w) for w in ws]).mean(0), dim=-1)
            td = np.array([tail_db(w) for w in ws])
            real[spk] = {"n": len(refs), "err (CER for CJK, else WER)": float(np.mean([err(it["text"], asr(w, it["lang"]))[0] for it, w in zip(refs, ws)])),
                         "tail_db": float(np.nanmean(td)), "tail_loud_frac": float(np.mean(td > -20))}
            print("real", spk, json.dumps(real[spk]), flush=True)
    per = []
    for r in rows:
        lang = items[r["i"]]["lang"] if r.get("i") is not None else ("zh" if CJK.search(r["text"]) else "en")
        rec = {"name": r["name"], "speaker": r["speaker"], "lang": lang, "ended": r["ended"], "frames": r["frames"]}
        if r["wav"]:
            w = load(f"{a.synth}/{r['name']}.wav")
            hyp = asr(w, lang)
            e, cjk = err(r["text"], hyp)
            rec.update(hyp=hyp, err=e, cjk=cjk, tail_db=tail_db(w),
                       sec=len(w) / SR, sim=float(emb(w) @ cen[r["speaker"]]) if r["speaker"] in cen else None)
            if r.get("i") is not None:
                rec["dur_ratio"] = len(w) / SR / items[r["i"]]["sec"]
        per.append(rec)
    summ = {}
    for spk in sorted({p["speaker"] for p in per}):
        ps = [p for p in per if p["speaker"] == spk]
        ok = [p for p in ps if "err" in p]
        td = np.array([p["tail_db"] for p in ok])
        summ[spk] = {"n": len(ps), "err (CER for CJK, else WER)": float(np.mean([p["err"] for p in ok])) if ok else None,
                     "sim": float(np.mean([p["sim"] for p in ok if p["sim"] is not None])) if cen.get(spk) is not None and ok else None,
                     "tail_db": float(np.nanmean(td)) if ok else None, "tail_loud_frac": float(np.mean(td > -20)) if ok else None,
                     "no_eos": float(np.mean([not p["ended"] for p in ps])), "undecodable": len(ps) - len(ok),
                     "dur_ratio_median": float(np.median([p["dur_ratio"] for p in ok if "dur_ratio" in p])) if any("dur_ratio" in p for p in ok) else None}
        print(spk, json.dumps(summ[spk]), flush=True)
    json.dump({"summary": summ, "real": real, "per_utt": per}, open(f"{a.synth}/eval.json", "w"), indent=1, ensure_ascii=False)
    print("EVAL_DONE", flush=True)


if __name__ == "__main__":
    main()
