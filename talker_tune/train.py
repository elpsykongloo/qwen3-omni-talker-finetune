"""Fine-tune the Qwen3-Omni Talker (and its MTP code predictor) on prepared labels.

  python -m talker_tune.train --omni Qwen/Qwen3-Omni-30B-A3B-Instruct --data data_dir --out run_dir \
      [--mode lora|dense] [--steps 800] [--batch 8] [--lr 1e-4] [--init-speaker chelsie[,name=ethan,...]] \
      [--spoken-pool pool_dir/pool.pt [--spoken-frac 0.5]]

Speakers named in items.jsonl that are not preset voices (chelsie / ethan / aiden) get a new codec id whose embedding
starts from --init-speaker and is trained; a preset name trains that preset slot instead. `name=preset` entries in
--init-speaker set the initial voice per speaker (e.g. female voices from chelsie, male voices from ethan).
With --spoken-pool (from talker_tune.spoken), a fraction of the samples get a spoken user turn of the same language
in place of the read-aloud prompt.
For several voices: at least two speakers of each gender per language and about 400 steps per speaker; with one
speaker per language, answers to spoken questions in one language fall back to the other language's voice.
Loss per batch: sum over the 16 codebooks of the per-frame cross-entropy plus the EOS cross-entropy, divided by frames.
Writes run_dir/step_{N}/adapter.safetensors + adapter_config.json, and run_dir/log.jsonl.
"""
import argparse, json, math, os, random, sys, time
import torch

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from talker_tune.core import TalkerOnly


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--omni", default="Qwen/Qwen3-Omni-30B-A3B-Instruct")
    ap.add_argument("--data", required=True)
    ap.add_argument("--out", required=True)
    ap.add_argument("--mode", default="lora", choices=["lora", "dense"])
    ap.add_argument("--steps", type=int, default=800)
    ap.add_argument("--batch", type=int, default=8)
    ap.add_argument("--lr", type=float, default=None, help="default 1e-4 (lora) / 1e-5 (dense)")
    ap.add_argument("--warmup", type=int, default=40)
    ap.add_argument("--r", type=int, default=16)
    ap.add_argument("--alpha", type=int, default=32)
    ap.add_argument("--text-projection", action="store_true", help="also train the Thinker->Talker text projection (new languages)")
    ap.add_argument("--init-speaker", default="chelsie",
                    help="preset voice new speakers start from; name=preset entries set it per speaker (chelsie,p259=ethan)")
    ap.add_argument("--save", default="", help="comma-separated extra steps to save (the last step is always saved)")
    ap.add_argument("--seed", type=int, default=0)
    ap.add_argument("--experts", default="grouped_mm", help="transformers experts_implementation (grouped_mm / eager / batched_mm)")
    ap.add_argument("--grad-ckpt", action="store_true")
    ap.add_argument("--lora-scope", default="both", choices=["both", "talker", "mtp"], help="LoRA on Talker + MTP, Talker only or MTP only")
    ap.add_argument("--limit-min", type=float, default=None, help="use only the first N minutes of training audio per speaker")
    ap.add_argument("--max-gpu-gb", type=float, default=None, help="cap this process's GPU memory (to emulate a smaller card)")
    ap.add_argument("--spoken-pool", default=None, help="pool.pt from talker_tune.spoken: spoken user turns to train with")
    ap.add_argument("--spoken-frac", type=float, default=0.5, help="fraction of training samples given a spoken user turn")
    a = ap.parse_args()
    lr = a.lr or (1e-4 if a.mode == "lora" else 1e-5)
    if a.max_gpu_gb:
        torch.cuda.set_per_process_memory_fraction(a.max_gpu_gb * 2 ** 30 / torch.cuda.get_device_properties(0).total_memory)
    items = [json.loads(l) for l in open(f"{a.data}/items.jsonl", encoding="utf-8")]
    codes = torch.load(f"{a.data}/codes.pt")
    tr = [i for i, it in enumerate(items) if it["split"] == "train"]
    if a.limit_min:
        rng0, keep, used = random.Random(0), [], {}
        for i in rng0.sample(tr, len(tr)):
            spk = items[i]["speaker"]
            if used.get(spk, 0) < a.limit_min * 60:
                keep.append(i); used[spk] = used.get(spk, 0) + items[i]["sec"]
        tr = sorted(keep)
        print("limited training audio (min):", {k: round(v / 60, 1) for k, v in used.items()}, flush=True)
    held = [i for i, it in enumerate(items) if it["split"] == "held"]
    os.makedirs(a.out, exist_ok=True)

    init = dict(s.rpartition("=")[::2] for s in a.init_speaker.lower().split(","))  # {"": default, name: preset}
    init_default = init.pop("", "chelsie")
    train_spk = sorted({items[i]["speaker"] for i in tr})
    assert set(init) <= set(train_spk), f"--init-speaker names speakers not in the training data: {sorted(set(init) - set(train_spk))}"

    T_ = TalkerOnly(a.omni, experts=a.experts)
    presets = set(T_.speakers)
    assert {init_default, *init.values()} <= presets, f"--init-speaker must name preset voices: {sorted(presets)}"
    new_rows, init_used = [], {}
    for spk in train_spk:
        preset = spk in T_.speakers
        src = init.get(spk, init_default)
        sid = T_.add_speaker(spk, src)
        if not preset:
            new_rows.append(sid); init_used[spk] = src
        print(f"speaker {spk}: id {sid} ({'preset slot' if preset else 'new, init from ' + src})", flush=True)
    ps = T_.trainable(a.mode, r=a.r, alpha=a.alpha, seed=a.seed, speaker_rows=new_rows, text_projection=a.text_projection,
                      scope=a.lora_scope)
    if a.grad_ckpt:
        T_.tl.model.gradient_checkpointing_enable(gradient_checkpointing_kwargs={"use_reentrant": False})
    T_.tl.train()
    opt = torch.optim.AdamW(ps, lr=lr, weight_decay=0.0, betas=(0.9, 0.99))
    sched = torch.optim.lr_scheduler.LambdaLR(
        opt, lambda s: min(1.0, (s + 1) / a.warmup) * 0.5 * (1 + math.cos(math.pi * min(1.0, s / a.steps))))
    meta = {"omni": a.omni, "mode": a.mode, "r": a.r, "alpha": a.alpha, "targets": getattr(T_, "targets", None), "data": os.path.basename(os.path.normpath(a.data)),
            "steps": a.steps, "lr": lr, "batch": a.batch, "seed": a.seed, "text_projection": a.text_projection,
            "init_speaker": init_used, "spoken_pool": a.spoken_pool and os.path.basename(a.spoken_pool),
            "spoken_frac": a.spoken_frac if a.spoken_pool else 0}
    save = {int(s) for s in a.save.split(",") if s} | {a.steps}
    rng = random.Random(1000 + a.seed)
    order, ptr, ids_cache = [], 0, {}
    pool = torch.load(a.spoken_pool) if a.spoken_pool else []
    by_lang = {}
    for p in pool:
        by_lang.setdefault(p["lang"], []).append(p)
    if pool:
        print(f"spoken user turns: {len(pool)} in pool ({ {k: len(v) for k, v in by_lang.items()} }), "
              f"{a.spoken_frac:.0%} of samples", flush=True)
        miss = sorted({items[i]["lang"] for i in tr} - set(by_lang))
        if miss:
            print(f"no spoken turns for {miss}: those sentences keep the read-aloud prompt", flush=True)

    def item(i, spoken=False):
        it = items[i]
        z = codes[i][None].long().cuda()
        if spoken and it["lang"] in by_lang:  # a spoken user turn of the same language in place of the text prompt
            src = by_lang[it["lang"]]
            p = src[rng.randrange(len(src))]
            return T_.ids_spoken(it["text"], p["ids"]), T_.speakers[it["speaker"]], z, p["hid"].cuda()
        if i not in ids_cache:
            ids_cache[i] = T_.ids(it["text"], it.get("user"), it["lang"])
        return ids_cache[i], T_.speakers[it["speaker"]], z

    @torch.no_grad()
    def held_ce():
        if not held:
            return {}
        T_.tl.eval()
        acc, n = torch.zeros(3), 0
        for j in range(0, len(held), 8):
            for nll, ne in T_.nll_batch([item(i) for i in held[j:j + 8]]):
                acc += torch.tensor([nll[:, :-1].sum().item(), nll[:, -1].sum().item(), ne.item()]); n += nll.shape[-1] - 1
        T_.tl.train()
        return {"held_ce_frame": acc[0].item() / n, "held_ce_last_frame": acc[1].item() / len(held), "held_eos": acc[2].item() / len(held)}

    log = open(f"{a.out}/log.jsonl", "a")
    t0 = time.time()
    for step in range(1, a.steps + 1):
        batch = []
        for _ in range(a.batch):
            if ptr == len(order):
                order = tr[:]; rng.shuffle(order); ptr = 0
            batch.append(item(order[ptr], bool(pool) and rng.random() < a.spoken_frac)); ptr += 1
        loss, tot, nfr = 0, torch.zeros(17), 0
        for nll, ne in T_.nll_batch(batch):
            T = nll.shape[-1]
            loss = loss + (nll.sum() + ne) / (T * a.batch)
            tot[:16] += nll.sum(1).detach().cpu(); tot[16] += ne.item(); nfr += T
        loss.backward()
        gn = torch.nn.utils.clip_grad_norm_(ps, 1.0).item()
        opt.step(); sched.step(); opt.zero_grad(set_to_none=True)
        if step % 10 == 0 or step == 1 or step in save:
            rec = {"step": step, "ce_frame": tot[:16].sum().item() / nfr, "ce_L0": tot[0].item() / nfr,
                   "ce_L1_15": tot[1:16].sum().item() / nfr, "eos": tot[16].item() / a.batch, "gnorm": gn,
                   "lr": sched.get_last_lr()[0], "sec": round(time.time() - t0, 1),
                   "mem_gb": round(torch.cuda.max_memory_allocated() / 2 ** 30, 1)}
            if step in save:
                rec.update(held_ce())
            log.write(json.dumps(rec) + "\n"); log.flush()
            if step % 50 == 0 or step == 1 or step in save:
                print(json.dumps(rec), flush=True)
        if step in save:
            T_.save_adapter(f"{a.out}/step_{step}", meta)
    print("TRAIN_DONE", a.out, flush=True)


if __name__ == "__main__":
    main()
