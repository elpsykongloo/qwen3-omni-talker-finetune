"""Merge a Talker adapter into a full Qwen3-Omni checkpoint (same layout as the original, so transformers and
vLLM-Omni load it unchanged). Only the shards holding changed tensors are rewritten; the rest are hard-linked
(or copied when linking is not possible).

  python -m talker_tune.export --omni Qwen/Qwen3-Omni-30B-A3B-Instruct --adapter run_dir/step_800 --out merged_dir
New speakers are added to config.json talker_config.speaker_id.
"""
import argparse, json, os, shutil
import torch
from safetensors import safe_open
from safetensors.torch import load_file, save_file


def changed_tensors(adapter, get):
    """{checkpoint key: new tensor}. `get(key)` returns the original tensor."""
    meta = json.load(open(f"{adapter}/adapter_config.json"))
    sd = load_file(f"{adapter}/adapter.safetensors")
    out = {}
    if meta["mode"] == "lora":
        scale = meta["alpha"] / meta["r"]
        for k in sd:
            if ".lora_A." not in k:
                continue
            mod = k.split(".lora_A.")[0]
            A, B = sd[k].float(), sd[k.replace(".lora_A.", ".lora_B.")].float()
            key = f"talker.{mod}.weight"
            W = get(key)
            out[key] = (W.float() + (B @ A) * scale).to(W.dtype)
        rest = [k for k in sd if "lora_" not in k and not k.startswith("speaker_row.")]
    else:
        rest = [k for k in sd if not k.startswith("speaker_row.")]
    for k in rest:  # dense weights (or the text projection trained with LoRA)
        key = f"talker.{k}"
        out[key] = sd[k].to(get(key).dtype)
    rows = {int(k.split(".")[1]): v for k, v in sd.items() if k.startswith("speaker_row.")}
    if rows:
        key = "talker.model.codec_embedding.weight"
        E = get(key).clone()
        for r, v in rows.items():
            E[r] = v.to(E.dtype)
        out[key] = E
    return out, meta


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--omni", default="Qwen/Qwen3-Omni-30B-A3B-Instruct")
    ap.add_argument("--adapter", required=True)
    ap.add_argument("--out", required=True)
    a = ap.parse_args()
    src = a.omni
    if not os.path.isdir(src):
        from huggingface_hub import snapshot_download
        src = snapshot_download(src)
    wmap = json.load(open(f"{src}/model.safetensors.index.json"))["weight_map"]

    def get(key):
        with safe_open(f"{src}/{wmap[key]}", "pt") as f:
            return f.get_tensor(key)
    new, meta = changed_tensors(a.adapter, get)
    missing = [k for k in new if k not in wmap]
    assert not missing, f"keys not in checkpoint: {missing[:5]}"
    os.makedirs(a.out, exist_ok=True)
    shards = {wmap[k] for k in new}
    for fn in sorted(os.listdir(src)):
        s, d = f"{src}/{fn}", f"{a.out}/{fn}"
        if os.path.isdir(s) or os.path.exists(d):
            continue
        if fn in shards:
            t = load_file(s)
            for k in new:
                if wmap[k] == fn:
                    assert t[k].shape == new[k].shape, (k, t[k].shape, new[k].shape)
                    t[k] = new[k].contiguous()
            save_file(t, d + ".tmp", metadata={"format": "pt"}); os.replace(d + ".tmp", d)
            print(f"rewrote {fn}: {sum(1 for k in new if wmap[k] == fn)} tensors", flush=True)
        elif fn == "config.json":
            cfg = json.load(open(s))
            cfg["talker_config"]["speaker_id"].update({k: v for k, v in meta["speakers"].items()})
            json.dump(cfg, open(d, "w"), indent=2, ensure_ascii=False)
        else:
            try:
                os.link(os.path.realpath(s), d)
            except OSError:
                shutil.copy2(s, d)
    print(f"EXPORT_DONE {len(new)} tensors changed -> {a.out}; speakers {meta['speakers']}", flush=True)


if __name__ == "__main__":
    main()
