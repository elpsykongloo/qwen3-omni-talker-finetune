"""Self-check: are your Talker labels distributed like Qwen3-Omni's own training labels?

Input: a JSONL manifest, one utterance per line: {"wav": "...", "text": "...", "lang": "zh"|"en", "codes": "....npy"}
where `codes` is YOUR [16, T] label array for that audio (T = ceil(n_samples_24k / 1920)). 30-100 utterances suffice.

Quick check (no Talker): code agreement of your labels with the recipe labels (qwen_mimi_encode.py) on the same audio.
  same GPU and bit-identical input samples: expect 1.000 everywhere; another GPU: layer 0 > 0.95, layers 1-4 about
  0.93-0.97 (bf16 kernel differences). bf16 Mimi is sensitive to the input itself: re-quantising the same audio to
  16-bit PCM already changes about 8% of the layer 1-4 codes, so compare codes made from the exact waveform you encoded.
  HF MimiModel fp32: layers 1-4 about 0.82;  HF bf16: about 0.85.
Talker check (--talker): teacher-forced log-likelihood under the frozen Qwen3-Omni Talker + MTP (all 16 layers),
  delta = mean over frames (last frame excluded) of [log p(your codes) - log p(recipe codes)], nats per frame;
  the last frame is reported separately (nats, summed over the 16 codebooks;
  unpadded transformers labels: about -15 to -50 on FLEURS).
  Reference: recipe 0; HF fp32 -1.2 (512 utterances, 64 speakers) / -0.8 (32 utterances, 2 speakers);
  HF bf16 -0.7 / -0.6. The size depends on speakers and material; the sign and order do not.
  Clearly negative = the Talker finds your labels less natural than the recipe's.

  python label_check.py manifest.jsonl [--talker --omni Qwen/Qwen3-Omni-30B-A3B-Instruct]
"""
import argparse, json, os
import numpy as np, torch, torch.nn.functional as F
from qwen_mimi_encode import load_mimi, load_wav, encode


class Talker:
    """Frozen Qwen3-Omni Talker. Thinker transformer layers are not loaded: for text-only prompts the Talker reads
    the Thinker's word embeddings only."""

    def __init__(self, omni, speaker="ethan"):
        from transformers import Qwen3OmniMoeForConditionalGeneration, AutoTokenizer, AutoConfig
        cfg = AutoConfig.from_pretrained(omni)
        cfg.thinker_config.text_config.num_hidden_layers = 0
        self.model = Qwen3OmniMoeForConditionalGeneration.from_pretrained(omni, config=cfg, dtype=torch.bfloat16,
                                                                          device_map="cuda", attn_implementation="sdpa").eval()
        self.tok = AutoTokenizer.from_pretrained(omni)
        self.cfg = self.model.config
        self.spk = self.cfg.talker_config.speaker_id[speaker]
        self.dev = torch.device("cuda")
        tl = self.tl = self.model.talker
        emb = self.model.thinker.get_input_embeddings()
        sp = torch.tensor([[self.cfg.tts_bos_token_id, self.cfg.tts_eos_token_id, self.cfg.tts_pad_token_id]], device=self.dev)
        with torch.no_grad():
            self.tts_bos, self.tts_eos, self.tts_pad = tl.text_projection(emb(sp).to(tl.dtype)).chunk(3, dim=1)
        self.emb0 = tl.get_input_embeddings()
        self.embk = tl.code_predictor.get_input_embeddings()

    def ids(self, text, lang):
        prompt = ("请朗读下面这句话：" if lang == "zh" else "Please read the following sentence aloud: ") + text
        s = "<|im_start|>user\n" + prompt + "<|im_end|>\n<|im_start|>assistant\n" + text + "<|im_end|>"
        return torch.tensor([self.tok(s, add_special_tokens=False)["input_ids"]], device=self.dev)

    @torch.no_grad()
    def logp(self, text, lang, Z):
        """Z [B,16,T] long (several label versions of one utterance). Returns log p [B,16,T]."""
        m, cfg, tl = self.model, self.cfg, self.tl
        input_ids = self.ids(text, lang)
        te = m.thinker.get_input_embeddings()(input_ids).to(tl.dtype)
        st = torch.nonzero(input_ids[0] == cfg.im_start_token_id).view(-1).tolist() + [input_ids.shape[1]]
        mm = torch.zeros_like(input_ids, dtype=torch.bool)
        u = m._get_talker_user_parts(st[0], st[1], mm, te, te)
        a, pid_a, trail = m._get_talker_assistant_parts(st[1], st[2], self.spk, te, self.tts_pad, self.tts_bos, self.tts_eos)
        pre, pid = torch.cat([u, a], 1), torch.cat([input_ids[:, st[0]:st[1]], pid_a], 1)
        B, _, T = Z.shape
        L = pre.shape[1]
        fs = self.emb0(Z[:, 0]) + sum(self.embk[k - 1](Z[:, k]) for k in range(1, 16))
        txt = torch.cat([trail, self.tts_pad.expand(1, max(0, T - 1 - trail.shape[1]), -1)], 1)[:, :T - 1]
        x = torch.cat([pre.expand(B, -1, -1), fs[:, :T - 1] + txt], 1)
        ids = torch.cat([pid, torch.full((1, T - 1), cfg.tts_pad_token_id, device=self.dev)], 1).expand(B, -1)
        tl.rope_deltas = None
        out = tl(inputs_embeds=x, attention_mask=torch.ones(ids.shape, dtype=torch.long, device=self.dev),
                 talker_input_ids=ids, output_hidden_states=True, use_cache=False)
        lp = torch.empty(B, 16, T, device=self.dev)
        lp[:, 0] = out.logits[:, L - 1:L - 1 + T].float().log_softmax(-1).gather(-1, Z[:, 0, :, None])[..., 0]
        h = out.hidden_states[0][-1][:, L - 1:L - 1 + T].reshape(B * T, 1, -1)
        seq = torch.cat([h, self.emb0(Z[:, 0]).reshape(B * T, 1, -1)]
                        + [self.embk[k - 1](Z[:, k]).reshape(B * T, 1, -1) for k in range(1, 15)], 1)
        cp = tl.code_predictor
        hs = cp.model(inputs_embeds=seq, use_cache=False).last_hidden_state
        for k in range(1, 16):
            lp[:, k] = cp.lm_head[k - 1](hs[:, k]).float().log_softmax(-1).gather(-1, Z[:, k].reshape(B * T, 1))[:, 0].view(B, T)
        return lp


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("manifest")
    ap.add_argument("--talker", action="store_true")
    ap.add_argument("--omni", default="Qwen/Qwen3-Omni-30B-A3B-Instruct")
    ap.add_argument("--weight", default=None)
    a = ap.parse_args()
    rows = [json.loads(l) for l in open(a.manifest, encoding="utf-8") if l.strip()]
    mimi = load_mimi(a.weight)
    pairs = []
    for r in rows:
        ref = encode(mimi, load_wav(r["wav"]))
        yours = torch.from_numpy(np.load(r["codes"]).astype(np.int64))
        assert yours.shape == ref.shape, f"{r['codes']}: shape {tuple(yours.shape)}, recipe {tuple(ref.shape)}"
        pairs.append((r, yours, ref))
    del mimi
    torch.cuda.empty_cache()
    agree = np.mean([(y[:, :-1] == z[:, :-1]).float().mean(1).numpy() for _, y, z in pairs], 0)
    last = np.mean([(y[:, -1] == z[:, -1]).float().mean().item() for _, y, z in pairs])
    print(f"agreement with recipe (excl. last frame): layer0 {agree[0]:.3f}  layers1-4 {agree[1:5].mean():.3f}  "
          f"layers5-15 {agree[5:].mean():.3f};  last frame (all layers) {last:.3f}")
    if not a.talker:
        return
    T_ = Talker(a.omni)
    d, dl = [], []
    for r, y, z in pairs:
        lp = T_.logp(r["text"], r["lang"], torch.stack([y, z]).cuda())
        d.append((lp[0, :, :-1].sum(0) - lp[1, :, :-1].sum(0)).mean().item())
        dl.append((lp[0, :, -1].sum() - lp[1, :, -1].sum()).item())
    d = np.array(d)
    rng = np.random.default_rng(0)
    bs = [d[rng.integers(0, len(d), len(d))].mean() for _ in range(2000)]
    print(f"Talker delta = log p(yours) - log p(recipe): {d.mean():+.3f} nats/frame "
          f"[{np.percentile(bs, 2.5):+.3f}, {np.percentile(bs, 97.5):+.3f}] over {len(d)} utterances")
    print(f"last frame: log p(yours) - log p(recipe) = {np.mean(dl):+.1f} nats")


if __name__ == "__main__":
    main()
