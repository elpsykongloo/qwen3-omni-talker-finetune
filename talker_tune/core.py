"""Qwen3-Omni Talker without the Thinker stack (transformers >= 5): prompt assembly, teacher-forced scoring,
LoRA / partial full fine-tuning, synthesis, adapter save/load.

For text-only prompts the Talker reads the Thinker's word embeddings and nothing else, so the Thinker is loaded with
zero decoder layers (about 20 GB of GPU memory for the rest instead of about 70 GB).
"""
import json, os, re
import torch, torch.nn.functional as F

READ_PROMPT = {"zh": "请朗读下面这句话：", "en": "Please read the following sentence aloud: "}
_T = r"model\.layers\.\d+\.(self_attn\.(q|k|v|o)_proj|mlp\.shared_expert\.(gate|up|down)_proj)"
_M = r"code_predictor\.model\.layers\.\d+\.(self_attn\.(q|k|v|o)_proj|mlp\.(gate|up|down)_proj)"
LORA_SCOPES = {"both": f"({_T}|{_M})", "talker": f"(?!code_predictor){_T}", "mtp": _M}
LORA_TARGETS = LORA_SCOPES["both"]


def guess_lang(text):
    return "zh" if re.search(r"[一-鿿]", text) else "en"


class TalkerOnly:
    def __init__(self, omni, device="cuda", dtype=torch.bfloat16, experts="grouped_mm"):
        """experts: transformers experts_implementation. grouped_mm is ~20x faster than eager for training."""
        from transformers import AutoConfig, AutoTokenizer, Qwen3OmniMoeForConditionalGeneration
        from transformers.utils import logging as hf_logging
        cfg = AutoConfig.from_pretrained(omni)
        cfg.thinker_config.text_config.num_hidden_layers = 0  # text prompts only need the Thinker's word embeddings
        kw = {"experts_implementation": experts} if experts else {}
        verb = hf_logging.get_verbosity(); hf_logging.set_verbosity_error()  # skipped Thinker layers are reported as unused
        try:
            m = Qwen3OmniMoeForConditionalGeneration.from_pretrained(omni, config=cfg, dtype=dtype, device_map=device,
                                                                     attn_implementation="sdpa", **kw)
        except (ValueError, ImportError, RuntimeError) as e:  # experts backend unavailable here
            print(f"experts_implementation={experts} unavailable ({e}); using eager", flush=True)
            m = Qwen3OmniMoeForConditionalGeneration.from_pretrained(omni, config=cfg, dtype=dtype, device_map=device,
                                                                     attn_implementation="sdpa")
        hf_logging.set_verbosity(verb)
        for name in ("audio_tower", "visual", "lm_head"):  # unused for text prompts
            if hasattr(m.thinker, name):
                delattr(m.thinker, name)
        m.eval().requires_grad_(False)
        torch.cuda.empty_cache()
        self.model, self.cfg, self.omni = m, m.config, omni
        self.tok = AutoTokenizer.from_pretrained(omni)
        self.dev = torch.device(device)
        tl = self.tl = m.talker
        self.emb0 = tl.get_input_embeddings()
        self.embk = tl.code_predictor.get_input_embeddings()
        self.eos = self.cfg.talker_config.codec_eos_token_id
        self.speakers = dict(self.cfg.talker_config.speaker_id)

    # ---------------------------------------------------------------- prompts
    def tts_embeds(self):
        """tts_bos, tts_eos, tts_pad as Talker inputs [1,1,D]; recomputed because text_projection may be trained."""
        c = self.cfg
        sp = torch.tensor([[c.tts_bos_token_id, c.tts_eos_token_id, c.tts_pad_token_id]], device=self.dev)
        return self.tl.text_projection(self.model.thinker.get_input_embeddings()(sp).to(self.tl.dtype)).chunk(3, dim=1)

    def ids(self, text, user=None, lang=None):
        """Chat ids for one spoken assistant turn. Default user turn: a read-aloud instruction (TTS-style data)."""
        if user is None:
            user = READ_PROMPT.get(lang or guess_lang(text), READ_PROMPT["en"]) + text
        s = "<|im_start|>user\n" + user + "<|im_end|>\n<|im_start|>assistant\n" + text + "<|im_end|>"
        return torch.tensor([self.tok(s, add_special_tokens=False)["input_ids"]], device=self.dev)

    def ids_spoken(self, text, user_ids):
        """Chat ids with a spoken user turn: `user_ids` [L] from spoken.py (user turn incl. audio placeholders)."""
        a = self.tok("<|im_start|>assistant\n" + text + "<|im_end|>", add_special_tokens=False)["input_ids"]
        return torch.cat([user_ids.to(self.dev), torch.tensor(a, device=self.dev)])[None]

    def prefix(self, input_ids, spk_id, tts=None, user_hidden=None):
        """Talker prefill embeds / ids and the trailing text that is fed one entry per generated frame.
        user_hidden: Thinker hidden states (accept_hidden_layer) at the audio positions of a spoken user turn."""
        m, cfg = self.model, self.cfg
        tts_bos, tts_eos, tts_pad = tts if tts is not None else self.tts_embeds()
        te = m.thinker.get_input_embeddings()(input_ids).to(self.tl.dtype)
        st = torch.nonzero(input_ids[0] == cfg.im_start_token_id).view(-1).tolist() + [input_ids.shape[1]]
        th, mm = te, torch.zeros_like(input_ids, dtype=torch.bool)
        if user_hidden is not None:
            mm = input_ids == cfg.thinker_config.audio_token_id
            assert mm.sum() == user_hidden.shape[0], (mm.sum(), user_hidden.shape)
            th = te.clone(); th[mm] = user_hidden.to(te.device, te.dtype)
        parts, pids, trailing = [], [], None
        for i in range(len(st) - 1):
            a, b = st[i], st[i + 1]
            role = input_ids[0, a + 1]
            if role == cfg.user_token_id:
                parts.append(m._get_talker_user_parts(a, b, mm, th, te)); pids.append(input_ids[:, a:b])
            elif role == cfg.assistant_token_id and i == len(st) - 2:
                e, ids_, trailing = m._get_talker_assistant_parts(a, b, spk_id, te, tts_pad, tts_bos, tts_eos)
                parts.append(e); pids.append(ids_)
        assert trailing is not None, "the last turn must be the assistant turn"
        return torch.cat(parts, 1), torch.cat(pids, 1), trailing

    def frame_sum(self, z):
        """z [B,16,T] -> summed codec embeddings [B,T,D] (the Talker's per-frame input)."""
        s = self.emb0(z[:, 0])
        for k in range(1, 16):
            s = s + self.embk[k - 1](z[:, k])
        return s

    # ---------------------------------------------------------------- scoring / training loss
    def nll_batch(self, items):
        """items: list of (input_ids [1,L], speaker id, Z [1,16,T] long[, user_hidden]). One right-padded Talker pass and
        one MTP pass for all items. Returns a list of (nll [16,T], eos_nll scalar)."""
        tl = self.tl
        tts = self.tts_embeds()
        tts_pad = tts[2]
        rows = []
        for input_ids, spk, Z, *uh in items:
            pre, pid, trail = self.prefix(input_ids, spk, tts, uh[0] if uh else None)
            T = Z.shape[-1]
            txt = torch.cat([trail, tts_pad.expand(1, max(0, T - trail.shape[1]), -1)], 1)[:, :T]
            x = torch.cat([pre, self.frame_sum(Z) + txt], 1)
            ids = torch.cat([pid, torch.full((1, T), self.cfg.tts_pad_token_id, device=self.dev)], 1)
            rows.append((x, ids, pre.shape[1], T, Z))
        S, B = max(r[0].shape[1] for r in rows), len(rows)
        X = torch.zeros(B, S, rows[0][0].shape[-1], dtype=tl.dtype, device=self.dev)
        I = torch.full((B, S), self.cfg.tts_pad_token_id, dtype=torch.long, device=self.dev)
        M = torch.zeros(B, S, dtype=torch.long, device=self.dev)
        for b, (x, ids, L, T, Z) in enumerate(rows):
            n = x.shape[1]
            X[b, :n] = x.to(tl.dtype); I[b, :n] = ids[0]; M[b, :n] = 1
        tl.rope_deltas = None
        with torch.autocast("cuda", dtype=torch.bfloat16, enabled=getattr(self, "mode", None) == "dense"):
            out = tl(inputs_embeds=X, attention_mask=M, talker_input_ids=I, output_hidden_states=True, use_cache=False)
        last = out.hidden_states[0][-1]
        res, hs_all, zs = [], [], []
        for b, (x, ids, L, T, Z) in enumerate(rows):
            lg = out.logits[b, L - 1:L + T].float()
            n0 = F.cross_entropy(lg[:T], Z[0, 0], reduction="none")
            ne = F.cross_entropy(lg[T:T + 1], torch.tensor([self.eos], device=self.dev))
            res.append((n0, ne)); hs_all.append(last[b, L - 1:L - 1 + T]); zs.append(Z[0])
        h, Zc = torch.cat(hs_all, 0), torch.cat(zs, -1)  # [sumT,D], [16,sumT]
        cp = tl.code_predictor
        seq = [h[:, None], self.emb0(Zc[0])[:, None]] + [self.embk[k - 1](Zc[k])[:, None] for k in range(1, 15)]
        with torch.autocast("cuda", dtype=torch.bfloat16, enabled=getattr(self, "mode", None) == "dense"):
            hs = cp.model(inputs_embeds=torch.cat(seq, 1).to(tl.dtype), use_cache=False).last_hidden_state
            nk = torch.stack([F.cross_entropy(cp.lm_head[k - 1](hs[:, k]).float(), Zc[k], reduction="none") for k in range(1, 16)])
        outl, o = [], 0
        for (n0, ne), (_, _, _, T, _) in zip(res, rows):
            outl.append((torch.cat([n0[None], nk[:, o:o + T]], 0), ne)); o += T
        return outl

    # ---------------------------------------------------------------- synthesis
    @torch.no_grad()
    def synthesize(self, input_ids, spk_id, max_frames=None, seed=None, top_k=50, top_p=1.0, temperature=0.9,
                   repetition_penalty=1.05, decode=True, user_hidden=None):
        """Returns (codes [16,T] long, wav float32 [n] at 24 kHz or None, ended_with_eos)."""
        tl = self.tl
        tts = self.tts_embeds()
        pre, pid, trail = self.prefix(input_ids, spk_id, tts, user_hidden)
        V = self.cfg.talker_config.text_config.vocab_size
        sup = [i for i in range(V - 1024, V) if i != self.eos]
        if max_frames is None:
            max_frames = 25 + 6 * trail.shape[1]  # generous: ~0.5 s per text token
        if seed is not None:
            torch.manual_seed(seed)
        tl.rope_deltas = None
        r = tl.generate(inputs_embeds=pre, trailing_text_hidden=trail, tts_pad_embed=tts[2], talker_input_ids=pid,
                        attention_mask=torch.ones(pid.shape, dtype=torch.long, device=self.dev),
                        max_new_tokens=max_frames, do_sample=True, top_k=top_k, top_p=top_p, temperature=temperature,
                        eos_token_id=self.eos, repetition_penalty=repetition_penalty, suppress_tokens=sup,
                        output_hidden_states=True, return_dict_in_generate=True)
        Z = torch.stack([hid[-1] for hid in r.hidden_states if hid[-1] is not None], 1).transpose(1, 2).long()
        bad = (Z[0, 0] >= 2048).nonzero()
        if len(bad):
            Z = Z[:, :, :bad[0, 0]]
        ended = len(bad) > 0 or r.sequences[0, -1].item() == self.eos
        w = None
        if decode and Z.shape[-1] >= 2 and not (Z[0, 1:] >= 2048).any():
            w = self.model.code2wav.chunked_decode(Z, chunk_size=300, left_context_size=25).float().reshape(-1).clamp(-1, 1).cpu()
        return Z[0].cpu(), w, ended

    # ---------------------------------------------------------------- trainable parts
    def add_speaker(self, name, init_from):
        """New speaker slot: the first unused codec id above the preset ones, embedding copied from `init_from`."""
        name = name.lower()
        if name in self.speakers:
            return self.speakers[name]
        sid = max(self.speakers.values()) + 1
        assert sid < self.cfg.talker_config.text_config.vocab_size
        w = self.emb0.weight
        with torch.no_grad():
            w[sid] = w[self.speakers[init_from.lower()]]
        self.speakers[name] = sid
        return sid

    def trainable(self, mode="lora", r=16, alpha=32, dropout=0.05, seed=0, speaker_rows=(), text_projection=False, scope="both"):
        """mode 'lora': LoRA on Talker attention + shared expert and MTP attention + MLP.
        mode 'dense': full fine-tuning of every Talker / MTP weight except the 128 routed experts and codec tables.
        speaker_rows: codec embedding rows (new speakers) trained in either mode. Returns the parameter list."""
        tl = self.tl
        ps = []
        if mode == "lora":
            from peft import LoraConfig, inject_adapter_in_model
            torch.manual_seed(seed)
            self.targets = LORA_SCOPES[scope]
            inject_adapter_in_model(LoraConfig(r=r, lora_alpha=alpha, lora_dropout=dropout, target_modules=self.targets), tl)
            for n, p in tl.named_parameters():
                if "lora_" in n:
                    p.data = p.data.float(); p.requires_grad_(True); ps.append(p)
        elif mode == "dense":  # fp32 master weights, bf16 autocast in nll_batch
            for n, p in tl.named_parameters():
                if ".experts." in n or "codec_embedding" in n or "codec_head" in n or n.startswith("hidden_projection"):
                    continue
                if n.startswith("text_projection") and not text_projection:
                    continue
                p.data = p.data.float(); p.requires_grad_(True); ps.append(p)
        else:
            raise ValueError(mode)
        if text_projection and mode == "lora":
            for p in tl.text_projection.parameters():
                p.data = p.data.float(); p.requires_grad_(True); ps.append(p)
            self.mode = "dense"  # fp32 weights inside the Talker need autocast
        else:
            self.mode = mode
        self._rows = sorted(set(speaker_rows))
        if self._rows:  # fp32 copies of the new speaker rows, swapped into the bf16 embedding output
            self.spk_param = torch.nn.Parameter(self.emb0.weight[self._rows].detach().float())
            rows = torch.tensor(self._rows, device=self.dev)

            def swap(mod, inp, out):
                hit = inp[0][..., None] == rows  # [..., n_rows]
                if hit.any():
                    out = torch.where(hit.any(-1, keepdim=True), (hit.to(out.dtype) @ self.spk_param.to(out.dtype)), out)
                return out
            self.emb0.register_forward_hook(swap)
            ps.append(self.spk_param)
        self.lora = mode == "lora"
        print(f"trainable {mode}: {sum(p.numel() for p in ps) / 1e6:.2f} M params"
              + (f" incl. speaker rows {self._rows}" if self._rows else ""), flush=True)
        return ps

    def adapter_state(self):
        """Only what changed: LoRA tensors or dense weights, plus new speaker rows."""
        sd = {}
        for n, p in self.tl.named_parameters():
            if p.requires_grad:
                sd[n] = p.detach().to(torch.float32 if "lora_" in n else torch.bfloat16).cpu().contiguous()
        for j, r_ in enumerate(getattr(self, "_rows", [])):
            sd[f"speaker_row.{r_}"] = self.spk_param[j].detach().float().cpu().clone()
        return sd

    def save_adapter(self, path, meta):
        from safetensors.torch import save_file
        os.makedirs(path, exist_ok=True)
        save_file(self.adapter_state(), f"{path}/adapter.safetensors.tmp")
        os.replace(f"{path}/adapter.safetensors.tmp", f"{path}/adapter.safetensors")
        meta = dict(meta, speakers=self.speakers, mode="lora" if self.lora else "dense")
        json.dump(meta, open(f"{path}/adapter_config.json", "w"), indent=1, ensure_ascii=False)

    def load_adapter(self, path):
        from safetensors.torch import load_file
        meta = json.load(open(f"{path}/adapter_config.json"))
        sd = load_file(f"{path}/adapter.safetensors")
        if meta["mode"] == "lora":
            from peft import LoraConfig, inject_adapter_in_model
            inject_adapter_in_model(LoraConfig(r=meta["r"], lora_alpha=meta["alpha"], lora_dropout=0.0,
                                               target_modules=meta.get("targets", LORA_TARGETS)), self.tl)
            for n, p in self.tl.named_parameters():
                if "lora_" in n:
                    p.data = p.data.float()
        rows = {int(k.split(".")[1]): v for k, v in sd.items() if k.startswith("speaker_row.")}
        sd = {k: v for k, v in sd.items() if not k.startswith("speaker_row.")}
        missing = self.tl.load_state_dict({k: v.to(self.tl.dtype) if "lora_" not in k else v for k, v in sd.items()}, strict=False)
        assert not missing.unexpected_keys, missing.unexpected_keys[:5]
        with torch.no_grad():
            for r_, v in rows.items():
                self.emb0.weight[r_] = v.to(self.emb0.weight.dtype)
        self.speakers.update(meta["speakers"])
        self.tl.eval()
        return meta
