"""Calibration code LMs for the Talker likelihood test.

A small Talker-shaped model with no text input: a frame transformer over the sum of the 16 code embeddings of the
previous frame predicts codebook 0; a depth transformer over [h_t, emb(c_0) .. emb(c_14)] predicts codebooks 1..15.
Each model is trained unconditionally on one encoder variant's codes. A model trained on variant C's labels prefers C's
codes, so comparing the Talker's preferences with these models' preferences tells which labels the Talker was trained
on. Scoring is per code, exactly like the Talker's teacher-forced log-likelihood.

  python calib_lm.py train --codes train_codes.npy --out lm_X.pt [--seed 0] [--steps 8000]
  python calib_lm.py train --codes A.npy+B.npy@0.2 --out lm_mix20.pt    # each chunk takes B's codes with prob 0.2
  python calib_lm.py score --lm lm_X.pt --codes codes.pt --out lp_X.pt [--device cpu]

train: --codes is an integer array [N, 16, T] of one variant's labels (we used 10 s chunks, T = 125). Needs a GPU.
score: --codes is a torch file {"meta": [...], "codes": {variant: [int tensor [16, T_i], ...]}} holding several label
  versions of the same utterances, like codes.pt of the dataset; writes {"meta": [...], "lp": {variant: [float32
  [16, T_i], ...]}}, the layout of the dataset's scores/*.pt.
The models take at most 160 frames (12.8 s) per sequence.
"""
import argparse, math, time
import numpy as np, torch, torch.nn as nn

D, NL, ND, NH, V, NQ = 512, 8, 3, 8, 2048, 16


class Blocks(nn.Module):
    def __init__(self, n):
        super().__init__()
        layer = nn.TransformerEncoderLayer(D, NH, 4 * D, dropout=0.1, batch_first=True, norm_first=True, activation="gelu")
        self.enc = nn.TransformerEncoder(layer, n, enable_nested_tensor=False)
        self.norm = nn.LayerNorm(D)

    def forward(self, x):
        T = x.shape[1]
        mask = torch.triu(torch.full((T, T), float("-inf"), device=x.device), 1)
        return self.norm(self.enc(x, mask=mask, is_causal=True))


class CodeLM(nn.Module):
    def __init__(self, maxT=160):
        super().__init__()
        self.temb = nn.ModuleList(nn.Embedding(V, D) for _ in range(NQ))
        self.bos = nn.Parameter(torch.zeros(D)); self.pos = nn.Embedding(maxT, D)
        self.tf = Blocks(NL)
        self.demb = nn.ModuleList(nn.Embedding(V, D) for _ in range(NQ - 1))
        self.dpos = nn.Embedding(NQ, D); self.dproj = nn.Linear(D, D)
        self.df = Blocks(ND)
        self.heads = nn.ModuleList(nn.Linear(D, V) for _ in range(NQ))

    def forward(self, z):  # z [B,16,T] long -> log p [B,16,T]
        B, _, T = z.shape
        if T > self.pos.num_embeddings:
            raise ValueError(f"{T} frames; the model takes at most {self.pos.num_embeddings} (12.8 s), split longer inputs")
        s = sum(self.temb[k](z[:, k]) for k in range(NQ))  # [B,T,D]
        x = torch.cat([self.bos.expand(B, 1, D), s[:, :-1]], 1) + self.pos(torch.arange(T, device=z.device))
        h = self.tf(x)  # [B,T,D]
        lp = torch.empty(B, NQ, T, device=z.device)
        lp[:, 0] = self.heads[0](h).float().log_softmax(-1).gather(-1, z[:, 0, :, None])[..., 0]
        dx = torch.cat([self.dproj(h).reshape(B * T, 1, D)] + [self.demb[k](z[:, k]).reshape(B * T, 1, D) for k in range(NQ - 1)], 1)
        dh = self.df(dx + self.dpos.weight[None])  # [BT,16,D]
        for k in range(1, NQ):
            lp[:, k] = self.heads[k](dh[:, k - 1]).float().log_softmax(-1).gather(-1, z[:, k].reshape(-1, 1))[:, 0].view(B, T)
        return lp


def load_lm(path, device="cuda"):
    m = CodeLM().to(device).eval()
    m.load_state_dict(torch.load(path, map_location=device, weights_only=True))
    return m


def train(codes, out, seed=0, steps=8000, bs=64):
    torch.manual_seed(seed); np.random.seed(seed)
    if "+" in codes:  # mixture "A.npy+B.npy@p": each chunk takes B's codes with prob p (fixed per chunk)
        pa, rest = codes.split("+"); pb, pr = rest.split("@")
        CA, CB = np.load(pa, mmap_mode="r"), np.load(pb, mmap_mode="r")
        assert CA.shape == CB.shape, (CA.shape, CB.shape)
        selB = np.random.default_rng(1234).random(len(CA)) < float(pr)
        print(f"mixture {pa} / {pb}: B share {selB.mean():.3f}", flush=True)
        C = CA
    else:
        C = np.load(codes, mmap_mode="r")  # [N,16,T]
        selB = None
    N = len(C)
    m = CodeLM().cuda()
    opt = torch.optim.AdamW(m.parameters(), lr=3e-4, betas=(0.9, 0.95), weight_decay=0.1)
    sched = torch.optim.lr_scheduler.LambdaLR(opt, lambda s: min(1, (s + 1) / 500) * 0.5 * (1 + math.cos(math.pi * min(s, steps) / steps)))
    rng = np.random.default_rng(seed); t0 = time.time()
    for step in range(steps):
        idx = np.sort(rng.choice(N, bs, replace=False))
        zc = C[idx] if selB is None else np.where(selB[idx][:, None, None], CB[idx], CA[idx])
        z = torch.from_numpy(zc.astype(np.int64)).cuda()
        with torch.autocast("cuda", dtype=torch.bfloat16):
            lp = m(z)
        loss = -lp.mean()
        opt.zero_grad(set_to_none=True); loss.backward()
        torch.nn.utils.clip_grad_norm_(m.parameters(), 1.0); opt.step(); sched.step()
        if step % 500 == 0 or step == steps - 1:
            print(f"step {step} loss/code {loss.item():.4f} ({time.time() - t0:.0f}s)", flush=True)
    torch.save(m.state_dict(), out)
    print(f"TRAIN_DONE {out}", flush=True)


@torch.no_grad()
def score(lm, codes, out, device="cuda"):
    d = torch.load(codes, weights_only=True)
    meta, cs = d["meta"], d["codes"]
    cands = list(cs)
    m = load_lm(lm, device)
    lp = {c: [] for c in cands}
    for i in range(len(meta)):
        Z = torch.stack([cs[c][i] for c in cands]).long().to(device)
        for c, x in zip(cands, m(Z).cpu()):
            lp[c].append(x)
    torch.save({"meta": meta, "lp": lp}, out)
    print(f"SCORE_DONE {out}: {len(meta)} utterances x {len(cands)} variants", flush=True)


def main():
    ap = argparse.ArgumentParser()
    sub = ap.add_subparsers(dest="cmd", required=True)
    t = sub.add_parser("train")
    t.add_argument("--codes", required=True, help="int array [N,16,T], or 'A.npy+B.npy@p' for a mixture")
    t.add_argument("--out", required=True)
    t.add_argument("--seed", type=int, default=0)
    t.add_argument("--steps", type=int, default=8000)
    t.add_argument("--batch", type=int, default=64)
    s = sub.add_parser("score")
    s.add_argument("--lm", required=True)
    s.add_argument("--codes", required=True, help='torch file {"meta": [...], "codes": {variant: [LongTensor [16,T], ...]}}')
    s.add_argument("--out", required=True)
    s.add_argument("--device", default="cuda" if torch.cuda.is_available() else "cpu")
    a = ap.parse_args()
    if a.cmd == "train":
        train(a.codes, a.out, a.seed, a.steps, a.batch)
    else:
        score(a.lm, a.codes, a.out, a.device)


if __name__ == "__main__":
    main()
