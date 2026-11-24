# Qwen3-Omni Talker Finetune

Label recipe, label checker and fine-tuning toolkit for the Talker of Qwen3-Omni.

Qwen3-Omni speaks through its **Talker**, which predicts 16 audio codebooks at 12.5 Hz; a multi-token predictor (MTP)
fills in codebooks 1–15 and **Code2Wav** turns codes into 24 kHz audio. Qwen did not release the encoder that turned
training audio into those codes, so new voices could not be trained on matching labels. This repository contains:

- **The recipe** to make labels that match the Talker's training labels (`qwen_mimi_encode.py`).
- **`label_check.py`**, which tells you whether your labels look like the Talker's own.
- **`talker_tune`**, a toolkit to fine-tune the Talker and MTP on new voices and use the result in transformers and
  vLLM-Omni.
- **`calib_lm.py`**, the calibration code language models used for the likelihood test.
- The technical report: *Reproducing the Speech Labels of Qwen3-Omni's Talker: Mimi in bf16, and a Last-Frame
  Pitfall*.

Data (test-set codes for all encoder variants, Talker log-likelihoods, the 13 calibration models) and sample voices:
[Hugging Face collection](https://huggingface.co/collections/mifanbushipeicai/qwen3-omni-talker-finetune-6aec41e50ac465e3f0ef9222).

## What we found

| Question | Answer | How we know |
|---|---|---|
| Which codec? | kyutai's Mimi, first 16 of 32 codebooks. | Talker, MTP and Code2Wav embedding tables are linearly predictable from Mimi's codebooks, layer by layer; a codec with Mimi's architecture but other weights (Qwen3-TTS-Tokenizer-12Hz) is not. |
| Which implementation? | Mimi run with kyutai's `moshi` package, weights cast to bf16. | Teacher-forced Talker likelihoods order the encoder implementations the same way as a code LM trained on such labels, and unlike LMs trained on fp32, fp16 or transformers-bf16 labels. The Talker's own sampled codes are also fit best by that LM. With `label_check` on FLEURS, the recipe is preferred over transformers labels in all 10 of the Talker's speech languages. |
| Any preprocessing? | No loudness normalization; no band-limiting of ≥ 20 % of the data. | Calibrated against LMs trained on those variants. |
| Same weights? | Consistent with yes, but not provable. | Likelihood tests cannot detect weight changes in directions nobody constructs. |
| Anything to watch out for? | **Pad the audio before encoding.** | transformers' `MimiModel.encode` computes the last frame differently from kyutai's encoder; after LoRA fine-tuning on such labels, about half of the generated sentences end with a noise burst. |

The idea that the codec is Mimi came from the community (abrar360 and SeungyounShin in
[QwenLM/Qwen2.5-Omni#219](https://github.com/QwenLM/Qwen2.5-Omni/issues/219)); we tested how exact it is and what
matters in practice.

## Recipe

```bash
pip install moshi==0.2.13 soundfile soxr
python qwen_mimi_encode.py --out codes/ a.wav b.wav        # one int16 .npy [16, ceil(n/1920)] per file
```

```python
from moshi.models import loaders
mimi = loaders.get_mimi(weight_path, device="cuda", num_codebooks=16).to(torch.bfloat16).eval()
n = -(-wav.shape[-1] // 1920)                          # frames at 12.5 Hz
x = torch.nn.functional.pad(wav, (0, wav.shape[-1]))   # zero-pad the tail
codes = mimi.encode(x.to(torch.bfloat16))[:, :16, :n]
```

The codes are bit-identical only on the same GPU type with the same torch version (the released codes: torch 2.9.1;
with torch 2.9.1 and `moshi` 0.2.13 an H100 reproduces the released H100 recipe codes, `E0bnPx2_h100`, bit for bit). Another GPU, or
torch 2.15 on the same GPU, changes about 1 % of the codebook-0 codes and 6–7 % of the codebook 1–4 codes (bf16
rounding).

If you use transformers' `MimiModel` instead, at least pad the input to a multiple of 1920 samples. For LoRA voice
adaptation, precision made no measurable difference in our tests; the padding did.

## Check your labels

```bash
python label_check.py manifest.jsonl                    # code agreement with the recipe (no Talker needed)
python label_check.py manifest.jsonl --talker           # + Talker log-likelihood difference, nats/frame
```

`manifest.jsonl`: `{"wav": "...", "text": "...", "lang": "zh"|"en", "codes": "....npy"}` per line; 30–100 utterances
are enough. The Talker value is log p(your codes) − log p(recipe codes) per frame, so negative means the Talker finds
your labels less likely than the recipe's. The report's Λ(C) = log p(fp32 reference) − log p(C) has the opposite
sign: Λ < 0 means C is preferred.
Reference values: recipe 0; transformers fp32 about −1.2; transformers bf16 about −0.7 (last frame
excluded). The last frame is reported separately and shows whether the tail was padded.

## Fine-tune the Talker (`talker_tune`)

Requirements: `torch>=2.9,<2.10`, `transformers>=5.20`, `peft`, `soundfile`, `soxr`, `moshi`; `jiwer` and
`sherpa-onnx` for evaluation. `moshi` pins `huggingface-hub<1.0`, which transformers 5 cannot use, so install it
without its dependencies (`requirements.txt` lists the ones it needs):

```bash
pip install -r requirements.txt
pip install --no-deps moshi==0.2.13
```

`talker_tune.spoken` also needs `torchvision` (the build matching your torch) and `pillow`. You also need the
Qwen3-Omni-30B-A3B-Instruct checkpoint (about 66 GB on disk). The Thinker is loaded without its decoder layers (only
its embeddings are needed for read-aloud training), so the Talker takes 7.2 GB and LoRA training peaks at 14 GB
allocated; it runs under a 22 GB per-process cap, so a 24 GB card is enough (0.45 s per step on an RTX PRO 6000).

```bash
# manifest.jsonl: {"audio": "wavs/0001.wav", "text": "Spoken text.", "speaker": "myvoice"}
python -m talker_tune.prepare manifest.jsonl --out data --held 20
python -m talker_tune.train --data data --out run --steps 800          # LoRA r16 on Talker + MTP
python -m talker_tune.synth --adapter run/step_800 --speaker myvoice --text "Hello, this is my new voice." --out samples
python -m talker_tune.evaluate --synth samples --data data             # CER/WER, speaker similarity, tail noise, stopping
python -m talker_tune.export --adapter run/step_800 --out Qwen3-Omni-myvoice
```

```python
model = Qwen3OmniMoeForConditionalGeneration.from_pretrained("Qwen3-Omni-myvoice", dtype=torch.bfloat16, device_map="cuda")
text_ids, audio = model.generate(**inputs, speaker="myvoice")
```

vLLM-Omni (tested with 0.30): serve the exported directory and pass `"speaker": "myvoice"` in the chat request. If you
use your own stage config, check the Talker stage's `max_tokens` (one token is 80 ms of speech; the bundled
`qwen3_omni_moe.yaml` uses 4096).

In chat (20 single-turn questions per language, transformers) the new voice holds for text questions and for spoken
Chinese questions (similarity 0.98 to the target speaker, no noisy endings); vLLM-Omni gives the same (10 text
questions per language: 0.985 / 0.988). Two caveats:

- The LoRA changes shared Talker weights, so the preset voices (Chelsie, Aiden, ...) also speak with the new voices.
  Keep the base checkpoint for them.
- For spoken English questions, a LoRA with one speaker per language answers in the Chinese speaker's voice
  (similarity 0.81 to p239's own answers, 0.97 to SSB0033). The base model keeps its voice on the same spoken
  questions, so this comes from the LoRA: it ties the voice to the language of the prompt. A follow-up with up to two
  female and two male speakers per language (3 seeds) fixed it: with 2 + 2 speakers per language and 400 steps per
  speaker (3,200 in total), spoken English questions give 0.983 (text questions 0.989) and error rates do not change.
  Neither part works alone: one speaker per language at four times the steps still drifts (0.83) and overfits
  (Chinese CER 19–22 %); 2 + 2 speakers at 200 steps each still drift for the female voices (0.80 for p239). With one
  female and one male speaker per language only the female voice drifts (0.86). So train at least two speakers of
  each gender per language and set `--steps` to about 400 × the number of speakers. Start female ids from Chelsie and
  male ids from Ethan, as in these runs (`--init-speaker chelsie,p259=ethan,p286=ethan`). Replacing half of the text
  prompts with same-language spoken questions (`talker_tune.spoken`, then `train --spoken-pool`) fixed the 1 + 1 case
  (0.988) but made the one-speaker case worse (English text questions and read-aloud drift too).

Useful options: `--lora-scope talker|mtp|both`, `--limit-min` (use only N minutes per speaker), `--mode dense`,
`--max-gpu-gb` (emulate a smaller card), `--init-speaker` (which preset voice new speakers start from; `name=preset`
entries set it per speaker), `--spoken-pool` / `--spoken-frac` (same-language spoken user turns built by
`talker_tune.spoken`, which runs the full Thinker once: about 70 GB of GPU memory, or CPU offload). More in
[`talker_tune/README.md`](talker_tune/README.md).

## Results

Two speakers (AISHELL-3 SSB0033, Chinese; VCTK p239, English), about 23 minutes each, rank-16 LoRA, 800 steps,
3 training seeds; 40 held-out sentences per speaker, each sampled with 3 seeds; CER with Paraformer, WER with
Parakeet-TDT:

| labels | CER zh | WER en | speaker similarity zh / en | last 80 ms loud (noise burst) zh / en |
|---|---|---|---|---|
| transformers fp32, unpadded | 3.08 % | 1.36 % | 0.969 / 0.958 | 49 % / 55 % |
| transformers bf16, unpadded | 3.19 % | 1.24 % | 0.968 / 0.958 | 44 % / 53 % |
| `moshi` bf16, unpadded | 3.22 % | 1.24 % | 0.970 / 0.959 | 0 % / 0.8 % |
| transformers fp32, padded | 3.45 % | 1.44 % | 0.969 / 0.958 | 0 % / 0 % |
| transformers bf16, padded | 3.45 % | 1.24 % | 0.968 / 0.956 | 0.3 % / 0.3 % |
| **recipe** (`moshi` bf16, padded) | 3.11 % | 1.16 % | 0.969 / 0.958 | 0 % / 0.3 % |
| real recordings | 2.35 % | 1.44 % | — | 0 % / 2.5 % |
| base model, preset voices | 12.8 % | 8.5 % | — | — |

For the five other label rows, every CER/WER and similarity difference to the recipe row has a bootstrap 95 % CI
containing 0. Padding alone removes the noise burst; precision does not matter here.

Toolkit runs (`talker_tune`, same two speakers, 800 steps, one seed, 40 held-out sentences per speaker sampled once;
CER with Paraformer, WER with Whisper-large-v3-turbo, as `talker_tune.evaluate` does):

| run | CER zh | WER en | speaker similarity zh / en | noise burst zh / en | no stop |
|---|---|---|---|---|---|
| new speaker ids, 23 min each | 3.3 % | 3.7 % | 0.969 / 0.960 | 0 % / 0 % | 0 % |
| replacing preset voices (Chelsie, Aiden) | 3.8 % | 3.0 % | 0.970 / 0.963 | 0 % / 0 % | 0 % |
| 10 min each | 4.8 % | 5.8 % | 0.965 / 0.959 | 0 % / 0 % | 1.3 % |
| 5 min each | 5.8 % | 5.9 % | 0.966 / 0.957 | 0 % / 2.5 % | 0 % |
| 2.5 min each | 5.7 % | 5.9 % | 0.968 / 0.955 | 0 % / 5 % | 0 % |
| LoRA on the Talker only | 3.0 % | 4.3 % | 0.966 / 0.958 | 0 % / 0 % | 0 % |
| LoRA on the MTP only | 11.3 % | 3.7 % | 0.962 / 0.880 | 18 % / 10 % | 1.3 % |
| real recordings, same ASR | 2.5 % | 3.4 % | — | 0 % / 2.5 % | — |

A few minutes of speech already give the voice; more data mainly improves intelligibility (one seed and 40 sentences
per cell, so differences of about one point are noise). The Talker carries the voice: training only the MTP is not
enough, training only the Talker is about as good as both.

## Calibration models

`calib_lm.py` trains and scores the small code LMs (Talker-shaped: a frame transformer for codebook 0 and a depth
transformer for codebooks 1–15, no text input) that calibrate the likelihood test. The 13 trained models are in the
dataset of the Hugging Face collection.

```bash
python calib_lm.py train --codes train_codes.npy --out lm_X.pt             # codes: int [N, 16, T]
python calib_lm.py score --lm lm_X.pt --codes codes.pt --out lp_X.pt       # per-code log p for every variant
```

`score` reads and writes the layouts of the dataset's `codes.pt` and `scores/*.pt`, and runs on CPU with
`--device cpu`. The models take at most 160 frames (12.8 s) per sequence.

## Other models using Mimi codes

CSM-1B: transformers' CSM training path encodes each segment with `MimiModel.encode` without padding; under the
released model this last frame costs +119 nats compared with kyutai-encoded labels, and padding removes the gap.
After LoRA fine-tuning as in the Unsloth notebook, 46 % of the generations end with a noise burst with the default
labels and 0 % with padded ones (one VCTK speaker, 2 seeds). We reported this to transformers and proposed a fix:
[huggingface/transformers#50396](https://github.com/huggingface/transformers/issues/50396).

## Citation

```bibtex
@misc{chen2026qwen3omnitalker,
  title  = {Reproducing the Speech Labels of {Qwen3-Omni}'s Talker: {Mimi} in bf16, and a Last-Frame Pitfall},
  author = {Chen, Yuxuan},
  year   = {2026},
  note   = {Technical report},
  url    = {https://github.com/elpsykongloo/qwen3-omni-talker-finetune}
}
```

## Acknowledgements

kyutai for Mimi and `moshi`; the Qwen team for Qwen3-Omni; abrar360 and SeungyounShin for the first public evidence
that the codec is Mimi; Hert4 and Unsloth for earlier Talker training code.

## License

Code: Apache-2.0. Mimi weights (`kyutai/mimi`, `kyutai/moshiko-pytorch-bf16`) are CC-BY 4.0 and the `moshi` package is MIT;
Qwen3-Omni weights are Apache-2.0. Fine-tuned voices inherit the rights of the recordings you train on.

`talker_tune` adapts the model to a new voice from a few minutes of recordings, so it could be used to imitate a
person without their consent. Adapt it to another person's voice only with that person's consent.
