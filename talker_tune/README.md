# talker_tune — fine-tune the Qwen3-Omni Talker on your own voice

Qwen3-Omni speaks through its Talker, which predicts 16 Mimi codebooks at 12.5 Hz. Qwen never released the encoder that
turns audio into those codes, so new voices could not be trained on matching labels. Our report identifies it: the Talker's labels
are kyutai Mimi codes (first 16 codebooks), closest to the `moshi` package encoder with weights cast to bf16. This
package uses that encoder, with one addition: the input is zero-padded at the tail so the last frame is not encoded
from a truncated window. Without that padding, about half of the generated sentences end with a burst of noise.

What you get:

- `prepare`: audio + text manifest to Talker labels (the recipe in the repository README; `--backend hf` uses transformers' MimiModel with the same padding).
- `train`: LoRA (default) or dense fine-tuning of the Talker and its MTP code predictor. The Thinker is loaded without its
  decoder layers, so training fits on a 24 GB GPU (peak 14 GB allocated; tested under a 22 GB cap). New voices get their own speaker id, but the LoRA changes the shared Talker weights, so the preset ids then
  speak with the new voices too (keep the base checkpoint, or unload the adapter, for the presets).
- `spoken`: a pool of spoken user turns (training sentences read by the base model in preset voices, or your own
  recordings) for `train --spoken-pool`. It runs the full Thinker once (about 70 GB of GPU memory, or CPU offload).
- `synth`: text to 24 kHz speech with the fine-tuned Talker (no Thinker generation needed).
- `export`: merge the adapter into a full Qwen3-Omni checkpoint with the original layout. transformers and vLLM-Omni
  (tested 0.30) load it as usual; call it with `speaker="<your voice>"`. In chat the new voice holds for text questions
  (similarity 0.98). For spoken questions, train more than one speaker of each gender per language, about 400 steps per
  speaker (`--steps` = 400 × speakers); with one speaker per language our English voice fell back to the Chinese one
  (see the [repository README](../README.md)).
  In vLLM-Omni with your own stage config, check the Talker stage's `max_tokens` (one token is 80 ms of speech).
- `evaluate`: CER (Paraformer, Chinese) / WER (Whisper, other languages), speaker similarity, end-of-utterance noise and
  stopping rate, with your held-out recordings scored the same way as a floor.

Requirements: `torch>=2.9,<2.10`, `transformers>=5.20`, `peft`, `soundfile`, `soxr`, `moshi` (for the default label
backend), `jiwer` and `sherpa-onnx` (evaluation); `torchvision` and `pillow` for `spoken`. Install `moshi==0.2.13` with
`--no-deps` after `requirements.txt` (see the [repository README](../README.md)). You also need the
Qwen3-Omni-30B-A3B-Instruct checkpoint (about 66 GB on disk).

## Quick start

```bash
# 1. manifest.jsonl: one utterance per line
#    {"audio": "wavs/0001.wav", "text": "Spoken text.", "speaker": "myvoice"}
python -m talker_tune.prepare manifest.jsonl --out data --held 20

# 2. train (LoRA r16 on Talker attention + shared expert and MTP attention + MLP)
python -m talker_tune.train --omni Qwen/Qwen3-Omni-30B-A3B-Instruct --data data --out run --steps 800

# 3. listen
python -m talker_tune.synth --omni Qwen/Qwen3-Omni-30B-A3B-Instruct --adapter run/step_800 --speaker myvoice \
    --text "Hello, this is my new voice." --out samples
python -m talker_tune.evaluate --synth samples --data data

# 4. use it in the full model
python -m talker_tune.export --omni Qwen/Qwen3-Omni-30B-A3B-Instruct --adapter run/step_800 --out Qwen3-Omni-myvoice
```

```python
model = Qwen3OmniMoeForConditionalGeneration.from_pretrained("Qwen3-Omni-myvoice", dtype=torch.bfloat16, device_map="cuda")
text_ids, audio = model.generate(**inputs, speaker="myvoice")
```

## Several voices

With one speaker per language, a LoRA ties the voice to the language of the prompt, and answers to spoken questions
fall back to the other language's voice. Train at least two speakers of each gender per language, about 400 steps per
speaker, with female voices starting from Chelsie and male voices from Ethan:

```bash
python -m talker_tune.train --data data --out run --steps 3200 \
    --init-speaker chelsie,ssb0603=ethan,ssb0590=ethan,p259=ethan,p286=ethan
```

With fewer speakers (one female and one male per language), give half of the samples a spoken user turn of the same
language. Do not do this with one speaker per language: it made text questions and read-aloud drift too.

```bash
python -m talker_tune.spoken --data data --out pool --n 200      # 200 turns per language, base model, chelsie/aiden
python -m talker_tune.train --data data --out run --steps 1600 --init-speaker chelsie,ssb0603=ethan,p259=ethan \
    --spoken-pool pool/pool.pt --spoken-frac 0.5
```

Similarity of the English voices on spoken questions in our runs (3 seeds; text questions 0.99 in each of these runs): one
speaker per language 0.80 (0.83 at four times the steps); one female and one male 0.91 (female 0.86), with spoken
turns 0.988; two female and two male at 200 steps each 0.91, at 400 steps each 0.983 (0.989 with spoken turns).

## Results

See the toolkit-runs table in the [repository README](../README.md#results) (new id vs preset slot, 2.5/5/10/23
minutes, LoRA scope), Section 5 and Appendices E–F of the report.

## Check your own labels

`label_check.py` (repository root) scores a set of labels under the Talker. Labels from the right encoder have the
lowest NLL, and the last frame shows at once whether the tail was padded.

## Citation

See the [repository README](../README.md#citation).
