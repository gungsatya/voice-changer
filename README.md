# 🌺 Balinese Dialect Voice Cloning & TTS Studio (VoxCPM2)

A Gradio app that captures Balinese accents (Logat & Bahasa Bali) from native speakers and
re-synthesizes new text in that voice using **OpenBMB VoxCPM2** (48 kHz output).

## Project layout

```
app.py              Gradio UI (3 tabs)
voice_manager.py    Voice profile CRUD + LoRA manifest sync
tts_engine.py       Lazy-loaded VoxCPM2 wrapper (Ultimate / Controllable cloning)
voice_profiles/<name>/reference.wav, metadata.json   (created at runtime)
dataset/lora_manifest.jsonl                          (rebuilt on every add/edit/delete)
outputs/                                             (generated speech)
```

## Setup

Requires Python 3.10–3.12 (tested on 3.12) and an NVIDIA GPU with ~6–8 GB VRAM (CPU works, but slowly).
Measured on a GTX 1660 Ti 6 GB: ~5.1 GB VRAM after load, ~5.5 GB peak while generating, ~45–55 s per
4–5 s clip (GTX 16xx cards have no native bfloat16, so RTX cards are much faster). Close other GPU apps.

```bash
py -3.12 -m venv .venv
```

```bash
.venv\Scripts\activate
```

Install the CUDA build of PyTorch that matches your driver (example: CUDA 12.8):

```bash
pip install torch torchaudio --index-url https://download.pytorch.org/whl/cu128
```

```bash
pip install -r requirements.txt
```

```bash
python app.py
```

Open http://127.0.0.1:7860. The first **Generate** click downloads `openbmb/VoxCPM2` from Hugging Face
(several GB) and loads it; later generations reuse the loaded model.

## Workflow

1. **Enroll** — upload or record a 5–15 s clip, type the *exact* transcript, add dialect notes, save.
   Audio is peak-normalized and stored as `voice_profiles/<name>/reference.wav`.
2. **Synthesize** — pick a profile, type text, choose a mode:
   - **Ultimate Cloning** (recommended): the clip is used both as the continuation prompt
     (`prompt_wav_path` + `prompt_text`) and as the isolated reference (`reference_wav_path`),
     which preserves Balinese cengkok, rhythm and timbre best. Accuracy depends on the transcript.
   - **Controllable Cloning**: only `reference_wav_path` is used, and an optional style is prepended
     as `(style)text`, e.g. `(speaking warmly with a thick Balinese accent)`. More expressive,
     slightly less faithful to the original prosody.
3. **Manage** — browse all profiles, fix transcript typos, edit notes, or delete a profile
   (tick the confirmation box, then press the red button). Every change refreshes all dropdowns and
   rebuilds `dataset/lora_manifest.jsonl` immediately.

### Advanced settings
- `cfg_value` (default 2.0): higher follows the prompt/text more strictly; too high can sound strained.
- `inference_timesteps` (default 10): more steps = slightly better quality, slower.
- `seed`: fixed for reproducible output; tick *Randomize Seed* to explore variations (the used seed is
  written back to the field).
- *Denoise reference audio*: loads the ZipEnhancer denoiser (extra VRAM). Off by default; toggling it
  reloads the model on the next generation.

## Recording high-quality Balinese reference audio

- **Length:** 5–15 s of continuous natural speech. Shorter clips lose accent detail; longer ones waste VRAM.
- **One speaker, quiet room:** no music, gamelan, motorbikes, fans, or reverb-heavy rooms. Phone
  recorders are fine if held ~20 cm from the mouth.
- **Natural speech, not reading:** ask the speaker to talk as they would with family — this is where
  the logat (intonation, vowel colouring, speed) lives. Greetings and everyday chat work well:
  *"Om Swastyastu, tiang Made saking Singaraja. Kenken kabare, bli? Sampun ngajeng?"*
- **Exact transcript:** write every word as spoken, including particles and fillers
  (*nggih, bli, gek, kenken, tiang, puniki, ampun, nah, ae*), repeated words and false starts.
  A mismatched transcript is the #1 cause of garbled Ultimate Cloning output — fix it in Tab 3.
- **Consistent spelling:** pick one spelling per word (e.g. always *tiang*, not *tyang*) so a future
  LoRA fine-tune sees consistent text.
- **No clipping:** keep peaks below the red zone; the app normalizes volume for you.
- **Several profiles per region:** record different speakers (Denpasar, Buleleng, Karangasem,
  Tabanan…) and note the region in *Dialect Notes* — that metadata is carried into the LoRA manifest.

## LoRA manifest

`dataset/lora_manifest.jsonl` uses VoxCPM's training columns (`audio`, `text`) plus
`profile_name` and `dialect_notes`, one line per active profile. Point VoxCPM's LoRA training
scripts at it once you have enough recordings.
