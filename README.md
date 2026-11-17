# 🌺 Balinese Dialect Voice Cloning & TTS Studio (VoxCPM2)

A Gradio app that captures Balinese accents (Logat & Bahasa Bali) from native speakers and
re-synthesizes new text in that voice using **OpenBMB VoxCPM2** (48 kHz output).

## Project layout

```
app.py              Gradio UI (4 tabs)
voice_manager.py    Voice profile + emotion CRUD, Drive folder import, LoRA manifest sync
prosody.py          Prosody (intonation) profiling per clip + optional prosody matching
emotion_script.py   Parser for emotion-tagged text: "[marah] ... [ceria] ..."
tts_engine.py       Lazy-loaded VoxCPM2 wrapper (Ultimate / Controllable / Emotion Script)
import_emotions.py  CLI: profile every clip in emotion_samples/ (cached, skips unchanged clips)
colab/VoiceChanger_Colab.ipynb   Google Colab launcher, all data on Google Drive

<data>/voice_profiles/<name>/reference.wav, metadata.json        base ("normal") voice + prosody
<data>/voice_profiles/<name>/emotions/<emotion>/reference.wav, metadata.json
<data>/emotion_samples/<name>/<emotion>.wav + <emotion>.txt       optional bulk-import source
<data>/dataset/lora_manifest.jsonl                                 (rebuilt on every change)
<data>/outputs/                                                    (generated speech)
```

`<data>` is the repo folder by default, or whatever the `VOICE_DATA_DIR` environment variable points
to (on Colab: a Google Drive folder).

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
2. **Emotion Profiles** — add marah / ceria / sedih / … clips per speaker (see *Emotion profiles* below).
3. **Synthesize** — pick a profile, type text, choose a mode:
   - **Ultimate Cloning** (recommended): the clip is used both as the continuation prompt
     (`prompt_wav_path` + `prompt_text`) and as the isolated reference (`reference_wav_path`),
     which preserves Balinese cengkok, rhythm and timbre best. Accuracy depends on the transcript.
   - **Controllable Cloning**: only `reference_wav_path` is used, and an optional style is prepended
     as `(style)text`, e.g. `(speaking warmly with a thick Balinese accent)`. More expressive,
     slightly less faithful to the original prosody.
4. **Manage** — browse all profiles, fix transcript typos, edit notes, or delete a profile
   (tick the confirmation box, then press the red button). Every change refreshes all dropdowns and
   rebuilds `dataset/lora_manifest.jsonl` immediately.

### Advanced settings
- `cfg_value` (default 2.0): higher follows the prompt/text more strictly; too high can sound strained.
- `inference_timesteps` (default 10): more steps = slightly better quality, slower.
- `seed`: fixed for reproducible output; tick *Randomize Seed* to explore variations (the used seed is
  written back to the field).
- *Denoise reference audio*: loads the ZipEnhancer denoiser (extra VRAM). Off by default; toggling it
  reloads the model on the next generation.

## Emotion profiles (marah, ceria, sedih, …)

Each speaker can have one clip per emotion. VoxCPM copies the intonation, energy and tempo of the clip
it is prompted with, so synthesising a segment from the *marah* clip gives angry intonation, from the
*ceria* clip cheerful intonation, and so on.

1. **Tab 1** — enroll the speaker. This base clip is automatically the `normal` emotion.
2. **Tab 2 · Emotion Profiles** — for the same speaker, add a 5–15 s clip + exact transcript for
   each emotion (`marah`, `ceria`, `sedih`, `takut`, … or any new lowercase name). Every clip is
   profiled once — pitch median, pitch range (semitones), loudness, speaking rate (syllables/s),
   pause ratio — and the result is cached in its `metadata.json`, shown in the table.
   Or drop files into `emotion_samples/<Profile>/<emotion>.wav` + `<emotion>.txt` and click
   *Import* (or run `python import_emotions.py`). Unchanged clips are skipped (SHA-1 of audio + transcript).
3. **Tab 3 · Emotion Script mode** — tag the text; a tag holds until the next tag, so any single
   word can switch emotion:

   ```
   [normal] Om Swastyastu. [marah] Kenapa kamu terlambat lagi? [ceria] Tapi tiang seneng pisan!
   Tiang [ceria]seneng [normal]pisan.
   ```

   *Tag setiap kata* rewrites the text as `[normal]word [normal]word …` so you can change the emotion
   word by word; *Preview segmen* shows how it will be split. Adjacent words with the same emotion are
   merged into one segment, each segment is generated in Ultimate mode from its emotion clip (same seed),
   and the pieces are joined with a configurable pause and short fades.
   *Prosody matching* (off by default) additionally pitch-shifts / time-stretches / re-levels each
   segment towards its emotion's cached profile (clamped to ±4 semitones, 0.8–1.25× tempo, ±6 dB).

Tips: phrase-level tags sound more natural than single-word tags (VoxCPM needs some context to
produce good prosody). Record all emotions of one speaker with the same mic and room.

## Running on Google Colab (data on Google Drive)

Open `colab/VoiceChanger_Colab.ipynb` in Colab (GPU runtime) and run the cells in order. It mounts
Drive, sets `VOICE_DATA_DIR=/content/drive/MyDrive/voice-changer` (profiles, emotion clips, cached
prosody profiles, outputs, manifest) and, optionally, `HF_HOME` to the same folder so the VoxCPM2
weights (~5 GB of Drive space) are downloaded only once. A fresh runtime therefore only reinstalls
pip packages — nothing is re-enrolled or re-profiled. The app is started with `python app.py --share`
for a public `gradio.live` link. Set the notebook's `BRANCH` to the branch that contains this feature
until it is merged into `main`.

## Recording high-quality Balinese reference audio

- **Length:** 5–15 s of continuous natural speech. Shorter clips lose accent detail; longer ones waste VRAM.
- **One speaker, quiet room:** no music, gamelan, motorbikes, fans, or reverb-heavy rooms. Phone
  recorders are fine if held ~20 cm from the mouth.
- **Natural speech, not reading:** ask the speaker to talk as they would with family — this is where
  the logat (intonation, vowel colouring, speed) lives. Greetings and everyday chat work well:
  *"Om Swastyastu, tiang Made saking Singaraja. Kenken kabare, bli? Sampun ngajeng?"*
- **Exact transcript:** write every word as spoken, including particles and fillers
  (*nggih, bli, gek, kenken, tiang, puniki, ampun, nah, ae*), repeated words and false starts.
  A mismatched transcript is the #1 cause of garbled Ultimate Cloning output — fix it in Tab 4.
- **Consistent spelling:** pick one spelling per word (e.g. always *tiang*, not *tyang*) so a future
  LoRA fine-tune sees consistent text.
- **No clipping:** keep peaks below the red zone; the app normalizes volume for you.
- **Several profiles per region:** record different speakers (Denpasar, Buleleng, Karangasem,
  Tabanan…) and note the region in *Dialect Notes* — that metadata is carried into the LoRA manifest.

## LoRA manifest

`dataset/lora_manifest.jsonl` uses VoxCPM's training columns (`audio`, `text`) plus
`profile_name`, `emotion` and `dialect_notes`, one line per clip (base clip + every emotion clip). Point VoxCPM's LoRA training
scripts at it once you have enough recordings.
