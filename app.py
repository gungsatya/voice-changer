"""Balinese Dialect Voice Cloning & TTS Studio (VoxCPM2) — Gradio UI."""

import argparse
import os
import random

import gradio as gr
import pandas as pd

import emotion_script
import voice_manager as vm
from tts_engine import engine

MODE_ULTIMATE = "Ultimate Cloning (Best for Authentic Balinese Dialect)"
MODE_CONTROLLABLE = "Controllable Cloning (With Style Prompt)"
MODE_EMOTION = "Emotion Script (ganti emosi per kata / kalimat)"

EMOTION_HELP = (
    "Tulis tag emosi di depan kata/kalimat: `[marah] Kenapa telat lagi? [normal] Ya sudah, [ceria] ayo makan!` "
    "Tag berlaku sampai tag berikutnya, jadi satu kata pun bisa punya emosi sendiri: "
    "`Tiang [ceria]seneng [normal]pisan.` Teks sebelum tag pertama memakai *Default Emotion*. "
    "Tip: per frasa terdengar lebih natural daripada per kata tunggal."
)
SEGMENT_HEADERS = ["#", "Emotion", "Text"]

TRANSCRIPT_HELP = (
    "Write EXACTLY what is spoken, word for word — including Balinese particles and fillers "
    "(nggih, bli, gek, kenken, tiang, puniki, ampun, sampun...). Do not correct grammar or translate."
)


def _table(rows):
    return pd.DataFrame(rows, columns=vm.TABLE_HEADERS)


def _emotion_table(name):
    return pd.DataFrame(vm.list_emotions(name) if name else [], columns=vm.EMOTION_TABLE_HEADERS)


def sync_ui(tts_value=None, manage_value=None, emotion_value=None):
    """Refresh every profile dropdown and the table, keeping selections that still exist."""
    names, rows = vm.list_profiles()
    keep = lambda v: v if v in names else None  # noqa: E731
    return (
        gr.update(choices=names, value=keep(tts_value)),
        gr.update(choices=names, value=keep(manage_value)),
        _table(rows),
        gr.update(choices=names, value=keep(emotion_value)),
    )


# ---------- Tab 1: Enrollment ----------
def enroll(name, notes, audio_path, transcript, tts_value, manage_value, emotion_value):
    try:
        meta = vm.create_profile(name, audio_path, transcript, notes)
    except ValueError as e:
        gr.Warning(str(e))
        return (f"❌ {e}", *sync_ui(tts_value, manage_value, emotion_value))
    gr.Info(f"Voice profile '{meta['profile_name']}' registered.")
    status = (
        f"✅ Saved '{meta['profile_name']}' ({meta['duration_sec']}s).\n"
        f"Stored in voice_profiles/{meta['profile_name']}/ and synced to dataset/lora_manifest.jsonl."
    )
    if not 5.0 <= meta["duration_sec"] <= 15.0:
        status += "\n⚠️ Duration is outside the recommended 5–15s range; cloning quality may suffer."
    return (status, *sync_ui(meta["profile_name"], manage_value, meta["profile_name"]))


# ---------- Tab 2: Emotion profiles ----------
def show_emotions(name):
    if not name:
        return _emotion_table(None), "Select a profile."
    try:
        emos = vm.emotion_names(name)
        return _emotion_table(name), f"Emosi tersedia untuk {name}: " + ", ".join(f"[{e}]" for e in emos)
    except ValueError as e:
        return _emotion_table(None), str(e)


def save_emotion(name, emotion, audio_path, transcript, notes, tts_value, manage_value):
    if not name:
        gr.Warning("Select a profile first.")
        return "❌ No profile selected.", *show_emotions(name), *sync_ui(tts_value, manage_value, name)
    try:
        meta = vm.add_emotion(name, emotion, audio_path, transcript, notes)
    except ValueError as e:
        gr.Warning(str(e))
        return f"❌ {e}", *show_emotions(name), *sync_ui(tts_value, manage_value, name)
    p = meta["prosody"]
    gr.Info(f"Emotion '{meta['emotion']}' saved for {name}.")
    status = (f"✅ Saved emotion '{meta['emotion']}' for {name} ({meta['duration_sec']}s). Prosody profile: "
              f"pitch {p['f0_median_hz']} Hz, range {p['pitch_range_st']} st, loudness {p['energy_db']} dB, "
              f"rate {p['syllables_per_sec']} syl/s. Cached in metadata.json — no re-profiling needed.")
    return status, *show_emotions(name), *sync_ui(tts_value, manage_value, name)


def remove_emotion(name, emotion, confirmed, tts_value, manage_value):
    if not name or not emotion:
        return "❌ Select a profile and an emotion.", False, *show_emotions(name), *sync_ui(tts_value, manage_value, name)
    if not confirmed:
        return (f"⚠️ Tick the confirmation box to delete '{emotion}'.", False, *show_emotions(name),
                *sync_ui(tts_value, manage_value, name))
    try:
        vm.delete_emotion(name, emotion)
    except ValueError as e:
        gr.Warning(str(e))
        return f"❌ {e}", False, *show_emotions(name), *sync_ui(tts_value, manage_value, name)
    return (f"🗑️ Deleted emotion '{emotion}' from {name}.", False, *show_emotions(name),
            *sync_ui(tts_value, manage_value, name))


def import_folder(folder, name, tts_value, manage_value):
    try:
        report = "\n".join(vm.import_emotion_folder(folder or None))
    except ValueError as e:
        report = f"❌ {e}"
    return report, *show_emotions(name), *sync_ui(tts_value, manage_value, name)


# ---------- Tab 3: Synthesis ----------
def show_tts_profile(name):
    if not name:
        return None, "", gr.update(choices=[vm.BASE_EMOTION], value=vm.BASE_EMOTION), ""
    try:
        p = vm.get_profile(name)
        emos = vm.emotion_names(name)
    except ValueError as e:
        return None, str(e), gr.update(choices=[vm.BASE_EMOTION], value=vm.BASE_EMOTION), ""
    info = f"Transcript: {p['prompt_text']}\nDialect notes: {p.get('dialect_notes') or '-'}\nDuration: {p['duration_sec']}s"
    emo_info = "Emosi tersedia: " + " ".join(f"`[{e}]`" for e in emos)
    return p["wav_path"], info, gr.update(choices=emos, value=vm.BASE_EMOTION), emo_info


def _resolve_script(name, text, default_emotion):
    """Parse the tagged text and attach each segment's emotion clip. Raises ValueError on problems."""
    segments = emotion_script.parse(text, default_emotion or vm.BASE_EMOTION)
    if not segments:
        raise ValueError("Enter the text to synthesize.")
    available = vm.emotion_names(name)
    missing = emotion_script.unknown_emotions(segments, available)
    if missing:
        raise ValueError(f"Unknown emotion(s) for '{name}': {', '.join(missing)}. "
                         f"Available: {', '.join(available)}. Add them in the Emotion Profiles tab.")
    resolved = []
    for emotion, chunk in segments:
        e = vm.get_emotion(name, emotion)
        resolved.append({"emotion": emotion, "text": chunk, "wav_path": e["wav_path"],
                         "prompt_text": e["prompt_text"], "prosody": e.get("prosody")})
    return resolved


def preview_segments(name, text, default_emotion):
    try:
        if not name:
            raise ValueError("Select a registered voice profile first.")
        segs = _resolve_script(name, text, default_emotion)
    except ValueError as e:
        gr.Warning(str(e))
        return pd.DataFrame([], columns=SEGMENT_HEADERS)
    return pd.DataFrame([[i + 1, s["emotion"], s["text"]] for i, s in enumerate(segs)], columns=SEGMENT_HEADERS)


def tag_words(text, default_emotion):
    return emotion_script.tag_each_word(text, default_emotion or vm.BASE_EMOTION)


def generate(name, mode, text, style, cfg, steps, seed, randomize, use_denoiser,
             default_emotion=vm.BASE_EMOTION, gap_ms=120, match_prosody=False):
    if not name:
        raise gr.Error("Select a registered voice profile first.")
    if not (text or "").strip():
        raise gr.Error("Enter the text to synthesize.")
    seed = random.randint(0, 2**31 - 1) if randomize else int(seed or 0)
    if mode == MODE_EMOTION:
        try:
            segs = _resolve_script(name, text, default_emotion)
            out_path, _, report = engine.synthesize_script(
                segs, cfg_value=cfg, timesteps=steps, seed=seed, use_denoiser=use_denoiser,
                profile_name=name, gap_ms=int(gap_ms), match_prosody=bool(match_prosody),
            )
        except (ValueError, RuntimeError) as e:
            raise gr.Error(str(e)) from e
        gr.Info(f"{len(report)} segment(s) saved to {out_path}")
        return out_path, seed
    try:
        p = vm.get_profile(name)  # read fresh so transcript edits from Tab 4 apply immediately
        out_path, _ = engine.synthesize(
            text=text,
            wav_path=p["wav_path"],
            prompt_text=p["prompt_text"],
            mode="ultimate" if mode == MODE_ULTIMATE else "controllable",
            style_prompt=style if mode == MODE_CONTROLLABLE else "",
            cfg_value=cfg,
            timesteps=steps,
            seed=seed,
            use_denoiser=use_denoiser,
            profile_name=name,
        )
    except (ValueError, RuntimeError) as e:
        raise gr.Error(str(e)) from e
    gr.Info(f"Saved to {out_path}")
    return out_path, seed


# ---------- Tab 4: Management ----------
def inspect_profile(name):
    if not name:
        return None, "", ""
    try:
        p = vm.get_profile(name)
    except ValueError as e:
        gr.Warning(str(e))
        return None, "", ""
    return p["wav_path"], p["prompt_text"], p.get("dialect_notes", "")


def update_metadata(name, transcript, notes, tts_value, emotion_value=None):
    if not name:
        gr.Warning("Select a profile to update.")
        return "❌ No profile selected.", *sync_ui(tts_value, name, emotion_value), *show_tts_profile(tts_value)
    try:
        vm.update_profile_metadata(name, transcript, notes)
    except ValueError as e:
        gr.Warning(str(e))
        return f"❌ {e}", *sync_ui(tts_value, name, emotion_value), *show_tts_profile(tts_value)
    gr.Info(f"Profile '{name}' updated.")
    return (f"✅ Updated transcript/notes for '{name}'. LoRA manifest re-synced.",
            *sync_ui(tts_value, name, emotion_value), *show_tts_profile(tts_value))


def delete_selected(name, confirmed, tts_value, emotion_value=None):
    if not name:
        gr.Warning("Select a profile to delete.")
        return "❌ No profile selected.", False, *sync_ui(tts_value, name, emotion_value)
    if not confirmed:
        gr.Warning("Tick the confirmation box first.")
        return (f"⚠️ Tick 'Yes, permanently delete' to remove '{name}'.", False,
                *sync_ui(tts_value, name, emotion_value))
    try:
        vm.delete_profile(name)
    except (ValueError, OSError) as e:
        gr.Warning(str(e))
        return f"❌ {e}", False, *sync_ui(tts_value, name, emotion_value)
    gr.Info(f"Profile '{name}' deleted.")
    return (f"🗑️ Deleted '{name}' and rebuilt dataset/lora_manifest.jsonl.", False,
            *sync_ui(tts_value, None, emotion_value))


def build_app() -> gr.Blocks:
    names, rows = vm.list_profiles()

    with gr.Blocks(title="Balinese Voice Cloning Studio") as demo:
        gr.Markdown(
            "# 🌺 Balinese Dialect Voice Cloning & TTS Studio\n"
            "Clone native Balinese speakers with **VoxCPM2** and synthesize Bahasa Bali, "
            "Balinese-accented Indonesian, or code-switched speech."
        )

        with gr.Tabs():
            with gr.Tab("1. Enroll Balinese Voice (Training)"):
                with gr.Row():
                    with gr.Column():
                        en_name = gr.Textbox(label="Profile Name", placeholder="Bli_Made_Denpasar",
                                             info="Letters, digits, '_' or '-' only.")
                        en_notes = gr.Textbox(label="Dialect / Region Notes",
                                              placeholder="Logat Buleleng kental, nada cepat")
                        en_audio = gr.Audio(label="Reference Audio (5–15 s, clean, single speaker)",
                                            sources=["upload", "microphone"], type="filepath")
                    with gr.Column():
                        en_text = gr.Textbox(label="Exact Transcript (prompt_text)", lines=5,
                                             placeholder="Om Swastyastu, tiang Made. Kenken kabare, bli?",
                                             info=TRANSCRIPT_HELP)
                        en_btn = gr.Button("Save & Register Voice Profile", variant="primary")
                        en_status = gr.Textbox(label="Status", interactive=False, lines=3)

            with gr.Tab("2. Emotion Profiles (marah / ceria / sedih ...)"):
                gr.Markdown(
                    "Tambahkan contoh suara per emosi untuk speaker yang sudah di-enroll. Setiap clip dianalisis "
                    "sekali (pitch, rentang intonasi, loudness, kecepatan bicara) dan hasilnya disimpan di "
                    f"`{vm.PROFILES_DIR}` — kalau folder ini di Google Drive, Colab tidak perlu profiling ulang. "
                    "Clip dasar dari tab 1 otomatis menjadi emosi `normal`."
                )
                with gr.Row(equal_height=True):
                    em_dd = gr.Dropdown(choices=names, value=None, label="Voice Profile", scale=4)
                    em_refresh = gr.Button("Refresh", size="sm", scale=1)
                with gr.Row():
                    with gr.Column():
                        em_emotion = gr.Dropdown(choices=vm.SUGGESTED_EMOTIONS, value="marah", allow_custom_value=True,
                                                 label="Emotion", info="Pilih atau ketik nama emosi baru (huruf kecil).")
                        em_audio = gr.Audio(label="Emotion Clip (5–15 s, same speaker)",
                                            sources=["upload", "microphone"], type="filepath")
                    with gr.Column():
                        em_text = gr.Textbox(label="Exact Transcript", lines=4, info=TRANSCRIPT_HELP)
                        em_notes = gr.Textbox(label="Notes (optional)", placeholder="Marah, nada tinggi & cepat")
                        em_save = gr.Button("Save / Replace Emotion Sample", variant="primary")
                em_status = gr.Textbox(label="Status", interactive=False, lines=3)
                em_info = gr.Markdown()
                em_table = gr.Dataframe(value=_emotion_table(None), headers=vm.EMOTION_TABLE_HEADERS,
                                        interactive=False, wrap=True, label="Emotion prosody profiles (cached)")
                with gr.Row():
                    em_confirm = gr.Checkbox(value=False, label="Yes, delete the selected emotion")
                    em_delete = gr.Button("Delete Emotion", variant="stop")
                with gr.Accordion("Bulk import from a folder (Google Drive)", open=False):
                    gr.Markdown(
                        "Struktur: `<folder>/<Profile>/<emotion>.wav` + `<emotion>.txt` (transkrip). "
                        "Contoh: `Bli_Made/normal.wav`, `Bli_Made/marah.wav`, `Bli_Made/ceria.wav`. "
                        "Profile baru dibuat otomatis dari `normal.*`; clip yang tidak berubah dilewati (cached)."
                    )
                    em_folder = gr.Textbox(label="Folder", value=str(vm.EMOTION_SAMPLES_DIR))
                    em_import = gr.Button("Import & Profile New/Changed Clips")
                    em_report = gr.Textbox(label="Import report", interactive=False, lines=6)

            with gr.Tab("3. Balinese TTS Synthesizer"):
                with gr.Row(equal_height=True):
                    tts_dd = gr.Dropdown(choices=names, value=None, label="Select Registered Voice Profile", scale=4)
                    tts_refresh = gr.Button("Refresh Profiles", variant="secondary", size="sm", scale=1)
                with gr.Accordion("Active Voice Profile Reference", open=False):
                    tts_ref_audio = gr.Audio(label="reference.wav", interactive=False, type="filepath")
                    tts_ref_info = gr.Textbox(label="Transcript & Dialect Notes", interactive=False, lines=3)
                tts_mode = gr.Radio([MODE_ULTIMATE, MODE_CONTROLLABLE, MODE_EMOTION], value=MODE_ULTIMATE,
                                    label="Cloning Mode")
                tts_text = gr.Textbox(label="Target Text to Synthesize", lines=4,
                                      placeholder="Rahajeng semeng, nggih. Tiang jagi ngaturang pengumuman puniki...")
                tts_style = gr.Textbox(label="Style Control (optional)", visible=False,
                                       placeholder="speaking warmly with a thick Balinese accent")
                with gr.Group(visible=False) as emo_group:
                    gr.Markdown(EMOTION_HELP)
                    emo_avail = gr.Markdown()
                    with gr.Row():
                        emo_default = gr.Dropdown(choices=[vm.BASE_EMOTION], value=vm.BASE_EMOTION,
                                                  label="Default Emotion (untuk teks tanpa tag)")
                        emo_gap = gr.Slider(0, 600, value=120, step=10, label="Jeda antar segmen (ms)")
                    emo_match = gr.Checkbox(
                        value=False,
                        label="Prosody matching: geser pitch/tempo/volume tiap segmen ke profil emosinya "
                              "(koreksi tambahan; bisa menambah artefak)")
                    with gr.Row():
                        emo_tag_btn = gr.Button("Tag setiap kata", size="sm")
                        emo_preview_btn = gr.Button("Preview segmen", size="sm")
                    emo_segments = gr.Dataframe(value=pd.DataFrame([], columns=SEGMENT_HEADERS),
                                                headers=SEGMENT_HEADERS, interactive=False, wrap=True,
                                                label="Segmen yang akan disintesis")
                with gr.Accordion("Advanced Generation Settings", open=False):
                    tts_cfg = gr.Slider(1.0, 5.0, value=2.0, step=0.1, label="cfg_value")
                    tts_steps = gr.Slider(5, 30, value=10, step=1, label="inference_timesteps")
                    with gr.Row():
                        tts_seed = gr.Number(value=42, precision=0, label="Seed")
                        tts_rand = gr.Checkbox(value=False, label="Randomize Seed")
                    tts_denoise = gr.Checkbox(value=False, label="Denoise reference audio (loads ZipEnhancer; extra VRAM, reloads model)")
                tts_btn = gr.Button("Generate Balinese Speech", variant="primary")
                tts_out = gr.Audio(label="Generated Speech (48 kHz)", type="filepath", autoplay=True, interactive=False)

            with gr.Tab("4. Manage Voice Profiles (Edit & Delete)"):
                mg_table = gr.Dataframe(value=_table(rows), headers=vm.TABLE_HEADERS, interactive=False, wrap=True)
                with gr.Row(equal_height=True):
                    mg_dd = gr.Dropdown(choices=names, value=None, label="Select Profile to Manage", scale=4)
                    mg_refresh = gr.Button("Refresh List", size="sm", scale=1)
                mg_audio = gr.Audio(label="Reference Audio Preview", interactive=False, type="filepath")
                mg_text = gr.Textbox(label="Exact Transcript (prompt_text)", lines=3, info=TRANSCRIPT_HELP)
                mg_notes = gr.Textbox(label="Dialect Notes")
                with gr.Row():
                    mg_update = gr.Button("Update Profile Metadata", variant="secondary")
                    mg_confirm = gr.Checkbox(value=False, label="Yes, permanently delete this profile")
                    mg_delete = gr.Button("Delete Selected Profile", variant="stop")
                mg_status = gr.Textbox(label="Status", interactive=False, lines=2)

        synced = [tts_dd, mg_dd, mg_table, em_dd]
        synced_in = [tts_dd, mg_dd, em_dd]
        tts_profile_out = [tts_ref_audio, tts_ref_info, emo_default, emo_avail]

        # Tab 1
        en_btn.click(enroll, [en_name, en_notes, en_audio, en_text, *synced_in], [en_status, *synced])

        # Tab 2
        em_view = [em_table, em_info]
        em_dd.change(show_emotions, em_dd, em_view)
        em_refresh.click(sync_ui, synced_in, synced).then(show_emotions, em_dd, em_view)
        em_save.click(save_emotion, [em_dd, em_emotion, em_audio, em_text, em_notes, tts_dd, mg_dd],
                      [em_status, *em_view, *synced]).then(show_tts_profile, tts_dd, tts_profile_out)
        em_delete.click(remove_emotion, [em_dd, em_emotion, em_confirm, tts_dd, mg_dd],
                        [em_status, em_confirm, *em_view, *synced]).then(show_tts_profile, tts_dd, tts_profile_out)
        em_import.click(import_folder, [em_folder, em_dd, tts_dd, mg_dd],
                        [em_report, *em_view, *synced]).then(show_tts_profile, tts_dd, tts_profile_out)

        # Tab 3
        tts_dd.change(show_tts_profile, tts_dd, tts_profile_out)
        tts_refresh.click(sync_ui, synced_in, synced)
        tts_mode.change(lambda m: (gr.update(visible=m == MODE_CONTROLLABLE), gr.update(visible=m == MODE_EMOTION)),
                        tts_mode, [tts_style, emo_group])
        emo_tag_btn.click(tag_words, [tts_text, emo_default], tts_text)
        emo_preview_btn.click(preview_segments, [tts_dd, tts_text, emo_default], emo_segments)
        tts_btn.click(
            generate,
            [tts_dd, tts_mode, tts_text, tts_style, tts_cfg, tts_steps, tts_seed, tts_rand, tts_denoise,
             emo_default, emo_gap, emo_match],
            [tts_out, tts_seed],
        )

        # Tab 4
        mg_dd.change(inspect_profile, mg_dd, [mg_audio, mg_text, mg_notes])
        mg_refresh.click(sync_ui, synced_in, synced)
        mg_update.click(update_metadata, [mg_dd, mg_text, mg_notes, tts_dd, em_dd],
                        [mg_status, *synced, *tts_profile_out])
        mg_delete.click(delete_selected, [mg_dd, mg_confirm, tts_dd, em_dd], [mg_status, mg_confirm, *synced])

        demo.load(sync_ui, synced_in, synced)  # pick up profiles added/removed on disk between page loads

    return demo


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--share", action="store_true", default=os.environ.get("GRADIO_SHARE") == "1",
                        help="Create a public gradio.live link (needed on Google Colab).")
    parser.add_argument("--port", type=int, default=int(os.environ.get("GRADIO_SERVER_PORT", 7860)))
    args = parser.parse_args()
    print(f"Data folder: {vm.DATA_ROOT}")
    vm.sync_lora_manifest()
    build_app().queue(default_concurrency_limit=1).launch(
        theme=gr.themes.Soft(),
        share=args.share,
        server_port=args.port,
        allowed_paths=[str(vm.DATA_ROOT)],  # profiles/outputs may live outside the repo (Google Drive)
    )
