"""Balinese Dialect Voice Cloning & TTS Studio (VoxCPM2) — Gradio UI."""

import random

import gradio as gr
import pandas as pd

import voice_manager as vm
from tts_engine import engine

MODE_ULTIMATE = "Ultimate Cloning (Best for Authentic Balinese Dialect)"
MODE_CONTROLLABLE = "Controllable Cloning (With Style Prompt)"

TRANSCRIPT_HELP = (
    "Write EXACTLY what is spoken, word for word — including Balinese particles and fillers "
    "(nggih, bli, gek, kenken, tiang, puniki, ampun, sampun...). Do not correct grammar or translate."
)


def _table(rows):
    return pd.DataFrame(rows, columns=vm.TABLE_HEADERS)


def sync_ui(tts_value=None, manage_value=None):
    """Refresh both profile dropdowns and the table, keeping selections that still exist."""
    names, rows = vm.list_profiles()
    keep = lambda v: v if v in names else None  # noqa: E731
    return (
        gr.update(choices=names, value=keep(tts_value)),
        gr.update(choices=names, value=keep(manage_value)),
        _table(rows),
    )


# ---------- Tab 1: Enrollment ----------
def enroll(name, notes, audio_path, transcript, tts_value, manage_value):
    try:
        meta = vm.create_profile(name, audio_path, transcript, notes)
    except ValueError as e:
        gr.Warning(str(e))
        return (f"❌ {e}", *sync_ui(tts_value, manage_value))
    gr.Info(f"Voice profile '{meta['profile_name']}' registered.")
    status = (
        f"✅ Saved '{meta['profile_name']}' ({meta['duration_sec']}s).\n"
        f"Stored in voice_profiles/{meta['profile_name']}/ and synced to dataset/lora_manifest.jsonl."
    )
    if not 5.0 <= meta["duration_sec"] <= 15.0:
        status += "\n⚠️ Duration is outside the recommended 5–15s range; cloning quality may suffer."
    return (status, *sync_ui(meta["profile_name"], manage_value))


# ---------- Tab 2: Synthesis ----------
def show_tts_profile(name):
    if not name:
        return None, ""
    try:
        p = vm.get_profile(name)
    except ValueError as e:
        return None, str(e)
    info = f"Transcript: {p['prompt_text']}\nDialect notes: {p.get('dialect_notes') or '-'}\nDuration: {p['duration_sec']}s"
    return p["wav_path"], info


def generate(name, mode, text, style, cfg, steps, seed, randomize, use_denoiser):
    if not name:
        raise gr.Error("Select a registered voice profile first.")
    if not (text or "").strip():
        raise gr.Error("Enter the text to synthesize.")
    seed = random.randint(0, 2**31 - 1) if randomize else int(seed or 0)
    try:
        p = vm.get_profile(name)  # read fresh so transcript edits from Tab 3 apply immediately
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


# ---------- Tab 3: Management ----------
def inspect_profile(name):
    if not name:
        return None, "", ""
    try:
        p = vm.get_profile(name)
    except ValueError as e:
        gr.Warning(str(e))
        return None, "", ""
    return p["wav_path"], p["prompt_text"], p.get("dialect_notes", "")


def update_metadata(name, transcript, notes, tts_value):
    if not name:
        gr.Warning("Select a profile to update.")
        return "❌ No profile selected.", *sync_ui(tts_value, name), *show_tts_profile(tts_value)
    try:
        vm.update_profile_metadata(name, transcript, notes)
    except ValueError as e:
        gr.Warning(str(e))
        return f"❌ {e}", *sync_ui(tts_value, name), *show_tts_profile(tts_value)
    gr.Info(f"Profile '{name}' updated.")
    return (f"✅ Updated transcript/notes for '{name}'. LoRA manifest re-synced.",
            *sync_ui(tts_value, name), *show_tts_profile(tts_value))


def delete_selected(name, confirmed, tts_value):
    if not name:
        gr.Warning("Select a profile to delete.")
        return "❌ No profile selected.", False, *sync_ui(tts_value, name)
    if not confirmed:
        gr.Warning("Tick the confirmation box first.")
        return f"⚠️ Tick 'Yes, permanently delete' to remove '{name}'.", False, *sync_ui(tts_value, name)
    try:
        vm.delete_profile(name)
    except (ValueError, OSError) as e:
        gr.Warning(str(e))
        return f"❌ {e}", False, *sync_ui(tts_value, name)
    gr.Info(f"Profile '{name}' deleted.")
    return (f"🗑️ Deleted '{name}' and rebuilt dataset/lora_manifest.jsonl.", False, *sync_ui(tts_value, None))


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

            with gr.Tab("2. Balinese TTS Synthesizer"):
                with gr.Row(equal_height=True):
                    tts_dd = gr.Dropdown(choices=names, value=None, label="Select Registered Voice Profile", scale=4)
                    tts_refresh = gr.Button("Refresh Profiles", variant="secondary", size="sm", scale=1)
                with gr.Accordion("Active Voice Profile Reference", open=False):
                    tts_ref_audio = gr.Audio(label="reference.wav", interactive=False, type="filepath")
                    tts_ref_info = gr.Textbox(label="Transcript & Dialect Notes", interactive=False, lines=3)
                tts_mode = gr.Radio([MODE_ULTIMATE, MODE_CONTROLLABLE], value=MODE_ULTIMATE, label="Cloning Mode")
                tts_text = gr.Textbox(label="Target Text to Synthesize", lines=4,
                                      placeholder="Rahajeng semeng, nggih. Tiang jagi ngaturang pengumuman puniki...")
                tts_style = gr.Textbox(label="Style Control (optional)", visible=False,
                                       placeholder="speaking warmly with a thick Balinese accent")
                with gr.Accordion("Advanced Generation Settings", open=False):
                    tts_cfg = gr.Slider(1.0, 5.0, value=2.0, step=0.1, label="cfg_value")
                    tts_steps = gr.Slider(5, 30, value=10, step=1, label="inference_timesteps")
                    with gr.Row():
                        tts_seed = gr.Number(value=42, precision=0, label="Seed")
                        tts_rand = gr.Checkbox(value=False, label="Randomize Seed")
                    tts_denoise = gr.Checkbox(value=False, label="Denoise reference audio (loads ZipEnhancer; extra VRAM, reloads model)")
                tts_btn = gr.Button("Generate Balinese Speech", variant="primary")
                tts_out = gr.Audio(label="Generated Speech (48 kHz)", type="filepath", autoplay=True, interactive=False)

            with gr.Tab("3. Manage Voice Profiles (Edit & Delete)"):
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

        synced = [tts_dd, mg_dd, mg_table]

        # Tab 1
        en_btn.click(enroll, [en_name, en_notes, en_audio, en_text, tts_dd, mg_dd], [en_status, *synced])

        # Tab 2
        tts_dd.change(show_tts_profile, tts_dd, [tts_ref_audio, tts_ref_info])
        tts_refresh.click(sync_ui, [tts_dd, mg_dd], synced)
        tts_mode.change(lambda m: gr.update(visible=m == MODE_CONTROLLABLE), tts_mode, tts_style)
        tts_btn.click(
            generate,
            [tts_dd, tts_mode, tts_text, tts_style, tts_cfg, tts_steps, tts_seed, tts_rand, tts_denoise],
            [tts_out, tts_seed],
        )

        # Tab 3
        mg_dd.change(inspect_profile, mg_dd, [mg_audio, mg_text, mg_notes])
        mg_refresh.click(sync_ui, [tts_dd, mg_dd], synced)
        mg_update.click(update_metadata, [mg_dd, mg_text, mg_notes, tts_dd],
                        [mg_status, *synced, tts_ref_audio, tts_ref_info])
        mg_delete.click(delete_selected, [mg_dd, mg_confirm, tts_dd], [mg_status, mg_confirm, *synced])

        demo.load(sync_ui, [tts_dd, mg_dd], synced)  # pick up profiles added/removed on disk between page loads

    return demo


if __name__ == "__main__":
    vm.sync_lora_manifest()
    build_app().queue(default_concurrency_limit=1).launch(theme=gr.themes.Soft())
