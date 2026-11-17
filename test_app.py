"""Self-check without downloading the model: python test_app.py"""

import json
import sys
import tempfile
import types
from pathlib import Path

import numpy as np
import soundfile as sf

import emotion_script
import voice_manager as vm

tmp = Path(tempfile.mkdtemp())
vm.PROFILES_DIR = tmp / "voice_profiles"
vm.MANIFEST_PATH = tmp / "dataset" / "lora_manifest.jsonl"


def tone(path, hz, sec=6.0, sr=16000, amp=0.3):
    """Voiced-ish test clip: a harmonic tone with syllable-like amplitude bursts."""
    t = np.arange(int(sr * sec)) / sr
    env = 0.5 + 0.5 * np.sin(2 * np.pi * 4 * t) ** 2
    y = amp * env * sum(np.sin(2 * np.pi * hz * k * t) / k for k in (1, 2, 3))
    sf.write(path, y, sr)
    return path


wav = tone(tmp / "in.wav", 140)


def raises(fn, *a):
    try:
        fn(*a)
    except ValueError:
        return True
    return False


# CRUD
meta = vm.create_profile("Bli_Made_Denpasar", str(wav), "Om Swastyastu, tiang Made.", "Logat Denpasar")
assert meta["duration_sec"] == 6.0
assert raises(vm.create_profile, "Bli_Made_Denpasar", str(wav), "x", "")  # duplicate
assert raises(vm.create_profile, "../evil", str(wav), "x", "")  # path traversal
assert raises(vm.create_profile, "Ok_Name", str(wav), "  ", "")  # empty transcript
short = tmp / "short.wav"
sf.write(short, np.ones(1600) * 0.1, 16000)
assert raises(vm.create_profile, "Too_Short", str(short), "x", "")
assert not (vm.PROFILES_DIR / "Too_Short").exists()
vm.create_profile("Mbok_Ayu_Buleleng", str(wav), "Kenken kabare, gek?", "Buleleng")

names, rows = vm.list_profiles()
assert names == ["Bli_Made_Denpasar", "Mbok_Ayu_Buleleng"] and len(rows[0]) == len(vm.TABLE_HEADERS)
assert abs(meta["prosody"]["f0_median_hz"] - 140) < 10  # prosody profiled once at enrollment
vm.update_profile_metadata("Bli_Made_Denpasar", "Om Swastyastu, tiang Made, nggih.", "edited")
assert vm.get_profile("Bli_Made_Denpasar")["prompt_text"].endswith("nggih.")
lines = [json.loads(line) for line in vm.MANIFEST_PATH.read_text(encoding="utf-8").splitlines()]
assert {"audio", "text"} <= lines[0].keys() and lines[0]["text"].endswith("nggih.")

vm.delete_profile("Mbok_Ayu_Buleleng")
assert not (vm.PROFILES_DIR / "Mbok_Ayu_Buleleng").exists()
assert len(vm.MANIFEST_PATH.read_text(encoding="utf-8").splitlines()) == 1
assert raises(vm.delete_profile, "..")

# Emotion profiles
angry = tone(tmp / "marah.wav", 220, amp=0.6)
em = vm.add_emotion("Bli_Made_Denpasar", "Marah", str(angry), "Kenapa kamu terlambat lagi!", "nada tinggi")
assert em["emotion"] == "marah" and abs(em["prosody"]["f0_median_hz"] - 220) < 15
assert em["prosody"]["syllables_per_sec"] > 0
assert vm.emotion_names("Bli_Made_Denpasar") == ["normal", "marah"]
assert vm.get_emotion("Bli_Made_Denpasar", "normal").get("is_base")
assert vm.get_emotion("Bli_Made_Denpasar", "marah")["wav_path"].endswith("emotions/marah/reference.wav")
assert raises(vm.add_emotion, "Bli_Made_Denpasar", "../x", str(angry), "x")  # bad emotion name
assert raises(vm.add_emotion, "Nobody", "marah", str(angry), "x")  # unknown profile
assert raises(vm.add_emotion, "Bli_Made_Denpasar", "marah", str(short), "x")  # invalid clip keeps old one
assert vm.get_emotion("Bli_Made_Denpasar", "marah")["prompt_text"].startswith("Kenapa")
assert raises(vm.get_emotion, "Bli_Made_Denpasar", "sedih")
assert len(vm.list_emotions("Bli_Made_Denpasar")[0]) == len(vm.EMOTION_TABLE_HEADERS)
assert {json.loads(x)["emotion"] for x in vm.MANIFEST_PATH.read_text(encoding="utf-8").splitlines()} == {"normal", "marah"}

# Script parsing
segs = emotion_script.parse("Halo bli. [marah] Kenapa telat? [MARAH]Lagi! [ceria]ayo [ceria] makan", "normal")
assert segs == [("normal", "Halo bli."), ("marah", "Kenapa telat? Lagi!"), ("ceria", "ayo makan")]
assert emotion_script.parse("[sedih]", "normal") == []
assert emotion_script.tag_each_word("Tiang [ceria]seneng pisan") == "[normal]Tiang [ceria]seneng [ceria]pisan"
assert emotion_script.unknown_emotions(segs, ["normal", "marah"]) == ["ceria"]

# Bulk import from a Drive-like folder: profiled once, cached afterwards
src = tmp / "emotion_samples" / "Gek_Ayu"
src.mkdir(parents=True)
tone(src / "normal.wav", 200)
(src / "normal.txt").write_text("Om Swastyastu, tiang Gek Ayu.", encoding="utf-8")
tone(src / "ceria.wav", 260)
(src / "ceria.txt").write_text("Aduh seneng pisan tiang!", encoding="utf-8")
tone(src / "sedih.wav", 180)  # no transcript -> reported, not imported
report = vm.import_emotion_folder(tmp / "emotion_samples")
assert sum(r.startswith("✅") for r in report) == 2 and sum(r.startswith("❌") for r in report) == 1, report
assert vm.emotion_names("Gek_Ayu") == ["normal", "ceria"]
report = vm.import_emotion_folder(tmp / "emotion_samples")
assert sum(r.startswith("⏭️") for r in report) == 2, report  # nothing re-profiled
(src / "ceria.txt").write_text("Aduh seneng pisan tiang, nggih!", encoding="utf-8")
report = vm.import_emotion_folder(tmp / "emotion_samples")
assert any("updated emotion 'ceria'" in r for r in report), report
vm.delete_emotion("Gek_Ayu", "ceria")
assert vm.emotion_names("Gek_Ayu") == ["normal"] and raises(vm.delete_emotion, "Gek_Ayu", "normal")
vm.delete_profile("Gek_Ayu")

# Engine with a fake VoxCPM: checks the kwargs each mode passes
calls = []


class FakeModel:
    tts_model = types.SimpleNamespace(sample_rate=48000)

    def generate(self, **kw):
        calls.append(kw)
        return np.zeros(4800, dtype=np.float32)


sys.modules["voxcpm"] = types.SimpleNamespace(VoxCPM=types.SimpleNamespace(from_pretrained=lambda *a, **k: FakeModel()))
import tts_engine  # noqa: E402

tts_engine.OUTPUT_DIR = tmp / "outputs"
ref = vm.get_profile("Bli_Made_Denpasar")["wav_path"]
out, sr = tts_engine.engine.synthesize("Rahajeng semeng", ref, "Om", mode="ultimate")
assert sr == 48000 and Path(out).is_file()
assert calls[-1]["prompt_wav_path"] == ref and calls[-1]["reference_wav_path"] == ref and "seed" not in calls[-1]
tts_engine.engine.synthesize("Rahajeng", ref, "Om", mode="controllable", style_prompt="warm")
assert calls[-1]["text"] == "(warm)Rahajeng" and "prompt_wav_path" not in calls[-1]

calls.clear()
marah = vm.get_emotion("Bli_Made_Denpasar", "marah")
segments = [
    {"emotion": "normal", "text": "Halo", "wav_path": ref, "prompt_text": "Om", "prosody": None},
    {"emotion": "marah", "text": "Kenapa telat", "wav_path": marah["wav_path"], "prompt_text": marah["prompt_text"],
     "prosody": marah["prosody"]},
]
out, sr, report = tts_engine.engine.synthesize_script(segments, gap_ms=100)
assert [c["prompt_wav_path"] for c in calls] == [ref, marah["wav_path"]]
assert calls[1]["prompt_text"] == marah["prompt_text"] and calls[1]["text"] == "Kenapa telat"
assert abs(sf.info(out).duration - (0.1 + 0.1 + 0.1)) < 0.01 and [r["emotion"] for r in report] == ["normal", "marah"]

# UI handlers
import app  # noqa: E402

app.build_app()
status, *_ = app.delete_selected("Bli_Made_Denpasar", False, None)
assert (vm.PROFILES_DIR / "Bli_Made_Denpasar").exists()  # unconfirmed delete is a no-op
audio_out, seed = app.generate("Bli_Made_Denpasar", app.MODE_ULTIMATE, "Suksma", "", 2.0, 10, 7, False, False)
assert seed == 7 and Path(audio_out).is_file()
calls.clear()
audio_out, _ = app.generate("Bli_Made_Denpasar", app.MODE_EMOTION, "Suksma [marah]bli!", "", 2.0, 10, 7, False,
                            False, "normal", 0, True)
assert Path(audio_out).is_file() and len(calls) == 2 and calls[1]["prompt_wav_path"] == marah["wav_path"]
assert len(app.preview_segments("Bli_Made_Denpasar", "a [marah] b", "normal")) == 2
try:
    app.generate("Bli_Made_Denpasar", app.MODE_EMOTION, "[sedih]x", "", 2.0, 10, 7, False, False)
    raise AssertionError("unknown emotion must fail")
except app.gr.Error:
    pass
status, *_ = app.save_emotion("Bli_Made_Denpasar", "ceria", str(tone(tmp / "c.wav", 250)), "Seneng pisan!", "", None, None)
assert status.startswith("✅") and "ceria" in vm.emotion_names("Bli_Made_Denpasar")
status, confirm, tts_upd, mg_upd, table, em_upd = app.delete_selected("Bli_Made_Denpasar", True, "Bli_Made_Denpasar")
assert tts_upd["value"] is None and tts_upd["choices"] == [] and len(table) == 0 and em_upd["choices"] == []

print("All checks passed.")
