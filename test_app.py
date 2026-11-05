"""Self-check without downloading the model: python test_app.py"""

import json
import sys
import tempfile
import types
from pathlib import Path

import numpy as np
import soundfile as sf

import voice_manager as vm

tmp = Path(tempfile.mkdtemp())
vm.PROFILES_DIR = tmp / "voice_profiles"
vm.MANIFEST_PATH = tmp / "dataset" / "lora_manifest.jsonl"

wav = tmp / "in.wav"
sf.write(wav, 0.3 * np.sin(np.linspace(0, 2000, 16000 * 6)), 16000)


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
assert names == ["Bli_Made_Denpasar", "Mbok_Ayu_Buleleng"] and len(rows[0]) == 5
vm.update_profile_metadata("Bli_Made_Denpasar", "Om Swastyastu, tiang Made, nggih.", "edited")
assert vm.get_profile("Bli_Made_Denpasar")["prompt_text"].endswith("nggih.")
lines = [json.loads(line) for line in vm.MANIFEST_PATH.read_text(encoding="utf-8").splitlines()]
assert {"audio", "text"} <= lines[0].keys() and lines[0]["text"].endswith("nggih.")

vm.delete_profile("Mbok_Ayu_Buleleng")
assert not (vm.PROFILES_DIR / "Mbok_Ayu_Buleleng").exists()
assert len(vm.MANIFEST_PATH.read_text(encoding="utf-8").splitlines()) == 1
assert raises(vm.delete_profile, "..")

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

# UI handlers
import app  # noqa: E402

app.build_app()
status, *_ = app.delete_selected("Bli_Made_Denpasar", False, None)
assert (vm.PROFILES_DIR / "Bli_Made_Denpasar").exists()  # unconfirmed delete is a no-op
audio_out, seed = app.generate("Bli_Made_Denpasar", app.MODE_ULTIMATE, "Suksma", "", 2.0, 10, 7, False, False)
assert seed == 7 and Path(audio_out).is_file()
status, confirm, tts_upd, mg_upd, table = app.delete_selected("Bli_Made_Denpasar", True, "Bli_Made_Denpasar")
assert tts_upd["value"] is None and tts_upd["choices"] == [] and len(table) == 0

print("All checks passed.")
