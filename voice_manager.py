"""CRUD for Balinese voice profiles stored under ./voice_profiles/<name>/."""

import json
import re
import shutil
from datetime import datetime
from pathlib import Path

import librosa
import numpy as np
import soundfile as sf

ROOT = Path(__file__).resolve().parent
PROFILES_DIR = ROOT / "voice_profiles"
MANIFEST_PATH = ROOT / "dataset" / "lora_manifest.jsonl"

NAME_RE = re.compile(r"^[A-Za-z0-9][A-Za-z0-9_-]{1,63}$")
MIN_SEC, MAX_SEC = 3.0, 30.0  # hard limits; 5–15 s is the recommended range
TABLE_HEADERS = ["Profile Name", "Dialect Notes", "Duration (s)", "Transcript Preview", "Created At"]


def _profile_dir(name: str) -> Path:
    """Validate name and return its folder. Rejects anything that could escape PROFILES_DIR."""
    name = (name or "").strip()
    if not NAME_RE.match(name):
        raise ValueError(
            "Profile name must be 2–64 chars: letters, digits, '_' or '-', starting with a letter/digit "
            "(e.g. Bli_Made_Denpasar)."
        )
    return PROFILES_DIR / name


def _read_meta(folder: Path) -> dict:
    return json.loads((folder / "metadata.json").read_text(encoding="utf-8"))


def _write_meta(folder: Path, meta: dict) -> None:
    tmp = folder / "metadata.json.tmp"
    tmp.write_text(json.dumps(meta, ensure_ascii=False, indent=2), encoding="utf-8")
    tmp.replace(folder / "metadata.json")


def create_profile(name: str, audio_path: str, transcript: str, notes: str = "") -> dict:
    folder = _profile_dir(name)
    transcript = (transcript or "").strip()
    if folder.exists():
        raise ValueError(f"Profile '{folder.name}' already exists. Pick another name or delete it first.")
    if not audio_path or not Path(audio_path).is_file():
        raise ValueError("Please upload or record a reference audio clip.")
    if not transcript:
        raise ValueError("Exact transcript is required — it must match the spoken words in the audio.")

    try:
        audio, sr = librosa.load(audio_path, sr=None, mono=True)
    except Exception as e:
        raise ValueError(f"Could not read audio file: {e}") from e
    duration = len(audio) / sr if sr else 0.0
    if not MIN_SEC <= duration <= MAX_SEC:
        raise ValueError(f"Audio is {duration:.1f}s; it must be between {MIN_SEC:.0f}s and {MAX_SEC:.0f}s (5–15s ideal).")
    peak = float(np.max(np.abs(audio)))
    if peak < 1e-3:
        raise ValueError("Audio appears to be silent.")

    folder.mkdir(parents=True)
    try:
        sf.write(folder / "reference.wav", (audio / peak * 0.95).astype(np.float32), sr, subtype="PCM_16")
        meta = {
            "profile_name": folder.name,
            "prompt_text": transcript,
            "dialect_notes": (notes or "").strip(),
            "duration_sec": round(duration, 2),
            "created_at": datetime.now().isoformat(timespec="seconds"),
        }
        _write_meta(folder, meta)
    except Exception:
        shutil.rmtree(folder, ignore_errors=True)  # never leave a half-written profile behind
        raise
    sync_lora_manifest()
    return meta


def list_profiles() -> tuple[list[str], list[list]]:
    """Return (dropdown names, table rows) for every valid profile, sorted by name."""
    names, rows = [], []
    if PROFILES_DIR.is_dir():
        for folder in sorted(PROFILES_DIR.iterdir(), key=lambda p: p.name.lower()):
            if not (folder / "metadata.json").is_file() or not (folder / "reference.wav").is_file():
                continue
            try:
                m = _read_meta(folder)
            except (json.JSONDecodeError, OSError):
                continue
            text = m.get("prompt_text", "")
            names.append(folder.name)
            rows.append([
                folder.name,
                m.get("dialect_notes", ""),
                m.get("duration_sec", 0.0),
                text if len(text) <= 60 else text[:57] + "...",
                m.get("created_at", ""),
            ])
    return names, rows


def get_profile(name: str) -> dict:
    folder = _profile_dir(name)
    if not (folder / "metadata.json").is_file():
        raise ValueError(f"Profile '{folder.name}' not found.")
    meta = _read_meta(folder)
    meta["wav_path"] = str(folder / "reference.wav")
    return meta


def update_profile_metadata(name: str, new_transcript: str, new_notes: str) -> dict:
    folder = _profile_dir(name)
    new_transcript = (new_transcript or "").strip()
    if not new_transcript:
        raise ValueError("Transcript cannot be empty.")
    if not (folder / "metadata.json").is_file():
        raise ValueError(f"Profile '{folder.name}' not found.")
    meta = _read_meta(folder)
    meta["prompt_text"] = new_transcript
    meta["dialect_notes"] = (new_notes or "").strip()
    meta["updated_at"] = datetime.now().isoformat(timespec="seconds")
    _write_meta(folder, meta)
    sync_lora_manifest()
    return meta


def delete_profile(name: str) -> None:
    folder = _profile_dir(name)
    if not folder.is_dir():
        raise ValueError(f"Profile '{folder.name}' not found.")
    shutil.rmtree(folder)
    sync_lora_manifest()


def sync_lora_manifest() -> int:
    """Rebuild the JSONL manifest (VoxCPM training format: 'audio' + 'text') from all active profiles."""
    MANIFEST_PATH.parent.mkdir(parents=True, exist_ok=True)
    names, _ = list_profiles()
    tmp = MANIFEST_PATH.with_suffix(".jsonl.tmp")
    with tmp.open("w", encoding="utf-8") as f:
        for n in names:
            m = _read_meta(PROFILES_DIR / n)
            f.write(json.dumps({
                "audio": str((PROFILES_DIR / n / "reference.wav").resolve()),
                "text": m["prompt_text"],
                "profile_name": n,
                "dialect_notes": m.get("dialect_notes", ""),
            }, ensure_ascii=False) + "\n")
    tmp.replace(MANIFEST_PATH)
    return len(names)
