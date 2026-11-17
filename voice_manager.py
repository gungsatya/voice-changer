"""CRUD for Balinese voice profiles and their emotion samples.

Layout (under DATA_ROOT, which defaults to the repo folder and is set to a Google Drive folder on
Colab via the VOICE_DATA_DIR environment variable so everything survives runtime restarts):

    voice_profiles/<Profile>/reference.wav            base ("normal") voice
    voice_profiles/<Profile>/metadata.json            transcript, notes, cached prosody profile
    voice_profiles/<Profile>/emotions/<emotion>/reference.wav
    voice_profiles/<Profile>/emotions/<emotion>/metadata.json
    dataset/lora_manifest.jsonl                       rebuilt on every change
"""

import hashlib
import json
import os
import re
import shutil
from datetime import datetime
from pathlib import Path

import librosa
import numpy as np
import soundfile as sf

import prosody

ROOT = Path(__file__).resolve().parent
DATA_ROOT = Path(os.environ.get("VOICE_DATA_DIR") or ROOT).expanduser().resolve()
PROFILES_DIR = DATA_ROOT / "voice_profiles"
MANIFEST_PATH = DATA_ROOT / "dataset" / "lora_manifest.jsonl"
EMOTION_SAMPLES_DIR = DATA_ROOT / "emotion_samples"

NAME_RE = re.compile(r"^[A-Za-z0-9][A-Za-z0-9_-]{1,63}$")
EMOTION_RE = re.compile(r"^[a-z][a-z0-9_-]{0,31}$")
MIN_SEC, MAX_SEC = 3.0, 30.0  # hard limits; 5–15 s is the recommended range
BASE_EMOTION = "normal"  # the profile's own reference.wav stands in for this emotion
SUGGESTED_EMOTIONS = ["normal", "marah", "ceria", "sedih", "takut", "terkejut", "tenang", "berbisik"]
AUDIO_EXTS = {".wav", ".mp3", ".flac", ".ogg", ".m4a"}

TABLE_HEADERS = ["Profile Name", "Dialect Notes", "Duration (s)", "Transcript Preview", "Emotions", "Created At"]
EMOTION_TABLE_HEADERS = ["Emotion", "Duration (s)", "Transcript Preview"] + [lbl for _, lbl in prosody.DISPLAY_FIELDS]


# ---------- helpers ----------
def _profile_dir(name: str) -> Path:
    """Validate name and return its folder. Rejects anything that could escape PROFILES_DIR."""
    name = (name or "").strip()
    if not NAME_RE.match(name):
        raise ValueError(
            "Profile name must be 2–64 chars: letters, digits, '_' or '-', starting with a letter/digit "
            "(e.g. Bli_Made_Denpasar)."
        )
    return PROFILES_DIR / name


def normalize_emotion(emotion: str) -> str:
    emotion = (emotion or "").strip().lower()
    if not EMOTION_RE.match(emotion):
        raise ValueError(
            "Emotion name must be 1–32 chars: lowercase letters, digits, '_' or '-', starting with a letter "
            "(e.g. marah, ceria, sedih)."
        )
    return emotion


def _read_meta(folder: Path) -> dict:
    return json.loads((folder / "metadata.json").read_text(encoding="utf-8"))


def _write_meta(folder: Path, meta: dict) -> None:
    tmp = folder / "metadata.json.tmp"
    tmp.write_text(json.dumps(meta, ensure_ascii=False, indent=2), encoding="utf-8")
    tmp.replace(folder / "metadata.json")


def _load_clip(audio_path: str) -> tuple[np.ndarray, int, float]:
    """Load, validate and peak-normalise a reference clip. Returns (audio, sr, duration)."""
    if not audio_path or not Path(audio_path).is_file():
        raise ValueError("Please upload or record a reference audio clip.")
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
    return (audio / peak * 0.95).astype(np.float32), sr, duration


def fingerprint(audio_path: str, transcript: str = "") -> str:
    """Content hash of a source clip + transcript, used to skip re-profiling unchanged files."""
    h = hashlib.sha1()
    with open(audio_path, "rb") as f:
        for block in iter(lambda: f.read(1 << 20), b""):
            h.update(block)
    h.update((transcript or "").strip().encode("utf-8"))
    return h.hexdigest()


def _save_clip(folder: Path, audio_path: str, transcript: str, extra: dict) -> dict:
    """Write reference.wav + metadata.json (with prosody profile) into a fresh folder."""
    transcript = (transcript or "").strip()
    if not transcript:
        raise ValueError("Exact transcript is required — it must match the spoken words in the audio.")
    audio, sr, duration = _load_clip(audio_path)
    folder.mkdir(parents=True)
    try:
        sf.write(folder / "reference.wav", audio, sr, subtype="PCM_16")
        meta = {
            **extra,
            "prompt_text": transcript,
            "duration_sec": round(duration, 2),
            "prosody": prosody.analyze(audio, sr, transcript),
            "source_sha1": fingerprint(audio_path, transcript),
            "created_at": datetime.now().isoformat(timespec="seconds"),
        }
        _write_meta(folder, meta)
    except Exception:
        shutil.rmtree(folder, ignore_errors=True)  # never leave a half-written clip behind
        raise
    return meta


def _ensure_prosody(folder: Path, meta: dict) -> dict:
    """Profiles created before emotion support have no cached prosody: compute it once and store it."""
    if meta.get("prosody") is None and (folder / "reference.wav").is_file():
        audio, sr = librosa.load(folder / "reference.wav", sr=None, mono=True)
        meta["prosody"] = prosody.analyze(audio, sr, meta.get("prompt_text", ""))
        _write_meta(folder, meta)
    return meta


# ---------- profiles ----------
def create_profile(name: str, audio_path: str, transcript: str, notes: str = "") -> dict:
    folder = _profile_dir(name)
    if folder.exists():
        raise ValueError(f"Profile '{folder.name}' already exists. Pick another name or delete it first.")
    meta = _save_clip(folder, audio_path, transcript,
                      {"profile_name": folder.name, "dialect_notes": (notes or "").strip()})
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
                ", ".join(emotion_names(folder.name)),
                m.get("created_at", ""),
            ])
    return names, rows


def get_profile(name: str) -> dict:
    folder = _profile_dir(name)
    if not (folder / "metadata.json").is_file():
        raise ValueError(f"Profile '{folder.name}' not found.")
    meta = _ensure_prosody(folder, _read_meta(folder))
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
    if meta.get("prompt_text") != new_transcript:
        meta["prosody"] = None  # speaking rate depends on the transcript: recompute below
    meta["prompt_text"] = new_transcript
    meta["dialect_notes"] = (new_notes or "").strip()
    meta["updated_at"] = datetime.now().isoformat(timespec="seconds")
    _write_meta(folder, meta)
    _ensure_prosody(folder, meta)
    sync_lora_manifest()
    return meta


def delete_profile(name: str) -> None:
    folder = _profile_dir(name)
    if not folder.is_dir():
        raise ValueError(f"Profile '{folder.name}' not found.")
    shutil.rmtree(folder)
    sync_lora_manifest()


# ---------- emotions ----------
def _emotion_dir(name: str, emotion: str) -> Path:
    return _profile_dir(name) / "emotions" / normalize_emotion(emotion)


def add_emotion(name: str, emotion: str, audio_path: str, transcript: str, notes: str = "",
                replace: bool = True) -> dict:
    """Add (or replace) one emotion sample of a profile, e.g. the 'marah' clip of Bli_Made."""
    profile = _profile_dir(name)
    if not (profile / "metadata.json").is_file():
        raise ValueError(f"Profile '{profile.name}' not found. Enroll the speaker first.")
    folder = _emotion_dir(name, emotion)
    if folder.exists():
        if not replace:
            raise ValueError(f"Emotion '{folder.name}' already exists for '{profile.name}'.")
        backup = folder.with_name(folder.name + ".old")
        shutil.rmtree(backup, ignore_errors=True)
        folder.rename(backup)
        try:
            meta = _save_clip(folder, audio_path, transcript, {"emotion": folder.name, "notes": (notes or "").strip()})
        except Exception:
            backup.rename(folder)  # keep the previous clip if the new one is invalid
            raise
        shutil.rmtree(backup, ignore_errors=True)
    else:
        meta = _save_clip(folder, audio_path, transcript, {"emotion": folder.name, "notes": (notes or "").strip()})
    sync_lora_manifest()
    return meta


def _explicit_emotions(name: str) -> list[str]:
    base = _profile_dir(name) / "emotions"
    if not base.is_dir():
        return []
    return sorted(
        d.name for d in base.iterdir()
        if EMOTION_RE.match(d.name) and (d / "metadata.json").is_file() and (d / "reference.wav").is_file()
    )


def emotion_names(name: str) -> list[str]:
    """All emotions usable in a script for this profile; 'normal' is always available (base clip)."""
    explicit = _explicit_emotions(name)
    return [BASE_EMOTION] + [e for e in explicit if e != BASE_EMOTION]


def get_emotion(name: str, emotion: str) -> dict:
    """Return {emotion, wav_path, prompt_text, prosody, ...} for one emotion of a profile."""
    emotion = normalize_emotion(emotion)
    folder = _emotion_dir(name, emotion)
    if (folder / "metadata.json").is_file() and (folder / "reference.wav").is_file():
        meta = _ensure_prosody(folder, _read_meta(folder))
        meta["wav_path"] = str(folder / "reference.wav")
        return meta
    if emotion == BASE_EMOTION:
        meta = get_profile(name)
        return {**meta, "emotion": BASE_EMOTION, "is_base": True}
    raise ValueError(
        f"Profile '{name}' has no '{emotion}' emotion. Available: {', '.join(emotion_names(name))}."
    )


def list_emotions(name: str) -> list[list]:
    """Table rows (EMOTION_TABLE_HEADERS) with the cached prosody profile of every emotion."""
    rows = []
    for e in emotion_names(name):
        m = get_emotion(name, e)
        text = m.get("prompt_text", "")
        label = e + (" (base clip)" if m.get("is_base") else "")
        p = m.get("prosody") or {}
        rows.append([label, m.get("duration_sec", 0.0), text if len(text) <= 50 else text[:47] + "..."]
                    + [p.get(k) for k, _ in prosody.DISPLAY_FIELDS])
    return rows


def delete_emotion(name: str, emotion: str) -> None:
    folder = _emotion_dir(name, emotion)
    if not folder.is_dir():
        if folder.name == BASE_EMOTION:
            raise ValueError("'normal' is the profile's base clip; delete the whole profile in Manage instead.")
        raise ValueError(f"Emotion '{folder.name}' not found for '{name}'.")
    shutil.rmtree(folder)
    sync_lora_manifest()


# ---------- bulk import from a (Google Drive) folder ----------
def import_emotion_folder(root: str | Path | None = None) -> list[str]:
    """Profile every clip under `root` once; unchanged clips are skipped on later runs.

    Expected layout (transcript in a .txt next to each clip, same stem):
        <root>/<Profile>/normal.wav + normal.txt
        <root>/<Profile>/marah.wav  + marah.txt
        <root>/<Profile>/ceria.mp3  + ceria.txt
    A profile that does not exist yet is created from normal.* (or the first clip found).
    Returns a human-readable report, one line per clip.
    """
    root = Path(root or EMOTION_SAMPLES_DIR).expanduser()
    if not root.is_dir():
        raise ValueError(f"Folder not found: {root}")
    report = []
    for pdir in sorted(p for p in root.iterdir() if p.is_dir()):
        clips = sorted(f for f in pdir.iterdir() if f.suffix.lower() in AUDIO_EXTS)
        if not clips:
            continue
        clips.sort(key=lambda f: f.stem.lower() != BASE_EMOTION)  # normal first
        for clip in clips:
            tag = f"{pdir.name}/{clip.name}"
            try:
                txt = clip.with_suffix(".txt")
                if not txt.is_file():
                    raise ValueError(f"missing transcript {txt.name}")
                transcript = txt.read_text(encoding="utf-8").strip()
                emotion = normalize_emotion(clip.stem)
                fp = fingerprint(str(clip), transcript)
                profile = _profile_dir(pdir.name)
                if not (profile / "metadata.json").is_file():
                    create_profile(pdir.name, str(clip), transcript, f"imported from {pdir}")
                    report.append(f"✅ {tag}: created profile (base clip = {emotion})")
                    if emotion == BASE_EMOTION:
                        continue
                edir = _emotion_dir(pdir.name, emotion)
                if (edir / "metadata.json").is_file():
                    current = _read_meta(edir).get("source_sha1")
                else:
                    current = None
                    if emotion == BASE_EMOTION and _read_meta(profile).get("source_sha1") == fp:
                        current = fp  # normal.* is already the base clip
                if current == fp:
                    report.append(f"⏭️ {tag}: unchanged, using cached profile")
                    continue
                add_emotion(pdir.name, emotion, str(clip), transcript, f"imported from {clip}")
                report.append(f"✅ {tag}: {'updated' if current else 'profiled'} emotion '{emotion}'")
            except (ValueError, OSError) as e:
                report.append(f"❌ {tag}: {e}")
    return report or [f"No clips found under {root} (expected <Profile>/<emotion>.wav + .txt)."]


# ---------- LoRA manifest ----------
def sync_lora_manifest() -> int:
    """Rebuild the JSONL manifest (VoxCPM training format: 'audio' + 'text') from every clip."""
    MANIFEST_PATH.parent.mkdir(parents=True, exist_ok=True)
    names, _ = list_profiles()
    tmp = MANIFEST_PATH.with_suffix(".jsonl.tmp")
    count = 0
    with tmp.open("w", encoding="utf-8") as f:
        for n in names:
            m = _read_meta(PROFILES_DIR / n)
            entries = [(PROFILES_DIR / n, m, BASE_EMOTION)]
            for e in _explicit_emotions(n):
                d = PROFILES_DIR / n / "emotions" / e
                entries.append((d, _read_meta(d), e))
            for folder, meta, emotion in entries:
                f.write(json.dumps({
                    "audio": str((folder / "reference.wav").resolve()),
                    "text": meta["prompt_text"],
                    "profile_name": n,
                    "emotion": emotion,
                    "dialect_notes": m.get("dialect_notes", ""),
                }, ensure_ascii=False) + "\n")
                count += 1
    tmp.replace(MANIFEST_PATH)
    return count
