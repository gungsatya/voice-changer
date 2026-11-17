"""Prosody (intonation) profiling for emotion voice profiles.

`analyze()` turns a clip into a small, JSON-serialisable "prosody profile" (pitch level and range,
loudness, speaking rate, pauses). It is computed once when a clip is enrolled and cached next to the
clip in metadata, so a restarted Colab runtime never has to re-profile the same recording.

`match()` optionally nudges a generated segment towards a target emotion's profile (pitch shift,
time stretch, gain). VoxCPM already imitates the intonation of the reference clip it is prompted
with; this is a bounded correction on top of that, not the main mechanism.
"""

import re

import librosa
import numpy as np

ANALYSIS_SR = 16000
HOP = 160  # 10 ms
VOWEL_GROUP_RE = re.compile(r"[aiueoəéèêâîôûäëïöü]+", re.IGNORECASE)

# Columns shown in the UI, in order: (key, label)
DISPLAY_FIELDS = [
    ("f0_median_hz", "Pitch median (Hz)"),
    ("pitch_range_st", "Pitch range (st)"),
    ("energy_db", "Loudness (dB)"),
    ("syllables_per_sec", "Rate (syl/s)"),
    ("pause_ratio", "Pause ratio"),
]


def count_syllables(text: str) -> int:
    """Approximate syllables for Balinese / Indonesian text: one per vowel group."""
    return len(VOWEL_GROUP_RE.findall(text or ""))


def _r(x, nd=2):
    return None if x is None or not np.isfinite(x) else round(float(x), nd)


def analyze(audio: np.ndarray, sr: int, transcript: str = "") -> dict:
    """Return a prosody profile for a mono clip. Values that cannot be measured are None."""
    y = np.asarray(audio, dtype=np.float32)
    if y.ndim > 1:
        y = y.mean(axis=-1 if y.shape[-1] <= 2 else 0)
    if sr != ANALYSIS_SR:
        y = librosa.resample(y, orig_sr=sr, target_sr=ANALYSIS_SR)
    total_sec = len(y) / ANALYSIS_SR
    out = {
        "f0_median_hz": None, "f0_p10_hz": None, "f0_p90_hz": None,
        "pitch_range_st": None, "pitch_std_st": None,
        "energy_db": None, "energy_std_db": None,
        "speech_sec": 0.0, "pause_ratio": None, "syllables_per_sec": None,
        "total_sec": _r(total_sec),
    }
    if total_sec < 0.2 or float(np.max(np.abs(y))) < 1e-4:
        return out

    # Pitch contour (pYIN), voiced frames only
    f0, voiced, _ = librosa.pyin(y, fmin=60, fmax=700, sr=ANALYSIS_SR, frame_length=1024, hop_length=HOP)
    f0v = f0[voiced & np.isfinite(f0)]
    if len(f0v) >= 10:
        med = float(np.median(f0v))
        p10, p90 = np.percentile(f0v, [10, 90])
        semis = 12 * np.log2(f0v / med)
        out.update(
            f0_median_hz=_r(med, 1), f0_p10_hz=_r(p10, 1), f0_p90_hz=_r(p90, 1),
            pitch_range_st=_r(12 * np.log2(p90 / p10)), pitch_std_st=_r(np.std(semis)),
        )

    # Loudness over the active part of the clip
    rms_db = 20 * np.log10(librosa.feature.rms(y=y, frame_length=1024, hop_length=HOP)[0] + 1e-8)
    active = rms_db[rms_db > rms_db.max() - 40]
    if len(active):
        out.update(energy_db=_r(np.mean(active), 1), energy_std_db=_r(np.std(active), 1))

    # Speech vs pauses, and speaking rate from the transcript
    intervals = librosa.effects.split(y, top_db=35, frame_length=1024, hop_length=HOP)
    speech_sec = float(sum(e - s for s, e in intervals)) / ANALYSIS_SR
    out["speech_sec"] = _r(speech_sec)
    out["pause_ratio"] = _r(1 - speech_sec / total_sec)
    syl = count_syllables(transcript)
    if syl and speech_sec > 0.2:
        out["syllables_per_sec"] = _r(syl / speech_sec)
    return out


def match(
    wav: np.ndarray,
    sr: int,
    text: str,
    target: dict,
    max_semitones: float = 4.0,
    rate_bounds: tuple[float, float] = (0.8, 1.25),
    max_gain_db: float = 6.0,
) -> tuple[np.ndarray, dict]:
    """Shift pitch / tempo / gain of `wav` towards `target` (a profile from analyze()).

    Every adjustment is clamped so a bad measurement can never wreck the audio.
    Returns (new_wav, {"pitch_shift_st", "tempo_rate", "gain_db"}).
    """
    applied = {"pitch_shift_st": 0.0, "tempo_rate": 1.0, "gain_db": 0.0}
    if not target:
        return wav, applied
    cur = analyze(wav, sr, text)
    y = np.asarray(wav, dtype=np.float32)

    if cur["f0_median_hz"] and target.get("f0_median_hz"):
        n = float(np.clip(12 * np.log2(target["f0_median_hz"] / cur["f0_median_hz"]), -max_semitones, max_semitones))
        if abs(n) >= 0.3:
            y = librosa.effects.pitch_shift(y, sr=sr, n_steps=n)
            applied["pitch_shift_st"] = round(n, 2)

    if cur["syllables_per_sec"] and target.get("syllables_per_sec"):
        rate = float(np.clip(target["syllables_per_sec"] / cur["syllables_per_sec"], *rate_bounds))
        if abs(rate - 1) >= 0.03:
            y = librosa.effects.time_stretch(y, rate=rate)
            applied["tempo_rate"] = round(rate, 3)

    if cur["energy_db"] is not None and target.get("energy_db") is not None:
        g = float(np.clip(target["energy_db"] - cur["energy_db"], -max_gain_db, max_gain_db))
        if abs(g) >= 0.5:
            y = y * (10 ** (g / 20))
            applied["gain_db"] = round(g, 1)

    return y.astype(np.float32), applied
