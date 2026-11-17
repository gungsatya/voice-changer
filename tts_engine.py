"""Lazy-loaded VoxCPM2 wrapper: Ultimate / Controllable cloning and multi-emotion scripts."""

import gc
import os
import random
import threading
from datetime import datetime
from pathlib import Path

import numpy as np
import soundfile as sf
import torch

import prosody

MODEL_ID = os.environ.get("VOXCPM_MODEL_ID", "openbmb/VoxCPM2")
OUTPUT_DIR = Path(os.environ.get("VOICE_DATA_DIR") or Path(__file__).resolve().parent).expanduser().resolve() / "outputs"
FADE_SEC = 0.01


class BalineseTTSEngine:
    def __init__(self, model_id: str = MODEL_ID):
        self.model_id = model_id
        self._model = None
        self._denoiser_loaded = False
        self._lock = threading.Lock()  # one GPU job at a time; also guards (re)loading

    def _get_model(self, use_denoiser: bool):
        if self._model is not None and self._denoiser_loaded == use_denoiser:
            return self._model
        self.unload()
        from voxcpm import VoxCPM  # heavy import: defer until first synthesis

        self._model = VoxCPM.from_pretrained(self.model_id, load_denoiser=use_denoiser)
        self._denoiser_loaded = use_denoiser
        return self._model

    def unload(self) -> None:
        self._model = None
        gc.collect()
        if torch.cuda.is_available():
            torch.cuda.empty_cache()

    @staticmethod
    def _seed_everything(seed: int) -> None:
        # VoxCPM.generate() has no seed argument, so seed the global RNGs it samples from.
        random.seed(seed)
        np.random.seed(seed % (2**32))
        torch.manual_seed(seed)
        if torch.cuda.is_available():
            torch.cuda.manual_seed_all(seed)

    def synthesize(
        self,
        text: str,
        wav_path: str,
        prompt_text: str,
        mode: str = "ultimate",
        style_prompt: str = "",
        cfg_value: float = 2.0,
        timesteps: int = 10,
        seed: int = 42,
        use_denoiser: bool = False,
        profile_name: str = "voice",
    ) -> tuple[str, int]:
        """Generate speech, save it to ./outputs/, return (path, sample_rate)."""
        text = (text or "").strip()
        if not text:
            raise ValueError("Target text is empty.")
        if not Path(wav_path).is_file():
            raise ValueError(f"Reference audio not found: {wav_path}")

        kwargs = dict(
            reference_wav_path=wav_path,
            cfg_value=float(cfg_value),
            inference_timesteps=int(timesteps),
            denoise=use_denoiser,
        )
        if mode == "ultimate":
            kwargs.update(text=text, prompt_wav_path=wav_path, prompt_text=prompt_text)
        else:
            style_prompt = (style_prompt or "").strip()
            kwargs["text"] = f"({style_prompt}){text}" if style_prompt else text

        with self._lock:
            wav, sr = self._generate([kwargs], use_denoiser, seed)[0]
        return self._save(wav, sr, f"{profile_name}_{mode}"), sr

    def _generate(self, kwargs_list: list[dict], use_denoiser: bool, seed: int) -> list[tuple[np.ndarray, int]]:
        """Run model.generate for each kwargs dict (caller holds the lock). Same seed for every call."""
        try:
            model = self._get_model(use_denoiser)
            sr = model.tts_model.sample_rate
            out = []
            for kw in kwargs_list:
                self._seed_everything(int(seed))
                with torch.inference_mode():
                    out.append((np.asarray(model.generate(**kw), dtype=np.float32).reshape(-1), sr))
            return out
        except torch.cuda.OutOfMemoryError as e:
            self.unload()
            raise RuntimeError(
                "CUDA out of memory. Disable the denoiser, shorten the text, or lower inference timesteps."
            ) from e
        finally:
            if torch.cuda.is_available():
                torch.cuda.empty_cache()

    @staticmethod
    def _save(wav: np.ndarray, sr: int, stem: str) -> str:
        OUTPUT_DIR.mkdir(parents=True, exist_ok=True)
        out = OUTPUT_DIR / f"{stem}_{datetime.now():%Y%m%d_%H%M%S_%f}.wav"
        sf.write(out, wav, sr)
        return str(out)

    def synthesize_script(
        self,
        segments: list[dict],
        cfg_value: float = 2.0,
        timesteps: int = 10,
        seed: int = 42,
        use_denoiser: bool = False,
        profile_name: str = "voice",
        gap_ms: int = 120,
        match_prosody: bool = False,
    ) -> tuple[str, int, list[dict]]:
        """Generate an emotion script.

        `segments` is a list of {"text", "emotion", "wav_path", "prompt_text", "prosody"}; each segment is
        cloned in Ultimate mode from its own emotion clip (so it inherits that clip's intonation) and the
        pieces are joined with `gap_ms` of silence. Returns (path, sample_rate, per-segment report).
        """
        if not segments:
            raise ValueError("Target text is empty.")
        kwargs_list = []
        for seg in segments:
            if not (seg.get("text") or "").strip():
                raise ValueError("Empty segment in emotion script.")
            if not Path(seg["wav_path"]).is_file():
                raise ValueError(f"Reference audio not found: {seg['wav_path']}")
            kwargs_list.append(dict(
                text=seg["text"].strip(),
                prompt_wav_path=seg["wav_path"],
                prompt_text=seg["prompt_text"],
                reference_wav_path=seg["wav_path"],
                cfg_value=float(cfg_value),
                inference_timesteps=int(timesteps),
                denoise=use_denoiser,
            ))

        with self._lock:
            results = self._generate(kwargs_list, use_denoiser, seed)

        sr = results[0][1]
        fade = max(1, int(FADE_SEC * sr))
        gap = np.zeros(int(sr * max(0, gap_ms) / 1000), dtype=np.float32)
        pieces, report = [], []
        for i, (seg, (wav, _)) in enumerate(zip(segments, results)):
            applied = {}
            if match_prosody and seg.get("prosody"):
                wav, applied = prosody.match(wav, sr, seg["text"], seg["prosody"])
            if len(wav) > 2 * fade:  # short fades avoid clicks at the joins
                ramp = np.linspace(0, 1, fade, dtype=np.float32)
                wav = wav.copy()
                wav[:fade] *= ramp
                wav[-fade:] *= ramp[::-1]
            if i:
                pieces.append(gap)
            pieces.append(wav)
            report.append({"emotion": seg["emotion"], "text": seg["text"],
                           "duration_sec": round(len(wav) / sr, 2), **applied})
        full = np.concatenate(pieces)
        peak = float(np.max(np.abs(full))) if len(full) else 0.0
        if peak > 0.99:
            full = full / peak * 0.99
        return self._save(full, sr, f"{profile_name}_emotion"), sr, report


engine = BalineseTTSEngine()
