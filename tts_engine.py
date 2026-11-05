"""Lazy-loaded VoxCPM2 wrapper with Ultimate and Controllable cloning modes."""

import gc
import random
import threading
from datetime import datetime
from pathlib import Path

import numpy as np
import soundfile as sf
import torch

MODEL_ID = "openbmb/VoxCPM2"
OUTPUT_DIR = Path(__file__).resolve().parent / "outputs"


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
            try:
                model = self._get_model(use_denoiser)
                self._seed_everything(int(seed))
                with torch.inference_mode():
                    wav = model.generate(**kwargs)
                sr = model.tts_model.sample_rate
            except torch.cuda.OutOfMemoryError as e:
                self.unload()
                raise RuntimeError(
                    "CUDA out of memory. Disable the denoiser, shorten the text, or lower inference timesteps."
                ) from e
            finally:
                if torch.cuda.is_available():
                    torch.cuda.empty_cache()

        OUTPUT_DIR.mkdir(parents=True, exist_ok=True)
        out = OUTPUT_DIR / f"{profile_name}_{mode}_{datetime.now():%Y%m%d_%H%M%S_%f}.wav"
        sf.write(out, wav, sr)
        return str(out), sr


engine = BalineseTTSEngine()
