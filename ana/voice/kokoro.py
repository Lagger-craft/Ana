"""Local text-to-speech via Kokoro (the default TTS backend).

Public names:
- VoiceError (re-exported from ana.voice.errors)
- KokoroTTS: synthesize text with kokoro-onnx and play the result with
  the shared paplay/aplay helper; the temporary WAV is deleted after
  playback. Same public contract as PiperTTS: ``speak(text)``.
- model_dir: the model directory — $ANA_KOKORO_DIR or
  ~/.local/share/kokoro.

``kokoro_onnx`` is imported lazily inside ``_synthesize`` — never at
module import time — so importing this module, and even constructing
``KokoroTTS``, works on machines without the package. Module-level
seams (``_synthesize``, ``play_wav``, ``find_player``) let tests run
without loading the model or touching the audio device.

Env knobs (read at speak() time): $ANA_KOKORO_DIR (model dir),
$ANA_KOKORO_VOICE (default ef_dora), $ANA_KOKORO_LANG (default es).
"""

from __future__ import annotations

import os
import tempfile
import wave
from pathlib import Path

from .errors import VoiceError
from .tts import find_player, play_wav
from ..waifu import sanitize_for_speech

try:
    import numpy as np
except ImportError:  # the optional voice extra may not be installed
    np = None

__all__ = ["VoiceError", "KokoroTTS", "model_dir"]

DEFAULT_VOICE = "ef_dora"   # Spanish female; em_alex / em_santa too
DEFAULT_LANG = "es"         # "e" does NOT work — kokoro wants "es"
MODEL_FILES = ("kokoro-v1.0.onnx", "voices-v1.0.bin")

KOKORO_MISSING_MSG = (
    "kokoro_onnx not installed — run: pip install kokoro-onnx")

# one engine per (model, voices) path — the ONNX session is expensive
_kokoro_cache: dict[tuple[str, str], object] = {}


def model_dir() -> Path:
    """Model directory: $ANA_KOKORO_DIR, else ~/.local/share/kokoro."""
    env = os.environ.get("ANA_KOKORO_DIR", "").strip()
    if env:
        return Path(env).expanduser()
    return Path.home() / ".local" / "share" / "kokoro"


def _require_models(root: Path) -> None:
    """Raise VoiceError naming both model files and where to get them."""
    missing = [name for name in MODEL_FILES if not (root / name).is_file()]
    if missing:
        raise VoiceError(
            f"Kokoro model files missing under {root} — "
            f"{MODEL_FILES[0]} and {MODEL_FILES[1]} are required "
            f"(missing: {', '.join(missing)}): download the "
            "'model-files-v1.0' assets from the thewh1teagle/kokoro-onnx "
            "GitHub release "
            "(https://github.com/thewh1teagle/kokoro-onnx/releases) "
            f"and put them in {root}")


def _synthesize(text: str, voice: str, lang: str):
    """(float32 samples, sample rate) for `text` — the real synth path.

    Module-level seam: tests monkeypatch it so no test ever loads the
    model. The ``kokoro_onnx`` import lives here (function scope) so
    this module imports cleanly without the package installed.
    """
    root = model_dir()
    try:
        from kokoro_onnx import Kokoro
    except ImportError:
        raise VoiceError(KOKORO_MISSING_MSG) from None
    key = (str(root / MODEL_FILES[0]), str(root / MODEL_FILES[1]))
    engine = _kokoro_cache.get(key)
    if engine is None:
        try:
            engine = Kokoro(*key)
        except Exception as exc:
            raise VoiceError(f"kokoro model load failed ({exc})") from None
        _kokoro_cache[key] = engine
    try:
        return engine.create(text, voice=voice, lang=lang)
    except Exception as exc:
        raise VoiceError(f"kokoro synthesis failed ({exc})") from None


def _write_wav(path: Path, samples, sample_rate: int) -> None:
    """Write float32 samples in [-1, 1] as a 16-bit mono PCM WAV."""
    if np is None:
        raise VoiceError(
            "numpy required for Kokoro TTS — pip install numpy") from None
    try:
        pcm = (np.clip(np.asarray(samples, dtype=np.float32), -1.0, 1.0)
               * 32767.0).astype(np.int16)
        with wave.open(str(path), "wb") as wf:
            wf.setnchannels(1)
            wf.setsampwidth(2)
            wf.setframerate(int(sample_rate))
            wf.writeframes(pcm.tobytes())
    except VoiceError:
        raise
    except Exception as exc:
        raise VoiceError(f"kokoro audio write failed ({exc})") from None


class KokoroTTS:
    """Text-to-speech with kokoro-onnx (Spanish voice, local, no root).

    ``voice`` / ``lang`` override the $ANA_KOKORO_VOICE /
    $ANA_KOKORO_LANG defaults when given. speak() sanitizes the text,
    synthesizes to a temp WAV, plays it, and deletes it. Raises
    VoiceError (never a raw import or file error) when the package,
    the model files, an audio player, or synthesis/playback is
    missing or fails.
    """

    def __init__(self, voice: str | None = None,
                 lang: str | None = None) -> None:
        self.voice = voice
        self.lang = lang

    def speak(self, text: str) -> None:
        """Speak `text` aloud; the temp WAV is removed after playback."""
        _require_models(model_dir())
        player = find_player()  # fail fast, before the expensive synth
        voice = self.voice if self.voice is not None else (
            os.environ.get("ANA_KOKORO_VOICE", "").strip() or DEFAULT_VOICE)
        lang = self.lang if self.lang is not None else (
            os.environ.get("ANA_KOKORO_LANG", "").strip() or DEFAULT_LANG)
        samples, sample_rate = _synthesize(
            sanitize_for_speech(text), voice, lang)
        with tempfile.TemporaryDirectory(prefix="ana-kokoro-") as tmp:
            wav = Path(tmp) / "reply.wav"
            _write_wav(wav, samples, sample_rate)
            play_wav(wav, player)
        # TemporaryDirectory cleaned up: the wav is gone after playback
