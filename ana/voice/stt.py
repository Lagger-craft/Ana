"""Local speech-to-text (Phase 2, S9): mic capture + faster-whisper.

Public names:
- VoiceError (re-exported from ana.voice.errors)
- WhisperSTT: configurable faster-whisper wrapper. The package is imported
  lazily on first transcribe() so this module imports and the class can be
  constructed without faster-whisper installed.
- record_to_wav: record `seconds` of microphone audio to a WAV file using
  the alsa-utils `arecord` binary (no Python audio dependencies).
"""

from __future__ import annotations

import shutil
import subprocess
from pathlib import Path

from .errors import VoiceError

__all__ = ["VoiceError", "WhisperSTT", "record_to_wav"]


def _segment_text(segment) -> str:
    """Extract text from a faster-whisper segment.

    Real segments expose ``.text``; plain ``(text, confidence)`` tuples
    (used by test doubles) are also accepted.
    """
    text = getattr(segment, "text", None)
    if text is None:
        text = segment[0]
    return str(text)


class WhisperSTT:
    """Speech-to-text via faster-whisper.

    Defaults follow S10: model "small" (multilingual, CPU-friendly),
    CPU device, int8 compute, Spanish. The model is loaded lazily on the
    first transcribe() call, so constructing WhisperSTT never requires
    the package to be installed.
    """

    def __init__(self, model_size: str = "small", device: str = "cpu",
                 compute_type: str = "int8",
                 language: str | None = "es") -> None:
        self.model_size = model_size
        self.device = device
        self.compute_type = compute_type
        self.language = language
        self._model = None  # built on first use

    def _load(self):
        try:
            from faster_whisper import WhisperModel
        except ImportError:
            raise VoiceError(
                "faster-whisper not installed — run: "
                ".venv/bin/pip install -e '.[voice]'") from None
        self._model = WhisperModel(self.model_size, device=self.device,
                                   compute_type=self.compute_type)
        return self._model

    def transcribe(self, wav_path: str | Path) -> str:
        """Transcribe a WAV file; returns the text with whitespace trimmed."""
        model = self._model if self._model is not None else self._load()
        segments, _info = model.transcribe(str(wav_path),
                                           language=self.language)
        text = "".join(_segment_text(segment) for segment in segments)
        return text.strip()


def record_to_wav(seconds: float, out_path: str | Path) -> Path:
    """Record `seconds` of microphone audio into out_path as 16 kHz mono
    S16_LE WAV, via the alsa-utils ``arecord`` binary.

    Raises VoiceError when arecord is not installed (pointer to alsa-utils
    and the pw-record alternative) or when the recording command fails.
    """
    if shutil.which("arecord") is None:
        raise VoiceError(
            "arecord not found — install alsa-utils (pacman -S alsa-utils); "
            "with PipeWire you can also use pw-record instead")
    out = Path(out_path)
    cmd = ["arecord", "-f", "S16_LE", "-r", "16000",
           "-d", str(int(seconds)), "-q", str(out)]
    try:
        subprocess.run(cmd, check=True)
    except subprocess.CalledProcessError as exc:
        raise VoiceError(
            f"recording failed (arecord exit {exc.returncode})") from None
    return out
