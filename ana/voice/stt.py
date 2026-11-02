"""Local speech-to-text (Phase 2, S9): mic capture + faster-whisper.

Public names:
- VoiceError (re-exported from ana.voice.errors)
- WhisperSTT: configurable faster-whisper wrapper. The package is imported
  lazily on first transcribe() so this module imports and the class can be
  constructed without faster-whisper installed. Transcription decodes the
  WAV with the stdlib `wave` module and hands faster-whisper a raw float32
  PCM array, so PyAV is never involved (PyAV 19 removed the
  `av.open(..., metadata_errors=...)` kwarg faster-whisper's path decoder
  uses).
- record_to_wav: push-to-talk recording — waits for a press to start the
  alsa-utils `arecord` binary (no Python audio dependencies) and stops it
  on a second press or at the max duration, whichever comes first.
- default_max_seconds: max recording seconds (ANA_VOICE_MAX_SECONDS or 12).
- stdin_press: default press seam (block on Enter, optional timeout).
"""

from __future__ import annotations

import os
import select
import shutil
import signal
import subprocess
import sys
import wave
from pathlib import Path

from .errors import VoiceError

__all__ = ["VoiceError", "WhisperSTT", "record_to_wav",
           "default_max_seconds", "stdin_press"]

DEFAULT_MAX_SECONDS = 12.0


def _segment_text(segment) -> str:
    """Extract text from a faster-whisper segment.

    Real segments expose ``.text``; plain ``(text, confidence)`` tuples
    (used by test doubles) are also accepted.
    """
    text = getattr(segment, "text", None)
    if text is None:
        text = segment[0]
    return str(text)


def _wav_pcm(wav_path: str | Path):
    """Decode a 16 kHz mono S16_LE WAV into a float32 array in [-1, 1].

    Uses only the stdlib `wave` module plus numpy — no av/PyAV. Raises
    VoiceError when numpy is missing or the file is not a readable WAV.
    """
    try:
        import numpy as np
    except ImportError:
        raise VoiceError(
            "numpy required for audio decode — pip install numpy") from None
    try:
        with wave.open(str(wav_path), "rb") as wf:
            frames = wf.readframes(wf.getnframes())
    except (wave.Error, OSError) as exc:
        raise VoiceError(f"cannot read WAV {wav_path}: {exc}") from None
    samples = np.frombuffer(frames, dtype=np.int16)
    return samples.astype(np.float32) / np.float32(32768.0)


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
        """Transcribe a WAV file; returns the text with whitespace trimmed.

        The WAV is read with the stdlib (we always record 16 kHz mono
        S16_LE) and passed to faster-whisper as a float32 array — the
        model skips its PyAV decoder entirely for ndarray input. Empty
        audio returns "" without loading the model.
        """
        audio = _wav_pcm(wav_path)
        if audio.size == 0:
            return ""
        model = self._model if self._model is not None else self._load()
        segments, _info = model.transcribe(audio, language=self.language)
        text = "".join(_segment_text(segment) for segment in segments)
        return text.strip()


def default_max_seconds() -> float:
    """Max recording seconds: $ANA_VOICE_MAX_SECONDS (seconds) or 12."""
    raw = os.environ.get("ANA_VOICE_MAX_SECONDS", "").strip()
    if not raw:
        return DEFAULT_MAX_SECONDS
    try:
        value = float(raw)
    except ValueError:
        raise VoiceError(
            f"invalid ANA_VOICE_MAX_SECONDS={raw!r} — use seconds, "
            f"e.g. {int(DEFAULT_MAX_SECONDS)}") from None
    if value <= 0:
        raise VoiceError(
            f"ANA_VOICE_MAX_SECONDS must be > 0, got {value:g}")
    return value


def stdin_press(timeout: float | None = None) -> bool:
    """Default press seam: block on Enter from stdin.

    With ``timeout``, return False when no Enter arrives in time (select
    on stdin; falls back to a blocking read when stdin is not
    selectable). Raises VoiceError on EOF (no input left).
    """
    def read() -> None:
        try:
            input()
        except EOFError:
            raise VoiceError(
                "input exhausted before recording started") from None

    if timeout is None:
        read()
        return True
    try:
        ready, _, _ = select.select([sys.stdin], [], [], timeout)
    except (OSError, ValueError):
        read()  # stdin not selectable; block without a timeout
        return True
    if not ready:
        return False
    read()
    return True


def _stop_recording(proc) -> None:
    """SIGINT arecord (it finalizes the WAV header), then reap it."""
    try:
        proc.send_signal(signal.SIGINT)
    except (ProcessLookupError, OSError):
        pass  # arecord already exited on its own
    try:
        proc.wait(timeout=10)
    except subprocess.TimeoutExpired:
        proc.kill()
        proc.wait()


def record_to_wav(out_path: str | Path, max_seconds: float | None = None,
                  press=None) -> Path:
    """Push-to-talk recording of microphone audio into out_path as
    16 kHz mono S16_LE WAV, via the alsa-utils ``arecord`` binary.

    ``press`` is a blocking click seam (``press()`` → started,
    ``press(timeout=...)`` → True on second press, False on timeout).
    The recording starts on the first press and stops on the second
    press or after ``max_seconds`` (default: $ANA_VOICE_MAX_SECONDS or
    12), whichever comes first; arecord then receives SIGINT so it
    finalizes the WAV header.

    Raises VoiceError when arecord is not installed (pointer to alsa-utils
    and the pw-record alternative) or when the recording command fails.
    """
    if shutil.which("arecord") is None:
        raise VoiceError(
            "arecord not found — install alsa-utils (pacman -S alsa-utils); "
            "with PipeWire you can also use pw-record instead")
    if press is None:
        press = stdin_press
    limit = default_max_seconds() if max_seconds is None else max_seconds
    out = Path(out_path)

    press()  # first press: start recording
    cmd = ["arecord", "-f", "S16_LE", "-r", "16000", "-q", str(out)]
    try:
        proc = subprocess.Popen(cmd)
    except OSError as exc:
        raise VoiceError(f"recording failed (arecord: {exc})") from None
    try:
        press(timeout=limit)  # second press or auto-stop at `limit`
    finally:
        _stop_recording(proc)
    if not out.exists():
        raise VoiceError(
            f"recording failed (arecord exit {proc.returncode})")
    return out
