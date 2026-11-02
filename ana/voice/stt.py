"""Local speech-to-text (S9/S11): mic capture + voxtype/faster-whisper.

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
- VoxtypeSTT: transcription via the locally installed voxtype CLI
  (`voxtype transcribe <wav>`) — the same engine the user dictates with.
  The transcript is the last non-empty stdout line (voxtype prints its
  progress logs around it).
- build_stt: backend resolver — explicit argument > $ANA_STT > "voxtype".
- default_max_seconds: max recording seconds (ANA_VOICE_MAX_SECONDS or 12).
- stdin_press: default press seam (block on Enter, optional timeout).
"""

from __future__ import annotations

import os
import re
import select
import shutil
import signal
import subprocess
import sys
import wave
from pathlib import Path

from .errors import VoiceError

__all__ = ["VoiceError", "WhisperSTT", "VoxtypeSTT", "build_stt",
           "record_to_wav", "default_max_seconds", "stdin_press"]

DEFAULT_MAX_SECONDS = 12.0
VALID_STT_BACKENDS = ("voxtype", "whisper")

_ANSI = re.compile(r"\x1b\[[0-9;]*m")


def _run(cmd, **kw):
    """Run a subprocess. Module-level seam so tests can monkeypatch it."""
    return subprocess.run(cmd, **kw)


def _last_stdout_line(text: str) -> str:
    """Last non-empty stdout line, ANSI stripped: the transcript.

    `voxtype transcribe` writes progress logs on stdout around the
    transcription; the transcript is always its final line.
    """
    lines = [line.strip() for line in _ANSI.sub("", text).splitlines()]
    return ([line for line in lines if line] or [""])[-1]


class VoxtypeSTT:
    """Speech-to-text via the locally installed voxtype CLI (S11).

    Ana records push-to-talk as before and hands the WAV to
    `voxtype transcribe`. A WAV with zero frames (press/press with
    nothing spoken) returns "" without running voxtype.
    """

    VOXTYPE_MISSING_MSG = (
        "voxtype not found — install it (AUR: voxtype, Omarchy: "
        "AI → Dictation) or set ANA_STT=whisper to fall back to "
        "faster-whisper")

    def __init__(self, binary: str | Path | None = None) -> None:
        self._binary = str(binary) if binary is not None else None

    def transcribe(self, wav_path: str | Path) -> str:
        audio = _wav_pcm(wav_path)
        if audio.size == 0:
            return ""
        vx = self._binary or shutil.which("voxtype")
        if not vx:
            raise VoiceError(self.VOXTYPE_MISSING_MSG)
        try:
            proc = _run([vx, "transcribe", str(wav_path)],
                        capture_output=True, text=True)
        except OSError as exc:
            raise VoiceError(f"voxtype transcribe failed ({exc})") from None
        if proc.returncode != 0:
            tail = (proc.stderr or "").strip().splitlines()
            detail = f": {tail[-1]}" if tail else ""
            raise VoiceError(
                f"voxtype transcribe failed (exit {proc.returncode})"
                f"{detail}")
        return _last_stdout_line(proc.stdout or "")


def build_stt(backend: str | None = None):
    """Resolve the STT backend: explicit arg > $ANA_STT > "voxtype"."""
    name = (backend if backend is not None
            else os.environ.get("ANA_STT", "")).strip().lower() or "voxtype"
    if name == "voxtype":
        return VoxtypeSTT()
    if name == "whisper":
        return WhisperSTT()
    opts = ", ".join(VALID_STT_BACKENDS)
    raise VoiceError(
        f"unknown STT backend {name!r} — options: {opts} "
        "(ANA_STT / --stt)")


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
    device = os.environ.get("ANA_MIC_DEVICE", "").strip()
    # -D pins the capture source: the system default can point at a
    # speaker monitor feed that records everything but speech (whisper
    # then hallucinates silence subtitles).
    cmd = (["arecord", "-D", device] if device else ["arecord"]) + \
        ["-f", "S16_LE", "-r", "16000", "-q", str(out)]
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
