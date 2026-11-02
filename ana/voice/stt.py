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
- capture_source: the mic to record from ($ANA_MIC_DEVICE, else the
  PipeWire default source, else "default"); used to show the mic and
  to make the capture explicit.
- mic_label: short node name of the capture source, for the REPL hint.
- peak_db: dBFS peak of a recorded WAV; the REPL warns about
  near-silent takes (a muted tap-mute mic yields whisper hallucinations
  instead of an error — “Gracias por ver el video” over silence).
- min_peak_db: warning threshold ($ANA_VOICE_MIN_DB or -35 dB).
- record_to_wav: push-to-talk recording — waits for a press to start
  and stops on the second press or at the max duration. PipeWire-node
  source → ``pw-record --target`` so ONLY that node is captured (never
  a system output or monitor mix); ALSA-style names (hw:, CARD=...)
  go through arecord -D.
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

try:
    import numpy as np
except ImportError:  # the optional voice extra may not be installed
    np = None

from .errors import VoiceError

__all__ = ["VoiceError", "WhisperSTT", "VoxtypeSTT", "build_stt",
           "capture_source", "mic_label", "peak_db", "min_peak_db",
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

    Raises VoiceError when no recorder is installed (alsa-utils' arecord
    or pipewire-utils' pw-record) or when the recording command fails.
    """
    if press is None:
        press = stdin_press
    limit = default_max_seconds() if max_seconds is None else max_seconds
    out = Path(out_path)

    recorder, head = _record_cmd(out, capture_source())
    if recorder is None:
        raise VoiceError(
            "no audio recorder found — install alsa-utils (arecord) or "
            "pipewire-utils (pw-record)")
    press()  # first press: start recording
    try:
        proc = subprocess.Popen(head)
    except OSError as exc:
        raise VoiceError(f"recording failed ({recorder}: {exc})") from None
    try:
        press(timeout=limit)  # second press or auto-stop at `limit`
    finally:
        _stop_recording(proc)
    if not out.exists():
        raise VoiceError(
            f"recording failed ({recorder} exit {proc.returncode})")
    return out


def _record_cmd(out: Path, source: str):
    """(recorder, final argv), or (None, None) with no recorder.

    A PipeWire node name (pactl / $ANA_MIC_DEVICE) records via
    ``pw-record --target`` so ONLY that node is captured — never a
    system output or monitor mix. ALSA-style names (hw:, pulse, ...)
    go through the alsa-utils ``arecord -D``. An empty/de"fault"
    source follows the PipeWire default source (pw-record auto, or
    arecord's default).
    """
    pw = shutil.which("pw-record")
    arecord = shutil.which("arecord")
    source = (source or "").strip()
    alsa_named = (bool(re.match(
        r"^(?:hw|plug|sysdefault|front|surround|phys):", source))
        or source in ("pulse", "pipewire"))
    if alsa_named:
        if arecord:
            head = ["arecord", "-f", "S16_LE", "-r", "16000", "-q"]
            if source != "default":
                head = ["arecord", "-D", source, *head]
            return "arecord", [*head, str(out)]
        if pw:
            return "pw-record", ["pw-record", "--target", source,
                                 "--rate", "16000", "--channels", "1",
                                 str(out)]
        return None, None
    if source and pw:
        return "pw-record", ["pw-record", "--target", source,
                             "--rate", "16000", "--channels", "1",
                             str(out)]
    if pw:  # empty or "default": pw-record auto-targets the PW default
        return "pw-record", ["pw-record", "--rate", "16000",
                             "--channels", "1", str(out)]
    if arecord:
        return "arecord", ["arecord", "-f", "S16_LE", "-r", "16000",
                           "-q", str(out)]
    return None, None


def capture_source() -> str:
    """The mic to record from: $ANA_MIC_DEVICE, else the PipeWire
    default source (pactl), else "default" (ALSA resolves the default)."""
    env = os.environ.get("ANA_MIC_DEVICE", "").strip()
    if env:
        return env
    try:
        proc = _run(["pactl", "get-default-source"],
                    capture_output=True, text=True)
    except OSError:
        return "default"
    raw = (proc.stdout or "").strip()
    if proc.returncode == 0 and raw:
        return raw
    return "default"


def mic_label(source: str | None = None) -> str:
    """Short label of the capture source for the REPL hint.

    'alsa_input.usb-0c76_Razer_Seiren_Mini-00.mono-fallback' →
    'Razer Seiren Mini'; anything unrecognized stays as-is.
    """
    src = (source if source is not None else capture_source()).strip()
    base = re.sub(r"^alsa_(?:input|output|capture)\.", "", src)
    base = re.sub(r"\.(?:mono-fallback|analog-stereo)$", "", base)
    base = re.sub(r"^usb-", "", base)
    base = base.replace("_", " ")
    base = re.sub(r"\b[0-9a-fA-F]{4}\b", "", base)  # vendor ids (0c76)
    base = re.sub(r"\d{5,}", "", base)              # long serial numbers
    base = re.sub(r"-\d{1,3}$", "", base)           # trailing index
    label = " ".join(base.split())
    return label or src


def peak_db(wav_path: str | Path) -> float:
    """Peak level of a WAV in dBFS; -96.0 when effectively silent."""
    audio = _wav_pcm(wav_path)
    if audio.size == 0:
        return -96.0
    peak = float(np.abs(audio).max())
    return float(np.log10(peak) * 20.0) if peak > 0 else -96.0


def min_peak_db() -> float:
    """Warning threshold: $ANA_VOICE_MIN_DB (dBFS) or -35."""
    raw = os.environ.get("ANA_VOICE_MIN_DB", "").strip()
    try:
        return float(raw) if raw else -35.0
    except ValueError:
        return -35.0
