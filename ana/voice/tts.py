"""Local text-to-speech (Phase 2, S9/S10): TTS backend resolution.

Public names:
- VoiceError (re-exported from ana.voice.errors)
- build_tts: backend resolver — explicit argument > $ANA_TTS > "kokoro"
  (mirrors build_stt; KokoroTTS is imported lazily inside it).
- PiperTTS: synthesize text with the ``piper`` binary and play the result
  with ``paplay`` (PipeWire/PulseAudio, default on Omarchy) falling back
  to ``aplay``. The temporary WAV is deleted after playback. Selected
  with ``ANA_TTS=piper`` / ``--tts piper``.
- resolve_voice: pick the Piper voice model — $ANA_PIPER_MODEL first,
  else the first es_*.onnx under ~/.local/share/piper/.
- find_player / play_wav: the shared paplay/aplay playback used by BOTH
  PiperTTS and KokoroTTS (ana.voice.kokoro).
- _run: module-level subprocess wrapper; tests monkeypatch this seam.
"""

from __future__ import annotations

import os
import shutil
import subprocess
import sys
import tempfile
from pathlib import Path

from .errors import VoiceError
from ..waifu import sanitize_for_speech

__all__ = ["VoiceError", "PiperTTS", "build_tts", "resolve_voice",
           "find_player", "play_wav"]

VALID_TTS_BACKENDS = ("kokoro", "piper")

PIPER_MISSING_MSG = (
    "piper not installed — install piper and a Spanish voice, "
    "e.g. es_MX-danepar or es_ES-carmen; or pip install piper-tts + "
    "download a voice")


def _run(cmd, **kw):
    """Run a subprocess. Module-level seam so tests can monkeypatch it."""
    return subprocess.run(cmd, **kw)


def _piper_binary() -> str | None:
    """Locate piper: PATH first, then the project venv's bin dir.

    The pip ``piper-tts`` package installs the binary into ``.venv/bin``,
    which is not on PATH when Ana is invoked as ``.venv/bin/ana``.
    """
    found = shutil.which("piper")
    if found:
        return found
    venv_bin = Path(sys.executable).parent / "piper"
    if venv_bin.is_file():
        return str(venv_bin)
    return None


def find_player() -> str:
    """The audio player: paplay (PipeWire/PulseAudio) or aplay (ALSA).

    Raises VoiceError when neither is installed — shared by every TTS
    backend so the message and lookup stay in one place.
    """
    player = shutil.which("paplay") or shutil.which("aplay")
    if player is None:
        raise VoiceError(
            "no audio player found — paplay (PipeWire/PulseAudio) or "
            "aplay (ALSA) is required to play replies")
    return player


def play_wav(wav: str | Path, player: str | None = None) -> None:
    """Play a WAV file; raises VoiceError when playback fails.

    ``player`` skips the PATH lookup when the caller already resolved
    it (PiperTTS checks before synthesizing).
    """
    player = player or find_player()
    try:
        _run([player, str(wav)], check=True)
    except subprocess.CalledProcessError as exc:
        raise VoiceError(
            f"playback failed ({player} exit {exc.returncode})") from None


def resolve_voice() -> str:
    """Resolve the Piper voice model path.

    Order: $ANA_PIPER_MODEL, else the first ``es_*.onnx`` found (flat or
    inside a per-voice folder) under ``~/.local/share/piper/``.
    Raises VoiceError with placement/download instructions when nothing
    is found.
    """
    env = os.environ.get("ANA_PIPER_MODEL")
    if env:
        return env
    root = Path.home() / ".local" / "share" / "piper"
    candidates = sorted(root.rglob("es_*.onnx"))
    if candidates:
        return str(candidates[0])
    raise VoiceError(
        "no Piper Spanish voice found — put an es_*.onnx (and its "
        f".onnx.json sidecar) under {root}, or set "
        "ANA_PIPER_MODEL=/path/to/voice.onnx; download a Spanish voice "
        "(e.g. es_ES-carmen or es_MX-danepar) from "
        "https://huggingface.co/rhasspy/piper-voices")


class PiperTTS:
    """Text-to-speech with the Piper binary (Spanish voice, local).

    ``voice`` is a model path; when None it is resolved at speak() time
    via resolve_voice(). speak() synthesizes to a temp WAV, plays it, and
    deletes it. Raises VoiceError when piper, a voice model, or an audio
    player is missing, or when synthesis/playback fails.
    """

    def __init__(self, voice: str | Path | None = None) -> None:
        self.voice = str(voice) if voice is not None else None

    def speak(self, text: str) -> None:
        """Speak `text` aloud; the temp WAV is removed after playback."""
        piper = _piper_binary()
        if piper is None:
            raise VoiceError(PIPER_MISSING_MSG)
        voice = self.voice if self.voice is not None else resolve_voice()
        player = find_player()
        with tempfile.TemporaryDirectory(prefix="ana-tts-") as tmp:
            wav = Path(tmp) / "reply.wav"
            try:
                # piper reads its text lines from stdin (no --input_file);
                # UTF-8 bytes so accented text survives the subprocess.
                _run([piper, "--model", voice,
                      "--output_file", str(wav)],
                     check=True,
                     input=(sanitize_for_speech(text) + "\n")
                     .encode("utf-8"))
            except subprocess.CalledProcessError as exc:
                raise VoiceError(
                    f"piper synthesis failed (exit {exc.returncode})"
                ) from None
            play_wav(wav, player)
        # TemporaryDirectory cleaned up: the wav is gone after playback


def build_tts(backend: str | None = None):
    """Resolve the TTS backend: explicit arg > $ANA_TTS > "kokoro".

    Construction is cheap and import-safe: KokoroTTS only loads
    kokoro_onnx (and the model) at speak() time, so an ANA_TTS=kokoro
    machine missing the package degrades at speak time instead of at
    REPL startup.
    """
    name = (backend if backend is not None
            else os.environ.get("ANA_TTS", "")).strip().lower() or "kokoro"
    if name == "kokoro":
        from .kokoro import KokoroTTS
        return KokoroTTS()
    if name == "piper":
        return PiperTTS()
    opts = ", ".join(VALID_TTS_BACKENDS)
    raise VoiceError(
        f"unknown TTS backend {name!r} — options: {opts} "
        "(ANA_TTS / --tts)")
