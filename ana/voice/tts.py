"""Local text-to-speech (Phase 2, S9/S10): Piper synthesis + playback.

Public names:
- VoiceError (re-exported from ana.voice.errors)
- PiperTTS: synthesize text with the ``piper`` binary and play the result
  with ``paplay`` (PipeWire/PulseAudio, default on Omarchy) falling back
  to ``aplay``. The temporary WAV is deleted after playback.
- resolve_voice: pick the Piper voice model — $ANA_PIPER_MODEL first,
  else the first es_*.onnx under ~/.local/share/piper/.
- _run: module-level subprocess wrapper; tests monkeypatch this seam.
"""

from __future__ import annotations

import os
import shutil
import subprocess
import tempfile
from pathlib import Path

from .errors import VoiceError

__all__ = ["VoiceError", "PiperTTS", "resolve_voice"]

PIPER_MISSING_MSG = (
    "piper not installed — install piper and a Spanish voice, "
    "e.g. es_MX-danepar or es_ES-carmen; or pip install piper-tts + "
    "download a voice")


def _run(cmd, **kw):
    """Run a subprocess. Module-level seam so tests can monkeypatch it."""
    return subprocess.run(cmd, **kw)


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
        piper = shutil.which("piper")
        if piper is None:
            raise VoiceError(PIPER_MISSING_MSG)
        voice = self.voice if self.voice is not None else resolve_voice()
        player = shutil.which("paplay") or shutil.which("aplay")
        if player is None:
            raise VoiceError(
                "no audio player found — paplay (PipeWire/PulseAudio) or "
                "aplay (ALSA) is required to play replies")
        with tempfile.TemporaryDirectory(prefix="ana-tts-") as tmp:
            wav = Path(tmp) / "reply.wav"
            try:
                _run([piper, "--model", voice,
                      "--output_file", str(wav)], check=True)
            except subprocess.CalledProcessError as exc:
                raise VoiceError(
                    f"piper synthesis failed (exit {exc.returncode})"
                ) from None
            try:
                _run([player, str(wav)], check=True)
            except subprocess.CalledProcessError as exc:
                raise VoiceError(
                    f"playback failed ({player} exit {exc.returncode})"
                ) from None
        # TemporaryDirectory cleaned up: the wav is gone after playback
