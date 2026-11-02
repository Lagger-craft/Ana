"""Local voice I/O for Ana (Phase 2, S9/S10): opt-in and fully local.

Public names:
- VoiceError: raised when a voice dependency, device, or model is missing
- WhisperSTT: faster-whisper speech-to-text wrapper (lazy import)
- record_to_wav: record the microphone to a WAV file via arecord
- PiperTTS: Piper text-to-speech synthesis and playback
"""

from .errors import VoiceError
from .stt import WhisperSTT, record_to_wav
from .tts import PiperTTS, resolve_voice

__all__ = ["VoiceError", "WhisperSTT", "record_to_wav", "PiperTTS",
           "resolve_voice"]
