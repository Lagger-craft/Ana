"""Local voice I/O for Ana (Phase 2, S9/S10): opt-in and fully local.

Public names:
- VoiceError: raised when a voice dependency, device, or model is missing
- WhisperSTT: faster-whisper speech-to-text wrapper (lazy import)
- record_to_wav: record the microphone to a WAV file via arecord
- KokoroTTS: kokoro-onnx text-to-speech (default backend, lazy import)
- PiperTTS: Piper text-to-speech (ANA_TTS=piper / --tts piper)
- build_tts: TTS backend resolver (explicit arg > $ANA_TTS > kokoro)
"""

from .errors import VoiceError
from .kokoro import KokoroTTS
from .stt import WhisperSTT, record_to_wav
from .tts import PiperTTS, build_tts, resolve_voice

__all__ = ["VoiceError", "WhisperSTT", "record_to_wav", "KokoroTTS",
           "PiperTTS", "build_tts", "resolve_voice"]
