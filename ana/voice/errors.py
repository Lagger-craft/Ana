"""Shared voice errors for Ana's local voice stack (Phase 2)."""


class VoiceError(RuntimeError):
    """A voice dependency, device, or voice model is unavailable.

    The CLI catches this to degrade gracefully to text mode instead of
    crashing the REPL.
    """
