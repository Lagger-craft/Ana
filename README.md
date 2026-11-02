# Ana — waifu virtual assistant for Hyprland/Linux

Ana is a personal AI assistant for Arch Linux (Omarchy + Hyprland). This is
**Phase 0**: a streaming chat CLI that talks to cloud providers (OpenAI,
Anthropic, OpenCode Go, Xiaomi MiMo) or a local Ollama, driven by a TOML
config with API keys kept out of the file (env vars only).

The full roadmap lives in `odd/tasks/assistant-core.md`:

| Phase | Scope |
|---|---|
| **0** ✅ | chat CLI, multi-provider config, streaming |
| **1** ✅✨ | tool-calling: Tavily web search, Hyprland read-only queries, system info, graceful no-tools fallback |
| **2** ✅ | voice loop: faster-whisper STT + Piper TTS (local, opt-in) |
| 3 | images (screenshots + files), local vision via Qwen2.5-VL-3B on the RX 7600 |
| 4 | waifu avatar overlay (layer-shell), emotion tags |
| 5 | Inochi2D rigging via an MCP-driven Inochi Creator (PNG fallback if the spike fails) |
| 6 | camera, local-only for privacy |

## Install

```sh
cd ~/Projects/Ana
python -m venv .venv
.venv/bin/pip install -e '.[dev]'
```

## Setup

For **OpenCode Go** (default provider): create a Personal API key in the
OpenCode Console (Account › Personal API keys) — key format
`oc_sk_...` — then:

```sh
export OPENCODE_API_KEY="oc_sk_..."   # OpenCode Go Console key
export OPENAI_API_KEY="..."           # optional
.venv/bin/ana init                     # writes ~/.config/ana/config.toml
$EDITOR ~/.config/ana/config.toml
```

The OpenCode Go models are served by API family on the same key:
- `opencode-go` (api_style chat): kimi-k3, glm-5.x, minimax, deepseek, qwen3.8+
- `opencode-go-claude` (api_style anthropic): claude-* models
- GPT* models need the Responses API, not supported yet.

Config keys: `default_provider`, `system_prompt`, and one
`[providers.NAME]` table per provider with `base_url`, `api_style`
(`chat` for OpenAI-compatible endpoints, `anthropic` for the Messages
API), `api_key_env`, `default_model`.

To let Ana search the web, also export `TAVILY_API_KEY`
(free tier: https://tavily.com).

## Usage

```sh
.venv/bin/ana                    # interactive chat REPL (tools on)
.venv/bin/ana --provider ollama  # force a provider for this session
.venv/bin/ana --no-tools         # plain streaming chat, no tool calls
.venv/bin/ana chat               # same as bare `ana`
.venv/bin/ana -m "busca que hay de nuevo en hyprland"  # may search with TAVILY_API_KEY
.venv/bin/ana -m "info del sistema"                    # picks system_info tool
.venv/bin/ana init               # (re)create the starter config
```

REPL commands: `/exit`, `/new` (clear history), `/model [name]`,
`/provider [name]`, `/tools [on|off|list]`, `/voice [on|off|status]`.

If a provider rejects the tools payload (e.g. a small local model), Ana
retries the turn without tools and keeps tools off for the session;
system tools are read-only for now (`hyprctl ... -j`, no dispatch).

Example, fully local (no cloud involved):

```sh
ollama serve &                   # if not already running
.venv/bin/ana --provider ollama -m "Say hello in three words"
```

Example, fully local (no cloud involved):

```sh
ollama serve &                   # if not already running
.venv/bin/ana --provider ollama --no-tools -m "Say hello in three words"
```

## Voice (Phase 2)

Bidirectional voice, **fully local** (S9/S10): faster-whisper for
speech-to-text, Piper for text-to-speech with a Spanish voice. Nothing is
sent to the cloud — recording, transcription, synthesis and playback all
happen on this machine. Voice is opt-in per session; plain text mode is
unchanged.

### Install

```sh
.venv/bin/pip install -e '.[voice]'    # adds faster-whisper>=1.0 + piper-tts
```

Version bounds are deliberately permissive: `faster-whisper>=1.0` and
`piper-tts` unpinned — the packages are not installed in this environment
yet (install was deferred), so bounds stay loose until a known-good pair
is verified. The group is **not** installed by default.

Besides the Python packages you need:

- `arecord` for mic capture — `sudo pacman -S alsa-utils`
  (the `pw-record` alternative from PipeWire is mentioned in the error)
- `paplay` for playback (PipeWire/PulseAudio, default on Omarchy);
  `aplay` is used as fallback
- a Piper **Spanish** voice model (below)

### Spanish voice model

Resolution order: `$ANA_PIPER_MODEL`, else the first `es_*.onnx` found
under `~/.local/share/piper/` (flat file or one folder per voice). If
nothing is found, Ana tells you where to put one. Example (es_ES-carmen
from the rhasspy/piper-voices collection):

```sh
mkdir -p ~/.local/share/piper/es_ES-carmen
wget -O ~/.local/share/piper/es_ES-carmen/es_ES-carmen.onnx \
  https://huggingface.co/rhasspy/piper-voices/resolve/main/es_ES/carmen/medium/es_ES-carmen.onnx
wget -O ~/.local/share/piper/es_ES-carmen/es_ES-carmen.onnx.json \
  https://huggingface.co/rhasspy/piper-voices/resolve/main/es_ES/carmen/medium/es_ES-carmen.onnx.json
# or point at any model you already have:
export ANA_PIPER_MODEL=/path/to/es_XXXX-voice.onnx
```

### Usage

```sh
.venv/bin/ana --provider ollama --voice   # start in voice mode
```

In the REPL, `/voice on|off|status` toggles voice for the session:

- **Voice round** (push-to-talk): with voice on, press Enter on the
  empty prompt to see the hint `(🎙 Enter para empezar, Enter para
  terminar (max 12s))`. The first Enter starts recording, the second
  stops it; recording also auto-stops at the max time. The audio is
  transcribed locally and run through the normal tool-enabled turn,
  then Ana speaks the final reply with Piper. Silence gets
  `(no escuché nada)` and returns to the prompt.
- The max recording time comes from `ANA_VOICE_MAX_SECONDS` (seconds,
  default `12`).
- **Typed input** still works while voice is on — the reply is spoken
  too. `/voice off` stops the playback and the push-to-talk rounds.
- If a dependency or device is missing (no `faster-whisper`, no `piper`,
  no `arecord`...), Ana emits `(voice unavailable: <reason>)` and keeps
  running in text mode — the REPL never crashes on voice errors.

### Privacy

All local, no cloud STT/TTS: mic audio goes straight to faster-whisper
(CPU, `small` model by default) and replies are synthesized by Piper on
this machine. The recorded WAV is deleted right after transcription.

## Tests

```sh
.venv/bin/python -m pytest -q
```
