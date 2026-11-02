# Ana — waifu virtual assistant for Hyprland/Linux

Ana is a personal AI assistant for Arch Linux (Omarchy + Hyprland). This is
**Phase 0**: a streaming chat CLI that talks to cloud providers (OpenAI,
Anthropic, OpenCode Zen, Xiaomi MiMo) or a local Ollama, driven by a TOML
config with API keys kept out of the file (env vars only).

The full roadmap lives in `odd/tasks/assistant-core.md`:

| Phase | Scope |
|---|---|
| **0** ✅ | chat CLI, multi-provider config, streaming |
| 1 | tool-calling: web search, Hyprland tools (`hyprctl`, `grim`, `wl-clipboard`) |
| 2 | voice loop: faster-whisper STT + Piper TTS |
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

```sh
export OPENCODE_API_KEY="..."   # https://opencode.ai/zen
export OPENAI_API_KEY="..."     # optional
.venv/bin/ana init              # writes ~/.config/ana/config.toml
$EDITOR ~/.config/ana/config.toml
```

Config keys: `default_provider`, `system_prompt`, and one
`[providers.NAME]` table per provider with `base_url`, `api_style`
(`chat` for OpenAI-compatible endpoints, `anthropic` for the Messages
API), `api_key_env`, `default_model`.

## Usage

```sh
.venv/bin/ana                    # interactive chat REPL
.venv/bin/ana --provider ollama  # force a provider for this session
.venv/bin/ana chat               # same as bare `ana`
.venv/bin/ana -m "hi there"     # one-shot prompt, prints and exits
.venv/bin/ana init               # (re)create the starter config
```

REPL commands: `/exit`, `/new` (clear history), `/model [name]`,
`/provider [name]`.

Example, fully local (no cloud involved):

```sh
ollama serve &                   # if not already running
.venv/bin/ana --provider ollama -m "Say hello in three words"
```

Known limitation: OpenCode Zen serves some GPT* models through the
Responses API (`/v1/responses`), which this client does not speak yet.
Claude/Grok/Qwen models on Zen work with `api_style = "chat"`.

## Tests

```sh
.venv/bin/python -m pytest -q
```
