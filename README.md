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
`/provider [name]`, `/tools [on|off|list]`.

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

## Tests

```sh
.venv/bin/python -m pytest -q
```
