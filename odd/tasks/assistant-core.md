# Feature: Ana assistant core

Waifu virtual assistant for Arch Linux (Omarchy + Hyprland). Multi-phase project; this document tracks implementation across sessions.

## Specs

- **S1**: "Arranquemos por la fase 0." — Phase 0 delivers a working chat CLI connected to a configured provider, with streaming output and session history.
- **S2**: Providers are cloud-first: OpenAI, Anthropic (Claude), Xiaomi MiMo, OpenCode Zen, and Ollama local as secondary option, selectable via config.
- **S3**: "Sobre el 'provedor' Opencode, me refiero a opencode go, con la api key." — OpenCode Go (Console subscription) provider support: API key created in the Console (Account › Personal API keys, `oc_sk_...`), used against the Console inference gateway. Gateway serves model families per docs: Chat Completions at `https://opencode.ai/inference/openai/v1/chat/completions`, Anthropic Messages at `https://opencode.ai/inference/anthropic/v1/messages`, Bearer auth.
- **S4**: "usemos de momento un modelo de local mas chico como el de Qwen2.5-VL-3B." — Vision phase (Phase 3+) defaults to Qwen2.5-VL-3B on the local RX 7600; recorded here for later phases, not implemented in Phase 0.
- **S5**: Camera input is deferred and, when built, must run with a local model for privacy ("Esto esta en un veremos mas adelante ... IA local o con alguna que sea privativa para mayor seguridad").
- **S7**: "La idea esque pueda realizarle preguntas, que indage por internet" — Phase 1 adds tool-calling: Ana may invoke registered tools during a chat turn and incorporate results before answering. Web search backend: **Tavily Search API** with `TAVILY_API_KEY` (user decision L8).
- **S6**: Voice conversation (faster-whisper STT + Piper TTS) and the waifu avatar (Inochi2D-with-MCP preferred, PNG/SVG fallback) are later phases.
- **S8**: Phase 1 system tools are read-only (hyprctl status/workspaces/activewindow, system info); destructive shell execution is out of scope until a permission model exists.
- **S9**: "Tambien que le pueda hablar por voice y ella tambien me responda con una voz." — Phase 2 adds bidirectional voice, fully local (privacy-first): faster-whisper STT and Piper TTS with a Spanish voice. Voice is opt-in per session (`--voice` / REPL `/voice on|off`), never forced.
- **S10**: Voice defaults and quality tradeoffs: STT default model "small" (multilingual, CPU-friendly on the 5700X); Piper Spanish voice; both configurable later. No cloud STT/TTS in this phase.

## Tasks

| ID | Linked | Route | Commit |
|---|---|---|---|
| T1 | S1 | git init + scaffold (pyproject, package, .gitignore) | d98bd44 |
| T2 | S2,S3 | config module: TOML providers, env-var key indirection, defaults | 78c4986 |
| T3 | S2 | provider client: chat-completions SSE streaming adapter | 5ae0ce3 |
| T4 | S1 | CLI chat loop: REPL cmds /exit /new /model /provider, one-shot -m | 9277cf7 |
| T5 | S1 | README docs + smoke test against local Ollama | 2e2afb6 |
| T6 | S3 | provider config: OpenCode Go Console gateway (correction of T2) | 4f7f159 |
| T7 | S7 | client tool-calling: structured turn events for both dialects | e2f1ece |
| T8 | S7 | agent loop: tool execution rounds with bounded iterations | fe1ab38 |
| T9 | S7,S8 | tools: Brave web search, hyprctl read-only, system info | c172ebd |
| T10 | S1,S7 | REPL integration (/tools on/off, graceful fallback), docs, smoke | 79c8b69 |
| T11 | S7 | Tavily Search API replaces Brave (L8 correction) | 37c7c1a |
| T12 | S9,S10 | voice packages: faster-whisper + piper surface (optional deps) | (this commit) |
| T13 | S9 | STT module: record mic → transcribe, fakeable | (this commit) |
| T14 | S9 | TTS module: synth + play, fakeable | (this commit) |
| T15 | S9,S10 | CLI voice loop: --voice + /voice, docs, live smoke | (this commit) |

## Log

- **L1** (verbatim, original request): "Una duda, puedes investigar que tan factible el uso de crear un asistente virtual que se pueda conectar con una api ya sea de IA local con ollama o con un provedor como OpenAI, Claude, Opencode o Xiaomi mimo. Ademas de tener en cuenta mi equipo actual, ya que estoy en arch linux con hyprland y Omarchy. La idea esque pueda realizarle preguntas, que indage por internet, que pueda pasarle imagenes, Tambien el uso de la camara del ordenador ( Esto esta en un veremos mas adelante ya que estoy pensando que sea con IA local o con alguna que sea privativa para mayor seguridad). Tambien que le pueda hablar por voice y ella tambien me responda con una voz. Sobre el tema de lenguaje de programacion habra que verlo porque al estar en linux necesito que responda bien al sistema, en especial a Wayland. Tambien tenemos que ver la arquitectura y el plan que llevaremos a cabo. Ya que la idea es tambien tenga una skin, tipo "waifu". Pero no se con que lo puedo generar."
- **L2** (decisions, research session): product decisions — cloud-first; Python orchestrator (possible Rust voice daemon later); Piper TTS; avatar Inochi2D-with-MCP preferred / PNG fallback; camera deferred. Language: "Arranquemos por la fase 0."
- **L3** (correction/clarification): VRAM concern → for the local vision route, use Qwen2.5-VL-3B first (S4); camera models must have decent privacy/permissions (S5). Provider clarification: "Sobre el 'provedor' Opencode, me refiero a opencode go, con la api key." (S3).
- **L5** (correction, verbatim): "Sobre los modelos de Zen no lo quiero. Estoy hablando del plan Opencode Go. Te permite entrar a tus ajustes de usuario y generar una api key. Esa es la que quiero hacer uso. Una vez esa aclaracion seria bueno continuar con la siguiente fase. Una vez me confirmes de que puedo interactuar aunque sea via chat, lo pruebo yo mismo para tener un test mas real." — replaced the Zen provider config with the two OpenCode Go gateway providers (S3 rewritten, T2 corrected by T6).
- **L4** (evidence): Phase 0 complete. 34 unit tests pass. Smoke test against local Ollama (`qwen2.5-coder:1.5b`, `api_style=chat` on `http://127.0.0.1:11434/v1`) streamed a real reply through the CLI (one-shot exit 0, REPL /model + /exit exit 0). Error paths verified live: missing config (exit 1 + `ana init` hint). Starter config written to `~/.config/ana/config.toml`. Known limitation: OpenCode Zen GPT* models need the Responses API, not yet supported (documented in README + config template).
- **L7** (evidence): Phase 1 complete. 63 tests pass. Client stream_turn tool-calling (chat + anthropic), bounded agent loop, 7 default tools. Live verify: hypr_workspaces -j ran against the real Hyprland session (workspace DP-2 JSON returned); system_info correct; REPL /tools list; ollama one-shot with tools enabled streamed a reply (exit 0). Brave search untested live pending user API key (BRAVE_API_KEY). Web search backend decision recorded in L6 (Brave, L6 user pick). OpenCode Go gateway config in place for user self-test with oc_sk_ key.
- **L9** (evidence): user live test — gateway requests logged on their account: bare API key works for free models (429 free-quota for external clients), paid models give 402 without an active Go subscription; user confirmed "Con local si funciona." via ollama `--model qwen3:8b`. Cloud-default decision still pending (Go subscription vs local-first); local path fully functional. Phase 2 started.
- **L8** (correction, verbatim): "Hay alguna otra api que se pueda usar como exa u alguna otra? Porque brave me pide poner tarjeta." — web search backend switched from Brave to Tavily (user decision L8); S7 updated.
- **L10** (evidence): Phase 2 voice complete (S9/S10): T12 optional `voice` extra (`faster-whisper>=1.0`, `piper-tts` unpinned — packages NOT installed yet, deferred for network/time risk; permissive bounds documented in README "Voice (Phase 2)"); T13 `ana/voice/stt.py` (WhisperSTT, lazy faster-whisper import, defaults small/cpu/int8/es; `record_to_wav` via `arecord`, error points to alsa-utils/pw-record); T14 `ana/voice/tts.py` (PiperTTS, module-level `_run` test seam, voice resolution ANA_PIPER_MODEL → `~/.local/share/piper/es_*.onnx`, `paplay` with `aplay` fallback, temp WAV deleted after playback); T15 REPL `/voice on|off|status` + `--voice` flag — Enter-to-talk round transcribed locally and run through the same `_do_turn`, final reply spoken via PiperTTS, VoiceError degrades to `(voice unavailable: …)` without leaving the REPL. 81 tests pass (63 existing + 18 new); differential base-vs-change check on `--help`/`--version`/REPL-EOF shows only the requested `--voice` delta. Live voice smoke still pending: requires `pip install -e '.[voice]'` + piper + a Spanish voice (install explicitly deferred by T12).
