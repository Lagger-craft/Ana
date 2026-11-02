# Feature: Ana assistant core

Waifu virtual assistant for Arch Linux (Omarchy + Hyprland). Multi-phase project; this document tracks implementation across sessions.

## Specs

- **S1**: "Arranquemos por la fase 0." — Phase 0 delivers a working chat CLI connected to a configured provider, with streaming output and session history.
- **S2**: Providers are cloud-first: OpenAI, Anthropic (Claude), Xiaomi MiMo, OpenCode Zen, and Ollama local as secondary option, selectable via config.
- **S3**: "Sobre el 'provedor' Opencode, me refiero a opencode go, con la api key." — OpenCode Zen gets first-class provider support (`https://opencode.ai/zen/v1`, API key auth).
- **S4**: "usemos de momento un modelo de local mas chico como el de Qwen2.5-VL-3B." — Vision phase (Phase 3+) defaults to Qwen2.5-VL-3B on the local RX 7600; recorded here for later phases, not implemented in Phase 0.
- **S5**: Camera input is deferred and, when built, must run with a local model for privacy ("Esto esta en un veremos mas adelante ... IA local o con alguna que sea privativa para mayor seguridad").
- **S6**: Voice conversation (faster-whisper STT + Piper TTS) and the waifu avatar (Inochi2D-with-MCP preferred, PNG/SVG fallback) are later phases.

## Tasks

| ID | Linked | Route | Commit |
|---|---|---|---|
| T1 | S1 | git init + scaffold (pyproject, package, .gitignore) | (pending) |
| T2 | S2,S3 | config module: TOML providers, env-var key indirection, defaults | (pending) |
| T3 | S2 | provider client: chat-completions SSE streaming adapter | (pending) |
| T4 | S1 | CLI chat loop: REPL cmds /exit /new /model /provider, one-shot -m | (pending) |
| T5 | S1 | README docs + smoke test against local Ollama | (pending) |

## Log

- **L1** (verbatim, original request): "Una duda, puedes investigar que tan factible el uso de crear un asistente virtual que se pueda conectar con una api ya sea de IA local con ollama o con un provedor como OpenAI, Claude, Opencode o Xiaomi mimo. Ademas de tener en cuenta mi equipo actual, ya que estoy en arch linux con hyprland y Omarchy. La idea esque pueda realizarle preguntas, que indage por internet, que pueda pasarle imagenes, Tambien el uso de la camara del ordenador ( Esto esta en un veremos mas adelante ya que estoy pensando que sea con IA local o con alguna que sea privativa para mayor seguridad). Tambien que le pueda hablar por voice y ella tambien me responda con una voz. Sobre el tema de lenguaje de programacion habra que verlo porque al estar en linux necesito que responda bien al sistema, en especial a Wayland. Tambien tenemos que ver la arquitectura y el plan que llevaremos a cabo. Ya que la idea es tambien tenga una skin, tipo "waifu". Pero no se con que lo puedo generar."
- **L2** (decisions, research session): product decisions — cloud-first; Python orchestrator (possible Rust voice daemon later); Piper TTS; avatar Inochi2D-with-MCP preferred / PNG fallback; camera deferred. Language: "Arranquemos por la fase 0."
- **L3** (correction/clarification): VRAM concern → for the local vision route, use Qwen2.5-VL-3B first (S4); camera models must have decent privacy/permissions (S5). Provider clarification: "Sobre el 'provedor' Opencode, me refiero a opencode go, con la api key." (S3).
