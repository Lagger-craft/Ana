# Ana — waifu virtual assistant for Linux (Wayland)

Ana is a personal AI assistant, tested daily on **Void Linux + KDE
Plasma 6 (Wayland, kwin_wayland)**. Arch Linux (Omarchy + Hyprland),
labwc and Niri remain supported as alternative compositor targets —
see [Keyboard shortcuts](#keyboard-shortcuts-s15). It began as
**Phase 0**, a streaming chat CLI that talks to cloud providers
(OpenAI, Anthropic, OpenCode Go, Xiaomi MiMo) or a local Ollama, driven
by a TOML config with API keys kept out of the file (env vars only).

The full roadmap lives in `odd/tasks/assistant-core.md`:

| Phase | Scope |
|---|---|
| **0** ✅ | chat CLI, multi-provider config, streaming |
| **1** ✅✨ | tool-calling: Tavily web search, `hypr_*` read-only queries (`hyprctl unavailable` outside Hyprland), system info, graceful no-tools fallback |
| **2** ✅ | voice loop: local STT (voxtype) + TTS — Kokoro default, Piper optional; voice ON by default (`--no-voice` to opt out) |
| **3** ✅ | images: local vision (Repl `/img`, one-shot `--image`), vision_model auto-pick |
| **4** ✅ | TUI (Textual): full-screen chat, same turn engine; opt-in `.tui` extra |
| **4** ✅ | waifu avatar overlay (layer-shell) — see [Waifu overlay art](#waifu-overlay-art) |
| 4 | emotion tags (overlay reactions) |
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
.venv/bin/ana talk               # headless voice hotkey round (see Voice)
```

REPL commands: `/exit`, `/new` (clear history), `/model [name]`,
`/provider [name]`, `/tools [on|off|list]`, `/voice [on|off|status]`,
`/img <ruta>` (queue images for the next message).

### TUI (Phase 4)

Opt-in interface for a full-screen Textual session. ONE turn engine:
provider, tools, image queue and vision pick behave exactly like the
REPL — the TUI is a frontend, not a fork.

```sh
.venv/bin/pip install -e '.[tui]'
.venv/bin/ana tui          # or: ana --tui --provider ollama
```

- Chat panel streams the same reply texts; the status bar shows model,
  provider, tools state and the pending-image count.
- Commands: `/model [name]`, `/provider [name]`, `/new`, `/img <ruta>`,
  `/voice [on|off|status]`, `/exit` (Ctrl+Q also quits). Voice is on by
  default: an empty Enter is push-to-talk, and `--stt`/`--tts` select
  the backends exactly like in the REPL.
- Headless-safe tests run through Textual's pilot (`tests/test_tui.py`).

### Images (Phase 3)

Attach a picture and ask about it, fully local (S4):

```sh
.venv/bin/ana -m "¿qué se ve en esta captura?" --image captura.png
# or in the REPL:
#   /img ~/Screenshots/foo.png     then just type your question
```

- Images attach to the **next message** (REPL stores them as pending;
  the queue clears after one send, `/new` clears it too).
- When a message carries images Ana picks the provider's
  `vision_model` automatically (template default for ollama:
  `qwen3-vl:4b`); an explicit `--model` / `/model` still wins.
- Reasoning models (qwen3*) surface a `(🤔 razonando...)` note while the
  model thinks, so the prompt never looks frozen.
- **Speed levers** (all config, no sudo): `vision_max_tokens` caps the
  output budget of image turns (measured on RX 7600: 256 ≈ 8 s but can
  answer EMPTY; 1024 ≈ 9 s with complete thinking; unset ≈ 18 s; the
  template ships 1024). Oversized shots are downscaled to
  `ANA_IMAGE_MAX_PX` (default 1024; 0 disables) before sending — the
  vision encoder's grid scales with pixels. `keep_alive = "1h"` keeps
  the model loaded between requests (default 5 min). And make sure
  ollama actually runs on your GPU: Arch's plain `ollama` package ships
  no GPU backends — install `ollama-vulkan` (`ollama ps` should say
  100% GPU after a vision turn; on CPU this blurred image took 2m17,
  same ask 17.8 s on the RX 7600).

```sh
.venv/bin/ana -m "describe la imagen" --image captura.png
```

If a provider rejects the tools payload (e.g. a small local model), Ana
retries the turn without tools and keeps tools off for the session;
system tools are read-only for now (`hyprctl ... -j`, no dispatch).
Built-ins include `current_time` (the model never guesses the time —
it calls the tool), `system_info`, `web_search` (Tavily key) and Ana's
own memory tools `memory_save` / `memory_search` (Engram — see
[Memory (Engram)](#memory-engram)).

### Local examples (no cloud)

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


### Waifu overlay art

PNGtuber frames live in `~/.local/share/ana/waifu/` (`idle.png`,
`talk.png`, optional `blink.png`; 600-1000px, transparent). The user
can hand-crop them, or run the one-shot pipeline (rembg isnet-anime,
anime-aware segmentation + shared-bbox trim so frames stay aligned):

```sh
scripts/waifu_prepare ~/Downloads/idle.jpg ~/Downloads/talk.jpg
```

Requires ImageMagick + `pip install rembg onnxruntime` (model downloads
on first run; the waifu shows idle when blink.png is absent).
Knobs: `$ANA_WAIFU_MONITOR` pins the output (default: NOT the avoid
one; `$ANA_WAIFU_AVOID`, default DP-1 for the gaming screen),
`$ANA_WAIFU_SIZE` (px edge, default 220), `$ANA_WAIFU_BLINK`
(`<seconds>:<ms>`, default 4 s / 150 ms). Events: JSON lines on the
FIFO — `say` sets the subtitle,
`state` swaps idle/talk/blink, `quit` closes the window. The REPL and
the TUI send `quit` when they exit, so the overlay closes with the chat
and `scripts/ana-quick` reopens a fresh one next time. `ana talk`
rounds do NOT send `quit` — the headless round has no chat window to
close, so the overlay stays up across hotkey rounds (the TUI/REPL
still close it on exit).

**Who sends them:** every frontend that talks to Ana — both `ana tui`
and the classic REPL (`ana`) push each reply as a `say` event, and voice
rounds wrap the spoken reply with `state: talk` → `state: idle` (S15b),
so the waifu reacts the same way no matter where you chat.

### Keyboard shortcuts (S15)

Two global shortcuts, currently bound on **KDE Plasma 6** — the
environment this project is tested on today. Both are *command
shortcuts* registered in System Settings → Shortcuts (search for
"Ana"); to rebind one, select it there and press the new key combo
(Plasma 6 stores command shortcuts in `~/.config/kglobalshortcutsrc`
under `[services]`, but the GUI is how you edit them):

- **`Meta+A` → "Ana — Abrir"** runs `scripts/ana-quick`: it starts
  `ana waifu` **only if no overlay is already live** (anchored
  `pgrep`, so a second instance never splits the FIFO events) and
  opens the TUI in your default terminal.
- **`Meta+Alt+A` → "Ana — Hablar"** runs `scripts/ana-talk-hotkey`,
  a thin wrapper built for the shortcut: it prepends `~/.local/bin`
  to `PATH` (kglobalaccel starts it from `kwin_wayland`, whose PATH
  has no `voxtype`), brings the waifu overlay up if it isn't running,
  then execs `ana talk` — the headless voice round documented under
  [Global voice hotkey](#global-voice-hotkey--ana-talk-s15c)
  (two-press toggle, capped by `ANA_VOICE_MAX_SECONDS`). No window
  needs focus.

#### Alternative compositors (Hyprland, labwc, Niri)

The same two scripts bind in the compositor's own config. On Hyprland
(Omarchy: `~/.config/hypr/bindings.lua`), adjust the paths to your
clone:

```lua
-- overlay (if not already running) + chat TUI, one key
o.bind("SUPER + A", "Ana waifu", "/home/<you>/Projects/Ana/scripts/ana-quick")
-- global push-to-talk toggle: press to record, press again to send
o.bind("SUPER + ALT + A", "Ana hablar", "/home/<you>/Projects/Ana/.venv/bin/ana talk")
```

- `SUPER + A` runs `scripts/ana-quick` — the same script the KDE
  **Ana — Abrir** shortcut runs.
- `SUPER + ALT + A` runs `ana talk` directly; on KDE the **Ana —
  Hablar** shortcut goes through `scripts/ana-talk-hotkey` instead
  (that wrapper works in any compositor config). Either way it is the
  headless voice round under [Global voice hotkey](#global-voice-hotkey--ana-talk-s15c)
  — no window needs focus.

Ana runs on any Wayland compositor with `wlr-layer-shell` (that is what
the overlay needs); outside Hyprland the `hypr_*` agent tools answer
`hyprctl unavailable: …` instead of the compositor state, and everything
else keeps working. For labwc (`~/.config/rc.xml`):

```xml
<keyboard>
  <keybind key="W-a">
    <action name="Execute" command="/home/<you>/Projects/Ana/scripts/ana-quick"/>
  </keybind>
  <keybind key="W-A-a">
    <action name="Execute" command="/home/<you>/Projects/Ana/.venv/bin/ana talk"/>
  </keybind>
</keyboard>
```

- Niri (`config.kdl`): `Mod+A { spawn "…/scripts/ana-quick"; }`.
- X11 sessions (Xfce, MATE, LXQt, IceWM) are **not** supported: the
  overlay is a layer-shell surface and has no X11 backend.
- `scripts/ana-quick` opens the TUI with `xdg-terminal-exec`, falling
  back to foot/alacritty/kitty/ghostty/wezterm/xterm/gnome-terminal/
  konsole when that helper is missing.

## Voice (Phase 2)

Bidirectional voice, **fully local** (S9/S10): voxtype for
speech-to-text, Kokoro (kokoro-onnx) for text-to-speech with a Spanish
voice — Piper stays available as an alternative TTS backend. Nothing is
sent to the cloud — recording, transcription, synthesis and playback all
happen on this machine. Voice is ON by default; `/voice off` (or
`ana --no-voice`) gives a text-only session.

### Speech-to-text backend

Default **voxtype** (S11): Ana records push-to-talk, then hands the WAV to
the locally installed `voxtype transcribe` CLI — the same engine you
dictate with (`voxtype.io`, AUR `voxtype` / Omarchy AI → Dictation).
Alternative: faster-whisper (`ANA_STT=whisper`, or `--stt whisper`).

- `--stt voxtype|whisper` selects per run; `$ANA_STT` does it per shell.
- **Capture source**: the push-to-talk recorder follows the PipeWire
  default source via `pw-record --target` — only that mic node is
  recorded, never the system audio mix. ALSA-style names (`hw:CARD=...`)
  go through `arecord -D`. `$ANA_MIC_DEVICE` pins a specific node.
- **Feedback in the REPL**: every voice round shows which mic it uses
  (`(🎙 mic: Razer Seiren Mini · ...)`) and warns when a take peaks
  near-silently (`ANA_VOICE_MIN_DB`, default -35 dBFS) — a muted tap-mute
  mic used to look like whisper hallucinations (“Gracias por ver el
  video”) instead of an error.

### Text-to-speech backend

Default **Kokoro** (kokoro-onnx, ~82M params): replies are synthesized
locally with a Spanish voice — no GPU, no system espeak, no root.
Alternative: Piper (`ANA_TTS=piper`, or `--tts piper`).

- `--tts kokoro|piper` selects per run; `$ANA_TTS` does it per shell.
- **Spanish voices**: `ef_dora` (female, default), `em_alex` and
  `em_santa` (male) — pick one with `$ANA_KOKORO_VOICE`, e.g.
  `export ANA_KOKORO_VOICE=em_alex`.
- `$ANA_KOKORO_DIR` points at the model directory (default
  `~/.local/share/kokoro`, see below).
- If a backend's package or model files are missing, Ana emits
  `(voice unavailable: …)` when the reply would be spoken and keeps
  running in text mode — the REPL never crashes on voice errors.

### Install

```sh
.venv/bin/pip install -e '.[voice]'    # adds faster-whisper>=1.0 + piper-tts + kokoro-onnx>=0.4
```

With the default voxtype backend, the `faster-whisper` extra is optional;
for replies you need the default TTS backend's package, `kokoro-onnx`
(or `piper-tts` when `ANA_TTS=piper`).

Version bounds are deliberately permissive: `faster-whisper>=1.0`,
`piper-tts` and `kokoro-onnx>=0.4` unpinned — `kokoro-onnx` 0.4.7 is the
verified pair on this machine, the others stay loose until a known-good
pair is pinned. The group is **not** installed by default.

Besides the Python packages you need:

- `arecord` for mic capture — Arch/Omarchy: `sudo pacman -S alsa-utils`,
  Void: `sudo xbps-install -S alsa-utils`
  (the `pw-record` alternative from PipeWire is mentioned in the error)
- `paplay` for playback (PipeWire/PulseAudio, default on Omarchy);
  `aplay` is used as fallback
- the Kokoro **model files** (below) for the default backend — or a
  Piper **Spanish** voice model if you use `ANA_TTS=piper`

### Kokoro model files (default backend)

Two files under `$ANA_KOKORO_DIR` (default `~/.local/share/kokoro`):
`kokoro-v1.0.onnx` and `voices-v1.0.bin`. Download them once from the
`model-files-v1.0` release assets of
[thewh1teagle/kokoro-onnx](https://github.com/thewh1teagle/kokoro-onnx/releases):

```sh
mkdir -p ~/.local/share/kokoro
wget -O ~/.local/share/kokoro/kokoro-v1.0.onnx \
  https://github.com/thewh1teagle/kokoro-onnx/releases/download/model-files-v1.0/kokoro-v1.0.onnx
wget -O ~/.local/share/kokoro/voices-v1.0.bin \
  https://github.com/thewh1teagle/kokoro-onnx/releases/download/model-files-v1.0/voices-v1.0.bin
# or keep them anywhere else:
export ANA_KOKORO_DIR=/path/to/kokoro-models
```

The Spanish voices (`ef_dora`, `em_alex`, `em_santa`) ship inside
`voices-v1.0.bin`; choose one with `ANA_KOKORO_VOICE`.

### Piper voice model (`ANA_TTS=piper`)

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
.venv/bin/ana --provider ollama             # voice on by default
# add --no-voice for a text-only session
```

In the REPL, `/voice on|off|status` toggles voice for the session:

- **Voice round** (push-to-talk): with voice on, press Enter on the
  empty prompt to see the hint `(🎙 Enter para empezar, Enter para
  terminar (max 12s))`. The first Enter starts recording, the second
  stops it; recording also auto-stops at the max time. The audio is
  transcribed locally and run through the normal tool-enabled turn,
  then Ana speaks the final reply with the TTS backend (Kokoro by
  default). Silence gets
  `(no escuché nada)` and returns to the prompt.
- The max recording time comes from `ANA_VOICE_MAX_SECONDS` (seconds,
  default `12`).
- **Media pause** (voxtype parity): while Ana records, MPRIS players
  (browser videos, etc.) are paused — playing media otherwise dominates
  the mic feed and gets transcribed instead of you. Paused players
  resume when the recording stops. `ANA_VOICE_PAUSE_MEDIA=0` opts out.
- **Typed input** still works while voice is on — the reply is spoken
  too. `/voice off` stops the playback and the push-to-talk rounds.
- If a dependency or device is missing (no `voxtype`, no
  `kokoro-onnx`, no `arecord`...), Ana emits `(voice unavailable: <reason>)` and keeps
  running in text mode — the REPL never crashes on voice errors.

### Global voice hotkey — `ana talk` (S15c)

A headless voice round with **no window and no terminal focus**, so a
global keybinding can talk to Ana from anywhere. On KDE Plasma it is
the **Ana — Hablar** command shortcut (`Meta+Alt+A`, rebind in System
Settings → Shortcuts), which executes:

```sh
# what the "Ana — Hablar" shortcut runs
/home/<you>/Projects/Ana/scripts/ana-talk-hotkey
# (PATH += ~/.local/bin for voxtype → ensure waifu overlay → exec `ana talk`)
```

On Hyprland (Omarchy) the same round binds directly in
`~/.config/hypr/bindings.lua`:

```sh
bind = SUPER ALT, A, exec, /home/<you>/Projects/Ana/.venv/bin/ana talk
```

Global bindings fire on key **press** only (no release) — KDE command
shortcuts and Hyprland alike — so this is a two-press toggle, not
hold-to-talk:

- **press 1** (no active round) → notify “🎙 grabando… (otra vez el
  atajo para cortar)”, then record the mic — auto-stops at
  `ANA_VOICE_MAX_SECONDS` (default 12 s) or when you press again,
  whichever comes first;
- **press 2** (recording) → touches the stop-file and exits 0
  immediately — it never blocks;
- **press 3 while Ana is past the mic** (transcribing/answering) →
  no-op with “ya estoy respondiendo”, so it can’t corrupt the round.

The transcript runs through the **same turn engine as the REPL**
(tools, provider config, error rollback, the spoken reply and the
waifu say/talk/idle events) — `talk` is orchestration, not a second code
path. Session state lives in `$XDG_RUNTIME_DIR/ana/{talk.json,talk.stop}`
beside the waifu FIFO and is cleared after every round; a crashed
worker’s dead pid is detected, so the next press starts fresh.

Feedback is headless: `notify-send` desktop notifications (recording
start, `(no escuché nada)`, `(voice unavailable: …)` and the first
120 chars of the reply) plus a `🎙 te escucho…` subtitle on the waifu
overlay when it’s running. Streamed turn output goes nowhere by
default (no TTY to write to); set `ANA_TALK_DEBUG=/path/to/log` to
append it to a file. `--stt`, `--tts` and `--config` work as everywhere
else: `.venv/bin/ana --stt whisper --tts piper talk`.

### Privacy

All local, no cloud STT/TTS: mic audio goes straight to faster-whisper
(CPU, `small` model by default) and replies are synthesized by Kokoro on
this machine. The recorded WAV is deleted right after transcription.

## Memory (Engram)

Ana keeps her own persistent memory, backed by the locally running
Engram server. Her recollections live in a SEPARATE namespace — project
`ana-asistente` (scope `project`) — so Ana's memories never mix with
your own Engram memories on the same server.

Two surfaces:

- **Tools the model drives**: `memory_save` (a short `title`, the
durable `content`, and a `type` of `fact` | `preference` | `persona`)
stores facts about you for FUTURE sessions; `memory_search` (a
`query`) brings them back formatted as `- [type] **title**: content`
lines, or an honest "no encontré recuerdos..." when nothing matches.
- **Session context**: at session start Ana's recent memory is
prepended to the system prompt — the REPL, `-m` one-shot and the TUI
all share the same turn engine, and the context is fetched at most
once per process.

`memory_save` treats the `title` as a natural key: saving a title that
already exists (matched ignoring case and surrounding whitespace)
updates that record in place (`PATCH /observations/{id}?expected_project=…`)
instead of creating a duplicate — the confirmation reads
`actualizado: <title>` rather than `guardado: <title>`. Ana's system
prompt reinforces this: when you state, correct or retract a durable
personal fact, she calls `memory_save` with the SAME title in that same
turn, so one topic stays one record.

Knobs (all optional):

| Env var | Default | Meaning |
|---|---|---|
| `ANA_ENGRAM_URL` | `http://127.0.0.1:7437` | Engram server base URL |
| `ANA_MEMORY_PROJECT` | `ana-asistente` | Ana's namespace |
| `ANA_MEMORY` | (on) | `ANA_MEMORY=0` disables everything |

Graceful degradation: every call has a short (3 s) timeout, and if
Engram is down or slow Ana prints a one-line note
(`memoria no disponible: ...`) and keeps the turn going in text mode —
a memory failure never crashes or stalls the chat. With `ANA_MEMORY=0`
the tools answer a one-line `memoria deshabilitada ...` note and the
system prompt stays exactly as it was. The REPL and the TUI end her
Engram session best-effort when they exit.

## Tests

```sh
.venv/bin/python -m pytest -q
```
