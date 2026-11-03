"""Waifu overlay window (S14): GTK4 + gtk4-layer-shell, JVía JSON stdin.

Runs `ana waifu`: a non-decorated always-on-top layer-surface anchored
top-right of the Hyprland screen. It reads JSON lines from stdin (tests
pipe; the CLI producers write to a FIFO — see waifu.fifo_path) and
applies them through ana.waifu.parse_event / WaifuState.coerce:

    {"type": "state", "state": "talk"}   avatar mouth state
    {"type": "say",   "text": "..."}     subtitle line
    {"type": "quit"}                     exit

Missing avatar PNGs degrade to a numbered placeholder (so the window is
testable without art); the real PNGs go to ~/.local/share/ana/waifu/.
"""

from __future__ import annotations

import os
import pathlib
import sys

from .waifu import (avatar_size, blink_params, choose_monitor,
                    DEFAULT_AVOID_MONITOR, fade_steps, parse_event,
                    resolve_asset, WaifuState, waifu_dir)

def _placeholder(name: str) -> bytes:
    """A plain rounded rect with the state name, so tests without art
    still show something stable."""
    color = {"idle": "#6c71c4", "talk": "#b58900"}.get(name, "#586e75")
    svg = f'''<svg xmlns="http://www.w3.org/2000/svg" width="220" height="220">
<rect x="10" y="10" width="200" height="200" rx="24" fill="{color}" stroke="#073642" stroke-width="4"/>
<text x="110" y="125" font-size="30" fill="#fdf6e3" text-anchor="middle"
 font-family="sans-serif">{name}</text></svg>'''
    return svg.encode("utf-8")  # raw SVG bytes: the file must BE the svg


def read_events_forever(apply=None, stream=None):
    """Blocking event loop: one JSON line per event, EOF quits.

    `apply(state, subtitle)` receives the parsed pair (Pure logic seam:
    tests drive it without GTK.) Returns True when events were consumed
    (None on clean EOF also means we are done).
    """
    stream = stream if stream is not None else sys.stdin
    state = WaifuState.IDLE
    subtitle = ""
    for line in stream:
        line = line.strip()
        if not line:
            continue
        if line == '{"type": "quit"}':
            return WaifuState.IDLE
        new_state, new_subtitle = parse_event(line)
        if new_state is not None:
            state = WaifuState.coerce(new_state)
        if new_subtitle is not None:
            subtitle = new_subtitle
        if apply is not None:
            apply(state.value, subtitle)
    return WaifuState.IDLE


ANA_WAIFU_DEBUG = os.environ.get("ANA_WAIFU_DEBUG") or \
    os.environ.get("ANA_TUI_DEBUG", "")


def _dbg(fmt: str, *args) -> None:
    """Opt-in trace to $ANA_WAIFU_DEBUG (file path)."""
    if not ANA_WAIFU_DEBUG:
        return
    try:
        import time
        with open(ANA_WAIFU_DEBUG, "a") as fh:
            fh.write(f"{time.strftime('%H:%M:%S')} "
                     + (fmt % args if args else fmt) + "\n")
    except OSError:
        pass


def _choose_gdk_monitor(shell):
    """Gdk.Monitor for the waifu, or None (no monitor choice possible)."""
    try:
        from gi.repository import Gdk

        display = Gdk.Display.get_default()
        monitor_list = display.get_monitors()
        names = []
        for i in range(monitor_list.get_n_items()):
            monitor = monitor_list.get_item(i)
            names.append(monitor.get_connector())
        preferred = os.environ.get("ANA_WAIFU_MONITOR", "").strip() or None
        avoid = (os.environ.get("ANA_WAIFU_AVOID", "").strip()
                 or DEFAULT_AVOID_MONITOR)
        picked = choose_monitor(names, preferred, avoid)
        for i in range(monitor_list.get_n_items()):
            monitor = monitor_list.get_item(i)
            if monitor.get_connector() == picked:
                _dbg("waifu monitor: %s (avoid=%s)", picked, avoid)
                return monitor
    except Exception as exc:  # wayland/compositor quirks stay cosmetic
        _dbg("monitor pick failed: %s", exc)
    return None


def _reexec_system_python() -> "NoReturn":
    """os.execl the system /usr/bin/python3 for this overlay process."""
    import pathlib

    pkg_parent = str(pathlib.Path(__file__).resolve().parent.parent)
    env = os.environ.copy()
    extra = env.get("PYTHONPATH", "")
    if pkg_parent not in extra.split(os.pathsep):
        env["PYTHONPATH"] = (pkg_parent + os.pathsep
                             + extra).rstrip(os.pathsep)
    os.execl("/usr/bin/python3", "python3", "-m", "ana.overlay",
             *sys.argv[1:])


def layer_shell_lib(root: str = "/usr/lib") -> str | None:
    """Path of libgtk4-layer-shell for the LD_PRELOAD re-exec, or None.

    Arch ships the development symlink `libgtk4-layer-shell.so`;
    runtime-only installs (Void/labwc) often ship only the soname
    `.so.0`. Try both, then give up — the caller then reports the
    missing protocol instead of silently skipping the re-exec.
    """
    for name in ("libgtk4-layer-shell.so", "libgtk4-layer-shell.so.0"):
        candidate = os.path.join(root, name)
        if os.path.exists(candidate):
            return candidate
    return None


def run_overlay() -> int:
    """Entry: Gtk4 layer-shell surface driven by JSON on stdin."""
    try:
        import gi
        gi.require_version("Gtk", "4.0")
        gi.require_version("Gtk4LayerShell", "1.0")
        from gi.repository import Gdk, Gtk4LayerShell, Gtk, GLib
        if not Gtk4LayerShell.is_supported():
            # libgtk4-layer-shell must load BEFORE libwayland-client;
            # the known fix is LD_PRELOAD — set it and re-exec once.
            lib = layer_shell_lib()
            if lib and not os.environ.get("_ANA_LAYER_PRELOAD"):
                os.environ["_ANA_LAYER_PRELOAD"] = "1"
                os.environ["LD_PRELOAD"] = os.environ.get("LD_PRELOAD", "") + \
                    f"{os.pathsep}{lib}"
                _reexec_system_python()
            print("error: the compositor does not support the layer-shell "
                  "protocol (requires a Wayland session)",
                  file=sys.stderr); return 1
    except ImportError:
        # the project venv isolates system site-packages; PyGTK lives in
        # /usr — re-exec with the system interpreter (overlay needs no
        # venv-only deps: stdlib + the ana package path).
        _reexec_system_python()
    except ValueError as exc:
        print(f"error: GTK4/gtk4-layer-shell version mismatch ({exc}) — "
              "install gtk-layer-shell (pacman -S gtk-layer-shell)",
              file=sys.stderr)
        return 1

    window = Gtk.Window()
    window.set_decorated(False)
    # GTK4 windows paint an opaque theme background by default; the waifu
    # must be a floating figure, not a white card
    provider = Gtk.CssProvider()
    provider.load_from_string(
        "window { background: transparent; } "
        "window > box { background: transparent; }")
    Gtk.StyleContext.add_provider_for_display(
        Gdk.Display.get_default(), provider,
        Gtk.STYLE_PROVIDER_PRIORITY_APPLICATION)
    # layer-shell init must happen before the window maps
    Gtk4LayerShell.init_for_window(window)

    # monitor choice: never on the user's gaming output (DP-1 by
    # default; $ANA_WAIFU_MONITOR pins, $ANA_WAIFU_AVOID changes the rule)
    picked = _choose_gdk_monitor(Gtk4LayerShell)
    if picked is not None:
        Gtk4LayerShell.set_monitor(window, picked)
    _dbg("init done; is_layer=%s supported=%s",
         Gtk4LayerShell.is_layer_window(window),
         Gtk4LayerShell.is_supported())
    Gtk4LayerShell.set_namespace(window, "ana-waifu")
    Gtk4LayerShell.set_layer(window, Gtk4LayerShell.Layer.OVERLAY)
    # must anchor at least one edge or the surface never maps; the waifu
    # lives top-right: TOP | RIGHT with a 12px margin
    for edge, anchored, margin in ((Gtk4LayerShell.Edge.TOP, True, 12),
                                   (Gtk4LayerShell.Edge.RIGHT, True, 12)):
        Gtk4LayerShell.set_anchor(window, edge, anchored)
        Gtk4LayerShell.set_margin(window, edge, margin)

    box = Gtk.Box(orientation=Gtk.Orientation.VERTICAL, spacing=6)
    avatar = Gtk.Image()
    avatar.set_pixel_size(avatar_size())
    subtitle = Gtk.Label(label="Ana 👋 · /say on the way")
    subtitle.set_wrap(True)
    subtitle.set_max_width_chars(46)
    box.append(avatar)
    box.append(subtitle)
    window.set_child(box)

    # fade-in: layer surfaces appear abruptly; a gentle ramp reads nicer
    window.set_opacity(0.0)
    ramp = fade_steps(steps=8)
    for i, opacity in enumerate(ramp):
        _loop_no = i * 40

        def _set(v=opacity):
            window.set_opacity(v)
            return False

        GLib.timeout_add(_loop_no, _set)

    placeholders: dict[str, str] = {}

    def placeholder_file(name: str) -> str:
        """SVG placeholder rendered to a temp file, cached per state."""
        import tempfile

        if name not in placeholders:
            data = _placeholder(name)
            with tempfile.NamedTemporaryFile(
                    prefix=f"ana-waifu-{name}-", suffix=".svg",
                    delete=False) as fh:
                fh.write(data)
                placeholders[name] = fh.name
        return placeholders[name]

    def apply(state_name: str, text: str | None) -> None:
        if text is not None:
            subtitle.set_text(text)
        png = resolve_asset(state_name)
        avatar.set_from_file(str(png) if png is not None
                             else placeholder_file(state_name))

    _dbg("window size after full setup: %s x %s",
         window.get_width(), window.get_height())
    try:
        from .waifu import fifo_path

        fifo = fifo_path()
        fifo.parent.mkdir(parents=True, exist_ok=True)
        if not fifo.exists():
            os.mkfifo(fifo)
            _dbg("fifo created: %s", fifo)
    except OSError as exc:  # fifos are optional niceties
        _dbg("fifo create failed: %s", exc)
    # the STDIN reader runs in a daemon thread: blocking reads MUST NOT
    # sit on the GTK main loop or the window paints nothing (a:0 ghost
    # surface — the exact reason the first overlay was invisible)
    import threading
    from gi.repository import GLib

    def map_apply(state_name: str, text):
        """Thread-safe apply: the widgets update on the GTK main loop."""
        GLib.idle_add(apply, state_name, text)

    def fifo_reader():
        """FIFO loop (blocks only off the loop; reopens after EOF)."""
        import time

        from .waifu import fifo_path

        path = fifo_path()
        while True:
            try:
                fd = os.open(path, os.O_RDONLY | os.O_NONBLOCK)
            except OSError:
                _dbg("fifo missing; retrying", str(path))
                time.sleep(1.0)
                continue
            buf = b""
            while True:
                try:
                    chunk = os.read(fd, 4096)
                except BlockingIOError:
                    time.sleep(0.05)
                    continue
                except OSError:
                    break
                if not chunk:   # writer closed; wait for the next
                    break
                buf += chunk
                while b"\n" in buf:
                    line, buf = buf.split(b"\n", 1)
                    text = line.decode("utf-8", "replace").strip()
                    if text:
                        state_name, subtitle = parse_event(text)
                        if state_name == "quit":
                            GLib.idle_add(window.destroy)
                            return
                        if state_name is not None:
                            map_apply(state_name, subtitle)
                        elif subtitle is not None:
                            map_apply("", subtitle)
            os.close(fd)
            time.sleep(0.5)   # reopen for the next producing session

    # blink: pure overlay-side life; blink.png optional (fallback: idle)
    interval_ms, blink_ms = blink_params()
    import random as _random

    def _blink_and_schedule():
        blink = resolve_asset("blink")
        fallback = resolve_asset("idle")
        shown = str(blink) if blink is not None else (
            str(fallback) if fallback is not None
            else placeholder_file("idle"))
        avatar.set_from_file(shown)

        def _back():
            state_now = str(resolve_asset(state_holder[0]) or "")
            if state_now:
                avatar.set_from_file(state_now)
            else:
                avatar.set_from_file(placeholder_file(state_holder[0]))
            return False

        GLib.timeout_add(blink_ms, _back)
        return bool(_random.random() > 0.05)

    state_holder = [WaifuState.IDLE.value]

    def _tick():
        if _blink_and_schedule():
            interval_ms2 = interval_ms
        else:
            interval_ms2 = 200
        GLib.timeout_add(interval_ms2, _tick)
        return False

    GLib.timeout_add(interval_ms, _tick)

    threading.Thread(target=fifo_reader, daemon=True).start()
    _dbg("fifo reader started; presenting")
    window.present()
    _loop = GLib.MainLoop()
    _loop.run()
    if state_id == WaifuState.IDLE:
        pass  # nothing further: EOF ends the stream and with it the app
    return 0


def main(argv=None) -> int:
    return run_overlay()


if __name__ == "__main__":  # pragma: no cover
    sys.exit(main())
