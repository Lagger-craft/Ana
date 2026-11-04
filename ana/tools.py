"""Default Ana tools: web search, Hyprland read-only queries, system info.

Phase 1 is read-only: hyprctl is queried, never dispatched; no shell
execution tool exists yet (needs a permission model first).
"""

from __future__ import annotations

import json
import os
import shutil
import subprocess
import sys
import tempfile

import httpx

from datetime import datetime, timezone as _timezone
from zoneinfo import ZoneInfo

from . import memory
from .agent import Tool, ToolRegistry, ToolResult

TAVILY_URL = "https://api.tavily.com/search"

HYPRCTL_READ_COMMANDS = {
    "hypr_active_window": "activewindow",
    "hypr_workspaces": "workspaces",
    "hypr_monitors": "monitors",
    "hypr_version": "version",
    "hypr_devices": "devices",
}


def _tavily_search(args: dict) -> str:
    """Search via Tavily Search API; degrade to an error message."""
    key = os.environ.get("TAVILY_API_KEY")
    if not key:
        return ("web search unavailable: TAVILY_API_KEY is not set. "
                "Ask the user to export it (https://tavily.com).")
    query = args.get("query") or ""
    count = max(1, min(int(args.get("count") or 5), 10))
    try:
        resp = httpx.post(
            TAVILY_URL,
            json={"query": query, "max_results": count,
                  "search_depth": "basic"},
            headers={"Authorization": f"Bearer {key}",
                     "Content-Type": "application/json"},
            timeout=15,
        )
        resp.raise_for_status()
        results = resp.json().get("results") or []
    except Exception as exc:  # network/parse errors must not kill the turn
        return f"web search unavailable: {exc.__class__.__name__}: {exc}"

    if not results:
        return f"no results for {query !r}"
    lines = []
    for i, r in enumerate(results, 1):
        desc = r.get("content") or "[no description]"
        lines.append(f"{i}. {r.get('title', '(no title)')} - {r.get('url', '')}"
                     f" - {desc}")
    return "\n".join(lines)


def _hyprctl_reader(subcommand: str):
    """Read-only hyprctl query that DEGRADES outside Hyprland.

    run_agent only traps AgentError, so any OSError here (no hyprctl on
    labwc/KDE/Niri, PATH stripped) or TimeoutExpired (hung compositor
    socket) would kill the whole turn. Both become an answer string the
    model can read and stop retrying.
    """
    def reader(args: dict) -> str:
        try:
            proc = subprocess.run(
                ["hyprctl", subcommand, "-j"], capture_output=True,
                text=True, timeout=10, check=False)
        except FileNotFoundError:
            return ("hyprctl unavailable: no Hyprland session on this "
                    "desktop (Wayland compositor without hyprctl) — do "
                    "not retry this tool")
        except subprocess.TimeoutExpired:
            return (f"hyprctl unavailable: `hyprctl {subcommand} -j` "
                    "timed out after 10s — do not retry this tool")
        except OSError as exc:
            return (f"hyprctl unavailable: {exc.__class__.__name__}: "
                    f"{exc} — do not retry this tool")
        out = (proc.stdout or proc.stderr or "").strip()
        return out if proc.returncode == 0 else f"hyprctl error: {out}"
    return reader


def _hypr_tool(display_name: str, subcommand: str) -> Tool:
    description = f"Run `hyprctl {subcommand} -j` and return its JSON status."
    return Tool(
        name=display_name,
        description=description,
        parameters={"type": "object", "properties": {}},
        fn=_hyprctl_reader(subcommand),
    )


def _system_info(args: dict) -> dict:
    load = os.getloadavg()
    return {
        "os": sys.platform,
        "kernel": os.uname().release,
        "host": os.uname().nodename,
        "python": sys.version.split()[0],
        "cpu_count": os.cpu_count() or 0,
        "load_avg": [round(x, 2) for x in load],
        "hyprland": bool(os.environ.get("HYPRLAND_INSTANCE_SIGNATURE")),
    }


def _current_time(args: dict) -> str:
    """Current wall-clock time: ISO 8601 line + a human-readable line.

    The model cannot know the present time; it must ask this tool rather
    than guess. Optional `timezone` accepts an IANA zone (Europe/Madrid);
    the default is the machine's local zone.
    """
    name = str(args.get("timezone", "")).strip()
    override = os.environ.get("ANA_TIMEZONE", "").strip()
    if not name and override:
        name = override
    now = datetime.now(_timezone.utc).astimezone(
        ZoneInfo(name) if name else None)
    return (now.isoformat(timespec="seconds") + "\n"
            + now.strftime("%A, %d %B %Y, %H:%M"))


def _memory_save(args: dict) -> str:
    """Persist via ana.memory; degrade to a one-line note, never raise.

    run_agent only traps AgentError, so ANY other exception here kills
    the whole turn — Engram being down (or memory disabled) must read
    as text the model understands and stops retrying.
    """
    title = str(args.get("title") or "").strip()
    content = str(args.get("content") or "").strip()
    if not title or not content:
        return ("memory_save needs a non-empty `title` and `content` "
                "— nothing saved.")
    kind = str(args.get("type") or "").strip() or "fact"
    try:
        return memory.save(title, content, type=kind)
    except Exception as exc:
        return f"memoria no disponible: {exc.__class__.__name__}: {exc}"


def _memory_search(args: dict) -> str:
    """Search via ana.memory; same graceful-degradation contract."""
    query = str(args.get("query") or "").strip()
    if not query:
        return "memory_search needs a non-empty `query`."
    try:
        return memory.search(query)
    except Exception as exc:
        return f"memoria no disponible: {exc.__class__.__name__}: {exc}"


def _run_spectacle(cmd: list[str], timeout: float = 10.0):
    """Subprocess seam for the screenshot capture (tests patch this)."""
    return subprocess.run(cmd, capture_output=True, timeout=timeout,
                          check=False)


def _screenshot(args: dict) -> "str | ToolResult":
    """Capture the active monitor via spectacle; same-turn image on success.

    run_agent only traps AgentError, so a missing binary, a timeout or a
    non-zero exit MUST read as an honest one-line string the model
    can understand and stop retrying — never raise. On success the
    capture is downscaled by the SHARED /img pipeline (see
    _encode_image: ANA_IMAGE_MAX_PX, default 1024 — a raw 2560×1080
    would blow the provider's 120s vision-turn timeout) and rides back
    as a ToolResult so the agent loop injects it into THIS turn (no
    file is kept: the tmp capture is always deleted).
    """
    if shutil.which("spectacle") is None:
        return ("screenshot unavailable: spectacle is not installed — "
                "cannot capture the screen (do not retry this tool)")
    fd, path = tempfile.mkstemp(suffix=".png")
    os.close(fd)
    try:
        try:
            proc = _run_spectacle(
                ["spectacle", "-b", "-m", "-n", "-o", path], timeout=10)
        except subprocess.TimeoutExpired:
            return ("screenshot unavailable: spectacle timed out after "
                    "10s (do not retry this tool)")
        except OSError as exc:
            return (f"screenshot unavailable: {exc.__class__.__name__}: "
                    f"{exc} (do not retry this tool)")
        if proc.returncode != 0:
            err = proc.stderr or b""
            if isinstance(err, bytes):
                err = err.decode("utf-8", "replace")
            err = str(err).strip()
            return f"screenshot failed: {err or 'spectacle exited non-zero'}"
        try:
            with open(path, "rb") as fh:
                data = fh.read()
        except OSError as exc:
            return f"screenshot failed: {exc.__class__.__name__}: {exc}"
        if not data:
            return "screenshot failed: spectacle produced an empty file"
        # Downscale with the SAME helper the /img pipeline uses
        # (ana.cli._encode_image applies ANA_IMAGE_MAX_PX before base64).
        # The import is deferred: ana.cli imports THIS module at load
        # time, so a module-level import would create a cli↔tools cycle.
        from .cli import _encode_image
        try:
            image = _encode_image(path)
        except ValueError as exc:  # not an image: honest note, never raise
            return f"screenshot failed: {exc}"
        return ToolResult(
            text="Screenshot captured and attached to this conversation.",
            image=image)
    finally:
        try:
            os.unlink(path)
        except OSError:
            pass


def build_default_tools() -> ToolRegistry:
    reg = ToolRegistry()
    reg.register(Tool(
        name="web_search",
        description=(
            "Search the public web via Tavily and return top results "
            "(title, url, description). Use for anything you are unsure of, "
            "news, or current events."),
        parameters={
            "type": "object",
            "properties": {
                "query": {"type": "string",
                          "description": "search query"},
                "count": {"type": "integer", "minimum": 1, "maximum": 10,
                          "description": "number of results (default 5)"},
            },
            "required": ["query"],
        },
        fn=_tavily_search,
    ))
    reg.register(Tool(
        name="current_time",
        description=(
            "Returns the current date and time on this machine (ISO 8601 "
            "plus a human-readable line). Call this whenever the user asks "
            "about the time, date, weekday, or any now-relative schedule; "
            "never guess the time."),
        parameters={
            "type": "object",
            "properties": {"timezone": {
                "type": "string",
                "description": "optional IANA timezone name "
                               "(e.g. Europe/Madrid); default: local zone"}},
            "required": [],
        },
        fn=_current_time,
    ))
    for display, sub in HYPRCTL_READ_COMMANDS.items():
        reg.register(_hypr_tool(display, sub))
    reg.register(Tool(
        name="system_info",
        description="Return OS/kernel/python/load/CPU info about this machine.",
        parameters={"type": "object", "properties": {}},
        fn=_system_info,
    ))
    # screen capture: the image is injected into the SAME turn by the
    # agent loop (see ana/agent.py ToolResult) — never stored in memory
    reg.register(Tool(
        name="screenshot",
        description=(
            "Capture the current screen (the active monitor) and attach "
            "the screenshot to this conversation. Use when the user asks "
            "what is on the screen, what a program is showing or doing, "
            "or what something looks like. Never store the image in "
            "memory."),
        parameters={"type": "object", "properties": {}},
        fn=_screenshot,
    ))
    # Ana's own Engram-backed memory (namespace: ana-asistente)
    reg.register(Tool(
        name="memory_save",
        description=(
            "Save a durable fact or preference about the user (name, likes, "
            "dislikes, workflow, recurring context) to Ana's long-term "
            "memory so FUTURE sessions remember it. Do not store transient "
            "details of the current chat. The title is the key: saving with "
            "an EXISTING title updates that memory in place — when the user "
            "corrects or refines a fact, reuse the exact same title and "
            "never create a second record for the same topic."),
        parameters={
            "type": "object",
            "properties": {
                "title": {"type": "string",
                          "description": "short label for the memory"},
                "content": {"type": "string",
                            "description": "the durable fact or preference "
                                           "to remember"},
                "type": {"type": "string",
                         "enum": ["fact", "preference", "persona"],
                         "description": "kind of memory: fact | preference | "
                                        "persona (default: fact)"},
            },
            "required": ["title", "content"],
        },
        fn=_memory_save,
    ))
    reg.register(Tool(
        name="memory_search",
        description=(
            "Search Ana's long-term memory — recollections saved in previous "
            "sessions — and return the matches. Use it before answering "
            "personal questions to recall what you already know about the "
            "user."),
        parameters={
            "type": "object",
            "properties": {
                "query": {"type": "string",
                          "description": "what to look for"},
            },
            "required": ["query"],
        },
        fn=_memory_search,
    ))
    return reg
