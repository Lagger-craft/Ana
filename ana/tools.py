"""Default Ana tools: web search, Hyprland read-only queries, system info.

Phase 1 is read-only: hyprctl is queried, never dispatched; no shell
execution tool exists yet (needs a permission model first).
"""

from __future__ import annotations

import json
import os
import subprocess
import sys

import httpx

from datetime import datetime, timezone as _timezone
from zoneinfo import ZoneInfo

from .agent import Tool, ToolRegistry

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
    return reg
