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
    def reader(args: dict) -> str:
        proc = subprocess.run(
            ["hyprctl", subcommand, "-j"], capture_output=True, text=True,
            timeout=10, check=False)
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
    for display, sub in HYPRCTL_READ_COMMANDS.items():
        reg.register(_hypr_tool(display, sub))
    reg.register(Tool(
        name="system_info",
        description="Return OS/kernel/python/load/CPU info about this machine.",
        parameters={"type": "object", "properties": {}},
        fn=_system_info,
    ))
    return reg
