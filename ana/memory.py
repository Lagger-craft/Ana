"""Ana's own persistent memory, backed by the local Engram server.

Everything Ana stores lives under her OWN namespace (project
`ana-asistente`, scope `project`), isolated from the user's memories on
the same server. Two surfaces: `session_context()` (a short markdown
context prepended to the system prompt at session start) and the
`save`/`search` pair the model drives through the `memory_save` /
`memory_search` tools.

Graceful degradation is mandatory: every call has a short timeout and
failures raise MemoryUnavailable — never a raw httpx error — so a slow
or dead Engram degrades to a one-line note instead of a dead turn.

Env knobs:
- ANA_ENGRAM_URL       server base (default http://127.0.0.1:7437)
- ANA_MEMORY_PROJECT   namespace (default ana-asistente)
- ANA_MEMORY=0         disables everything (empty context, note-only tools)

Test seam: `_get` / `_post` are the whole transport; tests monkeypatch
them, and the real implementations refuse to touch the network while a
pytest test is running (see `_in_pytest`).
"""

from __future__ import annotations

import os
import uuid
from pathlib import Path
from typing import Any

import httpx

DEFAULT_URL = "http://127.0.0.1:7437"
DEFAULT_PROJECT = "ana-asistente"
TIMEOUT = 3.0  # seconds — a slow Engram must never stall a turn

# One Ana session per process; registered lazily before the first write
# and ended best-effort by shutdown() at exit.
_session_id: str | None = None
_session_registered = False
_context_cache: str | None = None  # None = not fetched yet


class MemoryUnavailable(Exception):
    """Engram is down, slow or misbehaving; callers degrade to plain text."""


def _enabled() -> bool:
    """``ANA_MEMORY=0`` turns Ana's memory off entirely."""
    return os.environ.get("ANA_MEMORY", "").strip() != "0"


def _base_url() -> str:
    return ((os.environ.get("ANA_ENGRAM_URL", "").strip() or DEFAULT_URL)
            .rstrip("/"))


def _project() -> str:
    return os.environ.get("ANA_MEMORY_PROJECT", "").strip() or DEFAULT_PROJECT


def _data_dir() -> str:
    """Ana's state dir (XDG_DATA_HOME/ana) — the session's directory."""
    xdg = os.environ.get("XDG_DATA_HOME", "").strip()
    base = Path(xdg) if xdg else Path.home() / ".local" / "share"
    return str(base / "ana")


def _in_pytest() -> bool:
    """True while a pytest test is running.

    The suite fakes the `_get`/`_post` seams instead; the real transport
    refuses to run under pytest so no test can ever reach the live
    Engram server (it is up on the machine where the suite runs).
    """
    return os.environ.get("PYTEST_CURRENT_TEST") is not None


def _get(path: str, **params) -> Any:
    """GET seam: decoded JSON, or MemoryUnavailable (never raw httpx)."""
    if _in_pytest():
        raise MemoryUnavailable("engram: HTTP disabled during tests")
    try:
        resp = httpx.get(_base_url() + path, params=params, timeout=TIMEOUT)
        resp.raise_for_status()
        return resp.json()
    except Exception as exc:
        raise MemoryUnavailable(
            f"engram GET {path}: {exc.__class__.__name__}: {exc}") from exc


def _post(path: str, body: dict) -> Any:
    """POST seam: decoded JSON, or MemoryUnavailable (never raw httpx)."""
    if _in_pytest():
        raise MemoryUnavailable("engram: HTTP disabled during tests")
    try:
        resp = httpx.post(_base_url() + path, json=body, timeout=TIMEOUT)
        resp.raise_for_status()
        return resp.json()
    except Exception as exc:
        raise MemoryUnavailable(
            f"engram POST {path}: {exc.__class__.__name__}: {exc}") from exc


def _ensure_session() -> str:
    """Register this process's Ana session once, before the first write."""
    global _session_id, _session_registered
    if _session_id is None:
        _session_id = f"ana-{uuid.uuid4()}"
    if not _session_registered:
        _post("/sessions", {
            "id": _session_id,
            "project": _project(),
            "directory": _data_dir(),
            "ownership_mode": "project_owned",
        })
        _session_registered = True
    return _session_id


def save(title: str, content: str, type: str = "fact") -> str:
    """Persist one durable observation; raises MemoryUnavailable on failure.

    Returns a short confirmation suitable as a tool result; with memory
    disabled (``ANA_MEMORY=0``) a one-line note and no HTTP at all.
    """
    if not _enabled():
        return "memoria deshabilitada (ANA_MEMORY=0): nada se guardó."
    session_id = _ensure_session()   # registration failure raises first
    _post("/observations", {
        "session_id": session_id,
        "title": str(title),
        "content": str(content),
        "type": str(type or "fact"),
        "project": _project(),
        "scope": "project",
    })
    return f"guardado: {title}"


def search(query: str, limit: int = 5) -> str:
    """Format matching memories as markdown lines; raises MemoryUnavailable.

    Hits read ``- [type] **title**: content``; no hits is an honest
    one-line message. Read-only: search never registers or writes.
    """
    if not _enabled():
        return "memoria deshabilitada (ANA_MEMORY=0): no busqué recuerdos."
    limit = int(limit)
    hits = _get("/search", q=str(query), project=_project(), limit=limit)
    if not isinstance(hits, list) or not hits:
        return f"no encontré recuerdos para {query!r}."
    lines = [f"- [{hit.get('type', 'fact')}] **{hit.get('title', '')}**: "
             f"{hit.get('content', '')}" for hit in hits[:limit]]
    return "\n".join(lines)


def session_context() -> str:
    """Ana's memory context for the system prompt, cached per process.

    Fetched at most once per process; ``""`` when memory is disabled or
    Engram is unusable — the turn engine must keep working either way.
    """
    global _context_cache
    if not _enabled():
        return ""
    if _context_cache is None:
        try:
            data = _get("/context", project=_project(), max_bytes=2000)
        except MemoryUnavailable:
            _context_cache = ""
        else:
            _context_cache = (str(data.get("context", ""))
                              if isinstance(data, dict) else "")
    return _context_cache


def shutdown() -> None:
    """End this process's Engram session best-effort; never raises."""
    global _session_registered
    if not _session_registered or not _session_id:
        return
    try:
        _post(f"/sessions/{_session_id}/end", {"summary": ""})
    except Exception:
        pass  # the process is exiting; a lost end-call must not break it
    finally:
        _session_registered = False
