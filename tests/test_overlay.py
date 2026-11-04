"""ana/overlay.py — system-python re-exec must carry the environment.

The overlay re-execs /usr/bin/python3 so PyGTK (system site-packages)
loads, but the `ana` package lives in this repo: PYTHONPATH is what
lets the re-exec'd interpreter `import ana` from ANY cwd (the KDE
global hotkey launches with a KIO-job cwd, not the repo root).
os.execl IGNORES the env dict — only os.execve applies it — so the
re-exec must go through os.execve with the built env.
"""

from __future__ import annotations

import os
import sys
from pathlib import Path

import ana.overlay as overlay_mod


def test_reexec_system_python_uses_execve_with_pythonpath_and_preload(
        monkeypatch):
    captured: dict = {}

    def fake_execve(path, argv, env):
        captured["path"] = path
        captured["argv"] = argv
        captured["env"] = env

    def fake_execl(*_args, **_kwargs):
        # RED safety net: a REAL os.execl here would replace the pytest
        # process (GUI window / import failure) instead of failing cleanly
        raise AssertionError("os.execl called — it drops env, "
                             "the re-exec must use os.execve")

    monkeypatch.setattr(overlay_mod.os, "execve", fake_execve)
    monkeypatch.setattr(overlay_mod.os, "execl", fake_execl)
    # no extra CLI args, so sys.argv[1:] is empty and the re-exec argv
    # is exactly ["python3", "-m", "ana.overlay"]
    monkeypatch.setattr(sys, "argv", ["ana"])
    # the caller (run_overlay) exports these on os.environ BEFORE
    # invoking the re-exec; they must reach the child interpreter
    monkeypatch.setenv("LD_PRELOAD", "/usr/lib/libgtk4-layer-shell.so")
    monkeypatch.setenv("_ANA_LAYER_PRELOAD", "1")
    # a pre-existing PYTHONPATH: the repo root must be PREPENDED (first),
    # not appended or dropped
    monkeypatch.setenv("PYTHONPATH", "/sentinel/site-packages")

    overlay_mod._reexec_system_python()

    assert captured["path"] == "/usr/bin/python3"
    assert captured["argv"] == ["python3", "-m", "ana.overlay",
                                *sys.argv[1:]]
    # sys.argv was patched to ["ana"] above -> exactly:
    # ["python3", "-m", "ana.overlay"]
    repo_root = str(Path(overlay_mod.__file__).resolve().parent.parent)
    assert captured["env"]["PYTHONPATH"].split(os.pathsep)[0] == repo_root
    assert captured["env"]["LD_PRELOAD"] == "/usr/lib/libgtk4-layer-shell.so"
    assert captured["env"]["_ANA_LAYER_PRELOAD"] == "1"
