"""ana/overlay.py — non-GTK helper: layer-shell lib discovery (portability).

The overlay LD_PRELOADs libgtk4-layer-shell so it links before
libwayland-client. Arch ships the `.so` symlink; runtime-only installs
(Void labwc) often ship only `.so.0` — or nothing. Discovery must try
the candidates and return None instead of assuming Arch's layout.
Tests never import gi (GTK stays out of the suite).
"""

import pytest

import ana.overlay as overlay_mod


def test_prefers_plain_so_symlink(tmp_path, monkeypatch):
    lib = tmp_path / "libgtk4-layer-shell.so"
    lib.write_bytes(b"elf")
    found = overlay_mod.layer_shell_lib(tmp_path)
    assert found == str(lib)


def test_falls_back_to_soname_when_no_symlink(tmp_path):
    """Void-style runtime package: only libgtk4-layer-shell.so.0 exists."""
    lib = tmp_path / "libgtk4-layer-shell.so.0"
    lib.write_bytes(b"elf")
    found = overlay_mod.layer_shell_lib(tmp_path)
    assert found == str(lib)


def test_returns_none_when_absent(tmp_path):
    assert overlay_mod.layer_shell_lib(tmp_path) is None


def test_default_root_is_usr_lib():
    import os
    if os.path.exists("/usr/lib/libgtk4-layer-shell.so.0"):
        assert overlay_mod.layer_shell_lib() is not None
    else:
        pytest.skip("gtk4-layer-shell not installed on this machine")
