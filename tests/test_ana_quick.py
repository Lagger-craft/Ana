"""scripts/ana-quick terminal resolution (portability: any session).

The launcher must open the TUI with `xdg-terminal-exec` when present and
fall back to a known terminal otherwise, so a machine without
xdg-terminal-exec (e.g. a fresh Void/labwc install) still gets a window.
Tests stub the whole PATH: pgrep reports a live overlay (so no `nohup`
launch of the real overlay) and each terminal stub records its argv.
"""

import os
import stat
import subprocess
from pathlib import Path

import pytest

REPO = Path(__file__).resolve().parent.parent
SCRIPT = REPO / "scripts" / "ana-quick"


def _stub(dirpath: Path, name: str, argv_log: Path | None = None) -> None:
    """Executable stand-in: exit 0 (pgrep: overlay alive) or dump argv."""
    # absolute shebang: tests restrict PATH to this dir on purpose
    body = "#!/usr/bin/bash\n"
    if argv_log is not None:
        body += f'printf "%s\\n" "$@" >"{argv_log}"\n'
    body += "exit 0\n"
    path = dirpath / name
    path.write_text(body, encoding="utf-8")
    path.chmod(path.stat().st_mode | stat.S_IEXEC)


@pytest.fixture
def sandbox(tmp_path):
    (tmp_path / "pgrep").write_text("#!/usr/bin/bash\nexit 0\n",
                                    encoding="utf-8")
    (tmp_path / "pgrep").chmod(0o755)
    # the script needs coreutils (dirname); everything else stays stubbed
    # so terminal discovery sees ONLY what each test provides
    (tmp_path / "dirname").symlink_to("/usr/bin/dirname")
    return tmp_path


def test_prefers_xdg_terminal_exec(sandbox):
    argv_log = sandbox / "argv.txt"
    _stub(sandbox, "xdg-terminal-exec", argv_log)
    env = {**os.environ, "PATH": str(sandbox)}
    proc = subprocess.run(["/usr/bin/bash", str(SCRIPT)], env=env,
                          capture_output=True, text=True, timeout=30)
    assert proc.returncode == 0, proc.stderr
    args = argv_log.read_text(encoding="utf-8").splitlines()
    assert args[0].endswith("/ana")
    assert args[1] == "tui"


def test_falls_back_to_known_terminal(sandbox):
    """No xdg-terminal-exec → first of foot/alacritty/kitty/ghostty/wezterm."""
    argv_log = sandbox / "argv.txt"
    _stub(sandbox, "foot", argv_log)
    env = {**os.environ, "PATH": str(sandbox)}
    proc = subprocess.run(["/usr/bin/bash", str(SCRIPT)], env=env,
                          capture_output=True, text=True, timeout=30)
    assert proc.returncode == 0, proc.stderr
    args = argv_log.read_text(encoding="utf-8").splitlines()
    assert args[0].endswith("/ana")
    assert args[1] == "tui"


def test_no_terminal_at_all_fails_loudly(sandbox):
    """Empty PATH except pgrep: the script must state the problem (its own
    words), not exit with a bare 'command not found'."""
    env = {**os.environ, "PATH": str(sandbox)}
    proc = subprocess.run(["/usr/bin/bash", str(SCRIPT)], env=env,
                          capture_output=True, text=True, timeout=30)
    assert proc.returncode != 0
    assert "no terminal emulator" in (proc.stderr + proc.stdout).lower()
