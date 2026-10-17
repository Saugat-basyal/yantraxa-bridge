"""The one-line installers in apps/web/public, run for real against a fake `uv` and a fake
bridge in a throwaway HOME: every native call goes through the real shell's (or
PowerShell's parameter binder's) handling of its arguments.

install.ps1 0.2.0 died on every Windows machine before installing anything: `Invoke-Native`
was an advanced function (it has a [Parameter] attribute), so it had no `$args`, and
`Invoke-Native $uv --no-config …` failed with "A positional parameter cannot be found that
accepts argument '--no-config'". A dry run never reaches that call, so these tests run the
install itself. They need `pwsh` (PowerShell 7; `brew install powershell` on macOS) and skip
without it. Windows-only parts (the user PATH in the registry, the service) are not run here.
"""

from __future__ import annotations

import os
import shutil
import stat
import subprocess
from pathlib import Path

import pytest

from yantraxa_bridge import __version__

PUBLIC = Path(__file__).resolve().parents[2] / "apps" / "web" / "public"
PWSH = shutil.which("pwsh")

pytestmark = [
    pytest.mark.skipif(not PUBLIC.is_dir(), reason="apps/web/public is not here"),
    pytest.mark.skipif(os.name == "nt", reason="the fakes are POSIX scripts"),
]
needs_pwsh = pytest.mark.skipif(PWSH is None, reason="pwsh is not installed")


def script(path: Path, body: str) -> Path:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text("#!/bin/sh\n" + body)
    path.chmod(path.stat().st_mode | stat.S_IXUSR)
    return path


# Each argument on its own line, so a dropped, split or merged one shows.
ECHO_ARGS = 'for a in "$@"; do printf "[%s]\\n" "$a"; done\n'


@pytest.fixture
def fakes(tmp_path):
    """A HOME, and a PATH whose only `uv` is a fake that "installs" a fake bridge."""
    home = tmp_path / "home"
    home.mkdir()
    calls = tmp_path / "calls.log"
    bridge = script(
        tmp_path / "fake-bridge",
        f'echo "bridge $*" >> "{calls}"\n'
        'case "$1" in pair) echo "Connected as @fake.agent. It starts automatically when '
        'you log in." ;; esac\n',
    )
    script(
        tmp_path / "bin" / "uv",
        f'echo "uv $*" >> "{calls}"\n'
        'case "$*" in\n'
        '  *"tool install"*)\n'
        '    mkdir -p "$HOME/.local/bin"\n'
        f'    cp "{bridge}" "$HOME/.local/bin/yantraxa-bridge"\n'
        f'    cp "{bridge}" "$HOME/.local/bin/yantraxa-bridge.exe" ;;\n'
        f'  *"tool list"*) echo "yantraxa-bridge v{__version__}"; echo "- yantraxa-bridge" ;;\n'
        "esac\n",
    )
    env = {
        "HOME": str(home),
        "PATH": f"{tmp_path / 'bin'}:/usr/bin:/bin",
        "TERM": "dumb",
    }
    return home, calls, env


def lines(calls: Path) -> list[str]:
    return calls.read_text().splitlines()


def test_both_installers_pin_this_bridge_version():
    assert f'BRIDGE_VERSION="{__version__}"' in (PUBLIC / "install.sh").read_text()
    assert f'$BridgeVersion = "{__version__}"' in (PUBLIC / "install.ps1").read_text()


# ------------------------------------------------------------------ install.ps1


def pwsh(code: str, env: dict | None = None) -> subprocess.CompletedProcess:
    return subprocess.run(
        [PWSH, "-NoProfile", "-NonInteractive", "-Command", code],
        capture_output=True,
        text=True,
        timeout=120,
        env={**os.environ, **(env or {})},
        check=False,
    )


def invoke_native(tmp_path: Path, calls: str) -> subprocess.CompletedProcess:
    """Loads only `Invoke-Native` from install.ps1 (running the script would install) and
    makes `calls` with it against a command that prints its arguments."""
    echo = script(tmp_path / "echo-args", ECHO_ARGS)
    return pwsh(
        "$ErrorActionPreference = 'Stop'\n"
        f"$ast = [System.Management.Automation.Language.Parser]::ParseFile("
        f"'{PUBLIC / 'install.ps1'}', [ref]$null, [ref]$null)\n"
        "$function = $ast.Find({ param($n) $n -is "
        "[System.Management.Automation.Language.FunctionDefinitionAst] -and "
        "$n.Name -eq 'Invoke-Native' }, $true)\n"
        ". ([scriptblock]::Create($function.Extent.Text))\n"
        f"$echo = '{echo}'\n" + calls
    )


@needs_pwsh
def test_invoke_native_passes_double_dash_and_single_dash_arguments(tmp_path):
    done = invoke_native(
        tmp_path,
        'Invoke-Native $echo --no-config tool install "yantraxa-bridge==0.2.1" --python 3.12\n'
        "Write-Output '---'\n"
        "Invoke-Native $echo pair -DryRun -y --server 'https://a b'\n",
    )
    assert done.returncode == 0, done.stderr
    first, second = done.stdout.split("---\n")
    assert first.split() == [
        "[--no-config]",
        "[tool]",
        "[install]",
        "[yantraxa-bridge==0.2.1]",
        "[--python]",
        "[3.12]",
    ]
    assert second.splitlines() == ["[pair]", "[-DryRun]", "[-y]", "[--server]", "[https://a b]"]


@needs_pwsh
def test_invoke_native_with_an_array_passes_what_the_binder_would_eat(tmp_path):
    """Loose, `--` is dropped and `-C` binds to -Command; inside an array nothing is
    touched, which is why the script passes arrays."""
    done = invoke_native(
        tmp_path,
        "Invoke-Native $echo @('--', '-C', '-Command', 'x y', '-Arguments')\n"
        "Invoke-Native $echo @('one')\n"
        "Invoke-Native $echo\n",
    )
    assert done.returncode == 0, done.stderr
    assert done.stdout.splitlines() == [
        "[--]",
        "[-C]",
        "[-Command]",
        "[x y]",
        "[-Arguments]",
        "[one]",  # one element stays one argument, not three characters
    ]


@needs_pwsh
def test_invoke_native_throws_when_the_command_fails(tmp_path):
    done = invoke_native(tmp_path, "Invoke-Native /usr/bin/false @('x')\n")
    assert done.returncode != 0
    assert "/usr/bin/false exited with code 1." in done.stdout + done.stderr


@needs_pwsh
@pytest.mark.parametrize("no_service", [False, True])
def test_install_ps1_installs_and_pairs_without_asking(fakes, no_service):
    home, calls, env = fakes
    flags = " -NoService" if no_service else ""
    done = pwsh(
        "& ([scriptblock]::Create((Get-Content -Raw "
        f"'{PUBLIC / 'install.ps1'}'))) claude-code{flags}",
        env,
    )
    assert done.returncode == 0, done.stdout + done.stderr
    bridge = home / ".local" / "bin" / "yantraxa-bridge.exe"
    pair = ["bridge pair --server https://api.yantraxa.com --adapter claude-code --yes"]
    if no_service:
        pair[0] += " --no-service"
    assert lines(calls) == [
        f"uv --no-config tool install yantraxa-bridge=={__version__} --python 3.12",
        "uv --no-config tool list",
        "bridge --help",
        *pair,
    ]
    assert bridge.is_file()
    assert "Connected as @fake.agent." in done.stdout
    assert "[Y/n]" not in done.stdout


@needs_pwsh
def test_install_ps1_without_an_adapter_says_how_to_pair(fakes):
    _home, calls, env = fakes
    done = pwsh(f"& '{PUBLIC / 'install.ps1'}'", env)
    assert done.returncode == 0, done.stdout + done.stderr
    assert "Installed successfully. Next, run:" in done.stdout
    assert not any(line.startswith("bridge pair") for line in lines(calls))


@needs_pwsh
@pytest.mark.parametrize("adapter", ["", "claude-code"])
def test_install_ps1_dry_run_changes_nothing(fakes, adapter):
    home, calls, env = fakes
    done = pwsh(f"& '{PUBLIC / 'install.ps1'}' {adapter} -DryRun", env)
    assert done.returncode == 0, done.stdout + done.stderr
    assert "Dry run: no files, settings, or services were changed." in done.stdout
    assert not calls.exists()
    # pwsh keeps its own files in ~/.cache and ~/.local/share/powershell; the script wrote none.
    assert not (home / ".yantraxa").exists() and not (home / ".local" / "bin").exists()


# ------------------------------------------------------------------ install.sh


@pytest.mark.parametrize("no_service", [False, True])
def test_install_sh_installs_and_pairs_without_asking(fakes, no_service):
    _home, calls, env = fakes
    args = ["sh", str(PUBLIC / "install.sh"), "claude-code"]
    if no_service:
        args.append("--no-service")
    done = subprocess.run(
        args, capture_output=True, text=True, timeout=60, env=env, check=False, stdin=None
    )
    assert done.returncode == 0, done.stdout + done.stderr
    pair = "bridge pair --server https://api.yantraxa.com --adapter claude-code --yes"
    assert lines(calls) == [
        f"uv --no-config tool install yantraxa-bridge=={__version__} --python 3.12",
        "uv --no-config tool list",
        "bridge --help",
        pair + (" --no-service" if no_service else ""),
    ]
    assert "Connected as @fake.agent." in done.stdout
    assert "[Y/n]" not in done.stdout


def test_install_sh_dry_run_plan(fakes):
    _home, calls, env = fakes
    done = subprocess.run(
        ["sh", str(PUBLIC / "install.sh"), "--dry-run", "claude-code"],
        capture_output=True,
        text=True,
        timeout=60,
        env=env,
        check=False,
    )
    assert done.returncode == 0
    assert "Start the bridge now and at every login" in done.stdout
    assert not calls.exists()
