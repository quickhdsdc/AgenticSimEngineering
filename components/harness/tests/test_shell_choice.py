r"""Which shell the harness runs commands in, and why the first answer is wrong.

A live run on Windows failed every step of a task while reporting `shell:
bash`. `shutil.which("bash")` had returned `C:\Windows\System32\bash.exe` --
the WSL relay, which is not a shell on a machine with no distro installed, and
would be the *wrong* shell on one that has: a different machine, without the
venv, without `aas`, without the repository.

The lesson is narrow and worth pinning: "a file with that name exists on PATH"
is not the same question as "there is a working shell here", and on Windows
the two have different answers.
"""

from __future__ import annotations

import os
from pathlib import Path
from typing import List

import pytest

from a4s.environments import local


def _fake_shell(directory: Path, name: str, works: bool) -> Path:
    """A stand-in executable that either echoes its argument or fails like WSL."""
    directory.mkdir(parents=True, exist_ok=True)
    if os.name == "nt":
        script = directory / f"{name}.bat"
        body = "@echo %~2\n" if works else (
            "@echo WSL (1 - Relay) ERROR: execvpe(/bin/bash) failed 1>&2\n@exit /b 1\n"
        )
    else:
        script = directory / name
        body = ('#!/bin/sh\necho "$2"\n' if works else
                '#!/bin/sh\necho "execvpe(/bin/bash) failed" >&2\nexit 1\n')
    script.write_text(body, encoding="utf-8")
    script.chmod(0o755)
    return script


def test_a_shell_that_does_not_run_commands_is_not_chosen(
    tmp_path: Path, monkeypatch
) -> None:
    """The defect, reduced: a broken `bash` earlier on PATH than a working one.

    `shutil.which` returns the broken one, and nothing downstream looks again.
    """
    broken = tmp_path / "broken"
    working = tmp_path / "working"
    _fake_shell(broken, "bash", works=False)
    _fake_shell(working, "bash", works=True)
    monkeypatch.setenv("PATH", os.pathsep.join([str(broken), str(working)]))
    monkeypatch.delenv("A4S_SHELL", raising=False)

    chosen = local.find_shell()

    assert Path(chosen[0]).parent == working


def test_the_wsl_relay_is_skipped_by_where_it_lives(monkeypatch) -> None:
    """Not by probing, and deliberately so.

    A machine that *does* have a WSL distro passes the probe and is still the
    wrong answer: it would run every command inside another operating system,
    where this venv, `aas` and the repository do not exist.
    """
    assert local._looks_like_a_wsl_relay(r"C:\Windows\System32\bash.exe")
    assert local._looks_like_a_wsl_relay(
        r"C:\Users\x\AppData\Local\Microsoft\WindowsApps\bash.exe")
    assert not local._looks_like_a_wsl_relay(
        r"C:\Users\x\AppData\Local\Programs\Git\usr\bin\bash.exe")
    assert not local._looks_like_a_wsl_relay("/usr/bin/bash")


def test_every_candidate_on_path_is_considered_not_just_the_first(
    tmp_path: Path, monkeypatch
) -> None:
    """The working shell was three PATH entries below the broken one."""
    directories: List[str] = []
    for index in range(3):
        broken = tmp_path / f"broken{index}"
        _fake_shell(broken, "bash", works=False)
        directories.append(str(broken))
    working = tmp_path / "working"
    _fake_shell(working, "bash", works=True)
    directories.append(str(working))

    monkeypatch.setenv("PATH", os.pathsep.join(directories))
    monkeypatch.delenv("A4S_SHELL", raising=False)

    assert Path(local.find_shell()[0]).parent == working


def test_an_explicit_override_is_taken_at_its_word(monkeypatch) -> None:
    """Someone who sets `$A4S_SHELL` means it. `check_shell` still probes it,
    so `a4s doctor` can say it is broken rather than the run finding out."""
    monkeypatch.setenv("A4S_SHELL", "/opt/custom/shell")

    assert local.find_shell() == ["/opt/custom/shell", "-c"]


def test_doctor_can_tell_a_working_shell_from_a_present_one(
    tmp_path: Path, monkeypatch
) -> None:
    """`a4s doctor` passed every check on the run that then failed every
    command. The shell is now one of the things it reports."""
    broken = tmp_path / "broken"
    _fake_shell(broken, "bash", works=False)
    monkeypatch.setenv("PATH", str(broken))
    monkeypatch.delenv("A4S_SHELL", raising=False)

    report = local.check_shell()

    if os.name == "nt":
        # No POSIX shell survives the probe, so the fallback is cmd.exe: it
        # runs, and is exactly the case the warning exists for.
        assert report["posix"] is False
    else:
        assert report["works"] is False


@pytest.mark.skipif(os.name != "nt", reason="the relay only exists on Windows")
def test_a_real_windows_host_does_not_end_up_in_wsl() -> None:
    """The end of the story, against this machine's actual PATH."""
    report = local.check_shell()

    assert not local._looks_like_a_wsl_relay(str(report["path"]))
    assert report["works"] is True
    assert report["posix"] is True
