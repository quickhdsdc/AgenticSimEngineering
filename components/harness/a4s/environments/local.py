"""Run commands on the orchestrator host."""

from __future__ import annotations

import os
import shutil
import signal
import subprocess
import sys
import time
from pathlib import Path
from typing import Dict, List, Optional

from .base import Environment, ExecResult, clamp


# Windows ships two `bash.exe` that are not shells on this machine at all:
# the WSL relay in System32 and its App Execution Alias under WindowsApps.
# Both are on PATH ahead of Git Bash when a command prompt is the parent, and
# both are the wrong answer twice over -- without a distro installed they fail
# with `execvpe(/bin/bash) failed`, and *with* one they would run every command
# inside a different machine that has no venv, no `aas`, and no repository.
_WSL_RELAY_DIRS = ("system32", "windowsapps")


def _looks_like_a_wsl_relay(path: str) -> bool:
    return Path(path).parent.name.lower() in _WSL_RELAY_DIRS


def _shell_works(argv: List[str]) -> bool:
    """Does this shell actually run a command and hand back the output.

    A probe rather than a guess. `shutil.which` answers "is there a file with
    that name", which on Windows is a different question from "is there a shell
    here": the first candidate on PATH cost a whole live run, failing every
    step with a WSL error while the harness reported `shell: bash`.
    """
    try:
        probe = subprocess.run(
            [*argv, "echo a4s-shell-probe"],
            capture_output=True, text=True, timeout=10,
        )
    except (OSError, subprocess.SubprocessError):
        return False
    return probe.returncode == 0 and "a4s-shell-probe" in (probe.stdout or "")


def _posix_candidates() -> List[str]:
    """Every bash and sh on PATH, in PATH order, minus the WSL relays.

    Every one rather than `shutil.which`'s first: the first is exactly what
    went wrong, and the working shell was three entries further down.
    """
    found: List[str] = []
    directories = (os.getenv("PATH") or "").split(os.pathsep)
    for name in ("bash", "sh"):
        for directory in directories:
            if not directory:
                continue
            resolved = shutil.which(name, path=directory)
            if resolved and not _looks_like_a_wsl_relay(resolved) and resolved not in found:
                found.append(resolved)
    return found


def find_shell() -> List[str]:
    """Choose the shell commands actually run in, having checked that it runs.

    ``subprocess.Popen(shell=True)`` means ``cmd.exe`` on Windows. The whole
    design assumes a POSIX shell -- the skills are full of pipelines, ``$VAR``
    and heredocs -- and cmd.exe does not merely refuse those, it **exits 0 with
    empty stdout and empty stderr**. A live run lost its first two steps to
    exactly that: the agent wrote a correct ``bash -lc`` script, got back
    nothing at all, and concluded the files were missing.

    ``$A4S_SHELL`` overrides and is taken at its word -- someone who sets it
    means it -- though it is still probed so `a4s doctor` can say it is broken.
    Otherwise every bash and sh on PATH is probed in order and the first
    working one wins; cmd.exe only when the host truly has none.
    """
    override = os.getenv("A4S_SHELL")
    if override:
        return [override, "-c"]
    for candidate in _posix_candidates():
        if _shell_works([candidate, "-c"]):
            return [candidate, "-c"]
    if sys.platform == "win32":
        return [os.getenv("COMSPEC", "cmd.exe"), "/c"]
    return ["/bin/sh", "-c"]


def check_shell() -> Dict[str, object]:
    """What `a4s doctor` reports: which shell, and whether it works.

    Worth reporting because the failure it catches is silent from the outside.
    The run that found this printed `shell: bash`, passed every configuration
    check, and then failed every single command.
    """
    argv = find_shell()
    return {
        "argv": argv,
        "path": argv[0],
        "name": Path(argv[0]).stem,
        "works": _shell_works(argv),
        "posix": Path(argv[0]).stem.lower() not in ("cmd", "powershell", "pwsh"),
        "overridden": bool(os.getenv("A4S_SHELL")),
        "rejected": [
            found for found in
            (shutil.which(n, path=d) for n in ("bash", "sh")
             for d in (os.getenv("PATH") or "").split(os.pathsep) if d)
            if found and _looks_like_a_wsl_relay(found)
        ],
    }


def script_dir() -> Optional[str]:
    """Where this installation's console scripts live.

    ``aas`` has to be on the child's PATH or nothing works, and the first live
    run of this harness failed precisely because it was not: the agent fell back
    to hand-rolling base64-encoded ``curl`` calls and burned its whole step
    budget. Deriving it from the running interpreter means a venv works without
    the operator having exported anything.
    """
    base = Path(sys.executable).parent
    for candidate in (base, base / "Scripts", base / "bin"):
        if (candidate / "aas").exists() or (candidate / "aas.exe").exists():
            return str(candidate)
    return None


class LocalEnvironment(Environment):
    name = "local"

    def __init__(self, cwd: Optional[str] = None, env: Optional[Dict[str, str]] = None) -> None:
        self.cwd = cwd or os.getcwd()
        self.shell = find_shell()
        merged = {**os.environ, **(env or {})}
        # Git Bash rewrites arguments that look like POSIX paths into their
        # Windows equivalents before the program sees them. A bare `/` becomes
        # `C:/Users/.../Git/`, which is how `aas el ls <sm> /` reached BaSyx as
        # a request for an element at the Git installation directory and came
        # back a 400 with a Tomcat error page in it.
        #
        # Measured, not anticipated: it cost a live run one step, and the agent
        # recovered by trying a different verb -- so the harness would never
        # have heard about it. Turned off for the agent's shell only; the
        # arguments it passes are AAS identifiers and idShort paths, and none
        # of them wants translating.
        merged.setdefault("MSYS_NO_PATHCONV", "1")
        merged.setdefault("MSYS2_ARG_CONV_EXCL", "*")
        scripts = script_dir()
        if scripts:
            existing = merged.get("PATH", "")
            if scripts not in existing.split(os.pathsep):
                merged["PATH"] = scripts + os.pathsep + existing
        self.env = merged

    @property
    def shell_name(self) -> str:
        return Path(self.shell[0]).stem

    def run(
        self,
        command: str,
        *,
        cwd: Optional[str] = None,
        env: Optional[Dict[str, str]] = None,
        timeout: float = 120.0,
    ) -> ExecResult:
        started = time.monotonic()
        merged = {**self.env, **(env or {})}

        # New process group so a timeout kills the whole tree. A hung child
        # of a killed shell keeps a serial port or an OPC UA session open,
        # and the next command then fails for an unrelated-looking reason.
        popen_kwargs: Dict[str, object] = {}
        if sys.platform == "win32":
            popen_kwargs["creationflags"] = subprocess.CREATE_NEW_PROCESS_GROUP
        else:
            popen_kwargs["start_new_session"] = True

        process = subprocess.Popen(
            [*self.shell, command],
            cwd=cwd or self.cwd,
            env=merged,
            stdout=subprocess.PIPE,
            stderr=subprocess.PIPE,
            text=True,
            encoding="utf-8",
            errors="replace",
            **popen_kwargs,  # type: ignore[arg-type]
        )

        try:
            stdout, stderr = process.communicate(timeout=timeout)
            exit_code = process.returncode
        except subprocess.TimeoutExpired:
            self._kill_tree(process)
            stdout, stderr = process.communicate()
            exit_code = 124
            stderr = f"{stderr}\n[timed out after {timeout}s]"

        clamped_out, cut_out = clamp(stdout or "")
        clamped_err, cut_err = clamp(stderr or "")
        return ExecResult(
            exit_code=exit_code,
            stdout=clamped_out,
            stderr=clamped_err,
            truncated=cut_out or cut_err,
            duration_s=round(time.monotonic() - started, 3),
            environment=self.name,
        )

    @staticmethod
    def _kill_tree(process: subprocess.Popen) -> None:
        try:
            if sys.platform == "win32":
                process.send_signal(signal.CTRL_BREAK_EVENT)
                process.kill()
            else:
                os.killpg(os.getpgid(process.pid), signal.SIGKILL)
        except (ProcessLookupError, PermissionError, OSError):
            process.kill()
