"""`$A4S_HOME` is expanded, and the reason that is worth a test.

`.env` on this project carries `A4S_HOME=~/.a4s`, which is how a person writes
a path and not how `Path` reads one: without expansion it is a *relative*
directory literally named `~`, created under whatever the working directory
happened to be. The observable symptom was the state splitting in two -- the
retrieval index under `./~/.a4s/index/` in the project folder, the ledger and
the keys under the real home -- because the two are read at different moments
and only one of them saw the variable.

Split state is the failure that matters, not the odd folder name: a provenance
query answers from one ledger, and the write it is being asked about may have
gone to the other.

**Run in a subprocess, deliberately.** The values under test are module-level
constants, so exercising them in-process means `importlib.reload`, and a reload
swaps the class objects out from under every other module that already imported
a name from this one -- `pytest.raises(IndexError_)` in `test_index.py` stops
matching the exception `Index.load` now raises, and the failure appears in a
file this one never touched. A subprocess has its own interpreter and cannot
reach across.
"""

from __future__ import annotations

import json
import os
import subprocess
import sys
from pathlib import Path

PROBE = """
import json
from a4s import cli
from a4s.aasctl import index
from a4s.kernel import approval
print(json.dumps({
    "home": str(cli.DEFAULT_HOME),
    "index": str(index.DEFAULT_INDEX_HOME),
    "scopes": str(approval.DEFAULT_SCOPES),
}))
"""


def _paths_with_home(value: str) -> dict:
    environment = dict(os.environ, A4S_HOME=value)
    result = subprocess.run(
        [sys.executable, "-c", PROBE],
        capture_output=True, text=True, env=environment,
        cwd=str(Path(__file__).resolve().parent.parent),
    )
    assert result.returncode == 0, result.stderr
    return json.loads(result.stdout.strip().splitlines()[-1])


def test_a_tilde_in_a4s_home_becomes_the_real_home() -> None:
    paths = _paths_with_home("~/.a4s")

    for name, path in paths.items():
        assert "~" not in path, f"{name} keeps a literal tilde: {path}"
        assert Path(path).is_absolute(), f"{name} is relative to the cwd: {path}"

    assert Path(paths["home"]) == Path.home() / ".a4s"


def test_an_explicit_home_is_left_alone(tmp_path: Path) -> None:
    """Expansion must not rewrite a path that was already absolute -- a
    deployment pointing at `/var/lib/a4s` means it."""
    paths = _paths_with_home(str(tmp_path))

    assert Path(paths["home"]) == tmp_path
    assert Path(paths["index"]) == tmp_path / "index"
