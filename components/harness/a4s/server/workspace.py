"""Where the wizard's intermediate files live.

The DPP wizard is a four-stage pipeline whose stages do not run in one process:
the browser uploads a datasheet, an agent extracts entities from it later, a
person corrects them in the table, the agent matches them later still. Each
stage hands the next one a *file*, and the routes in `wizard_routes` are the
handles on those files. That is the predecessor's design and it is kept,
because the alternative -- holding the pipeline in server memory -- loses
everything when the process restarts, and a datasheet extraction is minutes of
model time.

Three directories under `$A4S_HOME/web/`, or `$A4S_WEB_WORKSPACE` if a
deployment wants them elsewhere:

    uploads/      what the browser sent, under its own name
    extractions/  entities, as the extractor wrote them and as the user fixed them
    matches/      match plans, as the matcher wrote them and as the user fixed them

**Every name that arrives from a request is checked, not trusted.** These
routes take a filename in the URL and turn it into a path, which is the exact
shape of a directory-traversal hole: `../../.ssh/id_rsa` reads a key,
`..%2f..%2fx` writes one. `safe_name` refuses anything with a separator or a
parent reference in it, and `under` re-checks the resolved path against the
directory afterwards -- two checks because the first is about the name and the
second is about what the filesystem did with it, and symlinks make those
different questions.
"""

from __future__ import annotations

import os
import re
from pathlib import Path
from typing import Optional

# Deliberately narrow, but wide enough for real datasheets: spaces, brackets
# and commas are ordinary in a filename a supplier sent, and rejecting them
# would make the wizard refuse the documents it exists to read. Everything
# outside this set is either a mistake or an attempt.
_UNSAFE = re.compile(r"[^A-Za-z0-9 ._@()\[\]{}+,&#'-]")


class WorkspaceError(ValueError):
    """The name cannot be turned into a path inside the workspace."""


def root() -> Path:
    """The workspace directory, read at call time.

    Call time rather than import time because `.env` is loaded by `main()`,
    which runs after this module is imported -- the same trap that split
    `$A4S_HOME` in two.
    """
    explicit = os.getenv("A4S_WEB_WORKSPACE")
    if explicit:
        return Path(explicit).expanduser()
    home = Path(os.getenv("A4S_HOME", Path.home() / ".a4s")).expanduser()
    return home / "web"


def uploads() -> Path:
    return _ensure(root() / "uploads")


def extractions() -> Path:
    return _ensure(root() / "extractions")


def matches() -> Path:
    return _ensure(root() / "matches")


def _ensure(path: Path) -> Path:
    path.mkdir(parents=True, exist_ok=True)
    return path


def safe_name(raw: str) -> str:
    """One path component, or an error.

    Rejects rather than sanitises. A silent rewrite turns
    `../../etc/passwd` into `etcpasswd`, which then reads or writes a file the
    caller did not name -- and the caller is told everything worked.
    """
    name = (raw or "").strip()
    if not name:
        raise WorkspaceError("empty filename")
    if name in (".", "..") or "/" in name or "\\" in name or "\0" in name:
        raise WorkspaceError(f"{raw!r} is not a plain filename")
    if _UNSAFE.search(name):
        bad = "".join(sorted(set(_UNSAFE.findall(name))))
        raise WorkspaceError(f"{raw!r} contains characters not allowed in a filename: {bad}")
    if len(name) > 200:
        raise WorkspaceError("filename is too long")
    return name


def under(directory: Path, name: str) -> Path:
    """`directory/name`, having checked the result is still under `directory`.

    The second half of the check. `safe_name` has already refused separators,
    so this catches what the filesystem itself does -- a symlinked workspace
    directory, a case-folding volume -- rather than what the caller typed.
    """
    candidate = (directory / safe_name(name)).resolve()
    base = directory.resolve()
    if base != candidate and base not in candidate.parents:
        raise WorkspaceError(f"{name!r} resolves outside the workspace")
    return candidate


def stem(filename: str) -> str:
    """The identifier the wizard uses across stages.

    The frontend passes whatever it has: `datasheet.pdf` in the PDF case, an
    already-stripped name for spreadsheets, `text_input_<hash>.json` for pasted
    text. One extension is removed so all four sources land on the same key,
    and only a known one -- a product name like `Type 400.5` must keep its
    tail.
    """
    name = safe_name(filename)
    for suffix in (".pdf", ".json", ".csv", ".xlsx", ".xls", ".md", ".txt", ".png", ".jpg"):
        if name.lower().endswith(suffix):
            return name[: -len(suffix)]
    return name


def extraction_file(filename: str, edited: bool = False) -> Path:
    suffix = "_edit.json" if edited else ".json"
    return under(extractions(), f"{stem(filename)}{suffix}")


def match_file(filename: str, aas_id_short: Optional[str] = None) -> Path:
    """Where the matcher's plan for one document against one AAS lives.

    The AAS is part of the name because the same datasheet is matched against
    different shells, and a single name per document would make the second
    match silently overwrite the first.
    """
    base = stem(filename)
    if aas_id_short and not base.endswith(f"_{safe_name(aas_id_short)}"):
        base = f"{base}_{safe_name(aas_id_short)}"
    return under(matches(), f"{base}_matching_result.json")


def edited_match_file(filename: str) -> Path:
    return under(matches(), f"{stem(filename)}_matching_result_edit.json")
