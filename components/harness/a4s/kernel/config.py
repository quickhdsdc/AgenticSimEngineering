"""Where configuration comes from, and in what order.

Two rules, both there for the same reason -- a shopfloor host is configured by
its operator, not by a file someone committed:

* The real environment always wins. ``.env`` fills gaps; it never overrides.
* Secrets are read, never echoed. :func:`describe` reports whether a variable
  is set and how long it is, which is enough to diagnose a paste error without
  putting a key in a log, a ledger entry, or a model's context.

No dotenv dependency: the format used here is the subset everyone actually
writes, and a dependency that parses a file this simple is a dependency that
can change under a deployment.
"""

from __future__ import annotations

import os
from pathlib import Path
from typing import Dict, List, Optional

# Search order for a .env: an explicit path, the working directory, then the
# package root. Stops at the first hit rather than merging, so which file is in
# force is always answerable.
_CANDIDATES = ("A4S_ENV_FILE",)


def find_env_file(explicit: Optional[str] = None) -> Optional[Path]:
    if explicit:
        path = Path(explicit)
        return path if path.is_file() else None
    for var in _CANDIDATES:
        raw = os.getenv(var)
        if raw and Path(raw).is_file():
            return Path(raw)
    for base in (Path.cwd(), Path(__file__).resolve().parents[2]):
        candidate = base / ".env"
        if candidate.is_file():
            return candidate
    return None


def load_env(explicit: Optional[str] = None) -> Optional[Path]:
    """Populate ``os.environ`` from a ``.env``, leaving existing values alone."""
    path = find_env_file(explicit)
    if path is None:
        return None
    for raw_line in path.read_text(encoding="utf-8").splitlines():
        line = raw_line.strip()
        if not line or line.startswith("#"):
            continue
        if line.startswith("export "):
            line = line[len("export "):].lstrip()
        key, sep, value = line.partition("=")
        if not sep:
            continue
        key = key.strip()
        value = value.strip()
        if len(value) >= 2 and value[0] == value[-1] and value[0] in ("'", '"'):
            value = value[1:-1]
        if key and key not in os.environ:
            os.environ[key] = value
    return path


def first(*names: str) -> Optional[str]:
    """The first of these variables that is set and non-empty."""
    for name in names:
        value = os.getenv(name)
        if value and value.strip():
            return value.strip()
    return None


def describe(*names: str) -> List[Dict[str, object]]:
    """Report presence and length per variable -- never the value."""
    out: List[Dict[str, object]] = []
    for name in names:
        value = os.getenv(name)
        out.append(
            {
                "name": name,
                "set": bool(value and value.strip()),
                "length": len(value.strip()) if value else 0,
            }
        )
    return out


def redact(value: Optional[str]) -> str:
    if not value:
        return "(unset)"
    return f"set, {len(value)} chars"
