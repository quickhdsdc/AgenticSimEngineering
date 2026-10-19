"""Project scratch layout and read compatibility for relocated evidence."""
from pathlib import Path

ROOT = Path(__file__).resolve().parents[2]
WORK = ROOT / "work"
RUNS = WORK / "runs"


def relocated_artifact(path):
    """Resolve an existing artifact without changing its hashed descriptor.

    Only the old project-local runs directory has a relocation mapping. Both
    lexical traversal and symlinks are checked through resolved paths.
    """
    original = (ROOT / path).resolve()
    if not original.is_relative_to(ROOT):
        raise ValueError("artifact path leaves the project")
    legacy = ROOT / "runs"
    target = (RUNS / original.relative_to(legacy)).resolve() if original.is_relative_to(legacy) else original
    if not target.is_relative_to(ROOT):
        raise ValueError("relocated artifact path leaves the project")
    return target
