"""Keep disposable pytest files under the repository's ignored temp directory."""

from pathlib import Path


def pytest_configure(config):
    # pytest creates basetemp itself, but not its parents on a fresh checkout.
    # Only normalize our configured default; explicit per-run overrides remain
    # the caller's responsibility and are useful for independent parallel runs.
    if config.option.basetemp == "temp/pytest/work":
        target = Path(config.rootpath) / "temp" / "pytest" / "work"
        target.parent.mkdir(parents=True, exist_ok=True)
        config.option.basetemp = str(target)
