"""The composition root — the one place that knows both sides.

Run with::

    uv sync --group service
    uv run python examples/serve_api.py
    # console: http://127.0.0.1:8000/   ·   raw API: .../docs

``opsworld.service`` knows how to serialise, register and schedule, and knows
nothing about a shopfloor or a warehouse. The domains know how to build their
own world models and encode their own decisions, and know nothing about HTTP.
This file introduces them, and that is all it does — which is architecture
constraint A6 held at the boundary where it is easiest to break.

The front end is written against ``GET /api/schema``: it lists the scenarios
with their controls, the world models with their capabilities, and — per world
model — **which solvers that world model can actually serve**. That last list
is computed from the capability declarations, so a UI never offers a gradient
solver for a model that has no gradient.

``frontend/`` is served from here rather than opened from disk, because the
console loads ``api.js`` through a dynamic ES-module import and ``file://``
refuses those. Serving it from this process also makes it same-origin, so the
console reaches the API without relying on the CORS allowance.
"""

from __future__ import annotations

import argparse
from pathlib import Path

import gradientshop.service as gradientshop_service
import intralogistics.service as intralogistics_service

from opsworld.service.app import create_app
from opsworld.service.jobs import JobRunner
from opsworld.service.registry import Registry

DOMAINS = ["gradientshop.service", "intralogistics.service"]

#: The console as exported from the design canvas. Left exactly as exported —
#: nothing is added to that folder, so a re-export cannot clobber our changes.
FRONTEND = Path(__file__).resolve().parents[1] / "frontend"
CONSOLE_ENTRY = "OpsworldConsole.dc.html"


def build_registry() -> Registry:
    registry = Registry()
    gradientshop_service.register(registry)
    intralogistics_service.register(registry)
    return registry


def mount_console(app, frontend: Path) -> bool:
    """Serve ``frontend/`` at ``/console``, with ``/`` redirecting to its entry.

    Returns whether it was mounted. A missing folder is not an error: the API
    is the product and the console is one client of it, so the server has to
    come up without it.
    """
    if not (frontend / CONSOLE_ENTRY).is_file():
        return False

    from fastapi.responses import RedirectResponse
    from fastapi.staticfiles import StaticFiles

    app.mount("/console", StaticFiles(directory=str(frontend)), name="console")

    @app.get("/", include_in_schema=False)
    def console_entry():
        return RedirectResponse(f"/console/{CONSOLE_ENTRY}")

    return True


def build_app(*, max_concurrent: int = 2, timeout: float = 300.0,
              frontend: Path | None = FRONTEND):
    app = create_app(
        build_registry(),
        domains=DOMAINS,
        runner=JobRunner(max_concurrent=max_concurrent, default_timeout=timeout),
    )
    app.state.console_mounted = (
        mount_console(app, frontend) if frontend is not None else False
    )
    return app


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--host", default="127.0.0.1",
                        help="research server; bind beyond localhost deliberately")
    parser.add_argument("--port", type=int, default=8000)
    parser.add_argument("--max-concurrent", type=int, default=2,
                        help="jobs running at once; each takes a whole process")
    parser.add_argument("--timeout", type=float, default=300.0,
                        help="wall clock per job, in seconds")
    parser.add_argument("--frontend", type=Path, default=FRONTEND,
                        help="folder holding the console; default: ./frontend")
    parser.add_argument("--no-frontend", action="store_true",
                        help="serve the API alone")
    args = parser.parse_args()

    import uvicorn

    frontend = None if args.no_frontend else args.frontend
    app = build_app(max_concurrent=args.max_concurrent, timeout=args.timeout,
                    frontend=frontend)

    registry = app.state.registry
    root = f"http://{args.host}:{args.port}"
    print(f"scenarios : {', '.join(sorted(registry.scenarios))}")
    print(f"worlds    : {', '.join(sorted(registry.worlds))}")
    print(f"solvers   : {', '.join(sorted(registry.solvers))}")
    if app.state.console_mounted:
        print(f"\nconsole   : {root}/")
    elif frontend is not None:
        print(f"\nconsole   : not mounted — no {CONSOLE_ENTRY} under {frontend}")
    print(f"schema    : {root}/api/schema")
    print(f"docs      : {root}/docs\n")

    uvicorn.run(app, host=args.host, port=args.port, log_level="info")


if __name__ == "__main__":
    main()
