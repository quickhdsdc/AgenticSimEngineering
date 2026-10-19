"""The application object, its CORS policy, and `a4s serve`."""

from __future__ import annotations

import argparse
import os
import sys
from typing import Any, List, Optional, Sequence

DEFAULT_PORT = 8000

# Where a Vite dev server runs. Listed explicitly rather than allowing `*`
# because the browser sends credentials on same-site requests and a wildcard
# origin with credentials is refused by every modern browser anyway -- so `*`
# would look permissive and fail. `$A4S_CORS_ORIGINS` extends it.
DEV_ORIGINS = (
    "http://localhost:5173", "http://127.0.0.1:5173",
    "http://localhost:3000", "http://127.0.0.1:3000",
    "http://localhost:8080", "http://127.0.0.1:8080",
)


def build_app(endpoint: Optional[str] = None, origins: Optional[Sequence[str]] = None) -> Any:
    """Assemble the FastAPI app.

    A factory rather than a module-level `app`, so the endpoint and the CORS
    policy are arguments rather than imports with side effects -- which is also
    what lets the tests build one against a scratch repository without
    reaching for environment variables.
    """
    try:
        from fastapi import FastAPI
        from fastapi.middleware.cors import CORSMiddleware
    except ImportError as exc:  # pragma: no cover - install guidance
        raise SystemExit(
            "the HTTP surface needs the `ui` extra: "
            'pip install -e ".[ui]"'
        ) from exc

    from . import aas_routes, chat, opc_routes, wizard_routes, write_routes, festo_demo_routes, festo_start_routes

    if endpoint:
        # Every route reads the endpoint through `Endpoint.from_env`, the same
        # path the CLI uses, so one variable configures both and a session
        # cannot end up talking to two repositories.
        os.environ["A4S_AAS_ENDPOINT"] = endpoint

    app = FastAPI(
        title="A4S harness",
        version="0.1.0",
        description="HTTP surface over the same library the `aas` command uses.",
    )
    app.add_middleware(
        CORSMiddleware,
        allow_origins=list(origins or _origins()),
        allow_credentials=True,
        allow_methods=["*"],
        allow_headers=["*"],
    )
    app.include_router(aas_routes.router)
    # The mutating half. It runs no approval gate -- that control is the
    # agent's, decided 2026-08-27 -- and every write it makes is appended to
    # the ledger instead. See `write_routes` for the argument.
    app.include_router(write_routes.router)
    # The DPP wizard's file store. Not the pipeline: the wizard is agent-driven
    # and every stage ends by prompting the chat socket. These routes are the
    # handles on the files the stages pass to each other.
    app.include_router(wizard_routes.router)
    # Live values from the machines the AAS points at. Reads freely; writes
    # only through an Operation the model declares, so the browser names a
    # path and never a node.
    app.include_router(opc_routes.router)
    # The one place a real `AgentLoop` runs. The approval gate runs with it,
    # asking through the socket -- this is the case it was built for, because
    # here there is both an agent and a person watching it.
    app.include_router(chat.router)
    app.include_router(festo_demo_routes.router)
    app.include_router(festo_start_routes.router)
    from . import live_monitor_routes
    app.include_router(live_monitor_routes.router)
    from . import production_routes
    app.include_router(production_routes.router)
    app.add_event_handler('shutdown', live_monitor_routes.service.close)
    app.add_event_handler('shutdown', live_monitor_routes.simulator.stop)

    @app.get("/api/service-links")
    def service_links() -> dict:
        from urllib.parse import urlsplit

        # Browser navigation is independent of repository/MES health and credentials.
        url = (os.getenv("A4S_MES_UI_URL") or os.getenv("A4S_MES_ENDPOINT") or "").strip()
        parsed = urlsplit(url)
        valid = parsed.scheme in ("http", "https") and parsed.hostname and not (
            parsed.username or parsed.password or parsed.query or parsed.fragment)
        return {"mes_url": url.rstrip("/") if valid else None}

    @app.get("/health")
    def health() -> dict:
        """Whether the repository answers, not merely whether this process is up.

        A health check that only proves the web server started is the kind that
        stays green while every request behind it fails.
        """
        import httpx

        from ..aasctl.client import Endpoint

        try:
            base = Endpoint.from_env(None).base_url
        except RuntimeError as exc:
            return {"ok": False, "repository": None, "detail": str(exc)}
        try:
            response = httpx.get(f"{base}/shells", params={"limit": 1}, timeout=5.0)
            response.raise_for_status()
            return {"ok": True, "repository": base}
        except Exception as exc:  # noqa: BLE001
            return {"ok": False, "repository": base,
                    "detail": f"{type(exc).__name__}: {exc}"}

    return app


def _origins() -> List[str]:
    extra = [o.strip() for o in (os.getenv("A4S_CORS_ORIGINS") or "").split(",") if o.strip()]
    return list(DEV_ORIGINS) + extra


def main(argv: Optional[Sequence[str]] = None) -> int:
    from ..kernel.config import load_env

    load_env()
    parser = argparse.ArgumentParser(
        prog="a4s-serve",
        description="Serve the HTTP surface the web UI talks to.",
    )
    parser.add_argument("--host", default="127.0.0.1",
                        help="bind address. 0.0.0.0 to accept from other hosts")
    parser.add_argument("--port", type=int, default=DEFAULT_PORT)
    parser.add_argument("--endpoint", help="AAS repository base URL (else $A4S_AAS_ENDPOINT)")
    parser.add_argument("--origin", action="append", metavar="URL",
                        help="extra CORS origin; repeatable")
    parser.add_argument("--reload", action="store_true", help="reload on source change")
    args = parser.parse_args(argv)

    endpoint = args.endpoint or os.getenv("A4S_AAS_ENDPOINT")
    if not endpoint:
        print("error: no AAS endpoint. Pass --endpoint or set A4S_AAS_ENDPOINT.",
              file=sys.stderr)
        return 2

    try:
        import uvicorn
    except ImportError:
        print('error: the HTTP surface needs the `ui` extra: pip install -e ".[ui]"',
              file=sys.stderr)
        return 2

    os.environ["A4S_AAS_ENDPOINT"] = endpoint
    if args.origin:
        os.environ["A4S_CORS_ORIGINS"] = ",".join(args.origin)

    print(f"repository: {endpoint}", file=sys.stderr)
    print(f"serving on http://{args.host}:{args.port}", file=sys.stderr)
    print(f"CORS:       {', '.join(_origins())}", file=sys.stderr)

    if args.reload:
        uvicorn.run("a4s.server.app:_reload_app", host=args.host, port=args.port,
                    reload=True, factory=True)
    else:
        uvicorn.run(build_app(endpoint), host=args.host, port=args.port)
    return 0


def _reload_app() -> Any:
    """Factory for `--reload`, which needs an import string rather than an object."""
    return build_app(os.getenv("A4S_AAS_ENDPOINT"))


if __name__ == "__main__":
    sys.exit(main())
