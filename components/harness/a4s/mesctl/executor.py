"""Coordination boundary; OpenMES owns business rules and records."""

from .adapter import OpenMesAdapter
from .client import MesError
from .execution import ExecutionContext


def execute(args, client, *, context=None):
    context = context or ExecutionContext.from_env(getattr(args, "trace_id", None))
    adapter = OpenMesAdapter(client)
    previous = client.execution_context
    client.execution_context = context
    metadata = {**context.as_dict(), "action": args.action.name}
    try:
        result = adapter.execute(args)
        outcome = (
            "previewed"
            if getattr(args, "dry_run", False)
            else "read_completed"
            if not args.action.mutates
            else "verified"
            if result.get("verified")
            else "accepted_unverified"
        )
        if adapter.request:
            metadata["request_fingerprint"] = adapter.request.fingerprint
        return {**result, "execution": {**metadata, "outcome": outcome}}
    except MesError as exc:
        outcome = "preflight_failed" if adapter.phase == "preflight" else "read_failed"
        if adapter.phase == "commit":
            outcome = (
                "rejected"
                if exc.status and 400 <= exc.status < 500 and exc.status != 408
                else "unknown"
            )
        if adapter.request:
            metadata["request_fingerprint"] = adapter.request.fingerprint
        exc.execution = {**metadata, "outcome": outcome}
        raise
    finally:
        client.execution_context = previous
