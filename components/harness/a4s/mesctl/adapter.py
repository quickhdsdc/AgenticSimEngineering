"""OpenMES wire adaptation and read-back; no local business records."""

import argparse
from dataclasses import dataclass
from datetime import datetime
import hashlib
import json
from .arguments import positive_id
from .client import MesError
from .verification import verify_postconditions

META = {
    "group",
    "verb",
    "api_path",
    "method",
    "resource",
    "command_name",
    "json",
    "target",
    "dry_run",
    "action",
    "trace_id",
    "action_words",
}


@dataclass(frozen=True)
class PreparedRequest:
    action: str
    method: str
    endpoint: str
    path: str
    body: dict

    @property
    def fingerprint(self):
        canonical = json.dumps(
            {
                "action": self.action,
                "method": self.method,
                "endpoint": self.endpoint,
                "path": self.path,
                "body": self.body,
            },
            sort_keys=True,
            separators=(",", ":"),
            ensure_ascii=True,
        )
        return hashlib.sha256(canonical.encode()).hexdigest()


class OpenMesAdapter:
    """One adapter per execution; only transient attempt metadata is kept."""

    def __init__(self, client):
        self.client = client
        self.phase = "preflight"
        self.request = None

    def execute(self, args):
        return _execute(args, self.client, self)


def _execute(args, client, state):
    action = args.action
    fields = {k: v for k, v in vars(args).items() if k not in META and v is not None}
    for key, collection, code in (
        ("line_id", "lines", "code"),
        ("work_order_id", "work-orders", "order_no"),
        ("issue_type_id", "issue-types", "code"),
    ):
        if key in fields:
            fields[key] = client.resolve(collection, fields[key], code)
    target_id = None
    path = args.api_path
    if "{id}" in path:
        if args.resource in ("lines", "work-orders"):
            target_id = client.resolve(
                args.resource, args.target, "code" if args.resource == "lines" else "order_no"
            )
        else:
            try:
                target_id = positive_id(args.target)
            except (ValueError, argparse.ArgumentTypeError):
                raise MesError("This command requires a positive numeric resource ID") from None
        path = path.replace("{id}", str(target_id))
    for key, limit in (
        ("title", 255),
        ("resolution_notes", 5000),
        ("reason", 2000),
        ("notes", 2000),
        ("order_no", 100),
        ("customer_order_no", 100),
    ):
        if key in fields and (not str(fields[key]).strip() or len(fields[key]) > limit):
            raise MesError(f"{key} must be non-empty and at most {limit} characters")
    if args.command_name == "issue create" and len(fields.get("description", "")) > 5000:
        raise MesError("Issue description must be at most 5000 characters")
    for start, end in (
        ("planned_start_at", "planned_end_at"),
        ("start", "end"),
        ("date_from", "date_to"),
    ):
        if start in fields and end in fields:
            try:
                a, b = (
                    datetime.fromisoformat(fields[k].replace("Z", "+00:00")) for k in (start, end)
                )
                if b <= a if start == "planned_start_at" else b < a:
                    raise ValueError()
                if start == "planned_start_at" and (a.tzinfo is None or b.tzinfo is None):
                    raise ValueError()
            except (ValueError, TypeError):
                raise MesError(
                    f"Invalid {start}/{end} interval; schedule timestamps require timezone offsets"
                ) from None
    if not action.mutates:
        state.phase = "read"
        return action.read(client, path, fields) if action.read else client.get(path, **fields)
    preflight = action.prepare(client, fields, target_id) if action.prepare else {}
    prepared = PreparedRequest(action.name, action.method, client.config.endpoint, path, fields)
    state.request = prepared
    if getattr(args, "dry_run", False):
        return {
            "dry_run": True,
            "method": args.method,
            "path": path,
            "body": fields,
            "preflight": preflight,
            "request_fingerprint": prepared.fingerprint,
            "action_contract": action.describe(),
        }

    state.phase = "commit"
    response = client.request(action.method, path, body=fields)
    state.phase = "verify"
    result = {
        "mutation": {
            "method": args.method,
            "path": path,
            "request_fingerprint": prepared.fingerprint,
        },
        "response": response,
    }
    # Never replay a mutation after a verification failure: it may already be committed.
    data = response.get("data", {}) if isinstance(response, dict) else {}
    resource_id = target_id or (data.get("id") if isinstance(data, dict) else None)
    result["affected_resources"] = [f"mes://{action.resource}/{resource_id}"] if resource_id else []
    try:
        if resource_id:
            result["read_back"] = client.get(f"/api/v1/{args.resource}/{resource_id}")
        if action.related_order:
            issue = _resource_data(result["read_back"]) if resource_id else data
            if not isinstance(issue, dict):
                raise MesError("Unexpected issue response during verification")
            if issue.get("work_order_id"):
                result["affected_resources"].append(f"mes://work-orders/{issue['work_order_id']}")
                result["order_after"] = client.get(f"/api/v1/work-orders/{issue['work_order_id']}")
                if _resource_data(result["order_after"]).get("id") != issue["work_order_id"]:
                    raise MesError("Read-back returned an unexpected related order ID")
        if resource_id:
            actual = _resource_data(result["read_back"])
            if actual.get("id") != resource_id:
                raise MesError("Read-back returned an unexpected resource ID")
            verify_postconditions(action, fields, actual, client.config.server_timezone or "UTC")
        result["verified"] = bool(resource_id)
    except MesError as exc:
        result.update(
            verified=False,
            verification_error=str(exc),
            warning="Mutation was accepted. Read the resource; do not repeat the mutation.",
        )
    return result


def _resource_data(response):
    if not isinstance(response, dict) or not isinstance(response.get("data"), dict):
        raise MesError("Unexpected resource response during verification; inspect current state")
    return response["data"]
