"""Order commands and request-shape checks, not an order state machine."""

from ..arguments import positive_id, quantity
from ..client import MesError

STATUSES = (
    "PENDING",
    "ACCEPTED",
    "IN_PROGRESS",
    "BLOCKED",
    "PAUSED",
    "CHANGE_HOLD",
    "DONE",
    "REJECTED",
    "CANCELLED",
)


def register(cmd, opt):
    for verb in ("ls", "current"):
        p = cmd(
            "order " + verb,
            "/api/v1/work-orders",
            "List orders" if verb == "ls" else "Find exactly one IN_PROGRESS order",
            resource="work-orders",
            read=current if verb == "current" else None,
        )
        opt(p, "line", "line_id", required=verb == "current")
        if verb == "ls":
            opt(p, "status", choices=STATUSES)
    cmd(
        "order show",
        "/api/v1/work-orders/{id}",
        "Read order with batches and issues",
        resource="work-orders",
        target=True,
    )
    for verb, method in (("create", "POST"), ("update", "PATCH")):
        p = cmd(
            "order " + verb,
            "/api/v1/work-orders" + ("/{id}" if verb == "update" else ""),
            "Create or update order fields",
            method,
            "work-orders",
            target=verb == "update",
            verify_fields=True,
            prepare=prepare if verb == "update" else None,
            subject_field="target" if verb == "update" else "order_no",
            effects=("OpenMES persists order fields and applies its own business validations.",),
        )
        if verb == "create":
            opt(p, "order-no", required=True)
            opt(p, "line", "line_id")
            opt(p, "product-type-id", type=positive_id)
        opt(p, "planned-qty", type=quantity, required=verb == "create")
        opt(p, "customer-order-no")
        opt(p, "priority", type=int)
        opt(p, "due-date")
        opt(p, "description")
    for verb in ("accept", "pause", "resume", "complete", "stop"):
        p = cmd(
            "order " + verb,
            "/api/v1/work-orders/{id}/" + verb,
            "Change production order state",
            "POST",
            "work-orders",
            target=True,
            expected_status={
                "accept": "ACCEPTED",
                "pause": "PAUSED",
                "resume": "IN_PROGRESS",
                "complete": "DONE",
            }.get(verb),
            effects=(
                ("OpenMES opens a stop and may place the order on change hold.",)
                if verb == "stop"
                else ("OpenMES changes order status; resume may close a structured stop.",)
            ),
        )
        if verb == "stop":
            opt(
                p,
                "type",
                required=True,
                choices=(
                    "OPERATIONAL",
                    "MATERIAL_SHORTAGE",
                    "MACHINE_FAILURE",
                    "QUALITY_HOLD",
                    "ENGINEERING_CHANGE",
                    "OTHER",
                ),
            )
            opt(p, "reason", required=True)
            opt(p, "issue-id", type=positive_id)
            opt(p, "batch-id", type=positive_id)
            opt(p, "downtime-reason-id", type=positive_id)
            opt(p, "requires-change", action="store_true", default=None)
        if verb == "resume":
            opt(p, "notes")
            opt(p, "change-request-id", type=positive_id)
    cmd(
        "order stops",
        "/api/v1/work-orders/{id}/stops",
        "Read structured stop history",
        resource="work-orders",
        target=True,
    )


def prepare(client, fields, target_id):
    if not fields:
        raise MesError("Supply at least one field to update")
    return {}


def current(client, path, fields):
    rows = client.rows(path, **fields, status="IN_PROGRESS")
    if len(rows) != 1:
        raise MesError(
            f"Expected exactly one IN_PROGRESS order; found {len(rows)}. Use mes order ls/show."
        )
    return {"data": rows[0]}
