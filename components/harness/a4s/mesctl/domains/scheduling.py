"""Conservative agent scheduling policy and upstream wall-time adaptation.

These checks narrow agent authority. OpenMES still decides business validity,
conflicts and persistence; preflight does not reserve capacity.
"""

from datetime import datetime, timezone
from zoneinfo import ZoneInfo, ZoneInfoNotFoundError
from ..client import MesError


def register(cmd, opt):
    p = cmd(
        "schedule board", "/api/v1/schedule/board", "Read planner and backlog", resource="schedule"
    )
    opt(p, "view", "view_mode", choices=("weekly", "daily", "hourly", "monthly"))
    opt(p, "start-date")
    opt(p, "line", "line_id")
    p = cmd(
        "schedule capacity", "/api/v1/schedule/capacity", "Read capacity grid", resource="schedule"
    )
    opt(p, "axis", choices=("line", "crew"), default="line")
    opt(p, "granularity", choices=("day", "week"), default="day")
    opt(p, "start-date")
    p = cmd(
        "schedule capacity-cell",
        "/api/v1/schedule/capacity/cell",
        "Read orders contributing to capacity",
        resource="schedule",
    )
    opt(p, "line", "line_id", required=True)
    opt(p, "start", required=True)
    opt(p, "end", required=True)
    for verb in ("move", "resize"):
        p = cmd(
            "schedule " + verb,
            "/api/v1/schedule/{id}" + ("/resize" if verb == "resize" else ""),
            "Change minute-level placement; verifies compatibility before move",
            "PUT",
            "work-orders",
            target=True,
            verify_fields=True,
            prepare=prepare if verb == "move" else resize,
            effects=(
                "OpenMES changes order placement and records a schedule change; no PLC operation.",
            ),
            policy_checks=(
                (
                    "Move only PENDING/ACCEPTED orders",
                    "Explicit product compatibility",
                    "Snapshot workstation routing",
                )
                if verb == "move"
                else ("Do not resize terminal orders",)
            )
            + ("Read capacity as evidence, not a reservation", "Normalize server-local timestamps"),
        )
        if verb == "move":
            opt(p, "line", "line_id", required=True)
        opt(p, "start", "planned_start_at", required=True)
        opt(p, "end", "planned_end_at", required=True)
        opt(p, "force-conflict", action="store_true", default=None)
    cmd(
        "schedule changes",
        "/api/v1/schedule/changes",
        "Read recent planner changes",
        resource="schedule-changes",
    )


def schedule_time(raw, zone):
    # Upstream persists datetime wall-clock components without converting the
    # supplied offset. Send server-local components to preserve the instant.
    if not zone:
        raise MesError("Set A4S_MES_TIMEZONE to OpenMES APP_TIMEZONE before scheduling")
    try:
        tz = timezone.utc if zone == "UTC" else ZoneInfo(zone)
    except (ZoneInfoNotFoundError, ValueError):
        raise MesError(
            "Unknown A4S_MES_TIMEZONE or missing timezone database; install the mes extra"
        ) from None
    local = datetime.fromisoformat(raw.replace("Z", "+00:00")).astimezone(tz)
    if local.replace(fold=0).utcoffset() != local.replace(fold=1).utcoffset():
        raise MesError(
            "Ambiguous DST wall time cannot be represented safely by this OpenMES scheduler"
        )
    return local.isoformat()


def prepare(client, fields, target_id, *, move=True):
    preflight = {}
    for key in ("planned_start_at", "planned_end_at"):
        fields[key] = schedule_time(fields[key], client.config.server_timezone)
    order = client.get(f"/api/v1/work-orders/{target_id}")["data"]
    if move:
        if order["status"] not in ("PENDING", "ACCEPTED"):
            raise MesError(
                "Only not-yet-started orders (PENDING/ACCEPTED) may be moved by this CLI"
            )
        products = client.rows(f"/api/v1/lines/{fields['line_id']}/product-types")
        if not any(p["id"] == order.get("product_type_id") for p in products):
            raise MesError("Target line does not explicitly support this order's product type")
        stations = client.rows(f"/api/v1/lines/{fields['line_id']}/workstations")
        allowed = {s["id"] for s in stations}
        snapshot = order.get("process_snapshot") or {}
        if any(
            s.get("workstation_id") and s["workstation_id"] not in allowed
            for s in snapshot.get("steps", [])
        ):
            raise MesError(
                "Process snapshot still assigns steps to another line; reconcile routing before moving"
            )
    elif order["status"] in ("DONE", "REJECTED", "CANCELLED"):
        raise MesError("Cannot resize a terminal order")
    preflight["capacity"] = client.get(
        "/api/v1/schedule/capacity",
        axis="line",
        granularity="day",
        start_date=fields["planned_start_at"][:10],
    )
    preflight["order_before"] = order
    return preflight


def resize(client, fields, target_id):
    return prepare(client, fields, target_id, move=False)
