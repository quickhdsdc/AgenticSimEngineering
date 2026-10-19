"""Resolve an AAS's stable OpenMES line code to live MES resources.

The AAS names the service and equipment; order IDs and scheduling belong only
to OpenMES. The A4S demo selects its newest complete A/B pair from MES itself.
"""

from ...mesctl.client import MesError


def resolve_lines(context, mes):
    if context.get("MESBaseUrl", "").rstrip("/") != mes.config.endpoint:
        raise MesError("ProductionContext does not reference the configured trusted MES")
    codes = (context.get("MESLineCode"), context.get("MESBackupLineCode"))
    if not codes[0] or not codes[1] or codes[0] == codes[1]:
        raise MesError("ProductionContext requires distinct MES line codes")
    lines = mes.rows("/api/v1/lines")
    resolved = []
    for code in codes:
        matches = [line for line in lines if line.get("code") == code]
        if len(matches) != 1:
            raise MesError(f"Expected one OpenMES line with code {code!r}; found {len(matches)}")
        resolved.append(matches[0])
    return tuple(resolved)


def current_demo_orders(mes, line, backup):
    """Read the latest complete demo pair, including an order moved to backup."""
    allowed_lines = {int(line["id"]), int(backup["id"])}
    groups = {}
    for order in mes.rows("/api/v1/work-orders"):
        number = order.get("order_no", "")
        if int(order.get("line_id") or 0) not in allowed_lines or not number.startswith("A4S-"):
            continue
        run, separator, suffix = number.rpartition("-")
        if separator and suffix in ("A", "B"):
            groups.setdefault(run, {})[suffix] = order
    complete = [(run, pair) for run, pair in groups.items() if set(pair) == {"A", "B"}]
    if not complete:
        raise MesError("No complete A4S demo order pair exists for this MES equipment")
    _, pair = max(complete, key=lambda item: max(int(o["id"]) for o in item[1].values()))
    return pair["A"], pair["B"]
