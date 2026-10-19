"""Read-only master data and monitoring commands."""


def register(cmd, opt):
    cmd("health", "/api/health", "Check OpenMES availability")
    cmd("auth whoami", "/api/auth/me", "Read authenticated identity")
    p = cmd("line ls", "/api/v1/lines", "List visible production lines", resource="lines")
    opt(p, "query", "q")
    opt(p, "include-inactive", "include_inactive", action="store_true", default=None)
    for name, suffix in (
        ("show", ""),
        ("workstations", "/workstations"),
        ("products", "/product-types"),
    ):
        cmd(
            "line " + name,
            "/api/v1/lines/{id}" + suffix,
            "Read line " + name,
            resource="lines",
            target=True,
        )
    cmd(
        "workstation show",
        "/api/v1/workstations/{id}",
        "Read workstation by numeric ID",
        resource="workstations",
        target=True,
    )
    cmd("product ls", "/api/v1/product-types", "List product types", resource="product-types")
    cmd(
        "product show",
        "/api/v1/product-types/{id}",
        "Read product by ID",
        resource="product-types",
        target=True,
    )
    cmd(
        "process show",
        "/api/v1/process-templates/{id}",
        "Read process template by ID",
        resource="process-templates",
        target=True,
    )
    cmd(
        "machine monitor",
        "/api/v1/machine-monitor",
        "Read live fleet state",
        resource="workstations",
    )
    cmd(
        "downtime reasons",
        "/api/v1/downtime-reasons",
        "List downtime reasons",
        resource="downtime-reasons",
    )
    p = cmd(
        "downtime ls", "/api/v1/downtimes", "Read latest 50 downtime records", resource="downtimes"
    )
    opt(p, "line", "line_id")
    opt(p, "date")
    p = cmd("oee show", "/api/v1/oee", "Read up to 100 OEE records", resource="oee")
    opt(p, "line", "line_id")
    opt(p, "from", "date_from")
    opt(p, "to", "date_to")
