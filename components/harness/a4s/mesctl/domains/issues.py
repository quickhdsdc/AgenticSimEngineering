"""Issue commands; blocking and recovery are OpenMES-owned side effects."""

from ..arguments import positive_id


def register(cmd, opt):
    cmd(
        "issue types",
        "/api/v1/issue-types",
        "List issue types and blocking flags",
        resource="issue-types",
    )
    p = cmd("issue ls", "/api/v1/issues", "List issues (20 per page)", resource="issues")
    opt(p, "line", "line_id")
    opt(p, "order", "work_order_id")
    opt(p, "status", choices=("OPEN", "ACKNOWLEDGED", "RESOLVED", "CLOSED"))
    opt(p, "page", type=positive_id)
    cmd("issue show", "/api/v1/issues/{id}", "Read issue", resource="issues", target=True)
    p = cmd(
        "issue create",
        "/api/v1/issues",
        "Create issue; blocking type also blocks order",
        "POST",
        "issues",
        expected_status="OPEN",
        verify_fields=True,
        related_order=True,
        prepare=prepare,
        subject_resource="work-orders",
        subject_field="work_order_id",
        effects=("A blocking issue type can set the related order to BLOCKED.",),
    )
    opt(p, "order", "work_order_id", required=True)
    opt(
        p,
        "type",
        "issue_type_id",
        required=True,
        help="Issue type ID or exact code, e.g. TOOL_FAILURE",
    )
    opt(p, "title", required=True)
    opt(p, "description")
    opt(p, "batch-step-id", type=positive_id)
    for verb in ("acknowledge", "resolve", "close"):
        p = cmd(
            "issue " + verb,
            "/api/v1/issues/{id}/" + verb,
            "Advance issue lifecycle",
            "POST",
            "issues",
            target=True,
            expected_status={
                "acknowledge": "ACKNOWLEDGED",
                "resolve": "RESOLVED",
                "close": "CLOSED",
            }[verb],
            verify_fields=verb == "resolve",
            related_order=True,
            effects=(
                "Changes issue lifecycle; resolution can recompute related order status from batches and remaining issues.",
            ),
        )
        if verb == "resolve":
            opt(
                p,
                "notes",
                "resolution_notes",
                required=True,
                help="Required resolution notes, maximum 5000 characters",
            )


def prepare(client, fields, target_id):
    return {"issue_type": client.get(f"/api/v1/issue-types/{fields['issue_type_id']}")}
