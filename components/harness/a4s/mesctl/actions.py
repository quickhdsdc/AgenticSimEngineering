"""Declarative integration contracts, not a second implementation of MES rules."""

from dataclasses import dataclass
from typing import Callable
from urllib.parse import quote


@dataclass(frozen=True)
class ActionSpec:
    name: str
    path: str | None
    description: str
    method: str = "GET"
    resource: str | None = None
    target: bool = False
    expected_status: str | None = None
    verify_fields: bool = False
    related_order: bool = False
    effects: tuple[str, ...] = ()
    policy_checks: tuple[str, ...] = ()
    subject_resource: str | None = None
    subject_field: str = "target"
    prepare: Callable | None = None
    read: Callable | None = None

    def __post_init__(self):
        if self.method not in ("GET", "POST", "PATCH", "PUT"):
            raise ValueError(f"Unsupported MES method: {self.method}")
        if self.mutates and (not self.resource or not self.effects):
            raise ValueError(f"Mutating action {self.name} must declare its resource and effects")

    @property
    def mutates(self):
        return self.method != "GET"

    def subject(self, args):
        resource = self.subject_resource or self.resource or "service"
        ref = getattr(args, self.subject_field, None)
        return "mes://" + resource + ("/" + quote(str(ref), safe="") if ref else "")

    def describe(self):
        return {
            "name": self.name,
            "description": self.description,
            "method": self.method,
            "path": self.path,
            "resource": self.resource,
            "risk": "business" if self.mutates else "read",
            "business_authority": "OpenMES",
            "dry_run": self.mutates,
            "approval": "required_by_default" if self.mutates else "read_policy",
            "effects": list(self.effects),
            "agent_policy_checks": list(self.policy_checks),
            "preflight": self.prepare.__name__ if self.prepare else None,
            "verification": {
                "resource_read_back": self.mutates,
                "expected_status": self.expected_status,
                "requested_fields": self.verify_fields,
                "related_order_read_back": self.related_order,
            },
            "retry": "inspect_before_retry; no automatic replay"
            if self.mutates
            else "safe_to_repeat_read",
        }
