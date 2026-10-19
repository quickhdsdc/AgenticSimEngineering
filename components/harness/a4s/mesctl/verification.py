"""Read-back assertions; no writes, status transitions or local business state."""

from datetime import datetime
from zoneinfo import ZoneInfo
from .client import MesError


def verify_postconditions(action, fields, actual, server_timezone="UTC"):
    """Check observable target state, without claiming atomicity with the write."""
    expected = {}
    if action.expected_status:
        expected["status"] = action.expected_status
    if action.verify_fields:
        expected.update({k: v for k, v in fields.items() if k != "force_conflict"})
    mismatches = []
    for key, value in expected.items():
        observed = actual.get(key)
        try:
            if key == "status" and getattr(action, 'resource', '') == 'maintenance-events':
                equal = str(observed).upper() == str(value).upper()
            elif key in ("planned_qty", "actual_cost"):
                equal = float(observed) == float(value)
            elif key in ("planned_start_at", "planned_end_at", "scheduled_at"):
                equal = datetime.fromisoformat(
                    observed.replace("Z", "+00:00")
                ) == datetime.fromisoformat(value.replace("Z", "+00:00"))
            elif key == "due_date":
                # Laravel serializes a local DATE as midnight converted to UTC.
                # Compare the calendar day in the configured MES timezone.
                day = str(observed)[:10]
                if isinstance(observed, str) and 'T' in observed:
                    day = datetime.fromisoformat(observed.replace('Z', '+00:00')).astimezone(
                        ZoneInfo(server_timezone)).date().isoformat()
                equal = day == str(value)[:10]
            else:
                equal = observed == value
        except (ValueError, TypeError, AttributeError):
            equal = False
        if not equal:
            mismatches.append(key)
    if mismatches:
        raise MesError("Read-back did not confirm requested fields: " + ", ".join(mismatches))
