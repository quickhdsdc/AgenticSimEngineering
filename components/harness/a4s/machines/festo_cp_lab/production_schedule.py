"""Current-time slots for a fresh local housing demo plan."""

from datetime import datetime, timedelta, timezone
import math
from zoneinfo import ZoneInfo

LEAD_MINUTES = 15
SLOT_MINUTES = 45


def next_start(now: datetime) -> datetime:
    """First five-minute boundary at least 15 minutes after the current clock."""
    if now.tzinfo is None or now.utcoffset() is None:
        raise ValueError('Current time must include a timezone')
    stamp = math.ceil((now.timestamp() + LEAD_MINUTES * 60) / 300) * 300
    return datetime.fromtimestamp(stamp, timezone.utc)


def lots(start: datetime, timezone_name: str):
    if start.tzinfo is None or start.utcoffset() is None:
        raise ValueError('Plan start must include a timezone')
    zone = ZoneInfo(timezone_name or 'UTC')
    first = start.astimezone(timezone.utc)
    result = []
    for index, suffix in enumerate(('A', 'B')):
        begin = first + timedelta(minutes=SLOT_MINUTES * index)
        end = begin + timedelta(minutes=SLOT_MINUTES)
        result.append({'lot': suffix, 'planned_start_at': begin.isoformat(),
                       'planned_end_at': end.isoformat(),
                       'due_date': end.astimezone(zone).date().isoformat()})
    return result
