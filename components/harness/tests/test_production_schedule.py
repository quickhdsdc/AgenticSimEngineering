"""Fresh demo slots follow the current clock and the MES local calendar."""

from datetime import datetime, timedelta, timezone

from a4s.machines.festo_cp_lab.production_schedule import lots, next_start


def test_new_schedule_tracks_now_in_five_minute_steps():
    now = datetime(2026, 10, 2, 11, 42, 1, tzinfo=timezone.utc)
    start = next_start(now)
    assert start == datetime(2026, 10, 2, 12, 0, tzinfo=timezone.utc)
    assert timedelta(minutes=15) <= start - now < timedelta(minutes=20)
    schedule = lots(start, 'Europe/Berlin')
    assert schedule[0]['planned_start_at'] == start.isoformat()
    assert schedule[0]['planned_end_at'] == schedule[1]['planned_start_at']
    assert schedule[1]['planned_end_at'] == (start + timedelta(minutes=90)).isoformat()


def test_due_date_uses_berlin_date_at_slot_end():
    start = datetime(2026, 10, 2, 20, 30, tzinfo=timezone.utc)
    schedule = lots(start, 'Europe/Berlin')
    assert [slot['due_date'] for slot in schedule] == ['2026-10-02', '2026-10-03']
