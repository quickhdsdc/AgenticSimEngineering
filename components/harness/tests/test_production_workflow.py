from types import SimpleNamespace

from a4s.mesctl.verification import verify_postconditions
from a4s.server.production_routes import in_window


def test_mes_date_readback_respects_server_calendar():
    action = SimpleNamespace(expected_status=None, verify_fields=True)
    verify_postconditions(action, {'due_date':'2026-09-25'},
                          {'due_date':'2026-09-24T22:00:00Z'}, 'Europe/Berlin')


def test_activity_time_window_excludes_old_run_and_keeps_boundary():
    records = [{'id':1,'created_at':'2026-09-25T07:00:00Z'},
               {'id':2,'created_at':'2026-09-25T08:00:00Z'}]
    from a4s.server.production_routes import epoch
    assert [r['id'] for r in in_window(records, epoch('2026-09-25T08:00:00Z'))] == [2]


def test_maintenance_lowercase_status_readback():
    action = SimpleNamespace(expected_status='PENDING', verify_fields=True, resource='maintenance-events')
    verify_postconditions(action, {}, {'status':'pending'})


def test_due_date_is_local_calendar_day_and_separate_from_schedule():
    from a4s.server.production_routes import delivery_summary
    order = {'due_date': '2026-09-27T22:00:00.000000Z',
             'planned_end_at': '2026-10-01T12:30:00.000000Z'}
    assert delivery_summary(order, 'Europe/Berlin') == {
        'due_day': '2026-09-28', 'planned_after_due': True}
    order['planned_end_at'] = '2026-09-28T13:15:00Z'
    assert delivery_summary(order, 'Europe/Berlin')['planned_after_due'] is False
