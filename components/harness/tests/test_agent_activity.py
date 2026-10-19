import json
import sqlite3

from a4s.ledger import Ledger
from a4s.server import agent_activity, production_routes


def test_ledger_pairs_tools_and_preserves_uncertain_execution(tmp_path, monkeypatch):
    monkeypatch.setattr(agent_activity, '_home', lambda: tmp_path)
    path = tmp_path / 'ledger.db'
    with Ledger(path, 'session-test', actor='web-chat') as ledger:
        ledger.append('turn.assistant', {'text': 'private internal content'})
        ledger.append('tool.call', {'tool': 'shell', 'call_id': 'one', 'arguments': {'command': 'secret'}})
        result = ledger.append('tool.result', {'tool': 'shell', 'call_id': 'one', 'output': 'secret token'})
        ledger.append('tool.call', {'tool': 'shell', 'call_id': 'pending'})
        ledger.append('mes.execution', {'stage': 'finished', 'execution': {
            'execution_id': 'exec-123', 'action': 'order show', 'outcome': 'read_completed'}})
        ledger.append('mes.execution', {'stage': 'finished', 'execution': {'outcome': 'accepted_unverified'}})
        ledger.append('approval', {'allowed': False, 'source': 'operator'})
        head = ledger.head()
    report = agent_activity.ledger_activity(0)
    assert len(report['activity']) == 5
    assert 'secret' not in json.dumps(report)
    assert 'private internal' not in json.dumps(report)
    assert any(e['evidence']['call_id'] == 'pending' and 'Awaiting result' in e['detail'] for e in report['activity'])
    assert any('accepted_unverified' in e['detail'] for e in report['activity'])
    assert any('read_completed' in e['detail'] for e in report['activity'])
    assert all(e['at'] >= result.ts for e in agent_activity.ledger_activity(result.ts)['activity'])
    assert agent_activity.ledger_activity(0, limit=1)['ledger_truncated']
    with sqlite3.connect(path) as db:
        assert db.execute('select hash from events order by seq desc limit 1').fetchone()[0] == head


def test_missing_ledger_does_not_create_database(tmp_path, monkeypatch):
    monkeypatch.setattr(agent_activity, '_home', lambda: tmp_path)
    assert not agent_activity.ledger_activity(0)['ledger_available']
    assert not (tmp_path / 'ledger.db').exists()


def test_activity_keeps_ledger_when_mes_fails_and_filters_reset_run(tmp_path, monkeypatch):
    monkeypatch.setattr(agent_activity, '_home', lambda: tmp_path)
    with Ledger(tmp_path / 'ledger.db', 'session-test') as ledger:
        ledger.append('tool.result', {'tool': 'shell', 'call_id': 'old', 'output': 'old'})
        recent = ledger.append('tool.result', {'tool': 'shell', 'call_id': 'new', 'output': 'new'})
    manifest = tmp_path / 'current.json'
    manifest.write_text(json.dumps({'started_at': recent.ts}))
    monkeypatch.setattr(production_routes, 'MANIFEST', manifest)

    def offline(*args):
        raise RuntimeError('MES offline')

    monkeypatch.setattr(production_routes, 'workflow', offline)
    data = production_routes.activity(0, True)
    assert len(data['activity']) == 1
    assert data['warnings']
    assert len(production_routes.activity(0, False)['activity']) == 2
