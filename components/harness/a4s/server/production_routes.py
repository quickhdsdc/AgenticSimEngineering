"""Read-only production/incident projection from OpenMES and recorded OPC evidence."""
from datetime import datetime, timedelta, timezone
import hashlib
import json
import os
from pathlib import Path
import subprocess
import sys
import threading
import time
import uuid
from zoneinfo import ZoneInfo

from fastapi import APIRouter, Body, HTTPException, Query

from .festo_demo_routes import snapshot
from .live_monitor_routes import service, target
from ..aasctl.client import Endpoint
from ..mesctl.client import Config, MesClient
from ..mesctl.client import MesError
from ..machines.festo_cp_lab.production_schedule import lots, next_start

router = APIRouter(prefix='/api/demo/festo')
MANIFEST = Path(os.getenv('A4S_WORK_DIRECTORY', str(Path(__file__).resolve().parents[2] / 'temp'))) / 'production-runs/current.json'
ROOT = Path(__file__).resolve().parents[2]
RESET_LOCK = threading.Lock()
RESET_PREVIEWS = {}
RESET_PREVIEW_LOCK = threading.Lock()
RESET_TIMEOUT_SECONDS = 600


def _reset_scope():
    """Inspect only the isolated local MES; never use AAS instance order IDs."""
    config = Config.from_env()
    if config.endpoint != 'http://localhost:8092':
        raise HTTPException(409, 'Production reset is available only for the local OpenMES at localhost:8092')
    with MesClient(config) as mes:
        lines = mes.rows('/api/v1/lines')
        resolved = {}
        for code in ('A4S-FESTO', 'A4S-BACKUP'):
            matches = [row for row in lines if row.get('code') == code]
            if len(matches) != 1:
                raise HTTPException(409, f'Expected one local MES line with code {code}')
            resolved[code] = matches[0]
        products = [row for row in mes.rows('/api/v1/product-types')
                    if row.get('code') == 'A4S-DEMO-PART']
        if len(products) != 1:
            raise HTTPException(409, 'Expected one A4S demo product type')
        all_orders = mes.rows('/api/v1/work-orders')
        unrelated = [row for row in all_orders if not str(row.get('order_no', '')).startswith('A4S-')]
        if unrelated:
            raise HTTPException(409, f'{len(unrelated)} non-A4S orders exist; local reset is blocked')
        active = [row for row in all_orders if row.get('status') == 'IN_PROGRESS'
                  or float(row.get('produced_qty') or 0) > 0]
        if active:
            raise HTTPException(409, 'An A4S order is active or has produced quantity; reset is blocked')
        line_ids = {int(row['id']) for row in resolved.values()}
        maintenance = [row for row in mes.rows('/api/v1/maintenance-events')
                       if int(row.get('line_id') or 0) in line_ids]
    orders = sorted(all_orders, key=lambda row: int(row['id']))
    maintenance.sort(key=lambda row: int(row['id']))
    evidence = {'endpoint': config.endpoint,
                'lines': [(code, resolved[code]['id']) for code in ('A4S-FESTO', 'A4S-BACKUP')],
                'product_id': products[0]['id'],
                'orders': [(row['id'], row.get('order_no'), row.get('status'),
                            row.get('produced_qty'), row.get('updated_at')) for row in orders],
                'maintenance': [(row['id'], row.get('status'), row.get('updated_at'))
                                for row in maintenance]}
    digest = hashlib.sha256(json.dumps(evidence, sort_keys=True).encode()).hexdigest()
    return {'digest': digest, 'endpoint': config.endpoint,
            'orders': [{'id': row['id'], 'order_no': row.get('order_no'),
                        'status': row.get('status')} for row in orders],
            'maintenance_count': len(maintenance),
            'lines': [resolved[code] for code in ('A4S-FESTO', 'A4S-BACKUP')]}


@router.post('/reset/preview')
def preview_reset():
    if RESET_LOCK.locked():
        raise HTTPException(409, 'Another MES reset is in progress')
    try:
        scope = _reset_scope()
    except (MesError, ValueError) as exc:
        raise HTTPException(503, f'Cannot preview local MES reset ({type(exc).__name__})') from exc
    plan_id = uuid.uuid4().hex
    expires_at = time.time() + 300
    start_at = next_start(datetime.now(timezone.utc))
    schedule = lots(start_at, Config.from_env().server_timezone)
    with RESET_PREVIEW_LOCK:
        for key in list(RESET_PREVIEWS):
            if RESET_PREVIEWS[key]['expires_at'] < time.time():
                del RESET_PREVIEWS[key]
        if len(RESET_PREVIEWS) >= 100:
            raise HTTPException(429, 'Too many pending reset previews; wait for expiry')
        RESET_PREVIEWS[plan_id] = {'digest': scope['digest'], 'expires_at': expires_at,
                                   'start_at': start_at.isoformat()}
    return {'id': plan_id, 'expires_at': expires_at, 'endpoint': scope['endpoint'],
            'orders': scope['orders'], 'maintenance_count': scope['maintenance_count'],
            'lines': scope['lines'], 'new_order_count': 2, 'schedule': schedule,
            'effects': ['Back up the local OpenMES PostgreSQL database before changes.',
                        'Archive existing A4S orders and maintenance on the two demo lines.',
                        'Clear their planner changes, refresh demo shifts, and create two accepted 90-piece orders.',
                        'Leave AAS instances, blueprints, OPC signals, accounts and unrelated MES configuration unchanged.']}


@router.post('/reset/execute')
def execute_reset(body: dict = Body(...)):
    if body.get('confirmed') is not True:
        raise HTTPException(403, 'Explicit reset confirmation is required')
    if not RESET_LOCK.acquire(blocking=False):
        raise HTTPException(409, 'Another MES reset is in progress')
    try:
        with RESET_PREVIEW_LOCK:
            preview = RESET_PREVIEWS.pop(str(body.get('id', '')), None)
        if not preview or preview['expires_at'] < time.time():
            raise HTTPException(409, 'Reset preview expired or was already used; preview again')
        try:
            scope = _reset_scope()
        except (MesError, ValueError) as exc:
            raise HTTPException(503, f'Cannot recheck local MES reset ({type(exc).__name__})') from exc
        if scope['digest'] != preview['digest']:
            raise HTTPException(409, 'MES state changed since preview; refresh and preview again')
        start_at = datetime.fromisoformat(preview['start_at'])
        if start_at < datetime.now(timezone.utc) + timedelta(minutes=5):
            raise HTTPException(409, 'Approved schedule is too close to the current time; preview again')
        try:
            completed = subprocess.run(
                [sys.executable, str(ROOT / 'scripts/reset-production-plan.py'),
                 '--start-at', preview['start_at'], '--apply'],
                cwd=ROOT, capture_output=True, text=True, encoding='utf-8',
                timeout=RESET_TIMEOUT_SECONDS, check=True,
            )
            outcome = json.loads(completed.stdout)
            manifest = json.loads(MANIFEST.read_text(encoding='utf-8'))
            if outcome.get('run_id') != manifest.get('run_id') or len(manifest.get('baseline', [])) != 2:
                raise ValueError('Reset manifest read-back does not match the completed run')
            with MesClient(Config.from_env()) as mes:
                for expected in manifest['baseline']:
                    actual = mes.get(f"/api/v1/work-orders/{expected['id']}")['data']
                    if (actual.get('order_no') != expected['order_no'] or
                            actual.get('status') != 'ACCEPTED' or
                            not actual.get('planned_start_at')):
                        raise ValueError('New MES order read-back differs from the reset manifest')
        except (subprocess.CalledProcessError, subprocess.TimeoutExpired, OSError,
                ValueError, MesError, KeyError) as exc:
            raise HTTPException(502, 'MES reset failed or its outcome is uncertain. Inspect OpenMES and the local backup before retrying; do not assume a retry is safe.') from exc
        return {'run_id': outcome['run_id'], 'orders': outcome['orders'],
                'archived_orders': outcome['archived_orders'],
                'archived_maintenance': outcome['archived_maintenance'],
                'backup': outcome['backup'], 'aas_modified': False}
    finally:
        RESET_LOCK.release()


def epoch(value):
    if isinstance(value, (int, float)):
        return value
    if not value:
        return 0
    return datetime.fromisoformat(value.replace('Z','+00:00')).timestamp()


def in_window(rows, since, field='created_at'):
    return [row for row in rows if epoch(row.get(field)) >= since]


def delivery_summary(order, timezone_name):
    zone = ZoneInfo(timezone_name or 'UTC')
    due = order.get('due_date')
    end = order.get('planned_end_at')
    def local_day(value):
        if 'T' not in value and ' ' not in value:
            return value[:10]
        parsed = datetime.fromisoformat(value.replace('Z', '+00:00'))
        if parsed.tzinfo is None:
            parsed = parsed.replace(tzinfo=zone)
        return parsed.astimezone(zone).date().isoformat()
    due_day = local_day(due) if due else None
    return {'due_day': due_day,
            'planned_after_due': bool(due_day and end and local_day(end) > due_day)}


@router.get('/activity')
def activity(hours: int = Query(1, ge=0, le=168), current_run: bool = True):
    from .agent_activity import ledger_activity

    since = time.time() - hours * 3600 if hours else 0
    warnings = []
    mes_url = None
    try:
        projection = workflow(hours, current_run)
        since = max(since, epoch(projection['started_at']) if current_run else 0)
        mes_url = projection.get('mes_url')
    except Exception:
        projection = {'activity': [], 'started_at': None}
        warnings.append('MES / OPC activity unavailable; showing available agent ledger records.')
        # The last local reset timestamp is only a fallback when MES cannot be
        # reached; never use it to select orders or attach an asset.
        if current_run and MANIFEST.exists():
            try:
                projection['started_at'] = json.loads(MANIFEST.read_text(encoding='utf-8')).get('started_at')
                since = max(since, epoch(projection['started_at']))
            except (OSError, ValueError, TypeError):
                pass
    try:
        result = ledger_activity(since)
        if not result['ledger_available']:
            warnings.append('Agent ledger has not been created yet.')
    except Exception as exc:
        result = {'activity': [], 'ledger_available': False, 'ledger_truncated': False}
        warnings.append(f'Agent ledger unavailable ({type(exc).__name__}).')
    result['activity'].extend(projection['activity'])
    result['activity'].sort(key=lambda row: row['at'], reverse=True)
    return {**result, 'started_at': projection.get('started_at'), 'warnings': warnings,
            'scope': 'Workspace agent ledger + current Festo MES / OPC events',
            'since': datetime.fromtimestamp(since, timezone.utc).isoformat(),
            'mes_url': mes_url}


@router.get('/workflow')
def workflow(hours: int = Query(1, ge=0, le=168), current_run: bool = True):
    try:
        state = snapshot()
        manifest = json.loads(MANIFEST.read_text(encoding='utf-8')) if MANIFEST.exists() else {}
        orders = [state['order_a'], state['order_b']]
        order_ids = {o['id'] for o in orders}
        same_run = (manifest.get('mes_endpoint') == state['mes_url'] and
                    {o.get('id') for o in manifest.get('baseline', [])} == order_ids)
        begin = min(epoch(o.get('created_at')) for o in orders)
        if not begin:
            begin = epoch(manifest.get('started_at')) if same_run else 0
        run_id = orders[0]['order_no'].rpartition('-')[0]
        since = max(time.time()-hours*3600 if hours else 0, begin if current_run else 0)
        with MesClient(Config.from_env()) as mes:
            maintenance = mes.rows('/api/v1/maintenance-events', line_id=state['line']['id'])
            raw_changes = mes.get('/api/v1/schedule/changes')
        changes = raw_changes.get('changes', raw_changes.get('data', []))
        changes = [c for c in changes if c.get('work_order_id') in order_ids]
        # Initial placements are baseline setup; subsequent edits are replanning evidence.
        changes = in_window(changes, max(since, epoch(manifest.get('ready_at')) if same_run else begin))
        maintenance = in_window(maintenance, since)
        events = []
        endpoint = Endpoint.from_env().base_url
        for source in ('live','simulator'):
            report = service.status(target(state['asset']['id'], endpoint, source))
            events.extend(e for e in report['events'] if e['at'] >= since)
        events.sort(key=lambda e:e['at'])
        faults = [e for e in events if e['kind'] in ('FAULT_RAISED','BASELINE_FAULT','FAULT_OBSERVED_AFTER_GAP')]
        baseline = {o['id']:o for o in manifest.get('baseline', [])} if same_run else {}
        activity = [{'at':e['at'],'type':'OPC','title':{
            'FAULT_RAISED':'New station fault detected','BASELINE_FAULT':'Fault present when monitoring began',
            'RECOVERY_OBSERVED':'Fault signals cleared','COMMUNICATION_LOST':'OPC connection unavailable',
            'COMMUNICATION_RESTORED':'OPC connection restored',
        }.get(e['kind'],e['kind']), 'detail': 'Agent analysis requested' if e.get('claimed') else 'Evidence recorded',
            'evidence':e} for e in events]
        for m in maintenance:
            activity.append({'at':epoch(m['created_at']),'type':'Maintenance','title':f"Maintenance #{m['id']} created",
                             'detail':m['title'], 'evidence':m})
        for c in changes:
            activity.append({'at':epoch(c['created_at']),'type':'Planning','title':f"{c.get('order_no')} rescheduled",
                             'detail':c.get('user') or 'MES planner', 'evidence':c})
        return {'run_id':run_id,'started_at':datetime.fromtimestamp(begin, timezone.utc).isoformat() if begin else None,
                'since':datetime.fromtimestamp(since,timezone.utc).isoformat(),
                'orders':[{'current':o,'initial':baseline.get(o['id']),
                           **delivery_summary(o, Config.from_env().server_timezone)} for o in orders],
                'lines':[state['line'],state['backup']], 'maintenance':maintenance,
                'changes':changes, 'faults':faults,
                'activity':sorted(activity,key=lambda e:e['at'],reverse=True),
                'mes_url':state['mes_url'],'read_at':time.time()}
    except Exception as exc:
        raise HTTPException(503, f'Production workflow unavailable ({type(exc).__name__})') from exc
