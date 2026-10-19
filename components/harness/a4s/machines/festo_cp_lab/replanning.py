"""Read current production references and propose maintenance-related MES placements."""
import argparse
from datetime import datetime, timedelta, timezone
import json
import shlex

from ...aasctl.client import BasyxClient, Endpoint
from ...kernel.config import load_env
from ...mesctl.client import Config, MesClient, MesError
from ...mesctl.commands import build_parser
from ...mesctl.executor import execute
from .mes_binding import current_demo_orders, resolve_lines


def instant(value):
    return datetime.fromisoformat(value.replace('Z', '+00:00'))


def proposal(asset, endpoint, maintenance_id=None):
    with BasyxClient(Endpoint.from_env(endpoint)) as aas:
        shell = aas.get_shell(asset)
        models = [aas.get_submodel(k['value']) for ref in shell.get('submodels', [])
                  for k in ref.get('keys', []) if k['type'] == 'Submodel']
    contexts = [m for m in models if m.get('idShort') == 'ProductionContext']
    if len(contexts) != 1:
        raise ValueError('Exactly one ProductionContext is required')
    context = {e['idShort']: e.get('value') for e in contexts[0]['submodelElements']}
    config = Config.from_env()
    with MesClient(config) as mes:
        def read(words):
            return execute(build_parser().parse_args(words), mes)
        line, backup = resolve_lines(context, mes)
        selected = current_demo_orders(mes, line, backup)
        orders = [read(['order', 'show', str(order['id'])])['data'] for order in selected]
        now = datetime.now(timezone.utc)
        maintenance = read(['maintenance','show',str(maintenance_id)])['data'] if maintenance_id else None
        if maintenance and (maintenance.get('line_id') != int(line['id']) or
                            maintenance.get('status') not in ('pending','in_progress')):
            raise ValueError('Maintenance must be pending/in progress on the affected line')
        # OpenMES reserves 60 minutes when its API event has no scheduled_end_at.
        start = instant(maintenance['scheduled_at']) if maintenance else max(
            instant(orders[0]['planned_start_at']), now + timedelta(minutes=10))
        end = instant(maintenance['scheduled_end_at']) if maintenance and maintenance.get('scheduled_end_at') else start + timedelta(minutes=60)
        placements = []
        reasons = ['Move the affected lot to compatible Cell 2 to protect its delivery time.',
                   'Restart the Cell 1 plan after the maintenance reservation and a 15-minute release allowance.']
        for index, order in enumerate(orders):
            duration = instant(order['planned_end_at']) - instant(order['planned_start_at'])
            begin = max(instant(order['planned_start_at']), now+timedelta(minutes=10)) if index == 0 else max(instant(order['planned_start_at']), end+timedelta(minutes=15))
            destination = backup['id'] if index == 0 else line['id']
            words = ['schedule','move',str(order['id']),'--line',str(destination),
                     '--start',begin.isoformat(),'--end',(begin+duration).isoformat(),'--dry-run']
            if order['line_id'] == int(destination) and abs((instant(order['planned_start_at'])-begin).total_seconds()) < 1:
                continue
            preview = read(words)  # Existing product/routing/capacity preflight, no commit.
            capacity = preview.get('preflight', {}).get('capacity', {})
            resources = capacity.get('grid', capacity).get('resources', [])
            resource = next((r for r in resources if r['id'] == int(destination)), {})
            cells = resource.get('cells', {})
            available = sum(float(c.get('available_h', 0)) for day,c in cells.items()
                            if begin.date().isoformat() <= day <= (begin+duration).date().isoformat())
            if available <= 0:
                raise MesError('Target line has no available capacity in the proposed dates')
            placements.append({'order_id':order['id'],'order_no':order['order_no'],
                'before':{k:order.get(k) for k in ('line_id','planned_start_at','planned_end_at','status')},
                'after':{'line_id':int(destination),'planned_start_at':begin.isoformat(),'planned_end_at':(begin+duration).isoformat()},
                'reason':reasons[index], 'preview_command':'mes '+shlex.join(words),
                'commit_command':'mes '+shlex.join(words[:-1]),'preview':preview})
        return {'current_time':now.isoformat(),'mes_line_code':line['code'],
                'maintenance':maintenance,'maintenance_start':start.isoformat(),'maintenance_end':end.isoformat(),
                'planning_allowance_minutes':60,'placements':placements,
                'note':'Read-only proposal. Maintenance and each order move require approval and read-back. Recheck timestamps and conflicts at execution.'}


def main():
    load_env()
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--asset', required=True)
    parser.add_argument('--endpoint', required=True)
    parser.add_argument('--maintenance-id', type=int)
    args = parser.parse_args()
    print(json.dumps(proposal(args.asset,args.endpoint,args.maintenance_id), indent=2))


if __name__ == '__main__':
    main()
