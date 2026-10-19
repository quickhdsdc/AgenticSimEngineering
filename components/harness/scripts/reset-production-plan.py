"""Reset the isolated local MES and seed a current, repeatable housing production plan."""
import argparse
from datetime import datetime, timedelta, timezone
import json
from pathlib import Path
import subprocess
import uuid

from a4s.machines.festo_cp_lab.production_schedule import lots, next_start

ROOT = Path(__file__).resolve().parents[1]


def main(argv=None):
    from a4s.kernel.config import load_env
    load_env(str(ROOT / '.env'))
    from a4s.mesctl.client import Config, MesClient
    from a4s.mesctl.commands import build_parser
    from a4s.mesctl.executor import execute
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--apply', action='store_true')
    parser.add_argument('--start-at', help='Approved ISO 8601 slot from the UI preview; otherwise use the current clock')
    args = parser.parse_args(argv)
    config = Config.from_env()
    if config.endpoint != 'http://localhost:8092':
        raise RuntimeError('This reset only supports the local OpenMES at localhost:8092')
    now = datetime.now(timezone.utc)
    start = datetime.fromisoformat(args.start_at.replace('Z', '+00:00')) if args.start_at else next_start(now)
    if start.tzinfo is None or start.utcoffset() is None:
        raise ValueError('--start-at must include a timezone')
    start = start.astimezone(timezone.utc)
    if args.apply and start < now + timedelta(minutes=5):
        raise ValueError('Approved schedule is too close to the current time; preview again')
    schedule = lots(start, config.server_timezone)
    run = now.strftime('%Y%m%dT%H%M%SZ') + '-' + uuid.uuid4().hex[:6]
    directory = ROOT / 'temp/production-runs' / run
    if not args.apply:
        print(json.dumps({'apply': False, 'start': start.isoformat(), 'schedule': schedule, 'orders': 2,
                          'scope': 'MES only: archive local A4S orders/maintenance, clear their planner changes, seed shifts and two new orders; no AAS or monitoring changes'}))
        return
    directory.mkdir(parents=True)
    compose = ['docker', 'compose', '--env-file', str(ROOT/'.env.openmes'), '-p', 'a4s-openmes',
               '-f', str(ROOT/'docker-compose.openmes.yml')]
    with (directory/'openmes-before.dump').open('wb') as backup:
        subprocess.run([*compose, 'exec', '-T', 'postgres', 'pg_dump', '-U', 'openmes', '-d', 'openmes', '-Fc'],
                       stdout=backup, check=True)
    if start < datetime.now(timezone.utc) + timedelta(minutes=5):
        raise RuntimeError('Backup took too long for the approved slots; no MES reset was sent. Preview again.')
    php = (ROOT/'scripts/openmes-reset-plan.php').read_text(encoding='utf-8')
    result = subprocess.run([*compose, 'exec', '-T', 'backend', 'php'], input=php,
                            text=True, encoding='utf-8', capture_output=True, check=True)
    markers = [line.removeprefix('A4S_RESET=') for line in result.stdout.splitlines()
               if line.startswith('A4S_RESET=')]
    if not markers:
        (directory/'reset-error.log').write_text(result.stdout + result.stderr, encoding='utf-8')
        raise RuntimeError(f'Reset transaction failed; inspect {directory}/reset-error.log')
    data = json.loads(markers[-1])
    baseline = []
    with MesClient(config) as mes:
        def call(words):
            result = execute(build_parser().parse_args(words), mes)
            with (directory/'execution.jsonl').open('a', encoding='utf-8') as log:
                log.write(json.dumps({'command': words, 'result': result})+'\n')
            if result.get('verified') is False:
                raise RuntimeError('MES verification failed; inspect execution.jsonl')
            return result
        for index, slot in enumerate(schedule):
            suffix = slot['lot']
            # Set due_date as well as the minute placement: OpenMES backlog uses it.
            created = call(['order','create','--order-no',f'A4S-381-{run}-{suffix}',
                '--line',str(data['line_id']),'--product-type-id',str(data['product_id']),
                '--planned-qty','90','--priority',str(2-index),'--due-date',slot['due_date'],
                '--description',f'Phone housing lot {suffix}. Stamp and turn over. 90 units / 45 minute planning slot.'])
            order_id = created['read_back']['data']['id']
            call(['order','accept',str(order_id)])
            call(['schedule','move',str(order_id),'--line',str(data['line_id']),
                  '--start',slot['planned_start_at'],'--end',slot['planned_end_at']])
            baseline.append(call(['order','show',str(order_id)])['data'])
    manifest = {'run_id':run,'started_at':now.isoformat(),'ready_at':datetime.now(timezone.utc).isoformat(),
                'mes_endpoint':config.endpoint, 'baseline':baseline, 'backup':str(directory/'openmes-before.dump'),**data}
    (directory/'baseline.json').write_text(json.dumps(manifest,indent=2),encoding='utf-8')
    current = ROOT/'temp/production-runs/current.json'
    staging = current.with_suffix('.json.tmp')
    staging.write_text(json.dumps(manifest,indent=2),encoding='utf-8')
    staging.replace(current)
    print(json.dumps({'run_id':run,'backup':manifest['backup'],
                      'orders':[{k:o[k] for k in ('id','order_no','planned_start_at','planned_end_at')} for o in baseline],
                      'aas_modified':False, 'archived_orders':data['archived_orders'],'archived_maintenance':data['archived_maintenance']},indent=2))


if __name__ == '__main__':
    main()
