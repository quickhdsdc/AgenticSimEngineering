"""Double-click launcher: verified local robot bundles, browser tabs, one console."""
import argparse
import json
from pathlib import Path
import shutil
import subprocess
import sys
import threading
from urllib.request import urlopen
import webbrowser

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
CASES = {
    'KUKA iiwa 14': ('examples/kuka-iiwa14-visual-study.json', 8765),
    'Unitree G1': ('examples/unitree-g1-visual-study.json', 8767),
}


def recent(directory, filename):
    return sorted((ROOT / directory).glob(f'*/{filename}'), key=lambda p: p.stat().st_mtime, reverse=True)


def prepare(task_path):
    from agenticsim.visualization.bundle import WEB, export, sha256, verify

    task = json.loads((ROOT / task_path).read_text(encoding='utf8'))
    for manifest_path in recent('work/runs/visualizations', 'bundle.json'):
        try:
            bundle = manifest_path.parent
            study = json.loads((bundle / 'study.json').read_text(encoding='utf8'))['payload']
            if study['task'] != task:
                continue
            manifest = json.loads(manifest_path.read_text(encoding='utf8'))
            if any(manifest['files'].get(p.name) != sha256(p) for p in WEB.iterdir() if p.is_file()):
                continue  # Re-export after viewer changes.
            return bundle, verify(bundle)
        except (OSError, ValueError, KeyError, TypeError):
            continue

    study_path = None
    for path in recent('work/runs/studies', 'study.json'):
        try:
            study = json.loads(path.read_text(encoding='utf8'))['payload']
            if study['task'] == task and study['status'] == 'completed' and all(
                    r.get('result', {}).get('provenance', {}).get('robot_playback') for r in study['runs']):
                study_path = path
                break
        except (OSError, ValueError, KeyError, TypeError):
            continue
    if study_path is None:
        worker = ROOT / '.venvs/newton/Scripts/python.exe'
        if not worker.is_file():
            raise RuntimeError('Newton environment missing. Run: powershell -File scripts/bootstrap.ps1 -Profile newton')
        print(f'Running real Newton simulation: {task_path}', flush=True)
        completed = subprocess.run([sys.executable, '-m', 'agenticsim', 'study', task_path,
                                    '--worker-python', str(worker)], cwd=ROOT, capture_output=True,
                                   text=True, encoding='utf8', errors='replace', timeout=700)
        if completed.returncode:
            raise RuntimeError(f'Simulation failed:\n{completed.stdout}\n{completed.stderr}')
        study_path = Path(json.loads(completed.stdout)['artifact'])
    if not (ROOT / 'components/visualization/node_modules/three/package.json').is_file():
        npm = shutil.which('npm.cmd' if sys.platform == 'win32' else 'npm')
        if not npm:
            raise RuntimeError('Node.js/npm missing. Install Node.js, then run: npm.cmd ci --prefix components/visualization')
        print('Installing local Three.js viewer resources...', flush=True)
        subprocess.run([npm, 'ci', '--prefix', 'components/visualization', '--no-audit', '--no-fund'],
                       cwd=ROOT, check=True)
    print(f'Exporting robot playback: {task_path}', flush=True)
    bundle = Path(export(study_path, ROOT / 'work/runs/visualizations')['bundle'])
    return bundle, verify(bundle)


def matching_server(port, manifest):
    try:
        with urlopen(f'http://127.0.0.1:{port}/bundle.json', timeout=2) as response:
            other = json.loads(response.read(256_000))
        return (other.get('study_id') == manifest['study_id']
                and other.get('study_payload_sha256') == manifest['study_payload_sha256']
                and other.get('files') == manifest['files'])
    except (OSError, ValueError):
        return False


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--no-browser', action='store_true', help='Print URLs without opening browser tabs')
    parser.add_argument('--check', action='store_true', help='Prepare and verify both bundles, then exit')
    args = parser.parse_args()
    from agenticsim.visualization.serve import server

    owned = []
    try:
        prepared = []
        for name, (task, port) in CASES.items():
            print(f'Preparing {name}...', flush=True)
            bundle, manifest = prepare(task)
            prepared.append((name, bundle, port, manifest))
        if args.check:
            print(json.dumps({'status': 'passed', 'bundles': {name: str(bundle) for name, bundle, _, _ in prepared}}))
            return
        for name, bundle, preferred, manifest in prepared:
            if matching_server(preferred, manifest):
                port = preferred
                print(f'{name}: using already running viewer')
            else:
                try:
                    httpd = server(bundle, preferred)
                except OSError as exc:
                    if getattr(exc, 'winerror', None) != 10048 and exc.errno not in (98, 10048):
                        raise
                    httpd = server(bundle, 0)  # Occupied port: do not stop another application.
                thread = threading.Thread(target=httpd.serve_forever, daemon=True)
                thread.start(); owned.append((httpd, thread))
                port = httpd.server_port
            url = f'http://127.0.0.1:{port}/'
            print(f'{name}: {url}', flush=True)
            if not args.no_browser:
                webbrowser.open_new_tab(url)
        if owned:
            input('\nKeep this window open. Press Enter or Ctrl+C to stop the viewers started here.\n')
        else:
            print('Both viewers are already running. You can close this window.')
    except (KeyboardInterrupt, EOFError):
        print('\nStopping viewers...')
    finally:
        for httpd, thread in owned:
            httpd.shutdown(); httpd.server_close(); thread.join(timeout=3)


if __name__ == '__main__':
    try:
        main()
    except (OSError, ValueError, KeyError, TypeError, RuntimeError, subprocess.SubprocessError) as exc:
        print(f'ERROR: {exc}', file=sys.stderr)
        sys.exit(1)
