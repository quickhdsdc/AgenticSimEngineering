"""Workspace-only PyChrono core environment, without optional GUI dependencies."""
import hashlib
import io
import json
from pathlib import Path
import subprocess
import tarfile
import urllib.request
import os

ROOT = Path(__file__).resolve().parents[1]
cache = ROOT / 'work/cache/mamba'
cache.mkdir(parents=True, exist_ok=True)
exe = cache / 'micromamba.exe'
if not exe.exists():
    raw = urllib.request.urlopen('https://micro.mamba.pm/api/micromamba/win-64/latest', timeout=120).read()
    with tarfile.open(fileobj=io.BytesIO(raw), mode='r:bz2') as archive:
        exe.write_bytes(archive.extractfile('Library/bin/micromamba.exe').read())
    (cache / 'bootstrap.json').write_text(json.dumps({'sha256': hashlib.sha256(raw).hexdigest(),
        'source': 'https://micro.mamba.pm/api/micromamba/win-64/latest'}), encoding='utf-8')
prefix = ROOT / '.venvs/chrono-native'
packages = ROOT / 'work/p'
packages.mkdir(exist_ok=True)
env = dict(os.environ, CONDA_PKGS_DIRS=str(packages))
lock = ROOT / 'environments/chrono/win64-explicit.txt'
specs = ['--file', str(lock)] if lock.exists() else ['python=3.12', 'pychrono=9.0.1', 'numpy=1.26.4']
action = 'install' if (prefix / 'conda-meta').exists() else 'create'
subprocess.run([str(exe), '--no-rc', '-r', str(ROOT / 'work/m'), action, '-y', '-p', str(prefix),
    '--override-channels', '-c', 'conda-forge', *specs], env=env, check=True)
rows = [json.loads(p.read_text()) for p in (prefix / 'conda-meta').glob('*.json')]
explicit = ['@EXPLICIT'] + [row['url'] + '#' + row['md5'] for row in sorted(rows, key=lambda r: r['name'])]
if not lock.exists():
    lock.write_text('\n'.join(explicit) + '\n', encoding='utf-8')
(cache / 'chrono-native-install.json').write_text(json.dumps({'packages': len(rows),
    'lock_sha256': hashlib.sha256(lock.read_bytes()).hexdigest(),
    'installer_sha256': hashlib.sha256(exe.read_bytes()).hexdigest()}, indent=2), encoding='utf-8')
