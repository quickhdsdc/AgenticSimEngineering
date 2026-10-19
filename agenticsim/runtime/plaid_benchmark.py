"""Sealed Tensile2D lifecycle with local holdout scoring and offline predictions."""
import argparse
import json
from pathlib import Path
import sys
import time
from uuid import uuid4

from agenticsim.adapters.execution import worker_environment
from agenticsim.adapters.process_tree import run_worker
from agenticsim.domain import require
from agenticsim.domain.plaid import validate_recipe
from agenticsim.runtime.benchmark import seal, verify
from agenticsim.runtime.calibration import directory_path, load_json, save_json, sha256
from agenticsim.runtime.paths import ROOT

WORKER_CODE = ROOT / 'agenticsim/adapters/plaid_worker.py'
OUTPUTS = {'prepare': ['dataset.json', 'source.json', 'train.csv', '.evaluation/data.npz', '.evaluation/splits.json'],
           'fit': ['recipe.json', 'fit.json', 'model.npz'],
           'score': ['score.json', 'holdout-field.png'], 'export': ['export.json', 'predictions.npz']}


def execute(phase, directory, *, recipe_path=None, worker_python=None, author='manual_cli'):
    require(phase in OUTPUTS, 'unknown PLAID phase')
    directory = directory_path(directory)
    recipe = None
    if phase == 'prepare':
        require(not directory.exists(), 'prepare requires a fresh directory')
        source = load_json(ROOT / 'examples/plaid-tensile2d-source.json')
        directory.mkdir(parents=True)
        (directory / '.evaluation').mkdir()
        save_json(directory / 'source.json', dict(source, worker_sha256=sha256(WORKER_CODE),
            contract_sha256=sha256(ROOT / 'agenticsim/domain/plaid.py')))
    else:
        verify(directory, 'prepare')
        source = load_json(directory / 'source.json')
        require(source['worker_sha256'] == sha256(WORKER_CODE) and
                source['contract_sha256'] == sha256(ROOT / 'agenticsim/domain/plaid.py'), 'PLAID algorithm changed')
        require(not (directory / (phase + '.json')).exists(), 'successful phase cannot be overwritten')
        if phase == 'fit':
            recipe = validate_recipe(load_json(recipe_path or ROOT / 'examples/plaid-tensile2d-recipe.json'))
            save_json(directory / 'recipe.json', recipe)
        else:
            verify(directory, 'fit')
        if phase == 'export':
            verify(directory, 'score')
    worker = Path(worker_python) if worker_python else ROOT / (
        '.venvs/plaid/Scripts/python.exe' if sys.platform == 'win32' else '.venvs/plaid/bin/python')
    require(worker.is_file(), 'install the isolated PLAID environment')
    started = time.monotonic()
    result = run_worker([str(worker), '-m', 'agenticsim.adapters.plaid_worker'],
                        payload=json.dumps({'phase': phase, 'directory': str(directory), 'source': source, 'recipe': recipe}),
                        cwd=ROOT, env=worker_environment(directory), timeout=600)
    (directory / '.evaluation' / (phase + '.stderr.log')).write_text(result.stderr, encoding='utf-8')
    require(result.returncode == 0, f'PLAID worker failed in {phase}; see private stderr log')
    report = json.loads(result.stdout)
    report.update(wall_seconds=time.monotonic()-started, author=author, worker_python=str(worker))
    save_json(directory / (phase + '.json' if phase != 'prepare' else 'dataset.json'), report)
    seal(directory, phase, OUTPUTS[phase])
    return report


def main(argv=None):
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('phase', choices=(*OUTPUTS, 'run'))
    parser.add_argument('--directory', type=Path)
    parser.add_argument('--recipe', type=Path)
    parser.add_argument('--worker-python', type=Path)
    args = parser.parse_args(argv)
    directory = args.directory or ROOT / 'work/runs/benchmarks' / ('plaid-' + uuid4().hex[:12])
    if args.phase != 'run' and args.directory is None:
        parser.error('manual phases require --directory')
    if args.phase == 'run':
        reports = {phase: execute(phase, directory, recipe_path=args.recipe, worker_python=args.worker_python,
                                 author='deterministic_baseline') for phase in OUTPUTS}
        report = {'directory': str(directory), 'results': reports}
    else:
        report = execute(args.phase, directory, recipe_path=args.recipe, worker_python=args.worker_python,
                         author='external_recipe' if args.recipe else 'manual_cli')
    print(json.dumps(report, ensure_ascii=False, indent=2, allow_nan=False))


if __name__ == '__main__':
    main()
