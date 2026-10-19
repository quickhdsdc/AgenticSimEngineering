"""Versioned constrained model generation pilots using native reference simulations."""
import argparse
import json
import math
from pathlib import Path
import sys
from uuid import uuid4

from agenticsim.adapters.execution import worker_environment
from agenticsim.adapters.process_tree import run_worker
from agenticsim.domain import require
from agenticsim.runtime.benchmark import seal, verify
from agenticsim.runtime.calibration import directory_path, load_json, save_json, sha256
from agenticsim.runtime.paths import ROOT

SOURCES = ROOT / 'components/model_benchmarks'


def gold_models(suite):
    if suite == 'factoryflow':
        return {
            'S1': {'nodes': [{'id': 'SRC', 'kind': 'source', 'inter_arrival_time': 1.2},
                            {'id': 'MACHINE', 'kind': 'machine', 'processing_delay': 2, 'work_capacity': 2},
                            {'id': 'SINK', 'kind': 'sink'}],
                   'buffers': [{'id': 'BUFFER1', 'from': 'SRC', 'to': 'MACHINE', 'capacity': 3},
                               {'id': 'BUFFER2', 'from': 'MACHINE', 'to': 'SINK', 'capacity': 3}]},
            'S2': {'nodes': [{'id': 'SRC', 'kind': 'source', 'inter_arrival_time': 1.2},
                            {'id': 'M1', 'kind': 'machine', 'processing_delay': 2, 'work_capacity': 1},
                            {'id': 'M2', 'kind': 'machine', 'processing_delay': 2, 'work_capacity': 1},
                            {'id': 'sink', 'kind': 'sink'}],
                   'buffers': [{'id': 'B_src_1', 'from': 'SRC', 'to': 'M1', 'capacity': 2},
                               {'id': 'B_1_2', 'from': 'M1', 'to': 'M2', 'capacity': 2},
                               {'id': 'B_2_sink', 'from': 'M2', 'to': 'sink', 'capacity': 2}]}}
    require(suite == 'chronobench', 'unknown modeling suite')
    return {
        'mass_spring_damper': {'kind': 'mass_spring_damper', 'gravity': [0, 0, 0], 'mass': 1,
            'inertia': [1, 1, 1], 'position': [-1, -3, 0], 'body_anchor': [0, 0, 0],
            'ground_anchor': [-1, 0, 0], 'rest_length': 1.5, 'spring_coefficient': 50, 'damping_coefficient': 1},
        'pendulum': {'kind': 'pendulum', 'gravity': [0, -9.81, 0], 'mass': 1,
            'inertia': [0.2, 1, 1], 'position': [1, 0, 1], 'pivot': [0, 0, 1]}}


def canonical(model, suite):
    if suite == 'factoryflow':
        return {key: sorted(value, key=lambda row: row['id']) for key, value in model.items()}
    return model


def validate_model(model, suite):
    require(isinstance(model, dict), 'model must be a JSON object')
    if suite == 'factoryflow':
        require(set(model) == {'nodes', 'buffers'}, 'factory model requires nodes and buffers only')
        require(isinstance(model['nodes'], list) and 2 <= len(model['nodes']) <= 20, 'invalid node count')
        require(isinstance(model['buffers'], list) and 1 <= len(model['buffers']) <= 30, 'invalid buffer count')
        ids = []
        for row in model['nodes']:
            require(isinstance(row, dict), 'node must be an object')
            kind = row.get('kind')
            fields = {'source': {'id', 'kind', 'inter_arrival_time'},
                      'machine': {'id', 'kind', 'processing_delay', 'work_capacity'},
                      'sink': {'id', 'kind'}}
            require(kind in fields and set(row) == fields[kind], 'invalid node fields')
            require(isinstance(row['id'], str) and row['id'].isalnum() and len(row['id']) <= 50, 'invalid node id')
            ids.append(row['id'])
            for key in fields[kind] - {'id', 'kind'}:
                number(row[key], 0.01, 100, integer=key == 'work_capacity')
        require(len(set(ids)) == len(ids), 'duplicate node id')
        edge_ids = []
        for row in model['buffers']:
            require(isinstance(row, dict) and set(row) == {'id', 'from', 'to', 'capacity'}, 'invalid buffer fields')
            require(isinstance(row['id'], str) and len(row['id']) <= 50, 'invalid buffer id')
            edge_ids.append(row['id'])
            require(row['from'] in ids and row['to'] in ids and row['from'] != row['to'], 'invalid buffer endpoints')
            number(row['capacity'], 1, 100, integer=True)
        require(len(set(edge_ids + ids)) == len(edge_ids + ids), 'duplicate graph id')
    else:
        require(model.get('kind') in gold_models(suite), 'unsupported Chrono model kind')
        expected = gold_models(suite)[model['kind']]
        require(set(model) == set(expected), 'invalid Chrono model fields')
        for key, value in model.items():
            if key == 'kind':
                continue
            if isinstance(expected[key], list):
                require(isinstance(value, list) and len(value) == 3, 'expected 3D vector')
                for entry in value:
                    number(entry, -100, 100)
                if key == 'inertia':
                    require(all(v > 0 for v in value), 'inertia must be positive')
            else:
                number(value, 0.001, 100)
    return model


def number(value, lower, upper, integer=False):
    require(type(value) in (int, float) and math.isfinite(value) and lower <= value <= upper,
            'numeric value is outside the supported finite range')
    require(not integer or type(value) is int, 'integer required')


def simulate(suite, task, model, horizon, worker, directory, *, reference=False, normalize=False):
    env = worker_environment(directory)
    if suite == 'chronobench':
        dll = worker.parent / 'Library/bin'
        env['PATH'] = str(dll) + ';' + env.get('PATH', '') if sys.platform == 'win32' else env.get('PATH', '')
    result = run_worker([str(worker), '-m', 'agenticsim.adapters.modelbench_worker'],
        payload=json.dumps({'suite': suite, 'task': task, 'model': model, 'horizon': horizon, 'reference': reference, 'normalize': normalize}),
        cwd=ROOT, env=env, timeout=60)
    label = f'{task}-{horizon}-' + ('reference-normalized' if normalize and reference else 'reference' if reference else 'candidate')
    log = directory / '.evaluation' / (label + '.stderr.log')
    log.write_text(result.stderr, encoding='utf-8')
    require(result.returncode == 0, f'native worker failed: {label}; see private worker log')
    return json.loads(result.stdout)


def difference(candidate, reference):
    if isinstance(reference, dict):
        require(set(candidate) == set(reference), 'worker result keys disagree')
        return max((difference(candidate[k], v) for k, v in reference.items()), default=0)
    if isinstance(reference, list):
        require(isinstance(candidate, list) and len(candidate) == len(reference), 'worker trajectory length mismatch')
        return max((difference(a, b) for a, b in zip(candidate, reference)), default=0)
    if type(reference) in (int, float):
        require(type(candidate) in (int, float) and math.isfinite(candidate) and math.isfinite(reference), 'nonfinite worker result')
        return abs(candidate - reference)
    require(candidate == reference, 'worker metadata disagrees')
    return 0


def execute(phase, directory, suite, submission=None, worker_python=None, *, author='external_submission'):
    directory = directory_path(directory)
    require(suite in ('factoryflow', 'chronobench'), 'unknown modeling suite')
    manifest = load_json(SOURCES / 'manifest.json')[suite]
    for path, digest in manifest['files'].items():
        require(sha256(SOURCES / suite / path) == digest, 'upstream pilot resource integrity mismatch')
    if phase == 'prepare':
        require(not directory.exists(), 'prepare requires a fresh directory')
        directory.mkdir(parents=True)
        (directory / '.evaluation').mkdir()
        cases = []
        for task, model in gold_models(suite).items():
            prompt = SOURCES / suite / (f'{task}_input.txt' if suite == 'factoryflow' else f'{task}/input1.txt')
            cases.append({'task_id': task, 'original_prompt': prompt.read_text(encoding='utf-8'),
                          'required_model_fields': list(model), 'source': 'reference_simulation'})
        save_json(directory / 'cases.json', cases)
        save_json(directory / '.evaluation/gold.json', gold_models(suite))
        save_json(directory / 'dataset.json', {'suite': suite, 'commit': manifest['commit'],
            'archive_sha256': manifest['archive_sha256'], 'tasks': len(cases),
            'contract': 'agenticsim-factory-capacity-pilot-v2' if suite == 'factoryflow' else 'agenticsim-constrained-model-pilot-v1',
            'source': 'reference_simulation', 'normalize_initial_resources': suite == 'factoryflow',
            'horizons': [25, 50] if suite == 'factoryflow' else [1, 2], 'tolerance': 1e-8,
            'physical_validation': 'not_established', 'published_score_comparable': False,
            'adaptation': 'JSON model authoring; delegated numeric context; finite headless runs; no J-LLM judge or GUI acceptance'})
        seal(directory, 'prepare', ['cases.json', 'dataset.json', '.evaluation/gold.json'])
        return load_json(directory / 'dataset.json')
    verify(directory, 'prepare')
    dataset = load_json(directory / 'dataset.json')
    require(dataset['suite'] == suite, 'suite differs from prepared dataset')
    require(phase == 'score', 'unknown modeling phase')
    require(not (directory / 'score.json').exists(), 'score is already frozen; start a new run')
    require(submission is not None, 'submission path is required')
    submission = Path(submission).resolve()
    require(submission.is_relative_to((ROOT / 'work/runs').resolve()) and submission.is_file(), 'submission must be in work/runs')
    require(submission.stat().st_size < 100_000, 'submission too large')
    rows = load_json(submission)
    gold = load_json(directory / '.evaluation/gold.json')
    require(isinstance(rows, dict) and set(rows) == set(gold), 'submit exactly the prepared task IDs')
    for model in rows.values():
        validate_model(model, suite)
    worker = Path(worker_python or ROOT / ('.venvs/chrono-native/python.exe' if suite == 'chronobench' else '.venvs/factoryflow/Scripts/python.exe')).resolve()
    require(worker.is_file(), 'install the isolated native worker environment')
    save_json(directory / '.evaluation/submission.json', rows)
    seal(directory, 'submission', ['.evaluation/submission.json'])
    results = []
    normalize = dataset.get('normalize_initial_resources', False)
    for task, model in rows.items():
        metrics = []
        for horizon in dataset['horizons']:
            ref = simulate(suite, task, None, horizon, worker, directory, reference=True, normalize=normalize)
            candidate = simulate(suite, task, model, horizon, worker, directory, normalize=normalize)
            save_json(directory / '.evaluation' / f'{task}-{horizon}-reference.json', ref)
            save_json(directory / f'{task}-{horizon}-candidate.json', candidate)
            row = {'horizon': horizon, 'max_absolute_residual': difference(candidate, ref)}
            if normalize:
                original = simulate(suite, task, None, horizon, worker, directory, reference=True)
                save_json(directory / '.evaluation' / f'{task}-{horizon}-original.json', original)
                row['original_unmodified_reference_kpi_residual'] = difference({k: candidate[k] for k in original}, original)
                row['capacities_realized'] = all(r['declared'] == r['realized'] for r in candidate['capacity_audit'])
            metrics.append(row)
        structure = canonical(model, suite) == canonical(gold[task], suite)
        results.append({'task_id': task, 'topology_and_parameters_match': structure, 'runs': metrics,
            'accepted': structure and all(m['max_absolute_residual'] <= dataset['tolerance'] and m.get('capacities_realized', True) for m in metrics)})
    score = {'suite': suite, 'contract': dataset['contract'], 'tasks': len(results), 'results': results,
        'passed': sum(r['accepted'] for r in results), 'accepted': all(r['accepted'] for r in results),
        'submission_sha256': sha256(submission), 'source': 'reference_simulation',
        'author': author, 'worker_python': str(worker), 'commit': dataset['commit'],
        'gui_rendered': False, 'physical_validation': 'not_established', 'published_score_comparable': False}
    save_json(directory / 'score.json', score)
    names = ['score.json', '.evaluation/submission.json'] + [p.relative_to(directory).as_posix()
        for p in directory.glob('*-candidate.json')] + [p.relative_to(directory).as_posix()
        for p in (directory / '.evaluation').glob('*-reference.json')] + [p.relative_to(directory).as_posix()
        for p in (directory / '.evaluation').glob('*-original.json')]
    seal(directory, 'score', names)
    return score


def main(argv=None):
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('phase', choices=('prepare', 'score', 'run'))
    parser.add_argument('--suite', required=True, choices=('factoryflow', 'chronobench'))
    parser.add_argument('--directory', type=Path)
    parser.add_argument('--submission', type=Path)
    parser.add_argument('--worker-python', type=Path)
    args = parser.parse_args(argv)
    directory = args.directory or ROOT / 'work/runs/modelbench' / (args.suite + '-' + uuid4().hex[:12])
    if args.phase == 'run':
        execute('prepare', directory, args.suite)
        # Explicit oracle baseline; this is never an Agent result.
        baseline = directory / '.evaluation/baseline.json'
        save_json(baseline, gold_models(args.suite))
        score = execute('score', directory, args.suite, baseline, args.worker_python,
                        author='oracle_reference_baseline')
    else:
        score = execute(args.phase, directory, args.suite, args.submission, args.worker_python)
    print(json.dumps({'directory': str(directory), **score}, indent=2))


if __name__ == '__main__':
    main()
