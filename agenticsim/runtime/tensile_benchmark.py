"""Sealed FEM identification, residual evaluation and official-format export."""
import argparse
import json
from pathlib import Path
import shutil
import sys
from uuid import uuid4

from agenticsim.adapters.execution import worker_environment
from agenticsim.adapters.process_tree import run_worker
from agenticsim.domain import require
from agenticsim.domain.tensile import validate_model_spec
from agenticsim.runtime.benchmark import seal, verify
from agenticsim.runtime.calibration import directory_path, load_json, save_json, sha256
from agenticsim.runtime.paths import ROOT

CODE = ('adapters/tensile_worker.py', 'adapters/tensile_fem.py', 'adapters/tensile_submission.py',
        'adapters/plaid_worker.py', 'domain/tensile.py', 'domain/plaid.py', 'runtime/tensile_benchmark.py')
OUTPUTS = {
    'prepare': ['source.json', 'dataset.json', 'train.csv', '.evaluation/data.npz', '.evaluation/splits.json',
                'protocol/manifest.json', 'protocol/submission_info.json', 'protocol/rules.json',
                'protocol/dataset_info.json', 'protocol/competition_info.json'],
    'model': ['generated-model.json'],
    'fit': ['fit.json', '.evaluation/fem-training-validation.npz', '.evaluation/fem-training-validation.npz.json'],
    'score': ['score.json', 'holdout-field.png', '.evaluation/fem-holdout.npz',
              '.evaluation/fem-holdout.npz.json', '.evaluation/holdout-predictions.npz'],
    'export': ['export.json', 'submission.pkl', 'ood-predictions.npz'],
}


def code_hashes():
    return {name:sha256(ROOT/'agenticsim'/name) for name in CODE}


def execute(phase, directory, *, model_spec=None, worker_python=None, author='manual_cli'):
    require(phase in OUTPUTS, 'unknown Tensile2D phase')
    directory = directory_path(directory)
    if phase == 'prepare':
        require(not directory.exists(), 'prepare requires a fresh directory')
        protocol = ROOT/'work/cache/datasets/plaid-tensile2d/protocol'
        manifest = load_json(protocol/'manifest.json')
        # Snapshot verification precedes copying; remote content is never executable.
        for name, digest in manifest['sha256'].items():
            require(sha256(protocol/name) == digest, 'official protocol snapshot changed')
        source = dict(load_json(ROOT/'examples/plaid-tensile2d-source.json'), code_hashes=code_hashes())
        directory.mkdir(parents=True)
        (directory/'.evaluation').mkdir()
        shutil.copytree(protocol, directory/'protocol')
        save_json(directory/'source.json', source)
    else:
        for preceding in list(OUTPUTS)[:list(OUTPUTS).index(phase)]:
            verify(directory, preceding)
        source = load_json(directory/'source.json')
        require(source['code_hashes'] == code_hashes(), 'Tensile2D implementation changed after preparation')
        require(not (directory/(phase+'.json')).exists() and
                not (phase == 'model' and (directory/'generated-model.json').exists()),
                'successful phase cannot be overwritten')
    if phase == 'model':
        spec = validate_model_spec(load_json(model_spec or ROOT/'examples/tensile2d-model.json'))
        report = {'specification':spec, 'model':'small-strain plane-strain TRI3 with learned residuals',
                  'boundary_mapping':'CGNS Bottom uy=0; BottomLeft ux=0; Top traction=-P',
                  'material_hypothesis':'Young=p5*scale; Poisson and scale identified from training-only elastic cases',
                  'nonlinear_law':'unknown; p1-p4 are residual features, not identified constitutive parameters',
                  'stress_recovery':'uniform incident-element average',
                  'qoi_convention':'in-plane von Mises sqrt(s11^2+s22^2-s11*s22+3*s12^2)',
                  'units':'publisher numerical units retained; physical unit mapping not established',
                  'validation':'fixed 300 train / 100 validation / 100 independent holdout; no official hidden labels',
                  'author':author}
        save_json(directory/'generated-model.json',report)
    else:
        worker = Path(worker_python) if worker_python else ROOT/(
            '.venvs/plaid/Scripts/python.exe' if sys.platform == 'win32' else '.venvs/plaid/bin/python')
        require(worker.is_file(), 'install the isolated PLAID environment')
        result = run_worker([str(worker), '-m', 'agenticsim.adapters.tensile_worker'],
                            payload=json.dumps({'phase':phase,'directory':str(directory),'source':source}),
                            cwd=ROOT,env=worker_environment(directory),timeout=600)
        (directory/'.evaluation'/(phase+'.stderr.log')).write_text(result.stderr,encoding='utf-8')
        require(result.returncode == 0, f'Tensile2D {phase} failed; see private stderr log')
        report = json.loads(result.stdout)
        report.update(author=author,worker_python=str(worker))
        save_json(directory/('dataset.json' if phase == 'prepare' else phase+'.json'),report)
    seal(directory,phase,OUTPUTS[phase])
    return report


def main(argv=None):
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('phase',choices=(*OUTPUTS,'run'))
    parser.add_argument('--directory',type=Path)
    parser.add_argument('--model-spec',type=Path)
    parser.add_argument('--worker-python',type=Path)
    args = parser.parse_args(argv)
    if args.phase != 'run' and args.directory is None:
        parser.error('manual phases require --directory')
    directory = args.directory or ROOT/'work/runs/benchmarks'/('tensile-'+uuid4().hex[:12])
    reports = {phase:execute(phase,directory,model_spec=args.model_spec,worker_python=args.worker_python,
                            author='external_model' if args.model_spec else 'deterministic_baseline')
               for phase in (OUTPUTS if args.phase == 'run' else [args.phase])}
    print(json.dumps({'directory':str(directory),'results':reports},indent=2,allow_nan=False))


if __name__ == '__main__':
    main()
