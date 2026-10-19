"""Leakage, metric and evidence failures relevant to Tensile2D evaluation."""
from pathlib import Path
import shutil
import subprocess
import unittest
from uuid import uuid4

from agenticsim.domain.plaid import split_ids, validate_recipe
from agenticsim.runtime.benchmark import seal
from agenticsim.runtime.calibration import save_json
from agenticsim.runtime.plaid_benchmark import execute
from agenticsim.runtime.paths import ROOT


class PlaidContractTests(unittest.TestCase):
    def test_split_is_case_disjoint_and_does_not_depend_on_input_order(self):
        result = split_ids(list(range(500)))
        self.assertEqual(result, split_ids(list(reversed(range(500)))))
        self.assertEqual({k: len(v) for k, v in result.items()}, {'train': 300, 'validation': 100, 'holdout': 100})
        self.assertEqual(len(set(sum(result.values(), []))), 500)
        with self.assertRaises(ValueError):
            split_ids([0]*500)

    def test_recipe_rejects_code_unknown_parameters_and_unbounded_search(self):
        valid = {'method': 'inverse_distance_mesh_knn', 'neighbors': [1, 4], 'geometry_weights': [1]}
        validate_recipe(valid)
        for update in ({'neighbors': [True]}, {'neighbors': [500]}, {'neighbors': [1, 1]},
                       {'geometry_weights': [float('nan')]}, {'python': 'print(1)'}, {'method': 'FEM'}):
            with self.assertRaises(ValueError):
                validate_recipe(dict(valid, **update))

    def test_tampered_prepare_rejected_before_any_worker(self):
        directory = ROOT / 'work/runs/tests' / uuid4().hex
        try:
            (directory / '.evaluation').mkdir(parents=True)
            save_json(directory / 'dataset.json', {'scope': 'holdout'})
            seal(directory, 'prepare', ['dataset.json'])
            save_json(directory / 'dataset.json', {'scope': 'official_score'})
            with self.assertRaisesRegex(ValueError, 'evidence changed'):
                execute('fit', directory, worker_python=Path('nonexistent'))
        finally:
            shutil.rmtree(directory, ignore_errors=True)

    def test_native_metric_and_no_pickle_class_execution(self):
        worker = ROOT / '.venvs/plaid/Scripts/python.exe'
        if not worker.is_file():
            self.skipTest('install isolated PLAID worker')
        code = '''
import pickle
import numpy as np
from agenticsim.adapters.plaid_worker import metrics, restricted_sample, Surrogate
class Fixed:
    samples = {1: {'fields':np.full((3,5),2.), 'scalars':np.full(3,4.)}}
    def predict(self, *args): return np.full((3,5),3.), np.full(3,5.)
result = metrics(Fixed(), [1], {'neighbors':1,'geometry_weight':1})
assert abs(result['total']-0.40625)<1e-12, result
assert result['rrmse']['U1']==0.5 and result['rrmse']['max_von_mises']==0.25
raw = pickle.dumps({'a':np.array([1.,2.])}, protocol=5)
assert restricted_sample(raw)['a'].tolist()==[1.,2.]
try: restricted_sample(pickle.dumps(eval))
except ValueError: pass
else: raise AssertionError('unrestricted pickle class loading')
samples = {i:{'xy':np.array([[0.,0.],[1.,0.],[0.,1.]]),
              'features':np.full(11,float(i)), 'fields':np.full((3,5),float(i+1)),
              'scalars':np.full(3,float(i+1))} for i in range(3)}
before = Surrogate(samples,[0,1]).predict(2,1,1)
samples[2]['fields'][:]=1e9; samples[2]['scalars'][:]=1e9
after = Surrogate(samples,[0,1]).predict(2,1,1)
assert all(np.array_equal(a,b) for a,b in zip(before,after)), 'prediction consumed target labels'
print('worker metric, label isolation and decoder verified')
'''
        result = subprocess.run([str(worker), '-c', code], cwd=ROOT, capture_output=True, text=True, timeout=45)
        self.assertEqual(result.returncode, 0, result.stderr)


if __name__ == '__main__':
    unittest.main()
