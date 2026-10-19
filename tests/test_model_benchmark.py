"""Failures that would otherwise let a modeling pilot report misleading success."""
import copy
import json
import shutil
import unittest
from uuid import uuid4

from agenticsim.runtime.model_benchmark import canonical, difference, execute, gold_models, validate_model
from agenticsim.runtime.paths import ROOT


class ModelBenchmarkTests(unittest.TestCase):
    def test_model_does_not_accept_code_nonfinite_or_unknown_fields(self):
        model = copy.deepcopy(gold_models('chronobench')['pendulum'])
        model['mass'] = float('nan')
        with self.assertRaises(ValueError):
            validate_model(model, 'chronobench')
        model['mass'] = 1
        model['python'] = 'import os'
        with self.assertRaises(ValueError):
            validate_model(model, 'chronobench')
        factory = copy.deepcopy(gold_models('factoryflow')['S1'])
        factory['buffers'][0]['to'] = 'missing'
        with self.assertRaises(ValueError):
            validate_model(factory, 'factoryflow')

    def test_topology_is_required_even_when_metrics_match(self):
        original = gold_models('factoryflow')['S2']
        reversed_model = copy.deepcopy(original)
        reversed_model['nodes'].reverse()
        self.assertEqual(canonical(original, 'factoryflow'), canonical(reversed_model, 'factoryflow'))
        reversed_model['buffers'][0]['capacity'] = 3
        self.assertNotEqual(canonical(original, 'factoryflow'), canonical(reversed_model, 'factoryflow'))
        self.assertEqual(difference({'received': 10}, {'received': 10}), 0)

    def test_worker_cannot_pass_nonfinite_or_short_trajectory(self):
        with self.assertRaises(ValueError):
            difference([float('nan')], [1.0])
        with self.assertRaises(ValueError):
            difference([[1, 2]], [[1, 2], [3, 4]])

    def test_tampered_dataset_rejected_before_worker_execution(self):
        directory = ROOT / 'work/runs/tests' / uuid4().hex
        try:
            execute('prepare', directory, 'factoryflow')
            dataset = json.loads((directory / 'dataset.json').read_text())
            dataset['tolerance'] = 1e9
            (directory / 'dataset.json').write_text(json.dumps(dataset))
            with self.assertRaises(ValueError):
                execute('score', directory, 'factoryflow')
        finally:
            shutil.rmtree(directory, ignore_errors=True)

    def test_wrong_buffer_capacity_rejected_despite_equal_throughput(self):
        worker = ROOT / '.venvs/factoryflow/Scripts/python.exe'
        if not worker.is_file():
            self.skipTest('install FactoryFlow worker')
        directory = ROOT / 'work/runs/tests' / uuid4().hex
        try:
            execute('prepare', directory, 'factoryflow')
            rows = gold_models('factoryflow')
            rows['S1']['buffers'][0]['capacity'] = 4
            submission = directory / 'submission.json'
            submission.write_text(json.dumps(rows))
            score = execute('score', directory, 'factoryflow', submission, worker)
            self.assertFalse(score['accepted'])
            self.assertFalse(score['results'][0]['topology_and_parameters_match'])
        finally:
            shutil.rmtree(directory, ignore_errors=True)

    def test_factoryflow_native_reference_parity(self):
        self._native_reference_parity('factoryflow')

    def test_chronobench_native_reference_parity(self):
        self._native_reference_parity('chronobench')

    def _native_reference_parity(self, suite):
        worker = ROOT / ('.venvs/chrono-native/python.exe' if suite == 'chronobench' else '.venvs/factoryflow/Scripts/python.exe')
        if not worker.is_file():
            self.skipTest('isolated model benchmark worker is absent')
        directory = ROOT / 'work/runs/tests' / uuid4().hex
        try:
            execute('prepare', directory, suite)
            submission = directory / 'submission.json'
            submission.write_text(json.dumps(gold_models(suite)))
            score = execute('score', directory, suite, submission, worker)
            self.assertTrue(score['accepted'], score)
            if suite == 'factoryflow':
                self.assertEqual(score['contract'], 'agenticsim-factory-capacity-pilot-v2')
                s1 = json.loads((directory / 'S1-25-candidate.json').read_text())
                self.assertEqual(s1['received'], 19)
                capacities = {r['id']: r for r in s1['capacity_audit']}
                self.assertEqual(capacities['MACHINE']['realized'], 2)
                self.assertEqual(capacities['BUFFER1']['realized'], 3)
                self.assertGreater(score['results'][0]['runs'][0]['original_unmodified_reference_kpi_residual'], 0)
            with self.assertRaises(ValueError):
                execute('score', directory, suite, submission, worker)
        finally:
            shutil.rmtree(directory, ignore_errors=True)
