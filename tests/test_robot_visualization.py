"""Real short robot solves followed by semantic and portable playback checks."""
import copy
import hashlib
import json
import os
from pathlib import Path
import tempfile
import unittest
from unittest.mock import patch

from agenticsim.adapters.asset_robot import AssetRobot
from agenticsim.adapters.execution import FileStudyStore, LocalWorker
from agenticsim.application import compare
from agenticsim.visualization.bundle import export, verify
from agenticsim.visualization.queue import digest
from agenticsim.visualization.robot import robot_replay, checked_json

ROOT = Path(__file__).resolve().parents[1]


@unittest.skipUnless(os.getenv('ASE_NEWTON_PYTHON'), 'set ASE_NEWTON_PYTHON for real robot playback checks')
class RobotPlayback(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        cls.temp = tempfile.TemporaryDirectory()
        cls.addClassCleanup(cls.temp.cleanup)
        cls.root = Path(cls.temp.name)
        cls.paths = []
        for name in ('kuka-iiwa14', 'unitree-g1'):
            task = json.loads((ROOT / f'examples/{name}-study.json').read_text(encoding='utf8'))
            task['execution'].update(device='cpu', duration_s=.02)
            task['candidates'] = task['candidates'][:1]
            study = compare(task, AssetRobot(name), LocalWorker(os.environ['ASE_NEWTON_PYTHON']), FileStudyStore(cls.root))
            if study['status'] != 'completed':
                raise AssertionError(study['runs'])
            cls.paths.append(cls.root / study['study_id'] / 'study.json')

    def test_original_meshes_full_base_pose_and_observation_alignment(self):
        for path in self.paths:
            replay, copies = robot_replay(path)
            run = replay['runs'][0]
            self.assertEqual(len(run['times']), 11)
            self.assertEqual(run['scene']['frame'], 'right-handed-Z-up')
            self.assertGreater(sum(m['vertex_count'] for m in run['scene']['meshes']), 30000)
            self.assertEqual(len(run['scene']['joint_bodies']), len(run['target_q']))
            self.assertTrue(any(key.endswith('ROBOT-LICENSE.txt') for key in copies))
            if replay['model']['id'] == 'unitree-g1':
                self.assertGreater(max(run['contact_counts'][1:]), 0)
                pelvis = run['scene']['bodies'].index('pelvis')
                self.assertGreater(run['body_poses'][0][pelvis][2], .5)

    def test_rejects_changed_units_joint_order_quaternion_and_observation_values(self):
        original = json.loads(self.paths[0].read_text())['payload']
        descriptor = original['runs'][0]['result']['provenance']['robot_playback']['trajectory']
        trace = json.loads(Path(descriptor['path']).read_text())
        for change in [lambda d: d.update(length_unit='mm'), lambda d: d.update(quaternion='wxyz'),
                       lambda d: d['joint_order'].reverse(), lambda d: d['body_poses'][0][0].__setitem__(6, 0),
                       lambda d: d['joint_values'][1].__setitem__(0, .5), lambda d: d['times'].__setitem__(1, .003)]:
            altered = copy.deepcopy(trace); change(altered)
            # Test owns this runs/ directory; do not alter the original solver evidence.
            artifact = ROOT / 'work/runs/robot-playback-tests' / self.root.name
            artifact.mkdir(parents=True, exist_ok=True)
            file = artifact / 'altered-trace.json'; file.write_text(json.dumps(altered), encoding='utf8')
            study = copy.deepcopy(original)
            study['runs'][0]['result']['provenance']['robot_playback']['trajectory'] = {
                'path': str(file), 'sha256': hashlib.sha256(file.read_bytes()).hexdigest()}
            path = self.root / 'altered-study.json'; path.write_text(json.dumps({'payload': study, 'sha256': digest(study)}))
            with self.subTest(change=change), self.assertRaises(ValueError):
                robot_replay(path)

    def test_bundle_reads_copied_pose_evidence_and_detects_mesh_tampering(self):
        result = export(self.paths[0], self.root / 'bundles'); bundle = Path(result['bundle'])
        def confined(path, checksum):
            self.assertTrue(path.resolve().is_relative_to(bundle))
            return checked_json(path, checksum)
        with patch('agenticsim.visualization.robot.checked_json', confined):
            verify(bundle)
        mesh = next((bundle / 'robot-source').rglob('*.bin'))
        raw = bytearray(mesh.read_bytes()); raw[-1] ^= 1; mesh.write_bytes(raw)
        with self.assertRaisesRegex(ValueError, 'checksum'):
            verify(bundle)
