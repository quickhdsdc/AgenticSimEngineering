import copy
import importlib.util
import json
from pathlib import Path
import tempfile
import threading
import unittest
from urllib.error import HTTPError
from urllib.request import urlopen

from agenticsim.adapters.des import SingleServer
from agenticsim.adapters.execution import FileStudyStore
from agenticsim.application import compare
from agenticsim.visualization.bundle import export, verify
from agenticsim.visualization.queue import digest, queue_replay
from agenticsim.visualization.serve import server

ROOT = Path(__file__).resolve().parents[1]
HAS_DES = all(importlib.util.find_spec(x) for x in ("numpy", "simpy"))
HAS_THREE = (ROOT / "components/visualization/node_modules/three/package.json").is_file()


@unittest.skipUnless(HAS_DES, "install des-reference for real FIFO playback checks")
class PlaybackTests(unittest.TestCase):
    def setUp(self):
        self.directory = tempfile.TemporaryDirectory()
        self.addCleanup(self.directory.cleanup)
        self.root = Path(self.directory.name)
        task = json.loads((ROOT / "examples/queue-study.json").read_text(encoding="utf-8"))
        self.study = compare(task, SingleServer(), SingleServer(), FileStudyStore(self.root))
        self.path = self.root / self.study["study_id"] / "study.json"

    def rewrite(self, study):
        self.path.write_text(json.dumps({"payload": study, "sha256": digest(study)}), encoding="utf-8")

    def test_real_fifo_events_assessment_and_source_mapping(self):
        replay = queue_replay(self.path)
        self.assertEqual([r["duration"] for r in replay["runs"]], [8., 4., 12.])
        self.assertEqual([r["assessment"]["status"] for r in replay["runs"]], ["accepted", "accepted", "rejected"])
        baseline = replay["runs"][0]
        self.assertEqual([j["arrival"] for j in baseline["jobs"]], [0, 1, 2, 3])
        self.assertEqual([j["start"] for j in baseline["jobs"]], [0, 2, 4, 6])
        self.assertEqual(len(baseline["events"]), 12)
        self.assertEqual(baseline["event_times"], [0, 1, 2, 3, 4, 6, 8])
        self.assertEqual(replay["input_source"], "synthetic_fixture")
        self.assertEqual(baseline["source"], "reference_simulation")
        self.assertEqual(replay["scene"]["geometry_source"], "synthetic_fixture")

    def test_corrupt_or_semantically_inconsistent_evidence_is_rejected(self):
        record = json.loads(self.path.read_text())
        record["payload"]["task"]["inputs"]["service"]["values"][0] = 10
        self.path.write_text(json.dumps(record))
        with self.assertRaisesRegex(ValueError, "checksum"):
            queue_replay(self.path)
        changes = [
            lambda s: s["task"]["inputs"]["service"].update(unit="ms"),
            lambda s: s["runs"][0]["result"]["observations"][1]["values"].__setitem__(0, 3),
            lambda s: s["runs"][0]["result"]["observations"][1].update(axis_unit="s"),
            lambda s: s["runs"][0]["assessment"].update(status="rejected"),
            lambda s: s["runs"][0].update(candidate_id="unregistered"),
            lambda s: s["decision"].update(best_observed_candidate="longer-service"),
            lambda s: s.update(status="running"),
        ]
        for change in changes:
            study = copy.deepcopy(self.study)
            change(study)
            self.rewrite(study)
            with self.subTest(change=change), self.assertRaises(ValueError):
                queue_replay(self.path)

    def test_failed_candidate_does_not_inherit_successful_frames(self):
        study = copy.deepcopy(self.study)
        study["status"] = "completed_with_errors"
        study["runs"][1] = {"run_id": "failed-run", "candidate_id": "shorter-service",
                            "status": "timed_out", "error": {"message": "timeout"}}
        study["decision"]["status"] = "inconclusive"
        study["decision"]["best_observed_candidate"] = "baseline"
        self.rewrite(study)
        replay = queue_replay(self.path)
        failed = replay["runs"][1]
        self.assertEqual(failed["jobs"], [])
        self.assertEqual(failed["events"], [])
        self.assertIsNone(failed["duration"])
        self.assertEqual(replay["runs"][0]["duration"], 8.)

    @unittest.skipUnless(HAS_THREE, "npm ci --prefix components/visualization")
    def test_portable_bundle_detects_tampering_and_serves_only_manifest_files(self):
        result = export(self.path, self.root / "exports")
        bundle = Path(result["bundle"])
        self.assertEqual(verify(bundle)["study_id"], self.study["study_id"])
        (bundle / "private.txt").write_text("not a public asset")
        with server(bundle, 0) as httpd:
            thread = threading.Thread(target=httpd.serve_forever, daemon=True)
            thread.start()
            try:
                url = f"http://127.0.0.1:{httpd.server_port}"
                with urlopen(url + "/") as response:
                    self.assertEqual(response.status, 200)
                for name in ("/private.txt", "/../private.txt", "/vendor/"):
                    with self.assertRaises(HTTPError) as error:
                        urlopen(url + name)
                    self.assertEqual(error.exception.code, 404)
            finally:
                httpd.shutdown()
                thread.join()
        (bundle / "replay.json").write_text("{}")
        with self.assertRaisesRegex(ValueError, "checksum"):
            verify(bundle)
