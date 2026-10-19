"""Persisted evidence survives rejected writes and detects corruption."""
import json
from pathlib import Path
import tempfile
import unittest
from unittest.mock import patch

from agenticsim.adapters.execution import FileStudyStore, worker_environment


class StorageTests(unittest.TestCase):
    def test_compute_worker_does_not_inherit_agent_or_plant_configuration(self):
        with patch.dict("os.environ", {"A4S_API_KEY": "fixture-secret", "AZURE_OPENAI_API_KEY": "fixture-secret",
                                       "A4S_AAS_ENDPOINT": "fixture-plant", "PATH": "fixture-path"}, clear=True):
            self.assertEqual(worker_environment("fixture-run"),
                             {"PATH": "fixture-path", "ASE_RUN_DIRECTORY": "fixture-run"})

    def test_failed_replacement_preserves_last_valid_snapshot(self):
        with tempfile.TemporaryDirectory() as directory:
            store = FileStudyStore(directory)
            study = store.create({"description": "工程数据"})
            before = store.load(study["study_id"])
            study["status"] = "running"
            with patch("agenticsim.adapters.execution.os.replace", side_effect=PermissionError("busy")):
                with self.assertRaises(PermissionError):
                    store.save(study)
            self.assertEqual(store.load(study["study_id"]), before)
            self.assertEqual(list(Path(directory).rglob("*.tmp")), [])
            store.save(study)
            self.assertEqual(store.load(study["study_id"])["status"], "running")

    def test_rejects_paths_and_tampered_evidence(self):
        with tempfile.TemporaryDirectory() as directory:
            store = FileStudyStore(directory)
            for identity in ("../outside", "", None):
                with self.assertRaises(ValueError):
                    store.save({"study_id": identity})
                with self.assertRaises(ValueError):
                    store.load(identity)
            study = store.create({})
            path = Path(directory) / study["study_id"] / "study.json"
            envelope = json.loads(path.read_text(encoding="utf-8"))
            envelope["payload"]["status"] = "fake-success"
            path.write_text(json.dumps(envelope), encoding="utf-8")
            with self.assertRaisesRegex(ValueError, "digest mismatch"):
                store.load(study["study_id"])
