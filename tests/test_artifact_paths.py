"""Preserve old evidence descriptors while confining relocation to the project."""
from pathlib import Path
import tempfile
import unittest
from unittest.mock import patch

from agenticsim.runtime.paths import relocated_artifact


class ArtifactPaths(unittest.TestCase):
    def test_legacy_relative_and_absolute_paths_read_same_unchanged_bytes(self):
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory).resolve()
            runs = root / "work/runs"
            target = runs / "workers/id/robot-scene.json"
            target.parent.mkdir(parents=True)
            raw = b'{"original":"hashed evidence"}'
            target.write_bytes(raw)
            with patch("agenticsim.runtime.paths.ROOT", root), patch("agenticsim.runtime.paths.RUNS", runs):
                for descriptor in ("runs/workers/id/robot-scene.json", root / "runs/workers/id/robot-scene.json",
                                   "work/runs/workers/id/robot-scene.json"):
                    self.assertEqual(relocated_artifact(descriptor).read_bytes(), raw)
                self.assertEqual(relocated_artifact("assets/model.xml"), root / "assets/model.xml")

    def test_outside_project_paths_are_rejected(self):
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory).resolve()
            with patch("agenticsim.runtime.paths.ROOT", root), patch("agenticsim.runtime.paths.RUNS", root / "work/runs"):
                for descriptor in ("../private.json", root.parent / "private.json"):
                    with self.assertRaisesRegex(ValueError, "leaves the project"):
                        relocated_artifact(descriptor)
