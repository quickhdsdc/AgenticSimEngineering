"""Architecture rules should reject regressions, not just inspect today's tree."""
from pathlib import Path
import unittest

from agenticsim.runtime.architecture import check, violations


class ArchitectureTests(unittest.TestCase):
    def test_current_product_import_boundaries(self):
        self.assertEqual(check(Path(__file__).resolve().parents[1]), [])

    def test_core_cannot_import_adapters_or_hide_sdk_in_a_function(self):
        for source in ["from agenticsim.adapters.execution import LocalWorker",
                       "def run():\n    import torch", "from a4s.kernel.loop import AgentLoop"]:
            self.assertTrue(violations(source, "agenticsim.domain.__init__", layer="domain"))
        self.assertTrue(violations("from ..adapters import execution", "agenticsim.ports.__init__", layer="ports"))
        self.assertFalse(violations("from ..domain import Evaluation", "agenticsim.ports.__init__", layer="ports"))

    def test_adapter_sdk_import_is_lazy(self):
        self.assertTrue(violations("if True:\n    import newton", "agenticsim.adapters.example"))
        self.assertFalse(violations("def evaluate():\n    import newton", "agenticsim.adapters.example"))
