"""Routing and persisted clarification, without an external model."""
import tempfile
import unittest
from pathlib import Path
from unittest.mock import patch

import importlib.util

HAS_HARNESS = importlib.util.find_spec("a4s") is not None
if HAS_HARNESS:
    from a4s.kernel.providers import AzureOpenAICompatible, Completion, ModelConfig, ToolCall
    from a4s.ledger import Ledger
    from agenticsim.runtime.dialogue import ROOT, DialogueTools, config_gaps, execute_turn


class RecordedProvider:
    def __init__(self, calls):
        self.calls = iter(calls)
        self.history = None

    def complete(self, messages, tools, config):
        self.history = list(messages)
        return Completion("", [next(self.calls)])


@unittest.skipUnless(HAS_HARNESS, "install harness extra for dialogue tests")
class TwinDialogueTests(unittest.TestCase):
    def test_runtime_resolves_checkout_resources_after_move(self):
        self.assertEqual(ROOT, Path(__file__).resolve().parents[1])
        self.assertTrue((ROOT / "examples/twin-operator-brief.md").is_file())

    def test_azure_v1_and_legacy_construct_different_clients(self):
        with patch("openai.OpenAI") as standard, patch("openai.AzureOpenAI") as legacy:
            cfg = ModelConfig(model="test-deployment", provider="azure", api_key="test-only",
                              base_url="https://example.invalid/openai/v1/")
            AzureOpenAICompatible(cfg)
            standard.assert_called_once()
            legacy.assert_not_called()
            cfg.base_url = "https://example.invalid"
            AzureOpenAICompatible(cfg)
            legacy.assert_called_once()

    def test_clarification_survives_restart_and_is_not_filled_with_fake_data(self):
        (ROOT / "work/runs").mkdir(parents=True, exist_ok=True)
        with tempfile.TemporaryDirectory(dir=ROOT / "work/runs") as directory:
            workspace = Path(directory)
            config = ModelConfig(model="offline-fixture")
            first = RecordedProvider([ToolCall("question", "ask_user", {"question": "What is the measured friction?"})])
            report = execute_turn(first, config, workspace, "Prepare a model; friction is unknown.")
            self.assertEqual(report["status"], "awaiting_operator")
            second = RecordedProvider([ToolCall("finish", "finish", {"summary": "Friction remains unknown; request calibration."})])
            report = execute_turn(second, config, workspace,
                                  "[delegated_operator] Keep friction unknown.", restored=True)
            self.assertEqual(report["status"], "finished")
            self.assertTrue(report["ledger_valid"])
            users = [m["content"] for m in second.history if m["role"] == "user"]
            self.assertEqual(len(users), 2)
            self.assertIn("Keep friction unknown", users[-1])
            with Ledger(workspace / "ledger.db", workspace.name) as ledger:
                tools = DialogueTools(ledger, workspace, {"python -m agenticsim capabilities"})
                self.assertIn("error:", tools.dispatch("shell", {"command": "curl example.invalid"}, "blocked"))
                self.assertIn("error:", tools.dispatch("read_file", {"path": ".env"}, "private"))
                hidden = workspace / 'artifacts/modelbench/.evaluation/gold.json'
                self.assertIn('error:', tools.dispatch('read_file', {'path': str(hidden)}, 'model-gold'))
                self.assertIn('error:', tools.dispatch('write_file', {
                    'path': str(workspace / 'artifacts/modelbench/cases.json'), 'content': '{}'}, 'model-tamper'))
                self.assertNotIn("error:", tools.dispatch("read_file", {"path": "README.md"}, "document-readme"))
                for name in ("PROGRESS.md", "TUTORIAL.md"):
                    self.assertIn("error:", tools.dispatch("read_file", {"path": name}, "private-document-" + name))
                self.assertIn("error:", tools.dispatch("read_file", {"path": "work/archive/docs/README.md"}, "archive"))
                hidden = workspace / "artifacts/calibration/.fixture/truth.json"
                heldout = workspace / "artifacts/calibration/excitation-c.csv"
                self.assertIn("error:", tools.dispatch("read_file", {"path": str(hidden)}, "truth"))
                self.assertIn("error:", tools.dispatch("read_file", {"path": str(heldout)}, "holdout"))
                self.assertIn("error:", tools.dispatch("write_file", {"path": str(heldout), "content": "fake"}, "evidence"))
                measured_holdout = workspace / "artifacts/identification/panda-seed-21.csv"
                self.assertIn("error:", tools.dispatch("read_file", {"path": str(measured_holdout)}, "measured-holdout"))
                self.assertIn("error:", tools.dispatch("write_file", {"path": str(measured_holdout), "content": "fake"}, "measured-evidence"))
                self.assertIn("error:", tools.dispatch("write_file", {"path": str(workspace / "artifacts/identification/fit.json"), "content": "fake"}, "fit-evidence"))
                benchmark = workspace / "artifacts/benchmark"
                self.assertIn("error:", tools.dispatch("read_file", {"path": str(benchmark / ".evaluation/labels.json")}, "benchmark-labels"))
                self.assertIn("error:", tools.dispatch("write_file", {"path": str(benchmark / "fit.json"), "content": "fake"}, "benchmark-fit"))
                semantic = workspace / "artifacts/semantic"
                self.assertIn("error:", tools.dispatch("read_file", {"path": str(semantic / ".evaluation/gold.json")}, "semantic-gold"))
                self.assertIn("error:", tools.dispatch("write_file", {"path": str(semantic / "cases.json"), "content": "fake"}, "semantic-cases"))
                plaid = workspace / 'artifacts/plaid'
                self.assertIn('error:', tools.dispatch('read_file', {'path': str(plaid / '.evaluation/data.npz')}, 'plaid-labels'))
                self.assertIn('error:', tools.dispatch('write_file', {'path': str(plaid / 'score.json'), 'content': '{}'}, 'plaid-score'))
                tensile = workspace / 'artifacts/tensile'
                self.assertIn('error:', tools.dispatch('read_file', {'path': str(tensile / '.evaluation/data.npz')}, 'tensile-labels'))
                self.assertIn('error:', tools.dispatch('write_file', {'path': str(tensile / 'score.json'), 'content': '{}'}, 'tensile-score'))
                self.assertIn("error:", tools.dispatch("write_file", {"path": "PROGRESS.md", "content": "bad"}, "outside"))
                local = workspace / "artifacts/intake.json"
                tools.dispatch("write_file", {"path": str(local), "content": '{"friction":null}'}, "draft")
                self.assertEqual(local.read_text(encoding="utf-8"), '{"friction":null}')

    def test_missing_public_credentials_reported(self):
        self.assertTrue(config_gaps(ModelConfig(model="test")))
        self.assertFalse(config_gaps(ModelConfig(model="test", base_url="http://127.0.0.1:8000/v1")))

    def test_tensile_shell_survives_generic_tool_timeout(self):
        with tempfile.TemporaryDirectory(dir=ROOT / 'work/runs') as directory:
            workspace = Path(directory)
            command = 'python -m agenticsim tensile fit --directory work/runs/example'
            with Ledger(workspace / 'ledger.db',workspace.name) as ledger:
                tools = DialogueTools(ledger,workspace,{command})
                with patch('agenticsim.runtime.dialogue.ToolRegistry._shell',return_value='ok') as shell:
                    self.assertEqual(tools._shell(command,timeout=120),'ok')
                    shell.assert_called_once_with(command,timeout=650)


if __name__ == "__main__":
    unittest.main()
