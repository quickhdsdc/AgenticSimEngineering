"""Real loopback OPC -> REST asset -> monitor -> original Harness -> read-only CLIs.

The scripted provider verifies dispatch and evidence, not LLM diagnosis quality.
The AAS HTTP fixture rejects mutations; no plant or MES service is configured.
"""
from http.server import BaseHTTPRequestHandler, ThreadingHTTPServer
import importlib.util
import json
import os
from pathlib import Path
import shlex
import socket
import sys
import tempfile
import threading
import time
import unittest
from unittest.mock import patch

ROOT = Path(__file__).resolve().parents[1]
AVAILABLE = all(importlib.util.find_spec(m) for m in ("a4s", "asyncua", "fastapi", "pypdfium2"))


@unittest.skipUnless(AVAILABLE, "install festo extra")
class FestoIntegration(unittest.TestCase):
    def test_fault_handoff_cli_evidence_recovery_and_disconnect(self):
        from a4s.environments.local import LocalEnvironment
        from a4s.aasctl.paths import encode_id
        from a4s.kernel.loop import AgentLoop
        from a4s.kernel.prompt import build_system_prompt
        from a4s.kernel.providers import Completion, ModelConfig, ToolCall
        from a4s.ledger import Ledger
        from a4s.machines.festo_cp_lab import fault_profile, opc_simulator
        from a4s.machines.festo_cp_lab.live_monitor import Monitor
        from a4s.server.live_monitor_routes import investigation_request, read_target
        from a4s.tools import ToolRegistry
        from agenticsim.adapters.execution import worker_environment

        with socket.socket() as reserved:
            reserved.bind(("127.0.0.1", 0))
            opc_endpoint = f"opc.tcp://127.0.0.1:{reserved.getsockname()[1]}/a4s-test/"
        asset_id = "urn:agenticsim:fixture:festo"
        requests = []

        class Repository(BaseHTTPRequestHandler):
            def do_GET(self):
                requests.append(("GET", self.path))
                document = documents.get(self.path)
                self.send_response(200 if document else 404)
                self.send_header("Content-Type", "application/json")
                self.end_headers()
                self.wfile.write(json.dumps(document or {"error": "missing"}).encode())

            def do_POST(self):
                requests.append(("WRITE", self.path))
                self.send_error(405)

            do_PUT = do_POST
            do_DELETE = do_POST

            def log_message(self, *_):
                pass

        (ROOT / "work/runs").mkdir(parents=True, exist_ok=True)
        with tempfile.TemporaryDirectory(dir=ROOT / "work/runs") as directory, \
                patch.dict(os.environ, {**worker_environment(directory), "HTTP_PROXY": "",
                           "HTTPS_PROXY": "", "ALL_PROXY": "", "NO_PROXY": "127.0.0.1,localhost"}, clear=True), \
                patch.object(opc_simulator, "ENDPOINT", opc_endpoint), \
                patch.object(fault_profile, "ENDPOINT", opc_endpoint):
            workspace = Path(directory)
            fixture_env = workspace / "fixture.env"
            fixture_env.write_text("# Isolated test: no private deployment configuration\n", encoding="utf-8")
            models = fault_profile.models({"kind": "Instance", "idShort": "ProductionContext",
                "submodelElements": [{"modelType": "Property", "idShort": "MESLineCode",
                                      "value": "FIXTURE-FESTO", "valueType": "xs:string"}]}, asset_id)
            shell = {"id": asset_id, "idShort": "FestoWorkCenterStamping_Fixture",
                     "assetInformation": {"assetKind": "Instance"}, "submodels": [
                         {"keys": [{"type": "Submodel", "value": m["id"]}]} for m in models]}
            documents = {"/shells/" + encode_id(asset_id): shell,
                         **{"/submodels/" + encode_id(m["id"]): m for m in models}}
            server = ThreadingHTTPServer(("127.0.0.1", 0), Repository)
            thread = threading.Thread(target=server.serve_forever, daemon=True)
            thread.start()
            selection = {"asset_id": asset_id, "endpoint": f"http://127.0.0.1:{server.server_port}",
                         "source": "simulator"}
            controller = opc_simulator.SimulatorController()
            monitor = Monitor(workspace / "monitor.db", read_target)
            try:
                self.assertTrue(controller.start()["running"])

                def sample():
                    frame = read_target(selection)
                    frame["observed_at"] = time.time()
                    return monitor.ingest(selection, frame)

                sample(); sample()
                controller.set_mode("fault")
                sample()
                events = sample()
                fault = next(e for e in events if e["kind"] == "FAULT_RAISED")
                self.assertEqual(sample(), [])
                event = monitor.claim(selection, fault["id"])
                self.assertIsNotNone(event)
                self.assertIsNone(monitor.claim(selection, fault["id"]))
                request = investigation_request(event)
                self.assertIn("SIMULATED OPC", request)
                values = {r["path"]: r for r in event["after"]["values"]}
                self.assertEqual(values["Press.PressureBar"]["value"], 3.8)
                self.assertTrue(values["Press.Fault"]["source_timestamp"])

                python = shlex.quote(Path(sys.executable).as_posix())
                context = f"{python} -m a4s.machines.festo_cp_lab.fault_context --asset {asset_id} --endpoint {selection['endpoint']}"
                knowledge = f"{python} -m a4s.machines.festo_cp_lab.knowledge --query 'pneumatic pressure flow Fault01 emergency stop' --limit 4"

                class ExactEnvironment:
                    name = "isolated-festo-fixture"

                    def __init__(self):
                        self.local = LocalEnvironment(cwd=str(workspace))
                        self.results = []
                        self.local.env = {**worker_environment(workspace), "PYTHONUTF8": "1",
                                          "A4S_ENV_FILE": str(fixture_env),
                                          "MSYS_NO_PATHCONV": "1", "MSYS2_ARG_CONV_EXCL": "*"}

                    def run(self, command, **kwargs):
                        if command not in (context, knowledge):
                            raise AssertionError("Unexpected fixture command")
                        result = self.local.run(command, **kwargs)
                        self.results.append(result)
                        return result

                class ScriptedProvider:
                    def __init__(self):
                        self.calls = iter([
                            ToolCall("context", "shell", {"command": context}),
                            ToolCall("manual", "shell", {"command": knowledge}),
                            ToolCall("review", "ask_user", {"question": "Review the synthetic pressure/flow evidence and proposed inspection before any MES change."})])

                    def complete(self, messages, tools, config):
                        return Completion("", [next(self.calls)])

                prompt = build_system_prompt(ROOT / "components/harness/a4s/skills",
                                             instruction=request, endpoint=selection["endpoint"])
                self.assertIn("fault-maintenance", prompt)
                self.assertIn("agenticsim capabilities", prompt)
                with Ledger(workspace / "ledger.db", "festo-integration") as ledger:
                    environment = ExactEnvironment()
                    registry = ToolRegistry(environment, ledger, workspace, roots=[workspace])
                    loop = AgentLoop(provider=ScriptedProvider(), model=ModelConfig(model="offline-fixture"),
                                     tools=registry, ledger=ledger, system_prompt=prompt)
                    result = loop.run(request)
                    self.assertEqual(result.status, "awaiting_operator")
                    replies = [json.loads(m["content"]) for m in result.messages
                               if m.get("role") == "tool" and m.get("tool_call_id") in ("context", "manual")]
                    self.assertEqual(len(replies), 2)
                    self.assertEqual(len(environment.results), 2)
                    self.assertTrue(all(r.exit_code == 0 for r in environment.results), environment.results)
                    self.assertIn(asset_id, json.dumps(replies[0]))
                    self.assertIn("Festo-Pneumatic-Fault-Service-Guide.pdf", json.dumps(replies[1]))
                    self.assertEqual(ledger.verify(), (True, None))

                controller.set_mode("normal")
                sample()
                self.assertIn("RECOVERY_OBSERVED", [e["kind"] for e in sample()])
                controller.stop()
                self.assertIn("COMMUNICATION_LOST", [e["kind"] for e in sample()])
                controller.start()
                controller.set_mode("fault")
                self.assertIn("COMMUNICATION_RESTORED", [e["kind"] for e in sample()])
                after_gap = next(e for e in sample() if e["kind"] == "FAULT_OBSERVED_AFTER_GAP")
                self.assertIsNone(monitor.claim(selection, after_gap["id"]))
                reloaded = Monitor(workspace / "monitor.db", read_target)
                self.assertIsNone(reloaded.claim(selection, fault["id"]))
                self.assertEqual(sum(e["kind"] == "FAULT_RAISED" for e in reloaded.status(selection)["events"]), 1)
                self.assertTrue(requests)
                self.assertTrue(all(method == "GET" for method, _ in requests))
            finally:
                monitor.close()
                controller.stop()
                server.shutdown()
                server.server_close()
                thread.join(timeout=2)


if __name__ == "__main__":
    unittest.main()
