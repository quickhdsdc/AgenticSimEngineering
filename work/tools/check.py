"""Named local test gates with explicit skips and persistent reports."""
import argparse
import importlib.util
import json
import os
from pathlib import Path
import subprocess
import sys
import unittest
import xml.etree.ElementTree as ET
from uuid import uuid4

ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT))
from scripts.doctor import inspect

CATEGORIES = {
    "DESIntegration": "des", "PlaybackTests": "des", "TwinDialogueTests": "harness",
    "AssetRobotIntegration": "robot", "NewtonNumerics": "robot", "RobotPlayback": "robot",
    "KukaCalibrationIntegration": "robot",
    "PandaIdentificationIntegration": "identification",
    "FestoIntegration": "festo",
}


def flatten(suite):
    for item in suite:
        if isinstance(item, unittest.TestSuite):
            yield from flatten(item)
        else:
            yield item


def selected(test, profile):
    category = CATEGORIES.get(type(test).__name__, "core")
    return category == profile


def unit_gate(profile, directory):
    suite = unittest.defaultTestLoader.discover(str(ROOT / "tests"), top_level_dir=str(ROOT))
    suite = unittest.TestSuite(t for t in flatten(suite) if selected(t, profile))
    with (directory / "unittest.log").open("w", encoding="utf-8") as log:
        result = unittest.TextTestRunner(stream=log, verbosity=2).run(suite)
    skips = [{"test": t.id(), "reason": reason} for t, reason in result.skipped]
    return {"status": "passed" if result.wasSuccessful() and not skips and result.testsRun else "failed",
            "tests": result.testsRun, "failures": len(result.failures), "errors": len(result.errors),
            "skipped": skips, "log": str(directory / "unittest.log")}


def command_gate(command, directory, name, cwd=ROOT, env=None):
    try:
        result = subprocess.run(command, cwd=cwd, env=env, capture_output=True, text=True,
                                encoding="utf-8", errors="replace", timeout=300)
        output, code = result.stdout + result.stderr, result.returncode
    except (OSError, subprocess.TimeoutExpired) as exc:
        output, code = type(exc).__name__, 124
    log = directory / (name + ".log")
    log.write_text(output, encoding="utf-8")
    return {"status": "passed" if code == 0 else "failed", "exit_code": code, "log": str(log)}


def run(profile, directory, worker=None):
    if profile == "browser":
        if not inspect("des")["ready"]:
            return {"status": "failed", "reason": "DES dependencies missing"}
        from agenticsim.adapters.des import SingleServer
        from agenticsim.adapters.execution import FileStudyStore, LocalWorker
        from agenticsim.application import compare
        from agenticsim.visualization.bundle import export
        task = json.loads((ROOT / "examples/queue-study.json").read_text(encoding="utf-8"))
        compare_result = compare(task, SingleServer(), LocalWorker(), FileStudyStore(directory / "studies"))
        artifact = directory / "studies" / compare_result["study_id"] / "study.json"
        bundle = export(artifact, directory / "visualizations")
        env = dict(os.environ, ASE_VIEWER_BUNDLE=bundle["bundle"], ASE_VIEWER_KIND="queue",
                   ASE_VIEWER_CASE=directory.name, ASE_VIEWER_PYTHON=sys.executable)
        npm = "npm.cmd" if os.name == "nt" else "npm"
        result = command_gate([npm, "test"], directory, "browser", ROOT / "components/visualization", env)
        report = ROOT / "work/runs/browser-tests" / (directory.name + ".json")
        if result["status"] == "passed":
            stats = json.loads(report.read_text(encoding="utf-8"))["stats"]
            result["tests"] = sum(stats.get(k, 0) for k in ("expected", "unexpected", "skipped", "flaky"))
            result["skipped"] = stats.get("skipped", 0)
            if not result["tests"] or result["skipped"] or stats.get("unexpected", 0):
                result["status"] = "failed"
        return result
    preflight = inspect(profile, worker)
    if not preflight["ready"]:
        return {"status": "failed", "preflight": preflight, "reason": "profile dependencies missing"}
    if profile == "robot":
        os.environ["ASE_NEWTON_PYTHON"] = preflight["worker_python"]
    if profile == "identification":
        os.environ["ASE_IDENTIFICATION_PYTHON"] = preflight["worker_python"]
    if profile == "des" and not (ROOT / "components/visualization/node_modules/three/package.json").is_file():
        return {"status": "failed", "reason": "npm ci --prefix components/visualization required for playback tests"}
    result = unit_gate(profile, directory)
    result["preflight"] = preflight
    if profile in ("harness", "festo"):
        if importlib.util.find_spec("pytest") is None:
            result.update(status="failed", reason="install dev extra")
            return result
        env = os.environ.copy()
        env.pop("PYTHONPATH", None)
        env["PYTHONUTF8"] = "1"
        if os.name == "nt":
            for base in (Path(os.getenv("ProgramFiles", "C:/Program Files")) / "Git/usr/bin",
                         Path(os.getenv("LOCALAPPDATA", "")) / "Programs/Git/usr/bin"):
                if (base / "bash.exe").is_file():
                    env["PATH"] = str(base) + os.pathsep + env.get("PATH", "")
                    break
        modules = ["test_loop.py", "test_approval.py", "test_boundaries.py"]
        if profile == "festo":
            modules = ["test_live_monitor.py", "test_live_monitor_routes.py", "test_fault_workflow.py",
                       "test_festo_assets.py", "test_festo_blueprint_context.py", "test_maintenance_cli.py",
                       "test_mes_binding.py", "test_production_schedule.py", "test_production_workflow.py",
                       "test_mes_reset.py", "test_mes_reset_routes.py", "test_agent_activity.py",
                       "test_festo_demo.py", "test_pdf_knowledge.py", "test_chat.py", "test_mes_architecture.py"]
            modules += ["test_festo_start.py", "test_festo.py", "test_opc.py"]
        command = [sys.executable, "-m", "pytest", *["tests/" + m for m in modules],
                   "-c", "pyproject.toml", "-q", "-m", "not integration", "-o", "addopts=",
                   "-o", "cache_dir=" + str(directory / "pytest-cache"),
                   "--basetemp=" + str(directory / "fixtures"),
                   "--junitxml=" + str(directory / "harness.xml")]
        component = command_gate(command, directory, profile, cwd=ROOT / "components/harness", env=env)
        if component["status"] == "passed":
            suites = list(ET.parse(directory / "harness.xml").getroot().iter("testsuite"))
            component["tests"] = sum(int(s.get("tests", 0)) for s in suites)
            component["skipped"] = sum(int(s.get("skipped", 0)) for s in suites)
            if not component["tests"] or component["skipped"]:
                component["status"] = "failed"
        result["component"] = component
        if component["status"] != "passed":
            result["status"] = "failed"
        if profile == "festo":
            npm = "npm.cmd" if os.name == "nt" else "npm"
            frontend = ROOT / "components/harness/frontend"
            result["frontend_tests"] = command_gate([npm, "test"], directory, "frontend-tests", frontend)
            result["frontend_build"] = command_gate([npm, "run", "build"], directory, "frontend-build", frontend)
            if any(result[k]["status"] != "passed" for k in ("frontend_tests", "frontend_build")):
                result["status"] = "failed"
    return result


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--profile", choices=["core", "des", "harness", "festo", "robot", "identification", "browser", "dev"], default="core")
    parser.add_argument("--worker-python")
    parser.add_argument("--output", type=Path, default=ROOT / "work/runs/checks")
    args = parser.parse_args()
    profiles = ["core", "des", "harness"] if args.profile == "dev" else [args.profile]
    directory = args.output.resolve() / ("check-" + uuid4().hex[:12])
    directory.mkdir(parents=True)
    gates = {}
    for profile in profiles:
        # Discovery caches decorators: each profile gets a fresh process in dev.
        if args.profile == "dev":
            gates[profile] = command_gate([sys.executable, str(Path(__file__).resolve()),
                                          "--profile", profile, "--output", str(directory / profile)],
                                         directory, profile)
        else:
            try:
                gates[profile] = run(profile, directory, args.worker_python)
            except Exception as exc:
                gates[profile] = {"status": "failed", "error_type": type(exc).__name__, "reason": str(exc)}
        print(profile + ": " + gates[profile]["status"], flush=True)
    report = {"profile": args.profile, "status": "passed" if all(g["status"] == "passed" for g in gates.values()) else "failed",
              "scope": "local tests only; selected test skips fail the gate", "gates": gates}
    (directory / "report.json").write_text(json.dumps(report, ensure_ascii=False, indent=2), encoding="utf-8")
    print("report: " + str(directory / "report.json"))
    return 0 if report["status"] == "passed" else 1


if __name__ == "__main__":
    raise SystemExit(main())
