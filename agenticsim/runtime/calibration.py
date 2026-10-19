"""CLI for synthetic data -> generated MJCF -> identification -> held-out check.

This is a separate calibration lifecycle, not a new EngineeringTask kind or
Agent loop. MuJoCo/NumPy run in an isolated, credential-stripped worker.
"""
import argparse
import hashlib
import json
from pathlib import Path
import subprocess
import sys
from uuid import uuid4

from agenticsim.adapters.execution import worker_environment
from agenticsim.adapters.process_tree import run_worker
from agenticsim.domain import require
from agenticsim.domain.calibration import validate_spec
from agenticsim.runtime.paths import ROOT, RUNS


def load_json(path):
    return json.loads(Path(path).read_text(encoding="utf-8"))


def save_json(path, data):
    Path(path).write_text(json.dumps(data, ensure_ascii=False, allow_nan=False, indent=2), encoding="utf-8")


def sha256(path):
    return hashlib.sha256(Path(path).read_bytes()).hexdigest()


def directory_path(path):
    directory = Path(path).resolve()
    require(directory.is_relative_to(RUNS.resolve()) and directory != RUNS.resolve(), "calibration directory must be below project work/runs")
    return directory


def artifact_path(directory, name):
    target = (directory / name).resolve()
    require(target.is_relative_to(directory.resolve()), "calibration artifact leaves its directory")
    return target


def execute(phase, directory, *, spec_path=None, worker_python=None):
    directory = directory_path(directory)
    require(phase in ("generate", "model", "fit", "validate"), "unknown calibration phase")
    if phase == "generate":
        spec = load_json(spec_path or ROOT / "examples/kuka-calibration.json")
        validate_spec(spec)
        require(not directory.exists(), "generate needs a new output directory; existing evidence is not overwritten")
        directory.mkdir(parents=True)
        save_json(directory / "intake.json", spec)
    else:
        require(directory.is_dir(), "generate the dataset first")
        validate_spec(load_json(directory / "intake.json"))
    outputs = {"generate": "dataset.json", "model": "model.json", "fit": "fit.json", "validate": "validation.json"}
    require(not (directory / outputs[phase]).exists(), "phase already recorded; use its existing result or a new case")
    worker = Path(worker_python) if worker_python else ROOT / (
        ".venvs/newton/Scripts/python.exe" if sys.platform == "win32" else ".venvs/newton/bin/python")
    require(worker.is_file(), "isolated Newton/MuJoCo worker environment is missing")
    logs = directory / "logs"
    logs.mkdir(exist_ok=True)
    payload = json.dumps({"phase": phase, "directory": str(directory)})
    try:
        result = run_worker([str(worker), "-m", "agenticsim.adapters.kuka_identification"], payload=payload,
                            cwd=ROOT, env=worker_environment(directory), timeout=120)
    except subprocess.TimeoutExpired as exc:
        (logs / (phase + "-stderr.log")).write_text(exc.stderr or "worker timed out", encoding="utf-8")
        raise TimeoutError("calibration worker timed out; inspect phase logs") from exc
    for suffix, value in (("stdout", result.stdout), ("stderr", result.stderr)):
        (logs / f"{phase}-{suffix}.log").write_text(value, encoding="utf-8")
    require(result.returncode == 0, f"{phase} worker failed; inspect {logs / (phase + '-stderr.log')}")
    summary = json.loads(result.stdout)
    require((directory / outputs[phase]).is_file(), "worker did not commit the phase result")
    return summary


def main(argv=None):
    if hasattr(sys.stdout, "reconfigure"):
        sys.stdout.reconfigure(encoding="utf-8")
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("phase", choices=("generate", "model", "fit", "validate"))
    parser.add_argument("--directory", type=Path, default=RUNS / "calibration" / ("case-" + uuid4().hex[:12]))
    parser.add_argument("--spec", type=Path)
    parser.add_argument("--worker-python", type=Path)
    args = parser.parse_args(argv)
    try:
        result = execute(args.phase, args.directory, spec_path=args.spec, worker_python=args.worker_python)
    except (ValueError, KeyError, TypeError, OSError, TimeoutError) as exc:
        parser.error(str(exc))
    print(json.dumps(result, ensure_ascii=False, allow_nan=False, indent=2))


if __name__ == "__main__":
    main()
