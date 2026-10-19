"""Four fixed phases for offline public Panda measurement identification."""
import argparse
import json
from pathlib import Path
import subprocess
import sys
from uuid import uuid4

from agenticsim.adapters.execution import worker_environment
from agenticsim.adapters.process_tree import run_worker
from agenticsim.domain import require
from agenticsim.domain.measurements import validate_spec
from agenticsim.runtime.calibration import directory_path, load_json, save_json, sha256
from agenticsim.runtime.paths import ROOT, RUNS

OUTPUTS = {"ingest": "dataset.json", "model": "model.json", "fit": "fit.json", "validate": "validation.json"}


def execute(phase, directory, *, spec_path=None, worker_python=None):
    require(phase in OUTPUTS, "unknown identification phase")
    directory = directory_path(directory)
    if phase == "ingest":
        spec = load_json(spec_path or ROOT / "examples/panda-identification.json")
        validate_spec(spec)
        require(not directory.exists(), "ingest requires a fresh directory")
        directory.mkdir(parents=True)
        save_json(directory / "intake.json", spec)
    else:
        require(directory.is_dir(), "ingest the dataset first")
        validate_spec(load_json(directory / "intake.json"))
    require(not (directory / OUTPUTS[phase]).exists(), "phase already recorded; use existing evidence or a new case")
    worker = Path(worker_python) if worker_python else ROOT / (
        ".venvs/identification/Scripts/python.exe" if sys.platform == "win32" else ".venvs/identification/bin/python")
    require(worker.is_file(), "prepare the isolated identification environment")
    logs = directory / "logs"
    logs.mkdir(exist_ok=True)
    try:
        result = run_worker([str(worker), "-m", "agenticsim.adapters.panda_identification"],
                            payload=json.dumps({"phase": phase, "directory": str(directory)}),
                            cwd=ROOT, env=worker_environment(directory), timeout=180)
    except subprocess.TimeoutExpired as exc:
        (logs / (phase + "-stderr.log")).write_text(exc.stderr or "worker timed out", encoding="utf-8")
        raise TimeoutError("identification worker timed out") from exc
    for suffix, value in (("stdout", result.stdout), ("stderr", result.stderr)):
        (logs / f"{phase}-{suffix}.log").write_text(value, encoding="utf-8")
    require(result.returncode == 0, f"{phase} worker failed; inspect {logs / (phase + '-stderr.log')}")
    require((directory / OUTPUTS[phase]).is_file(), "worker did not commit phase evidence")
    receipts = directory / "receipts"
    receipts.mkdir(exist_ok=True)
    save_json(receipts / (phase + ".json"), {"phase": phase, "file": OUTPUTS[phase],
                                             "sha256": sha256(directory / OUTPUTS[phase])})
    return json.loads(result.stdout)


def main(argv=None):
    if hasattr(sys.stdout, "reconfigure"):
        sys.stdout.reconfigure(encoding="utf-8")
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("phase", choices=tuple(OUTPUTS))
    parser.add_argument("--directory", type=Path, default=RUNS / "identification" / ("case-" + uuid4().hex[:12]))
    parser.add_argument("--spec", type=Path)
    parser.add_argument("--worker-python", type=Path)
    args = parser.parse_args(argv)
    try:
        result = execute(args.phase, args.directory, spec_path=args.spec, worker_python=args.worker_python)
    except (ValueError, KeyError, TypeError, OSError, TimeoutError) as exc:
        parser.error(str(exc))
    print(json.dumps(result, indent=2, allow_nan=False))


if __name__ == "__main__":
    main()
