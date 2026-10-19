"""Verify a core wheel from outside the checkout, without optional runtimes."""
import argparse
import json
from pathlib import Path
import subprocess
import sys
import tempfile
import venv
import zipfile

ROOT = Path(__file__).resolve().parents[2]


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--wheel", type=Path)
    args = parser.parse_args()
    wheels = sorted((ROOT / "work/runs/distribution").glob("*.whl"))
    wheel = args.wheel or (wheels[-1] if wheels else None)
    if wheel is None:
        parser.error("run uv build --wheel --out-dir work/runs/distribution first")
    with zipfile.ZipFile(wheel) as archive:
        names = archive.namelist()
        required = {"agenticsim/runtime/dialogue.py", "agenticsim/visualization/web/index.html"}
        if not required.issubset(names) or any(n.startswith(("components/", "scripts/", "work/")) or "/.env" in n for n in names):
            raise ValueError("wheel package contents do not match the product boundary")
    with tempfile.TemporaryDirectory() as temporary:
        directory = Path(temporary)
        venv.EnvBuilder(with_pip=False).create(directory / "env")
        python = directory / ("env/Scripts/python.exe" if sys.platform == "win32" else "env/bin/python")
        subprocess.run(["uv", "pip", "install", "--no-deps", "--python", str(python), str(wheel.resolve())],
                       check=True, capture_output=True)
        probe = subprocess.run([str(python), "-I", "-c",
            "import agenticsim, importlib.util, json; "
            "print(json.dumps({'origin':agenticsim.__file__, 'harness':importlib.util.find_spec('a4s') is not None}))"],
            cwd=directory, check=True, capture_output=True, text=True)
        data = json.loads(probe.stdout)
        if Path(data["origin"]).resolve().is_relative_to(ROOT) or data["harness"]:
            raise ValueError("distribution probe used the checkout or optional Harness")
        for command in ("describe", "capabilities"):
            result = subprocess.run([str(python), "-I", "-m", "agenticsim", command], cwd=directory,
                                    check=True, capture_output=True, text=True, encoding="utf-8")
            json.loads(result.stdout)
        # Source tests in the minimal wheel environment detect accidental
        # optional imports. The preceding -I CLI probe tests installed code.
        subprocess.run([str(python), str(ROOT / "work/tools/check.py"), "--profile", "core"],
                       cwd=ROOT, check=True, capture_output=True, text=True)
    print(json.dumps({"status": "passed", "scope": "installed core wheel, no checkout or Harness",
                      "wheel": str(wheel), "commands": ["describe", "capabilities"],
                      "source_core_tests_in_minimal_environment": "passed"}))


if __name__ == "__main__":
    main()
