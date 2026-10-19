"""Read-only capability preflight; full migration assets are a separate profile."""
from pathlib import Path
import argparse
import importlib.util
import json
import subprocess
import sys

ROOT = Path(__file__).resolve().parents[1]
ASSETS = ["components/harness/a4s/kernel/loop.py", "components/harness/BaSyxMinimal/docker-compose.yml",
          "components/harness/_others/OpenMes-main/backend/Dockerfile", "components/des/src/opsworld/world.py",
          "components/dynamics/projects/mbs_isaac/newton_backend.py",
          "components/fem/openform_poc/formpipe/openradioss.py",
          "components/fem/openform_poc/tools/OpenRadioss/exec/engine_win64.exe",
          "components/fem/Samples/7-Q2603019/0220M000ND_stamping-Al 1050 Stamping.zip"]


PROFILES = {
    "tensile": (["agenticsim", "a4s"], ["examples/tensile2d-model.json", "examples/tensile2d-benchmark-brief.md",
                               "work/cache/datasets/plaid-tensile2d/protocol/manifest.json",
                               "work/cache/datasets/plaid-tensile2d/data/all_samples-00000-of-00002.parquet",
                               "work/cache/datasets/plaid-tensile2d/data/all_samples-00001-of-00002.parquet"]),
    "plaid": (["agenticsim"], ["examples/plaid-tensile2d-source.json", "environments/plaid/requirements.txt",
                               "work/cache/datasets/plaid-tensile2d/data/all_samples-00000-of-00002.parquet",
                               "work/cache/datasets/plaid-tensile2d/data/all_samples-00001-of-00002.parquet"]),
    "core": (["agenticsim"], ["agenticsim/domain/__init__.py", "examples/queue-study.json"]),
    "factoryflow": (["agenticsim"], ["components/model_benchmarks/manifest.json", "environments/factoryflow/requirements.txt"]),
    "chronobench": (["agenticsim"], ["components/model_benchmarks/manifest.json", "environments/chrono/environment.yml"]),
    "benchmark": (["agenticsim"], ["examples/screw-benchmark.json", "examples/benchmark-catalog.json",
                                    "work/cache/datasets/screw-s01-v1.1.3/s01_thread-degradation.zip"]),
    "semantic": (["agenticsim", "a4s"], ["examples/semantic-benchmark-brief.md", "environments/semantic/requirements.txt"]),
    "des": (["agenticsim", "numpy", "simpy"], ["components/des/src/gradientshop/reference/simpy_engine.py"]),
    "harness": (["agenticsim", "a4s"], ["components/harness/a4s/kernel/loop.py"]),
    "festo": (["agenticsim", "a4s", "asyncua", "fastapi", "pypdfium2", "tzdata"], [
        "components/harness/a4s/machines/festo_cp_lab/live_monitor.py",
        "components/harness/a4s/skills/fault-maintenance/SKILL.md",
        "components/harness/a4s/machines/festo_cp_lab/documents/Festo-Pneumatic-Fault-Service-Guide.pdf",
        "components/harness/BaSyxMinimal/aas/festo-cp-lab.json"]),
    "robot": (["agenticsim"], ["examples/kuka-iiwa14-study.json", "assets/robots/menagerie/manifest.json",
                                 "assets/robots/menagerie/kuka_iiwa_14/iiwa14.xml",
                                 "assets/robots/menagerie/unitree_g1/g1.xml"]),
    "identification": (["agenticsim"], ["examples/panda-identification.json", "assets/robots/panda-identification/manifest.json",
                                        "assets/robots/panda-identification/panda.xml", "assets/robots/panda-identification/LICENSE",
                                        "work/cache/datasets/merl-lip4rid-v1/PUB-5510-LIP4RID.zip"]),
    "fem": (["agenticsim", "mold_agent"], ASSETS[-2:]),
    "full": (["agenticsim", "a4s", "mold_agent"], ASSETS),
}


def inspect(profile="full", worker_python=None):
    names, assets = PROFILES[profile]
    missing = [p for p in assets if not (ROOT / p).is_file()]
    modules = {}
    external = []
    for name in names:
        spec = importlib.util.find_spec(name)
        origin = Path(spec.origin).resolve() if spec and spec.origin else None
        modules[name] = str(origin) if origin else None
        if name in {"agenticsim", "a4s", "mold_agent"} and origin and not origin.is_relative_to(ROOT):
            external.append(name)
    report = {"profile": profile, "root": str(ROOT), "python": sys.executable,
                      "python_in_project": Path(sys.executable).resolve().is_relative_to(ROOT),
                      "modules": modules, "external_project_modules": external,
                      "missing_assets": missing,
                      "scope": "local readiness; no live services, credentials or network probes"}
    if profile in ("plaid", "tensile"):
        worker = Path(worker_python) if worker_python else ROOT / (
            ".venvs/plaid/Scripts/python.exe" if sys.platform == 'win32' else '.venvs/plaid/bin/python')
        report['worker_python'] = str(worker)
        try:
            code = "import numpy, scipy, pyarrow, yaml, matplotlib; assert numpy.__version__=='2.2.6' and pyarrow.__version__=='19.0.1'; print('worker-ready')"
            probe = subprocess.run([str(worker), '-c', code], cwd=ROOT, capture_output=True, text=True, timeout=45)
            report['worker_ready'] = probe.returncode == 0 and 'worker-ready' in probe.stdout
        except (OSError, subprocess.TimeoutExpired):
            report['worker_ready'] = False
    if profile in ("factoryflow", "chronobench"):
        worker = Path(worker_python) if worker_python else ROOT / (
            ".venvs/chrono-native/python.exe" if profile == "chronobench" else ".venvs/factoryflow/Scripts/python.exe")
        report["worker_python"] = str(worker)
        try:
            code = ("from agenticsim.adapters.modelbench_worker import chrono_import; chrono_import(); print('worker-ready')"
                    if profile == "chronobench" else "import simpy, graphviz; print('worker-ready')")
            probe = subprocess.run([str(worker), "-c", code], cwd=ROOT, capture_output=True, text=True, timeout=45)
            report["worker_ready"] = probe.returncode == 0 and "worker-ready" in probe.stdout
        except (OSError, subprocess.TimeoutExpired):
            report["worker_ready"] = False
    if profile in ("robot", "identification", "semantic"):
        environment = "newton" if profile == "robot" else profile
        worker = Path(worker_python) if worker_python else ROOT / (
            f".venvs/{environment}/Scripts/python.exe" if sys.platform == "win32" else f".venvs/{environment}/bin/python")
        report["worker_python"] = str(worker)
        if not worker.is_file():
            missing.append(str(worker))
        else:
            try:
                code = ("import newton, mujoco, warp; print('worker-ready')" if profile == "robot" else
                        "import numpy, pandas, scipy, mujoco; assert pandas.__version__=='2.2.3' and mujoco.__version__=='3.8.1'; print('worker-ready')")
                if profile == "semantic":
                    code = ("from importlib.metadata import version; from aas_test_engines.file import check_json_data; "
                            "assert version('aas-test-engines')=='1.0.3'; print('worker-ready')")
                probe = subprocess.run([str(worker), "-c", code],
                                       cwd=ROOT, capture_output=True, text=True, timeout=45)
                report["worker_ready"] = probe.returncode == 0 and "worker-ready" in probe.stdout
            except (OSError, subprocess.TimeoutExpired):
                report["worker_ready"] = False
    report["ready"] = not (missing or external or any(v is None for v in modules.values()))
    if profile in ("factoryflow", "chronobench", "plaid", "tensile"):
        report["ready"] = report["ready"] and report.get("worker_ready", False)
    if profile in ("robot", "identification", "semantic"):
        report["ready"] = report["ready"] and report.get("worker_ready", False)
    return report


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--profile", choices=PROFILES, default="full")
    parser.add_argument("--worker-python")
    args = parser.parse_args()
    report = inspect(args.profile, args.worker_python)
    print(json.dumps(report, ensure_ascii=False, indent=2))
    return 0 if report["ready"] else 1


if __name__ == "__main__":
    raise SystemExit(main())
