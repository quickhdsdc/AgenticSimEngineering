"""JSON CLI usable by a harness shell tool; no LLM or external service needed."""
import argparse
import hashlib
import json
import platform
import sys
from datetime import datetime, timezone
from pathlib import Path
from uuid import uuid4
from .demo import CoolingBackend, experiment


def main() -> None:
    if hasattr(sys.stdout, "reconfigure"):
        sys.stdout.reconfigure(encoding="utf-8")
    parser = argparse.ArgumentParser(description=__doc__)
    commands = parser.add_subparsers(dest="command", required=True)
    commands.add_parser("describe", help="Describe the executable demo model")
    commands.add_parser("capabilities", help="List CLI capabilities and their integration status as JSON")
    tensile = commands.add_parser("tensile", help="Tensile2D FEM identification, residual benchmark and official-format bundle")
    tensile.add_argument("phase", choices=("prepare", "model", "fit", "score", "export", "run"))
    tensile.add_argument("--directory", type=Path)
    tensile.add_argument("--model-spec", type=Path)
    tensile.add_argument("--worker-python", type=Path)
    plaid = commands.add_parser("plaid", help="Pinned Tensile2D surrogate and local holdout benchmark")
    plaid.add_argument("phase", choices=("prepare", "fit", "score", "export", "run"))
    plaid.add_argument("--directory", type=Path)
    plaid.add_argument("--recipe", type=Path)
    plaid.add_argument("--worker-python", type=Path)
    modelbench = commands.add_parser("modelbench", help="FactoryFlow / ChronoBench constrained native modeling pilots")
    modelbench.add_argument("phase", choices=("prepare", "score", "run"))
    modelbench.add_argument("--suite", required=True, choices=("factoryflow", "chronobench"))
    modelbench.add_argument("--directory", type=Path)
    modelbench.add_argument("--submission", type=Path)
    modelbench.add_argument("--worker-python", type=Path)
    semantic = commands.add_parser("semantic", help="Official AAS checker and synthetic mapping benchmark")
    semantic.add_argument("phase", choices=("prepare", "check", "score", "run"))
    semantic.add_argument("--directory", type=Path)
    semantic.add_argument("--submission", type=Path)
    semantic.add_argument("--worker-python", type=Path)
    benchmark = commands.add_parser("benchmark", help="Pinned real screw-driving grouped offline benchmark")
    benchmark.add_argument("phase", choices=("prepare", "fit", "score", "run"))
    benchmark.add_argument("--directory", type=Path)
    benchmark.add_argument("--spec", type=Path)
    calibration = commands.add_parser("calibrate", help="Synthetic KUKA data, model generation, identification and held-out checks")
    calibration.add_argument("phase", choices=("generate", "model", "fit", "validate"))
    calibration.add_argument("--directory", type=Path)
    calibration.add_argument("--spec", type=Path)
    calibration.add_argument("--worker-python", type=Path)
    identification = commands.add_parser("identify", help="Offline MERL Panda measurements, model, fit and holdout validation")
    identification.add_argument("phase", choices=("ingest", "model", "fit", "validate"))
    identification.add_argument("--directory", type=Path)
    identification.add_argument("--spec", type=Path)
    identification.add_argument("--worker-python", type=Path)
    visual = commands.add_parser("visualize", help="Export or serve verified read-only 3D Study playback")
    visual_commands = visual.add_subparsers(dest="visual_command", required=True)
    export = visual_commands.add_parser("export", help="Export a FIFO or registered robot Study with local Three.js resources")
    export.add_argument("study", type=Path)
    export.add_argument("--output", type=Path, default=Path("work/runs/visualizations"))
    serve = visual_commands.add_parser("serve", help="Serve a verified viewer bundle on 127.0.0.1")
    serve.add_argument("bundle", type=Path)
    serve.add_argument("--port", type=int, default=8765)
    demo = commands.add_parser("demo", help="Run synthetic parameter identification")
    demo.add_argument("--output", type=Path, default=Path("work/runs"))
    study = commands.add_parser("study", help="Compare a versioned task using registered isolated backends")
    study.add_argument("task", type=Path)
    study.add_argument("--output", type=Path, default=Path("work/runs/studies"))
    study.add_argument("--worker-python", type=Path, help="Python environment containing the selected backend dependencies")
    args = parser.parse_args()
    if args.command == "tensile":
        from .runtime.tensile_benchmark import main as tensile_main
        options = [args.phase]
        for flag in ("directory", "model_spec", "worker_python"):
            value = getattr(args, flag)
            if value is not None:
                options += ["--" + flag.replace("_", "-"), str(value)]
        tensile_main(options)
        return
    if args.command == "plaid":
        from .runtime.plaid_benchmark import main as plaid_main
        options = [args.phase]
        for flag in ("directory", "recipe", "worker_python"):
            value = getattr(args, flag)
            if value is not None:
                options += ["--" + flag.replace("_", "-"), str(value)]
        plaid_main(options)
        return
    if args.command == "modelbench":
        from .runtime.model_benchmark import main as modelbench_main
        options = [args.phase, "--suite", args.suite]
        for flag in ("directory", "submission", "worker_python"):
            value = getattr(args, flag)
            if value is not None:
                options += ["--" + flag.replace("_", "-"), str(value)]
        modelbench_main(options)
        return
    if args.command == "semantic":
        from .runtime.semantic_benchmark import main as semantic_main
        options = [args.phase]
        for flag in ("directory", "submission", "worker_python"):
            value = getattr(args, flag)
            if value is not None:
                options += ["--" + flag.replace("_", "-"), str(value)]
        semantic_main(options)
        return
    if args.command == "benchmark":
        from .runtime.benchmark import main as benchmark_main
        options = [args.phase]
        for flag in ("directory", "spec"):
            value = getattr(args, flag)
            if value is not None:
                options += ["--" + flag, str(value)]
        benchmark_main(options)
        return
    if args.command in ("calibrate", "identify"):
        if args.command == "calibrate":
            from .runtime.calibration import main as calibration_main
        else:
            from .runtime.identification import main as calibration_main
        options = [args.phase]
        for flag in ("directory", "spec", "worker_python"):
            value = getattr(args, flag)
            if value is not None:
                options += ["--" + flag.replace("_", "-"), str(value)]
        calibration_main(options)
        return
    if args.command == "visualize":
        try:
            if args.visual_command == "export":
                from .visualization.bundle import export
                print(json.dumps(export(args.study, args.output), ensure_ascii=False))
            else:
                from .visualization.serve import serve
                serve(args.bundle, args.port)
        except (ValueError, KeyError, TypeError, OSError) as exc:
            parser.error(str(exc))
        return
    if args.command == "capabilities":
        from .capabilities import catalog
        print(json.dumps(catalog(), ensure_ascii=False, indent=2))
        return
    if args.command == "study":
        from .application import compare
        from .runtime.registry import evaluator_for
        from .adapters.execution import FileStudyStore, LocalWorker
        try:
            document = json.loads(args.task.read_text(encoding="utf-8"))
            result = compare(document, evaluator_for(document), LocalWorker(args.worker_python), FileStudyStore(args.output))
        except (ValueError, KeyError, TypeError) as exc:
            parser.error(str(exc))
        print(json.dumps({"study_id": result["study_id"], "status": result["status"],
                          "decision": result["decision"],
                          "artifact": str((args.output / result["study_id"] / "study.json").resolve())}))
        if result["status"] == "completed_with_errors":
            raise SystemExit(1)
        return
    if args.command == "describe":
        print(json.dumps(CoolingBackend.card.to_dict(), ensure_ascii=False, indent=2))
        return
    result = experiment()
    canonical = json.dumps(result, sort_keys=True, allow_nan=False).encode()
    result["evidence_sha256"] = hashlib.sha256(canonical).hexdigest()
    result["runtime"] = {"python": platform.python_version(), "platform": platform.platform()}
    result["created_at"] = datetime.now(timezone.utc).isoformat()
    result["run_id"] = str(uuid4())
    args.output.mkdir(parents=True, exist_ok=True)
    destination = args.output / f"{result['run_id']}.json"
    with destination.open("x", encoding="utf-8") as handle:
        json.dump(result, handle, ensure_ascii=False, indent=2, allow_nan=False)
    print(json.dumps({"run_id": result["run_id"], "artifact": str(destination.resolve()),
                      "best": result["calibration"]["best"]}, ensure_ascii=False))


if __name__ == "__main__":
    main()
