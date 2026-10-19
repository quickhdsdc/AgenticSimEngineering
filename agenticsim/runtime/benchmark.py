"""Reproducible offline benchmarks exposed as deterministic Harness CLI tools."""
import argparse
from datetime import datetime, timezone
import json
import math
from pathlib import Path
import time
from uuid import uuid4

from agenticsim.adapters.screw_dataset import ARCHIVE_SHA256, FEATURES, read_archive
from agenticsim.domain import require
from agenticsim.domain.benchmark import anomaly_score, choose_threshold, classification_metrics, fit_normal, grouped_split
from agenticsim.runtime.calibration import directory_path, load_json, save_json, sha256
from agenticsim.runtime.paths import ROOT, RUNS

OUTPUTS = {"prepare": "dataset.json", "fit": "fit.json", "score": "score.json"}


def validate_spec(spec):
    require(spec.get("schema_version") == "1.0" and spec.get("case_id") == "screw-s01-post-cycle-quality-v1",
            "unsupported benchmark case")
    require(spec.get("source") == "measurement" and spec.get("archive_sha256") == ARCHIVE_SHA256,
            "benchmark requires pinned real measurements")
    require(type(spec.get("split_seed")) is int, "integer split seed required")
    require(spec.get("split") == "workpiece grouped 60/20/20, SHA-256 seed:workpiece ordering", "unsupported split")
    floors = spec.get("scale_floors", {})
    require(set(floors) == set(FEATURES) and all(type(v) in (int, float) and math.isfinite(v) and v > 0
                                             for v in floors.values()), "invalid feature scales")
    candidates = spec.get("threshold_candidates")
    require(isinstance(candidates, list) and bool(candidates) and all(type(v) in (int, float)
            and math.isfinite(v) and v > 0 for v in candidates), "invalid threshold candidates")
    window = spec.get("reference_torque_window_nm")
    require(isinstance(window, list) and len(window) == 2 and all(type(v) in (int, float) and math.isfinite(v)
            for v in window) and window[0] < window[1], "invalid reference torque window")
    acceptance = spec.get("acceptance", {})
    require(acceptance.get("scope") == "dataset_quality_classification_only", "invalid acceptance scope")
    require(all(type(acceptance.get(k)) in (int, float) and 0 <= acceptance[k] <= 1 for k in
                ("minimum_average_precision", "maximum_false_positive_rate")), "invalid acceptance limits")


def seal(directory, phase, names):
    save_json(directory / ".evaluation" / (phase + "-receipt.json"),
              {name: sha256(directory / name) for name in names})


def verify(directory, phase):
    for name, expected in load_json(directory / ".evaluation" / (phase + "-receipt.json")).items():
        target = (directory / name).resolve()
        require(target.is_relative_to(directory.resolve()) and sha256(target) == expected,
                "benchmark evidence changed: " + name)


def execute(phase, directory, spec_path=None):
    require(phase in OUTPUTS, "unknown benchmark phase")
    directory = directory_path(directory)
    started = time.monotonic()
    if phase == "prepare":
        spec = load_json(spec_path or ROOT / "examples/screw-benchmark.json")
        validate_spec(spec)
        archive = (ROOT / spec["archive"]).resolve()
        require(archive.is_relative_to((ROOT / "work/cache/datasets").resolve()), "archive must be in project dataset cache")
        require(not directory.exists(), "prepare requires a fresh benchmark directory")
        records, labels, quality, notice = read_archive(archive)
        splits = grouped_split(records, spec["split_seed"])
        require(all({labels[r["operation_id"]] for r in part} == {0, 1} for part in splits.values()),
                "each partition requires both classes")
        directory.mkdir(parents=True)
        private = directory / ".evaluation"
        private.mkdir()
        save_json(directory / "intake.json", spec)
        (directory / "SOURCE-README.txt").write_text(notice, encoding="utf-8")
        save_json(private / "labels.json", labels)
        for part, values in splits.items():
            save_json(directory / (".evaluation/test.json" if part == "test" else part + ".json"), values)
        dataset = {"schema_version": "1.0", "case_id": spec["case_id"], "source": "measurement",
                   "archive_sha256": ARCHIVE_SHA256, "source_url": spec["dataset"],
                   "license": "MIT in pinned archive README; keep original notice and attribution",
                   "features": FEATURES, "mapping": {"torque": "N*m", "angle": "deg", "time": "s"},
                   "timestamp_timezone": "unknown; timestamp is descriptive, not a live clock",
                   "split_seed": spec["split_seed"], "split_policy": spec["split"],
                   "partitions": {p: {"operations": len(v), "workpieces": len({r["workpiece_id"] for r in v})}
                                  for p, v in splits.items()}, "data_quality": quality,
                   "evaluation_scope": "post-operation station-label prediction; no early warning",
                   "assumptions": spec["assumptions"], "physical_validation": "not_established"}
        save_json(directory / "dataset.json", dataset)
        seal(directory, "prepare", ["intake.json", "dataset.json", "train.json", "validation.json",
                                    ".evaluation/test.json", ".evaluation/labels.json", "SOURCE-README.txt"])
    else:
        require(directory.is_dir(), "prepare the benchmark first")
        require(not (directory / OUTPUTS[phase]).exists(), "phase already recorded; start a new benchmark to repeat")
        verify(directory, "prepare")
        spec = load_json(directory / "intake.json")
        validate_spec(spec)
        labels = load_json(directory / ".evaluation/labels.json")
        if phase == "fit":
            train, validation = load_json(directory / "train.json"), load_json(directory / "validation.json")
            model = fit_normal(train, labels, FEATURES, spec["scale_floors"])
            scores = [anomaly_score(r, model) for r in validation]
            truth = [labels[r["operation_id"]] for r in validation]
            threshold = choose_threshold(truth, scores, spec["threshold_candidates"])
            save_json(directory / "fit.json", {"algorithm": "maximum absolute median/MAD deviation",
                      "source": "learned_prediction", "training_source": "measurement", "model": model,
                      "threshold": threshold, "threshold_source": "fixed grid, maximum validation F1; ties choose higher",
                      "validation_metrics": classification_metrics(truth, scores, threshold),
                      "test_access": "none during fitting", "physical_validation": "not_established"})
            seal(directory, "fit", ["fit.json"])
        else:
            verify(directory, "fit")
            fitted = load_json(directory / "fit.json")
            records = load_json(directory / ".evaluation/test.json")
            truth = [labels[r["operation_id"]] for r in records]
            scores = [anomaly_score(r, fitted["model"]) for r in records]
            metrics = classification_metrics(truth, scores, fitted["threshold"])
            low, high = spec["reference_torque_window_nm"]
            # The last graph sample follows tool release; it is not target torque.
            # A peak-torque window is a simple comparator, not the station's own classifier.
            rule_scores = [max(low - r["features"]["peak_torque_nm"], r["features"]["peak_torque_nm"] - high)
                           for r in records]
            predictions = [{"operation_id": r["operation_id"], "workpiece_id": r["workpiece_id"],
                            "cycle": r["cycle"], "timestamp_local": r["timestamp_local"], "location": r["location"],
                            "source": "learned_prediction", "score": s, "flag": s >= fitted["threshold"]}
                           for r, s in zip(records, scores)]
            predictions.sort(key=lambda r: r["cycle"])
            save_json(directory / "predictions.json", predictions)
            accepted = (metrics["average_precision"] >= spec["acceptance"]["minimum_average_precision"]
                        and metrics["false_positive_rate"] <= spec["acceptance"]["maximum_false_positive_rate"])
            report = {"schema_version": "1.0", "case_id": spec["case_id"], "source": "measurement",
                      "detector_output_source": "learned_prediction", "status": "scored", "accepted": accepted,
                      "acceptance": spec["acceptance"], "test_metrics": metrics,
                      "baselines": {"always_normal": classification_metrics(truth, [0.] * len(truth), 1.),
                                    "reference_peak_torque_window": classification_metrics(truth, rule_scores, 0.)},
                      "threshold": fitted["threshold"], "fit_sha256": sha256(directory / "fit.json"),
                      "archive_sha256": ARCHIVE_SHA256, "predictions_sha256": sha256(directory / "predictions.json"),
                      "metric_definition": "average precision with grouped ties; not trapezoidal PR-AUC",
                      "limitations": ["Labels are station outcomes, not independent physical inspection",
                                      "Completed-operation prediction, including process termination; no early warning",
                                      "Grouped holdout from one station/scenario; no cross-site acceptance",
                                      "No live OPC, MES actions, repair success or causal diagnosis measured"],
                      "physical_validation": "not_established", "created_at": datetime.now(timezone.utc).isoformat()}
            save_json(directory / "score.json", report)
            seal(directory, "score", ["score.json", "predictions.json"])
    output = load_json(directory / OUTPUTS[phase])
    # Compact tool results keep raw observations and private labels out of the context.
    return {"phase": phase, "artifact": str(directory / OUTPUTS[phase]), "elapsed_seconds": time.monotonic() - started,
            **{key: output[key] for key in ("case_id", "partitions", "threshold", "test_metrics", "baselines", "accepted")
               if key in output}, "physical_validation": "not_established"}


def main(argv=None):
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("phase", choices=(*OUTPUTS, "run"))
    parser.add_argument("--directory", type=Path)
    parser.add_argument("--spec", type=Path)
    args = parser.parse_args(argv)
    if args.phase in ("fit", "score") and args.directory is None:
        parser.error("fit/score require the directory returned by prepare")
    directory = args.directory or RUNS / "benchmarks" / ("screw-" + uuid4().hex[:12])
    try:
        if args.phase == "run":
            result = {phase: execute(phase, directory, args.spec if phase == "prepare" else None) for phase in OUTPUTS}
        else:
            result = execute(args.phase, directory, args.spec)
    except (ValueError, KeyError, TypeError, OSError) as exc:
        parser.error(str(exc))
    print(json.dumps(result, indent=2, allow_nan=False))


if __name__ == "__main__":
    main()
