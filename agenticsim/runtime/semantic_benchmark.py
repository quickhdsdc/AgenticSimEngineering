"""AAS conformance plus explicit mapping challenges, using one existing Harness."""
import argparse
import json
from pathlib import Path
import subprocess
import sys
from uuid import uuid4

from agenticsim.adapters.execution import worker_environment
from agenticsim.adapters.process_tree import run_worker
from agenticsim.domain import require
from agenticsim.domain.semantics import assess
from agenticsim.runtime.benchmark import seal, verify
from agenticsim.runtime.calibration import directory_path, load_json, save_json, sha256
from agenticsim.runtime.paths import ROOT, RUNS

PHASES = {"prepare": "dataset.json", "check": "checks.json", "score": "score.json"}


def evaluate_submission(rows, gold):
    require(isinstance(rows, list) and len(rows) == len(gold), "submit exactly one answer per case")
    answers = {}
    for row in rows:
        require(isinstance(row, dict) and row.get("case_id") in gold and row["case_id"] not in answers,
                "unknown or duplicate case answer")
        require(row.get("decision") in ("accept", "reject", "clarify"), "invalid decision")
        require(isinstance(row.get("codes"), list) and all(isinstance(c, str) for c in row["codes"])
                and len(set(row["codes"])) == len(row["codes"]), "invalid reason codes")
        answers[row["case_id"]] = row
    decisions = sum(answers[key]["decision"] == expected["decision"] for key, expected in gold.items())
    exact = sum(answers[key]["decision"] == expected["decision"] and sorted(answers[key]["codes"]) == expected["codes"]
                for key, expected in gold.items())
    return {"cases": len(gold), "correct_decisions": decisions, "exact_decision_and_codes": exact,
            "decision_accuracy": decisions / len(gold), "exact_accuracy": exact / len(gold),
            "accepted": exact == len(gold)}


def execute(phase, directory, *, submission=None, worker_python=None):
    directory = directory_path(directory)
    require(phase in PHASES, "unknown semantic benchmark phase")
    if phase == "prepare":
        require(not directory.exists(), "prepare requires a new semantic directory")
        from agenticsim.adapters.semantic_cases import cases
        inputs, gold = cases()
        directory.mkdir(parents=True)
        (directory / ".evaluation").mkdir()
        save_json(directory / "cases.json", inputs)
        save_json(directory / ".evaluation/gold.json", gold)
        save_json(directory / "dataset.json", {"schema_version": "1.0", "case_id": "aas-mapping-fixtures-v1",
                  "source": "synthetic_fixture", "cases": len(inputs), "authoring": "vendored A4S AAS factories",
                  "semantic_ids": "project URNs, not official IDTA submodel template IDs",
                  "gold_source": "predeclared manual fixture mutations; not generated from checker outputs",
                  "scope": "closed local fixture set; not equipment measurement or API/template certification",
                  "physical_validation": "not_established"})
        seal(directory, "prepare", ["cases.json", "dataset.json", ".evaluation/gold.json"])
    else:
        verify(directory, "prepare")
        require(not (directory / PHASES[phase]).exists(), "phase already recorded; use another run")
        if phase == "check":
            worker = worker_python or ROOT / (
                ".venvs/semantic/Scripts/python.exe" if sys.platform == "win32" else ".venvs/semantic/bin/python")
            require(Path(worker).is_file(), "install the isolated semantic environment")
            inputs = load_json(directory / "cases.json")
            logs = directory / "official-checks"
            logs.mkdir()
            result = run_worker([str(worker), "-m", "agenticsim.adapters.aas_conformance"],
                payload=json.dumps({"cases": inputs, "directory": str(logs)}), cwd=ROOT,
                env=worker_environment(directory), timeout=60)
            (logs / "stdout.log").write_text(result.stdout, encoding="utf-8")
            (logs / "stderr.log").write_text(result.stderr, encoding="utf-8")
            require(result.returncode == 0, "official checker worker failed; inspect official-checks/stderr.log")
            official = json.loads(result.stdout)
            require([r["case_id"] for r in official["results"]] == [c["case_id"] for c in inputs], "checker case mismatch")
            findings = [{**row, **assess(case, row["schema_ok"])} for case, row in zip(inputs, official["results"])]
            save_json(directory / "checks.json", {"tool": official["tool"], "version": official["version"],
                      "scope": official["scope"], "source": "synthetic_fixture", "results": findings,
                      "mapping_check": "project source facts and explicit units/frames/ranges; separate from official checks"})
            seal(directory, "check", ["checks.json", *["official-checks/" + r["case_id"] + ".json" for r in findings]])
        else:
            verify(directory, "check")
            require(submission is not None, "score requires a submission JSON")
            path = Path(submission).resolve()
            require(path.is_relative_to(directory.parent.resolve()) and ".evaluation" not in path.parts,
                    "submission must be under the case parent and outside private evaluation")
            gold = load_json(directory / ".evaluation/gold.json")
            rows = load_json(path)
            official = load_json(directory / "checks.json")["results"]
            schema_only = [{"case_id": r["case_id"], "decision": "accept" if r["schema_ok"] else "reject",
                            "codes": [] if r["schema_ok"] else ["AAS_SCHEMA"]} for r in official]
            deterministic = [{key: r[key] for key in ("case_id", "decision", "codes")} for r in official]
            save_json(directory / "submission.json", rows)
            save_json(directory / "score.json", {"case_id": "aas-mapping-fixtures-v1", "source": "synthetic_fixture",
                      "submission": evaluate_submission(rows, gold), "submission_sha256": sha256(directory / "submission.json"),
                      "baselines": {"schema_only": evaluate_submission(schema_only, gold),
                                    "schema_and_explicit_mapping": evaluate_submission(deterministic, gold)},
                      "official_schema_passes": sum(r["schema_ok"] for r in official),
                      "limitations": ["Sixteen designed fixtures, fixed prompts; no unseen generalization claim",
                                      "Custom project submodel, not an IDTA template or certified live API",
                                      "Source/mapping facts are synthetic; no physical endpoint accepted",
                                      "Checker-assisted Agent interpretation, not autonomous semantic authoring"],
                      "physical_validation": "not_established"})
            seal(directory, "score", ["submission.json", "score.json"])
    output = load_json(directory / PHASES[phase])
    return {"phase": phase, "artifact": str(directory / PHASES[phase]), "source": "synthetic_fixture",
            **{key: output[key] for key in ("cases", "results", "submission", "baselines", "official_schema_passes") if key in output}}


def main(argv=None):
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("phase", choices=(*PHASES, "run"))
    parser.add_argument("--directory", type=Path)
    parser.add_argument("--submission", type=Path)
    parser.add_argument("--worker-python", type=Path)
    args = parser.parse_args(argv)
    if args.phase in ("check", "score") and not args.directory:
        parser.error("check/score require the prepared directory")
    directory = args.directory or RUNS / "benchmarks" / ("semantic-" + uuid4().hex[:12])
    try:
        if args.phase == "run":
            execute("prepare", directory)
            result = execute("check", directory, worker_python=args.worker_python)
            path = directory / "baseline-submission.json"
            save_json(path, [{k: r[k] for k in ("case_id", "decision", "codes")} for r in result["results"]])
            result = execute("score", directory, submission=path)
        else:
            result = execute(args.phase, directory, submission=args.submission, worker_python=args.worker_python)
    except (ValueError, KeyError, TypeError, OSError, subprocess.TimeoutExpired) as exc:
        parser.error(str(exc))
    print(json.dumps(result, indent=2, allow_nan=False))


if __name__ == "__main__":
    main()
