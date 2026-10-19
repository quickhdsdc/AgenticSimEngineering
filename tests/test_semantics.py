"""AAS schema is not sufficient for identity, units or measurement provenance."""
import copy
import importlib.util
import json
import os
from pathlib import Path
import tempfile
import unittest

from agenticsim.domain.semantics import assess, mapping_findings
from agenticsim.runtime.semantic_benchmark import evaluate_submission, execute
from agenticsim.runtime.paths import RUNS

HAS_HARNESS = importlib.util.find_spec("a4s") is not None


@unittest.skipUnless(HAS_HARNESS, "install vendored harness for AAS authoring")
class SemanticTests(unittest.TestCase):
    def test_mapping_changes_are_distinct_from_schema_validity(self):
        from agenticsim.adapters.semantic_cases import base_case
        case = base_case()
        self.assertEqual(assess(case, True), {"decision": "accept", "codes": []})
        case["aas"]["assetAdministrationShells"][0]["assetInformation"]["globalAssetId"] = "urn:other"
        self.assertEqual(assess(case, True), {"decision": "reject", "codes": ["ASSET_ID"]})
        self.assertEqual(assess(case, False), {"decision": "reject", "codes": ["AAS_SCHEMA"]})

    def test_convertible_units_unknown_timezone_and_nonfinite_values(self):
        from agenticsim.adapters.semantic_cases import base_case
        case = base_case()
        case["mapping"]["signals"][0].update(source_unit="Ncm", source_value=140., scale=.01)
        self.assertEqual(mapping_findings(case), [])
        case["mapping"]["utc_timestamp_required"] = True
        self.assertEqual(assess(case, True)["decision"], "clarify")
        case["aas"]["submodels"][0]["submodelElements"][0]["value"] = "NaN"
        result = assess(case, True)
        self.assertEqual(result["decision"], "reject")
        self.assertIn("VALUE", result["codes"])
        self.assertIn("TIMEZONE_UNKNOWN", result["codes"])

    def test_scorer_rejects_duplicates_missing_cases_and_invalid_decisions(self):
        gold = {"a": {"decision": "accept", "codes": []}, "b": {"decision": "reject", "codes": ["UNIT"]}}
        good = [{"case_id": key, **row} for key, row in gold.items()]
        self.assertEqual(evaluate_submission(good, gold)["exact_accuracy"], 1.)
        for rows in (good[:1], [good[0], good[0]], [{**good[0], "decision": "guess"}, good[1]]):
            with self.assertRaises(ValueError):
                evaluate_submission(rows, gold)
        wrong = copy.deepcopy(good)
        wrong[1]["codes"] = []
        self.assertEqual(evaluate_submission(wrong, gold)["correct_decisions"], 2)
        self.assertFalse(evaluate_submission(wrong, gold)["accepted"])

    @unittest.skipUnless(os.getenv("ASE_SEMANTIC_PYTHON"), "set ASE_SEMANTIC_PYTHON for official conformance checks")
    def test_official_checker_frozen_gold_and_schema_only_baseline(self):
        RUNS.mkdir(parents=True, exist_ok=True)
        with tempfile.TemporaryDirectory(dir=RUNS) as temporary:
            path = Path(temporary) / "semantic"
            execute("prepare", path)
            result = execute("check", path, worker_python=Path(os.environ["ASE_SEMANTIC_PYTHON"]))
            rows = [{k: r[k] for k in ("case_id", "decision", "codes")} for r in result["results"]]
            submission = Path(temporary) / "submission.json"
            submission.write_text(json.dumps(rows))
            score = execute("score", path, submission=submission)
            self.assertEqual(score["official_schema_passes"], 14)
            self.assertEqual(score["baselines"]["schema_only"]["correct_decisions"], 4)
            self.assertTrue(score["submission"]["accepted"])
            with self.assertRaises(ValueError):
                execute("score", path, submission=submission)
            # A different run with a deliberately wrong submission must not be accepted.
            path2 = Path(temporary) / "bad"
            execute("prepare", path2)
            execute("check", path2, worker_python=Path(os.environ["ASE_SEMANTIC_PYTHON"]))
            rows[4]["decision"] = "accept"
            rows[4]["codes"] = []
            submission.write_text(json.dumps(rows))
            self.assertFalse(execute("score", path2, submission=submission)["submission"]["accepted"])
            # Tampering after the phase must prevent further scoring.
            (path2 / "cases.json").write_text("[]")
            with self.assertRaisesRegex(ValueError, "evidence changed"):
                execute("score", path2, submission=submission)


if __name__ == "__main__":
    unittest.main()
