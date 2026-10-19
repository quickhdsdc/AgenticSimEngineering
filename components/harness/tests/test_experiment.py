"""The comparison harness itself: what it holds constant, and how it judges.

An experiment whose scaffolding is untested measures the scaffolding. These
tests reach no network and call no model -- they check the parts that decide
whether the comparison is fair and whether a verdict means anything.
"""

from __future__ import annotations

import json
from pathlib import Path
from typing import Any, Dict


from a4s.experiment.run import Outcome, render, summarise
from a4s.experiment.tasks import TASKS, Context


def outcome(arm: str, task: str = "t", passed: bool = True, **kw: Any) -> Outcome:
    defaults: Dict[str, Any] = dict(
        repeat=1, status="finished", detail="", steps=5, input_tokens=1000,
        output_tokens=100, seconds=10.0, tool_calls=4, session="s", schema_bytes=500,
    )
    defaults.update(kw)
    return Outcome(arm=arm, task=task, passed=passed, **defaults)


# ---------------------------------------------------------------------------
# the task suite
# ---------------------------------------------------------------------------

def test_every_task_has_a_verifier_and_a_budget() -> None:
    assert TASKS, "the suite is empty"
    for task in TASKS:
        assert callable(task.check), f"{task.name} has no verifier"
        assert task.max_steps > 0, f"{task.name} has no step budget"
        assert task.instruction.strip(), f"{task.name} has no instruction"


def test_task_names_are_unique() -> None:
    names = [t.name for t in TASKS]
    assert len(names) == len(set(names))


def test_a_task_that_creates_things_also_tears_them_down() -> None:
    """An arm inheriting the other arm's work is not measuring anything."""
    from a4s.experiment.tasks import SCRATCH_PREFIX

    for task in TASKS:
        creates = SCRATCH_PREFIX in task.instruction
        if creates or task.setup is not None:
            assert task.teardown is not None, f"{task.name} leaves state behind"


def test_instructions_name_no_tool_and_no_command() -> None:
    """An instruction that says `aas smt instantiate` tells one arm the answer.

    The task must describe the goal, not the means, or the comparison is rigged
    before it starts. "shell" and "AAS" are exempt: they are IEC 63278
    vocabulary -- an Asset Administration Shell is the thing, not the tool.
    """
    import re

    invocation = re.compile(r"aas\s+(shell|sm|el|get|set|op|smt|index|find|match|populate|validate)")
    for task in TASKS:
        lowered = task.instruction.lower()
        assert not invocation.search(lowered), f"{task.name} spells out an `aas` command"
        for banned in ("aas_", "input_pdf2entity", "a4s extract", "--"):
            assert banned not in lowered, f"{task.name} mentions {banned!r}"


# ---------------------------------------------------------------------------
# verdicts
# ---------------------------------------------------------------------------

def test_a_report_check_needs_every_named_path(tmp_path: Path) -> None:
    from a4s.experiment.tasks import _check_retrieve

    context = Context(endpoint="http://x", workdir=tmp_path)
    context.summary = "It belongs at TechnicalPropertyAreas.CapacityEnergyVoltage.RatedCapacity."
    assert not _check_retrieve(context).passed          # MinVoltage not named
    context.summary += " And MinVoltage is the other one."
    assert _check_retrieve(context).passed


def test_report_checks_ignore_case(tmp_path: Path) -> None:
    from a4s.experiment.tasks import _check_read

    context = Context(endpoint="http://x", workdir=tmp_path)
    context.summary = "ratedcapacity, inside capacityenergyvoltage"
    assert _check_read(context).passed


def test_the_completeness_check_rejects_a_false_positive(tmp_path: Path) -> None:
    """Reporting a filled slot as a gap is wrong in the direction that matters:
    it sends someone to look for data that is already there."""
    from a4s.experiment.tasks import _check_complete

    context = Context(endpoint="http://x", workdir=tmp_path)
    context.summary = "Empty: BatteryMass, WarrantyPeriod."
    assert _check_complete(context).passed
    context.summary += " Also ManufacturerName."
    verdict = _check_complete(context)
    assert not verdict.passed
    assert "although it has a value" in verdict.detail


def test_a_failed_verdict_says_what_was_missing(tmp_path: Path) -> None:
    from a4s.experiment.tasks import _check_read

    context = Context(endpoint="http://x", workdir=tmp_path)
    context.summary = "I could not find it."
    verdict = _check_read(context)
    assert not verdict.passed
    assert "RatedCapacity" in verdict.detail


# ---------------------------------------------------------------------------
# aggregation
# ---------------------------------------------------------------------------

def test_means_include_the_failed_attempts() -> None:
    """Averaging only successes flatters whichever arm fails on the expensive
    tasks -- which is exactly the arm that should look worse."""
    outcomes = [
        outcome("cli", steps=2, passed=True),
        outcome("cli", steps=20, passed=False),
    ]
    assert summarise(outcomes)["arms"]["cli"]["steps_mean"] == 11.0


def test_pass_counts_are_per_arm_and_per_task() -> None:
    outcomes = [
        outcome("baseline", "read", True), outcome("cli", "read", True),
        outcome("baseline", "chain", False), outcome("cli", "chain", True),
    ]
    report = summarise(outcomes)
    assert report["arms"]["baseline"]["passed"] == 1
    assert report["arms"]["cli"]["passed"] == 2
    assert report["tasks"]["chain"]["baseline"]["passed"] == 0
    assert report["tasks"]["chain"]["cli"]["passed"] == 1


def test_schema_size_is_reported_per_arm() -> None:
    """The token cost of a resident tool surface is the first-order claim, so
    it is measured rather than asserted."""
    outcomes = [outcome("baseline", schema_bytes=11732), outcome("cli", schema_bytes=2260)]
    report = summarise(outcomes)
    assert report["arms"]["baseline"]["schema_bytes"] > report["arms"]["cli"]["schema_bytes"]


def test_the_table_lists_every_failure_with_its_reason() -> None:
    text = render([
        outcome("cli", "chain", False, detail="RatedCapacity=None (want '68')", status="finished"),
        outcome("baseline", "chain", True),
    ])
    assert "Failures" in text
    assert "RatedCapacity=None" in text


def test_the_table_is_ascii_only() -> None:
    """It is printed to a Windows console that defaults to cp1252."""
    text = render([outcome("cli"), outcome("baseline")])
    text.encode("ascii")


# ---------------------------------------------------------------------------
# the baseline arm
# ---------------------------------------------------------------------------

def test_the_baseline_has_no_shell(tmp_path: Path) -> None:
    """With a shell it could run `aas` and the two arms would be one."""
    from a4s.environments.local import LocalEnvironment
    from a4s.experiment.baseline import BaselineRegistry
    from a4s.ledger import Ledger

    ledger = Ledger(tmp_path / "l.db", "t", actor="t")
    registry = BaselineRegistry(LocalEnvironment(cwd=str(tmp_path)), ledger, tmp_path, "http://x")
    try:
        assert "shell" not in registry.names()
        assert "finish" in registry.names()
    finally:
        ledger.close()


def test_the_baseline_covers_every_capability_the_cli_has(tmp_path: Path) -> None:
    """Withholding a capability from one arm measures a capability gap and
    calls it a surface effect."""
    from a4s.environments.local import LocalEnvironment
    from a4s.experiment.baseline import BaselineRegistry
    from a4s.ledger import Ledger

    ledger = Ledger(tmp_path / "l.db", "t", actor="t")
    registry = BaselineRegistry(LocalEnvironment(cwd=str(tmp_path)), ledger, tmp_path, "http://x")
    try:
        names = set(registry.names())
    finally:
        ledger.close()

    # One baseline tool per `aas` verb group that does something distinct.
    for capability in (
        "aas_explore", "aas_list_submodels", "aas_list_elements", "aas_read_property",
        "aas_describe_property", "aas_write_property", "aas_create_shell",
        "aas_create_submodel", "aas_create_element", "aas_list_templates",
        "aas_template_outline", "aas_from_smt", "aas_template_diff", "aas_validate",
        "aas_parse", "aas_search_property", "aas_match_inputs", "aas_populate_inputs",
        "input_pdf2entity", "aas_operation_delegation",
    ):
        assert capability in names, f"the baseline arm cannot {capability}"


def test_the_baseline_surface_is_the_larger_one(tmp_path: Path) -> None:
    from a4s.environments.local import LocalEnvironment
    from a4s.experiment.baseline import BaselineRegistry
    from a4s.ledger import Ledger
    from a4s.tools import ToolRegistry

    ledger = Ledger(tmp_path / "l.db", "t", actor="t")
    environment = LocalEnvironment(cwd=str(tmp_path))
    try:
        baseline = BaselineRegistry(environment, ledger, tmp_path, "http://x")
        cli = ToolRegistry(environment, ledger, tmp_path)
        assert len(baseline.names()) > len(cli.names()) * 4
        assert len(json.dumps(baseline.schemas())) > len(json.dumps(cli.schemas())) * 3
    finally:
        ledger.close()


def test_every_baseline_tool_declares_usable_parameters(tmp_path: Path) -> None:
    """A malformed schema would make the baseline fail for a reason that has
    nothing to do with its surface."""
    from a4s.environments.local import LocalEnvironment
    from a4s.experiment.baseline import BaselineRegistry
    from a4s.ledger import Ledger

    ledger = Ledger(tmp_path / "l.db", "t", actor="t")
    registry = BaselineRegistry(LocalEnvironment(cwd=str(tmp_path)), ledger, tmp_path, "http://x")
    try:
        for schema in registry.schemas():
            function = schema["function"]
            parameters = function["parameters"]
            assert function["description"].strip(), f"{function['name']} has no description"
            assert parameters["type"] == "object"
            for name, spec in parameters["properties"].items():
                assert "type" in spec, f"{function['name']}.{name} has no type"
                assert "_required" not in spec, f"{function['name']}.{name} leaked _required"
                assert spec.get("description", "").strip(), f"{function['name']}.{name} undocumented"
            for name in parameters["required"]:
                assert name in parameters["properties"]
    finally:
        ledger.close()
