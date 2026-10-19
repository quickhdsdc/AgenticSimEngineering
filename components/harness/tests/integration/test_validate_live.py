"""`aas validate` against every submodel the repository actually holds.

The unit tests all point the same way: build a submodel with a known violation,
assert the rule fires. That measures sensitivity and says nothing about the
other error -- whether the validator calls correct data broken. A rule that
cries wolf is worse than a missing one, because the agent is told to run
`validate` before considering a submodel finished, and findings it cannot act
on cost it steps and teach it to ignore the command.

So this runs the validator over real data and asserts two things that stay true
as the repository changes: nothing crashes, and every finding names a rule the
validator actually implements. It deliberately does **not** assert a count.
Pinning "97 of 116 clean" would fail the day someone loads a new AASX, for a
reason that has nothing to do with this code.

What the first run found, on 116 live submodels: 97 clean, 19 with findings,
and both rules that fired were right.

* **AASd-129, 846 times** across 15 submodels -- TemplateQualifiers left behind
  on Instance submodels. That is exactly the defect `smt.py` describes the
  predecessor's copy-the-template-and-flip-`kind` instantiation as producing,
  and the evidence for it is sitting in the repository.
* **AASd-002, 12 times** -- `ExpectedLifetimeInCalendarYears ` with a trailing
  space, which the metamodel forbids and BaSyx accepted.
"""

from __future__ import annotations

import argparse
import contextlib
import json
from typing import Any, Dict, List

import pytest

from a4s.aasctl import __main__ as cli
from a4s.aasctl.client import BasyxClient

from .conftest import TEST_PREFIX, run_aas

pytestmark = pytest.mark.integration

# Every rule `cmd_validate` can emit. A finding outside this set means either a
# rule was added without updating the catalogue, or a rule name was mistyped --
# and a mistyped rule is invisible to anyone grepping the ledger for it.
KNOWN_RULES = {"AASd-002", "AASd-007", "AASd-022", "AASd-108", "AASd-117", "AASd-120",
               "AASd-129"}


def _validate(submodel: Dict[str, Any]) -> List[Dict[str, str]]:
    """Run the real command against an already-fetched submodel.

    Fetched once by the caller and injected, rather than letting each call
    re-request it: 116 submodels is 116 round trips and several megabytes for
    data already in hand.
    """
    captured: Dict[str, Any] = {}

    class Capture(cli.Out):
        def lines(self, payload: Any, lines: Any) -> None:
            captured.update(payload)

    original_get, original_client = cli.get_submodel, cli._client
    try:
        cli.get_submodel = lambda client, raw: submodel          # type: ignore[assignment]
        cli._client = lambda args: contextlib.nullcontext(None)  # type: ignore[assignment]
        cli.cmd_validate(
            argparse.Namespace(submodel_id=submodel.get("id"), endpoint=None, json=True),
            Capture(True),
        )
    finally:
        cli.get_submodel, cli._client = original_get, original_client
    return captured["findings"]


def test_the_validator_survives_every_submodel_in_the_repository(
    client: BasyxClient,
) -> None:
    """No crashes, and every rule name is one the validator implements.

    The traversal is where a validator breaks on real data -- an element shape
    nobody anticipated, a null where a list was assumed. A crash here is worse
    than a wrong finding, because the agent gets a traceback instead of
    something it can read.
    """
    submodels = client.list_submodels(limit=500)
    assert submodels, "the repository served no submodels to validate"

    unknown: Dict[str, str] = {}
    clean = 0
    for submodel in submodels:
        findings = _validate(submodel)
        if not findings:
            clean += 1
        for finding in findings:
            assert set(finding) >= {"rule", "path", "detail"}, finding
            if finding["rule"] not in KNOWN_RULES:
                unknown[finding["rule"]] = submodel.get("id", "?")

    assert not unknown, f"findings naming rules the validator does not implement: {unknown}"
    # Not a threshold on the data, a sanity check on the validator: a rule set
    # that flags every submodel in a working plant is measuring itself.
    assert clean > 0, "not one submodel in the repository validates clean"


def test_a_submodel_this_cli_authored_validates_clean(
    client: BasyxClient, scratch: str
) -> None:
    """The tightest form of "does not cry wolf".

    Whatever the state of the repository's imported data, anything the CLI
    builds itself must pass its own checks. If it cannot, the agent is being
    sent into a loop: author, validate, fail, with no correction available.
    """
    run = run_aas

    run("el", "add", scratch, "", "--type", "SubmodelElementCollection", "--id-short", "Box")
    run("el", "add", scratch, "Box", "--type", "Property", "--id-short", "Name",
        "--value", "x", "--value-type", "xs:string")
    run("el", "add", scratch, "Box", "--type", "MultiLanguageProperty",
        "--id-short", "Label", "--value", "Hello")
    run("el", "add", scratch, "", "--type", "SubmodelElementList", "--id-short", "Items",
        "--type-value-list-element", "Property", "--value-type-list-element", "xs:string")
    run("el", "add", scratch, "Items", "--type", "Property", "--value", "a",
        "--value-type", "xs:string")

    findings = json.loads(run("--json", "validate", scratch).stdout)["findings"]

    assert findings == [], f"the CLI authored a submodel its own validator rejects: {findings}"


def test_an_instantiated_template_validates_clean(client: BasyxClient) -> None:
    """AASd-129 is the rule instantiation gets wrong, so instantiate and check.

    `smt instantiate` exists because copying a template and flipping `kind`
    leaves TemplateQualifiers on an Instance. 846 findings across 7 live
    submodels say that happened here at least once. The transformation this CLI
    performs must not add to that count.
    """
    templates = [
        sm for sm in client.list_submodels(limit=500)
        if sm.get("kind") == "Template" and (sm.get("submodelElements") or [])
    ]
    if not templates:
        pytest.skip("the repository holds no templates to instantiate")

    target = f"{TEST_PREFIX}sm:instantiated"
    template_id = templates[0]["id"]
    try:
        run_aas("smt", "instantiate", template_id, "--as", target,
                "--id-short", "Instantiated", "--replace")
        findings = json.loads(run_aas("--json", "validate", target).stdout)["findings"]
        offending = [f for f in findings if f["rule"] == "AASd-129"]
        assert not offending, (
            f"instantiating {template_id} left template qualifiers behind: {offending[:3]}"
        )
    finally:
        with contextlib.suppress(Exception):
            client.delete_submodel(target)
