"""`aas smt author`: a structure file becomes a real, instantiable template.

The gap this closes: `smt instantiate` requires a `kind: Template` submodel to
already exist, so a product class nobody has written a template for was
unreachable. Every other verb could reach anything in the metamodel; this one
could only reach what someone had already modelled.

The claim under test is not "it writes JSON to BaSyx". It is that the template
it writes is **usable** -- that `smt show` reads the cardinalities back, that
`smt instantiate` produces a clean instance from it, and that the instance does
not inherit the AASd-129 violation that 846 findings in this repository say is
the mistake people actually make.
"""

from __future__ import annotations

import contextlib
import json
from pathlib import Path
from typing import Any, Dict

import pytest

from a4s.aasctl.client import AasHttpError, BasyxClient

from .conftest import TEST_PREFIX, run_aas

pytestmark = pytest.mark.integration


STRUCTURE: Dict[str, Any] = {
    "idShort": "WeldingRobotSpec",
    "description": "Technical specification of a welding robot",
    "source": "fixture",
    "slots": [
        {"idShort": "ManufacturerName", "modelType": "Property",
         "valueType": "xs:string", "cardinality": "One", "source": "p.1"},
        {"idShort": "Reach", "modelType": "Property", "valueType": "xs:double",
         "unit": "mm", "cardinality": "One", "source": "p.1"},
        {"idShort": "AxisData", "modelType": "SubmodelElementList",
         "typeValueListElement": "SubmodelElementCollection",
         "cardinality": "OneToMany", "source": "p.2",
         "children": [
             {"idShort": "Axis", "modelType": "SubmodelElementCollection",
              "cardinality": "One", "source": "p.2",
              "children": [
                  {"idShort": "MaxTorque", "modelType": "Property",
                   "valueType": "xs:double", "unit": "Nm", "cardinality": "One",
                   "source": "p.2"},
              ]},
         ]},
        {"idShort": "PowerSupply", "modelType": "SubmodelElementCollection",
         "cardinality": "ZeroToOne", "source": "p.3",
         "children": [
             {"idShort": "Voltage", "modelType": "Property", "valueType": "xs:double",
              "unit": "V", "cardinality": "One", "source": "p.3"},
             {"idShort": "Phases", "modelType": "Property", "valueType": "xs:integer",
              "cardinality": "ZeroToOne", "source": "p.3"},
         ]},
    ],
}


@pytest.fixture
def structure_file(tmp_path: Path) -> Path:
    path = tmp_path / "structure.json"
    path.write_text(json.dumps(STRUCTURE), encoding="utf-8")
    return path


@pytest.fixture
def template_id(client: BasyxClient):
    identifier = f"{TEST_PREFIX}smt:robot"
    yield identifier
    with contextlib.suppress(AasHttpError):
        client.delete_submodel(identifier)


def test_the_authored_template_is_a_template_and_validates_clean(
    client: BasyxClient, structure_file: Path, template_id: str
) -> None:
    """`kind` and the constraint check, which are the two things that make it
    usable rather than merely present."""
    run_aas("smt", "author", template_id, "--structure", str(structure_file), "--replace")

    submodel = client.get_submodel(template_id)
    assert submodel["kind"] == "Template"
    assert submodel["idShort"] == "WeldingRobotSpec"

    findings = run_aas("--json", "validate", template_id).json()["findings"]
    assert findings == [], findings


def test_a_list_child_loses_its_name_on_the_way_in(
    client: BasyxClient, structure_file: Path, template_id: str
) -> None:
    """AASd-120, and the asymmetry it forces.

    The structure file names the list child "Axis" because that is how a person
    reads what an entry is. The metamodel forbids the name on the wire. So the
    document keeps it and `smt author` drops it -- verified from both ends,
    since dropping it in the wrong place would either break the review or break
    the template.
    """
    run_aas("smt", "author", template_id, "--structure", str(structure_file), "--replace")

    axes = next(e for e in client.get_submodel(template_id)["submodelElements"]
                if e["idShort"] == "AxisData")
    entry = (axes.get("value") or [])[0]
    assert "idShort" not in entry, f"the list child kept its name: {entry}"
    assert entry["modelType"] == "SubmodelElementCollection"
    # The name survives where it is useful.
    assert json.loads(structure_file.read_text(encoding="utf-8"))[
        "slots"][2]["children"][0]["idShort"] == "Axis"


def test_the_cardinalities_read_back_through_smt_show(
    structure_file: Path, template_id: str
) -> None:
    """The round trip that makes it a template rather than a shape.

    `smt show` reads cardinality through `smt.cardinality`, which understands
    four dialects. Writing one it cannot read would produce a template whose
    slots all look mandatory -- the failure being silent is what makes it worth
    a test rather than an inspection.
    """
    run_aas("smt", "author", template_id, "--structure", str(structure_file), "--replace")

    outline = run_aas("smt", "show", template_id).stdout
    assert "1..1   ManufacturerName" in outline
    assert "1..*   AxisData" in outline
    assert "0..1   PowerSupply" in outline
    assert "0..1     PowerSupply.Phases" in outline


def test_an_instance_of_it_carries_no_template_qualifiers(
    client: BasyxClient, structure_file: Path, template_id: str
) -> None:
    """The end of the chain, and the mistake this repository is full of.

    846 AASd-129 findings across 15 live submodels come from instantiations
    that copied a template and flipped `kind`. A template authored here has to
    be one that `smt instantiate` can strip cleanly, or it adds to that count
    every time it is used.
    """
    instance = f"{TEST_PREFIX}sm:robot-instance"
    run_aas("smt", "author", template_id, "--structure", str(structure_file), "--replace")
    try:
        run_aas("smt", "instantiate", template_id, "--as", instance,
                "--id-short", "RobotSpec", "--replace")

        submodel = client.get_submodel(instance)
        assert submodel["kind"] == "Instance"

        findings = run_aas("--json", "validate", instance).json()["findings"]
        assert not [f for f in findings if f["rule"] == "AASd-129"], findings

        # Mandatory kept, optional dropped: the cardinality was not merely
        # stored, it was acted on.
        names = {e["idShort"] for e in submodel["submodelElements"]}
        assert {"ManufacturerName", "Reach", "AxisData"} <= names
        assert "PowerSupply" not in names, "an optional slot was instantiated"
    finally:
        with contextlib.suppress(AasHttpError):
            client.delete_submodel(instance)


def test_a_structure_with_known_problems_is_refused_by_default(
    client: BasyxClient, tmp_path: Path, template_id: str
) -> None:
    """Refusing is the default because a wrong template is worse than a wrong
    instance: every instance built from it inherits the mistake."""
    broken = dict(STRUCTURE)
    broken["slots"] = [
        {"idShort": "Twin", "modelType": "Property", "valueType": "xs:string"},
        {"idShort": "Twin", "modelType": "Property", "valueType": "xs:string"},
    ]
    path = tmp_path / "broken.json"
    path.write_text(json.dumps(broken), encoding="utf-8")

    result = run_aas("--json", "smt", "author", template_id, "--structure", str(path))

    payload = result.json()
    assert payload["created"] is False
    assert any(p["rule"] == "AASd-022" for p in payload["problems"])
    with pytest.raises(AasHttpError):
        client.get_submodel(template_id)


def test_force_builds_it_anyway_and_validate_agrees_it_is_broken(
    client: BasyxClient, tmp_path: Path, template_id: str
) -> None:
    """The escape hatch has to actually produce the damage it warns about.

    If `--force` quietly repaired the structure the warning would be a lie, and
    an operator who used it would not learn what the proposal was wrong about.
    """
    broken = dict(STRUCTURE)
    broken["slots"] = [
        {"idShort": "Twin", "modelType": "Property", "valueType": "xs:string"},
        {"idShort": "Twin", "modelType": "Property", "valueType": "xs:string"},
    ]
    path = tmp_path / "broken.json"
    path.write_text(json.dumps(broken), encoding="utf-8")

    run_aas("smt", "author", template_id, "--structure", str(path), "--force", "--replace")

    findings = run_aas("--json", "validate", template_id).json()["findings"]
    assert any(f["rule"] == "AASd-022" for f in findings), findings


def test_a_missing_structure_file_names_the_command_that_makes_one(
    template_id: str, tmp_path: Path
) -> None:
    result = run_aas("smt", "author", template_id,
                     "--structure", str(tmp_path / "absent.json"), expect_ok=False)

    assert not result.ok
    assert "--as-template" in result.stderr


def test_dry_run_prints_the_submodel_and_writes_nothing(
    client: BasyxClient, structure_file: Path, template_id: str
) -> None:
    result = run_aas("--json", "smt", "author", template_id,
                     "--structure", str(structure_file), "--dry-run")

    payload = result.json()
    assert payload["kind"] == "Template"
    assert payload["submodelElements"]
    with pytest.raises(AasHttpError):
        client.get_submodel(template_id)


def test_attach_to_binds_it_to_a_shell(
    client: BasyxClient, structure_file: Path, template_id: str, scratch_shell: str
) -> None:
    run_aas("smt", "author", template_id, "--structure", str(structure_file),
            "--attach-to", scratch_shell, "--replace")

    refs = client.list_submodel_refs(scratch_shell)
    assert any(
        key.get("value") == template_id
        for ref in refs for key in (ref.get("keys") or [])
    ), refs
