"""Every metamodel element type, created, read, updated, deleted, validated.

The audit that produced this file found four types with *zero* mentions
anywhere in `tests/` -- `Blob`, `File`, `ReferenceElement`,
`RelationshipElement` -- and a fifth, `Entity`, whose apparent coverage was
`match.Entity`, an unrelated dataclass that happens to share the name. None of
them is exotic: `File` carries the datasheet a passport cites, and
`RelationshipElement` is how a bill of materials is expressed.

Deletion had no coverage at all, for any type. It is the only irreversible
operation the CLI offers.

What the first run established, against BaSyx 2.0.0-SNAPSHOT:

* **Every type creates, reads, updates, deletes and validates cleanly** --
  including the five that had never been exercised. Three update paths were
  predicted broken and are not: `Range`, `ReferenceElement`, `Blob` and `File`
  all update, but only through `--json-value`, because their `$value` is an
  object rather than a scalar. A bare string draws HTTP 500 from BaSyx and the
  hint `cmd_set` prints for that 500 names `--lang`, which is the right advice
  for a MultiLanguageProperty and misleading for these four.
* **A duplicate idShort corrupts the submodel**, silently. Serious enough to
  have been fixed rather than noted: `el add` now reads the siblings before
  writing, and `validate` carries AASd-022 so damage that already exists can
  be found. The repository's own behaviour is still pinned here, going around
  the guard, so a BaSyx version that starts conflicting properly shows up as a
  failure and the guard can be reconsidered.
* **A Blob does not round-trip between the two read endpoints.** `$value`
  decodes the base64 and the element does not.

The three things that can collide are three different problems, and only one
of them was ever dangerous -- see the comment above the duplicate tests.

Ordering inside a case is deliberate: create, read back, update, validate,
delete, confirm absent. Validation runs *before* deletion so that a type whose
creation produces a constraint violation is reported as a create defect rather
than disappearing with the element.
"""

from __future__ import annotations

import base64
from dataclasses import dataclass, field
from typing import Any, Callable, Dict, List, Optional

import pytest

from a4s.aasctl.client import AasHttpError, BasyxClient
from a4s.aasctl.paths import ElementPath

from .conftest import TEST_PREFIX, assert_absent, children_of, element_at, run_aas, value_at

pytestmark = pytest.mark.integration


# A one-pixel PNG. Small enough that the output clamp cannot truncate it, and
# real enough that a repository which validates content against contentType
# has something valid to look at.
PNG_1PX = base64.b64encode(
    bytes.fromhex(
        "89504e470d0a1a0a0000000d4948445200000001000000010806000000"
        "1f15c4890000000a49444154789c6360000002000100ffff03000006000557bfabd4"
        "0000000049454e44ae426082"
    )
).decode("ascii")


@dataclass
class Case:
    """One element type and how to exercise it."""

    model_type: str
    create: List[str]                      # extra flags for `el add`
    update: Optional[List[str]] = None     # extra flags for `aas set`, None = no value
    new_value: Optional[str] = None        # the positional VALUE for `aas set`
    # Checks the element as POSTed comes back the way it went in. Takes the
    # element dict, returns nothing, raises on disagreement.
    check_created: Optional[Callable[[Dict[str, Any]], None]] = None
    note: str = ""
    marks: List[Any] = field(default_factory=list)

    @property
    def id_short(self) -> str:
        return f"Probe{self.model_type}"


def _has(key: str, expected: Any) -> Callable[[Dict[str, Any]], None]:
    def check(element: Dict[str, Any]) -> None:
        assert element.get(key) == expected, (
            f"{key}: sent {expected!r}, read back {element.get(key)!r}"
        )

    return check


def _check_mlp(element: Dict[str, Any]) -> None:
    """A MultiLanguageProperty has two serialisations and both are correct.

    The *element* carries `[{"language": "en", "text": "..."}]`; its `$value`
    under ValueOnly carries `[{"en": "..."}]`. `make_element` produces the
    first and `_lang_strings` the second, which is right -- but it means a
    round-trip has to know which endpoint it read from. Pinning it here is the
    only place that distinction is written down as an executable claim.
    """
    value = element.get("value") or []
    assert value, "MultiLanguageProperty came back with no value"
    assert value[0].get("language") == "en", f"element shape expected, got {value[0]}"
    assert value[0].get("text") == "Hello", f"text not preserved: {value[0]}"


def _check_relationship(element: Dict[str, Any]) -> None:
    for end in ("first", "second"):
        reference = element.get(end)
        assert isinstance(reference, dict), f"{end} is not a Reference: {reference!r}"
        keys = reference.get("keys") or []
        assert keys, f"{end} has no keys"


CASES: List[Case] = [
    Case(
        model_type="Property",
        create=["--value", "SN-4711", "--value-type", "xs:string"],
        update=[], new_value="SN-9999",
        check_created=_has("value", "SN-4711"),
    ),
    Case(
        model_type="MultiLanguageProperty",
        create=["--value", "Hello"],
        update=["--lang"], new_value="Hallo@de|Hello@en",
        check_created=_check_mlp,
        note="create takes the element shape, update the ValueOnly shape",
    ),
    Case(
        model_type="Range",
        create=["--min", "-20", "--max", "85", "--value-type", "xs:double"],
        update=["--json-value"], new_value='{"min": "-30", "max": "90"}',
        check_created=_has("min", "-20"),
        note="predicted failure: no dedicated update path exists for a Range",
    ),
    Case(
        # Measured, not assumed: `$value` for a Blob is
        # `{"contentType": ..., "value": ...}`, so a bare string draws a 500.
        model_type="Blob",
        create=["--content-type", "image/png", "--value", PNG_1PX],
        update=["--json-value"],
        new_value='{"contentType": "image/png", "value": "Ynll"}',
        check_created=_has("contentType", "image/png"),
        note="structured $value; zero prior coverage",
    ),
    Case(
        model_type="File",
        create=["--content-type", "application/pdf", "--value", "/aasx/datasheet.pdf"],
        update=["--json-value"],
        new_value='{"contentType": "application/pdf", "value": "/aasx/revised.pdf"}',
        check_created=_has("value", "/aasx/datasheet.pdf"),
        note="structured $value; the type a passport cites its source with",
    ),
    Case(
        model_type="ReferenceElement",
        create=["--value", "https://example.com/ids/aas/Manufacturer"],
        update=["--json-value"],
        new_value=(
            '{"type": "ExternalReference", "keys": '
            '[{"type": "GlobalReference", "value": "https://example.com/ids/aas/Other"}]}'
        ),
        note="predicted failure: value is a Reference object, not a scalar",
    ),
    Case(
        model_type="RelationshipElement",
        create=[
            "--first", "https://example.com/ids/asset/module",
            "--second", "https://example.com/ids/asset/cell",
        ],
        check_created=_check_relationship,
        note="no update path: first/second are settable only at creation",
    ),
    Case(
        model_type="SubmodelElementCollection",
        create=[],
        check_created=_has("modelType", "SubmodelElementCollection"),
    ),
    Case(
        model_type="SubmodelElementList",
        create=[
            "--type-value-list-element", "Property",
            "--value-type-list-element", "xs:string",
        ],
        check_created=_has("typeValueListElement", "Property"),
        note="children and index behaviour are exercised in test_list_semantics.py",
    ),
    Case(
        model_type="Operation",
        create=["--input", "speed=xs:double", "--output", "accepted=xs:boolean"],
        check_created=lambda e: (
            _assert_operation_variables(e)
        ),
    ),
    Case(
        model_type="Entity",
        create=["--global-asset-id", "https://example.com/ids/asset/cell-001"],
        check_created=_has("entityType", "SelfManagedEntity"),
        note="predicted problem: entityType is hardcoded and `el add` cannot set it",
    ),
]


def _assert_operation_variables(element: Dict[str, Any]) -> None:
    inputs = element.get("inputVariables") or []
    outputs = element.get("outputVariables") or []
    assert len(inputs) == 1, f"expected one input variable, got {inputs}"
    assert len(outputs) == 1, f"expected one output variable, got {outputs}"
    assert inputs[0]["value"]["idShort"] == "speed"
    assert outputs[0]["value"]["valueType"] == "xs:boolean"


def _id(case: Case) -> str:
    return case.model_type


# ---------------------------------------------------------------------------
# the matrix
# ---------------------------------------------------------------------------

@pytest.mark.parametrize("case", CASES, ids=_id)
def test_create_then_read_back(client: BasyxClient, scratch: str, case: Case) -> None:
    """What went in comes back out, with its type-specific attributes intact.

    Read through `get_element`, not `get_value`: the ValueOnly serialisation
    drops `contentType`, `typeValueListElement` and the Operation variable
    declarations, which are exactly the attributes these types are defined by.
    """
    run_aas(
        "el", "add", scratch, "",
        "--type", case.model_type, "--id-short", case.id_short, *case.create,
    )

    element = element_at(client, scratch, case.id_short)
    assert element.get("modelType") == case.model_type
    assert element.get("idShort") == case.id_short
    if case.check_created is not None:
        case.check_created(element)


@pytest.mark.parametrize(
    "case", [c for c in CASES if c.update is not None], ids=_id
)
def test_update_changes_the_stored_value(
    client: BasyxClient, scratch: str, case: Case
) -> None:
    """`aas set` moves the value, and reading it back proves it moved.

    Comparing against the value read *before* the write rather than against a
    literal: several of these types serialise their value differently through
    `$value` than through the element, and asserting on the difference tests
    the write without asserting on a serialisation this test has no business
    pinning.
    """
    run_aas(
        "el", "add", scratch, "",
        "--type", case.model_type, "--id-short", case.id_short, *case.create,
    )
    before = value_at(client, scratch, case.id_short)

    run_aas("set", scratch, case.id_short, case.new_value, *(case.update or []))

    after = value_at(client, scratch, case.id_short)
    assert after != before, (
        f"{case.model_type}: value unchanged after `aas set`. "
        f"before={before!r} after={after!r}"
    )


@pytest.mark.parametrize("case", CASES, ids=_id)
def test_a_freshly_created_element_violates_nothing(
    client: BasyxClient, scratch: str, case: Case
) -> None:
    """The CLI must not be able to author an element the CLI then rejects.

    `aas validate` is what an agent is told to run before considering a
    submodel finished. If `el add` can produce something that fails it, the
    agent is being sent into a loop it cannot exit -- and the constraint that
    catches it (AASd-120 on a list child, AASd-007 on a valueType) is one the
    CLI could have enforced at construction time instead.
    """
    run_aas(
        "el", "add", scratch, "",
        "--type", case.model_type, "--id-short", case.id_short, *case.create,
    )

    result = run_aas("--json", "validate", scratch)
    findings = result.json()["findings"]
    assert findings == [], f"{case.model_type} authored a violation: {findings}"


@pytest.mark.parametrize("case", CASES, ids=_id)
def test_delete_makes_it_absent_not_empty(
    client: BasyxClient, scratch: str, case: Case
) -> None:
    """Deletion had no coverage for any type before this test.

    The distinction it enforces -- 404 rather than 200-with-null -- is the one
    an agent acts on. "Absent" means create it; "present and unset" means write
    to it. An endpoint that conflates them sends the agent down the wrong path
    with no error to read.
    """
    run_aas(
        "el", "add", scratch, "",
        "--type", case.model_type, "--id-short", case.id_short, *case.create,
    )
    element_at(client, scratch, case.id_short)          # present before

    run_aas("el", "rm", scratch, case.id_short)

    assert_absent(client, scratch, case.id_short)


# ---------------------------------------------------------------------------
# error contracts
# ---------------------------------------------------------------------------

# -- duplicate identifiers ---------------------------------------------------
#
# Three different things can collide and the repository treats them as three
# different problems, which is why they need three sets of tests:
#
#   shell `id`      global Identifier    BaSyx: 409, original intact  -- safe
#   submodel `id`   global Identifier    BaSyx: 409, original intact  -- safe
#   element idShort unique among siblings (AASd-022)
#                     at a submodel root   BaSyx: appends, answers 404
#                     inside a collection  BaSyx: appends, answers 200
#
# Only the third is dangerous, and its nested form is the worse of the two:
# the write reports success. `cmd_el_add` now refuses both before writing.

def test_a_duplicate_shell_id_is_refused_by_the_repository(
    client: BasyxClient, scratch_shell: str
) -> None:
    """A global Identifier collision is BaSyx's own job and it does it.

    Worth pinning rather than assuming: the guard added to `el add` exists
    because element idShorts are *not* protected this way, and the difference
    between the two is the whole reason the guard is needed at one level and
    not the other.
    """
    result = run_aas("shell", "create", scratch_shell, "--id-short", "Second",
                     expect_ok=False)

    assert not result.ok, "a colliding shell id was accepted"
    assert client.get_shell(scratch_shell)["idShort"] == "ScratchShell", (
        "the original shell was overwritten"
    )


def test_a_duplicate_submodel_id_is_refused_by_the_repository(
    client: BasyxClient, scratch: str
) -> None:
    result = run_aas("sm", "create", scratch, "--id-short", "SecondSubmodel",
                     expect_ok=False)

    assert not result.ok, "a colliding submodel id was accepted"
    assert client.get_submodel(scratch)["idShort"] == "ScratchSubmodel", (
        "the original submodel was overwritten"
    )


def test_the_repository_corrupts_a_submodel_given_a_duplicate_at_the_root(
    client: BasyxClient, scratch: str
) -> None:
    """The defect the guard defends against, pinned by going around the guard.

    Written through the client rather than the CLI on purpose: this asserts
    what BaSyx does, so that a version which starts conflicting properly shows
    up as a failure here and the guard can then be reconsidered.

    What it does today: appends a second child with the same idShort, answers
    404, and leaves *both* unaddressable -- every later read of that path
    answers 404 because the lookup is ambiguous. `el rm` cannot remove either;
    only a whole-submodel PUT can.
    """
    from a4s.aasctl.model import make_element

    run_aas("el", "add", scratch, "", "--type", "Property", "--id-short", "Twin",
            "--value", "first", "--value-type", "xs:string")

    with pytest.raises(AasHttpError):
        client.add_element(
            scratch, ElementPath.parse(""),
            make_element("Property", "Twin", value="second", value_type="xs:string"),
        )

    id_shorts = [child.get("idShort") for child in children_of(client, scratch)]
    assert id_shorts.count("Twin") == 2, (
        f"expected the duplicate to have been appended anyway, got {id_shorts!r}"
    )
    with pytest.raises(AasHttpError) as excinfo:
        value_at(client, scratch, "Twin")
    assert excinfo.value.status == 404, "the surviving element stayed readable"


def test_the_repository_accepts_a_duplicate_inside_a_collection_without_an_error(
    client: BasyxClient, scratch: str
) -> None:
    """The worse of the two: nested, the duplicate write reports success.

    At a submodel root the caller at least gets a 404 to react to. One level
    down there is no error at all -- HTTP 200, and the collection now holds two
    children with the same idShort, both unreadable.
    """
    from a4s.aasctl.model import make_element

    run_aas("el", "add", scratch, "", "--type", "SubmodelElementCollection",
            "--id-short", "Box")
    run_aas("el", "add", scratch, "Box", "--type", "Property", "--id-short", "Inner",
            "--value", "first", "--value-type", "xs:string")

    client.add_element(                                   # no exception raised
        scratch, ElementPath.parse("Box"),
        make_element("Property", "Inner", value="second", value_type="xs:string"),
    )

    inner = [c.get("idShort") for c in children_of(client, scratch, "Box")]
    assert inner.count("Inner") == 2, f"expected two children, got {inner!r}"


def test_el_add_refuses_a_duplicate_at_the_root(
    client: BasyxClient, scratch: str
) -> None:
    """The guard. One GET before the write, and the damage never happens."""
    add = ["el", "add", scratch, "", "--type", "Property", "--id-short", "Twin",
           "--value-type", "xs:string"]
    run_aas(*add, "--value", "first")

    again = run_aas(*add, "--value", "second", expect_ok=False)

    assert not again.ok, "the duplicate was written"
    assert "already exists" in again.stderr.lower(), (
        f"the refusal did not name the cause: {again.stderr!r}"
    )
    assert "aas set" in again.stderr, "the refusal did not name the recovery"
    assert value_at(client, scratch, "Twin") == "first", "the original was disturbed"
    assert [c.get("idShort") for c in children_of(client, scratch)] == ["Twin"]


def test_el_add_refuses_a_duplicate_inside_a_collection(
    client: BasyxClient, scratch: str
) -> None:
    """The nested case has to be guarded too, and it reads a different shape.

    A submodel root answers with AAS Part 2's paged envelope; a collection
    answers with the element and its children under `value`. A guard that
    handled only the first would pass everything nested straight through --
    which is the case with no error to catch it.
    """
    run_aas("el", "add", scratch, "", "--type", "SubmodelElementCollection",
            "--id-short", "Box")
    add = ["el", "add", scratch, "Box", "--type", "Property", "--id-short", "Inner",
           "--value-type", "xs:string"]
    run_aas(*add, "--value", "first")

    again = run_aas(*add, "--value", "second", expect_ok=False)

    assert not again.ok, "the nested duplicate was written"
    assert value_at(client, scratch, "Box.Inner") == "first"
    assert [c.get("idShort") for c in children_of(client, scratch, "Box")] == ["Inner"]


def test_el_add_still_appends_to_a_list(client: BasyxClient, scratch: str) -> None:
    """The guard must not block the one case where repetition is correct.

    An SML child carries no idShort, so there is nothing to collide and every
    append is legitimate. A uniqueness check that did not exempt lists would
    make `SubmodelElementList` -- the type this CLI exists to reach -- writable
    exactly once.
    """
    run_aas("el", "add", scratch, "", "--type", "SubmodelElementList",
            "--id-short", "Items", "--type-value-list-element", "Property",
            "--value-type-list-element", "xs:string")
    for value in ("a", "b", "c"):
        run_aas("el", "add", scratch, "Items", "--type", "Property",
                "--value", value, "--value-type", "xs:string")

    assert [c.get("value") for c in children_of(client, scratch, "Items")] == ["a", "b", "c"]


def test_validate_reports_a_duplicate_id_short(client: BasyxClient, scratch: str) -> None:
    """AASd-022, so damage that already exists can be found.

    The guard stops new corruption; it does nothing about a repository that is
    already holding some -- from an earlier version of this CLI, from another
    client, or from an AASX file loaded at boot. Without this rule `aas
    validate` called such a submodel clean, which is how the damage stayed
    invisible.
    """
    from a4s.aasctl.model import make_element

    run_aas("el", "add", scratch, "", "--type", "Property", "--id-short", "Twin",
            "--value", "first", "--value-type", "xs:string")
    with pytest.raises(AasHttpError):
        client.add_element(
            scratch, ElementPath.parse(""),
            make_element("Property", "Twin", value="second", value_type="xs:string"),
        )

    findings = run_aas("--json", "validate", scratch).json()["findings"]

    duplicates = [f for f in findings if f["rule"] == "AASd-022"]
    assert duplicates, f"the duplicate went unreported. findings={findings}"
    assert duplicates[0]["path"] == "Twin"
    assert "2 siblings" in duplicates[0]["detail"]


def test_a_blob_reads_back_differently_through_value_than_through_the_element(
    client: BasyxClient, scratch: str
) -> None:
    """`$value` decodes the base64; the element does not.

    Writing `aGk=` and reading the element gives `aGk=` back, but reading
    `$value` gives `hi`. Neither is wrong -- ValueOnly serialises the decoded
    content -- but an agent that writes through one endpoint and verifies
    through the other concludes its write was mangled and retries.

    Pinned here so `aas get` on a Blob is never assumed to round-trip.
    """
    run_aas("el", "add", scratch, "", "--type", "Blob", "--id-short", "Payload",
            "--content-type", "text/plain", "--value", "aGk=")

    assert element_at(client, scratch, "Payload")["value"] == "aGk="
    assert value_at(client, scratch, "Payload")["value"] == "hi"


def test_writing_a_bare_string_to_a_multilanguage_property_is_explained(
    client: BasyxClient, scratch: str
) -> None:
    """The 500 that says nothing has to be turned into a hint that says what.

    BaSyx answers 500, not 400, when a scalar is written to an element whose
    `$value` is structured, and the status carries no explanation. `cmd_set`
    catches that specific combination and names `--lang`. A live run lost four
    steps to this before the hint existed, so it is worth a test.
    """
    run_aas("el", "add", scratch, "", "--type", "MultiLanguageProperty",
            "--id-short", "Designation", "--value", "Hello")

    result = run_aas("set", scratch, "Designation", "Bonjour", expect_ok=False)

    assert not result.ok, "a bare string was accepted for a MultiLanguageProperty"
    assert "--lang" in result.stderr, (
        f"the hint did not name the fix: {result.stderr!r}"
    )


def test_a_missing_parent_names_the_command_that_lists_paths(scratch: str) -> None:
    """A 404 on `el add` must not read like a 404 on the submodel itself."""
    result = run_aas(
        "el", "add", scratch, "NoSuchCollection",
        "--type", "Property", "--id-short", "Orphan", "--value", "x",
        expect_ok=False,
    )

    assert not result.ok
    assert "sm tree" in result.stderr, (
        f"the hint did not name a recovery command: {result.stderr!r}"
    )


def test_an_unknown_value_type_is_refused_before_the_request(scratch: str) -> None:
    """The xsd enumeration is closed, and argparse rejects outside it.

    Worth an integration test rather than only a unit test because the
    enumeration is the natural target for constrained decoding: what matters
    is that an invalid `--value-type` never reaches the repository, not merely
    that `make_element` would have raised.
    """
    result = run_aas(
        "el", "add", scratch, "", "--type", "Property", "--id-short", "Bad",
        "--value", "1", "--value-type", "xs:notAType",
        expect_ok=False,
    )
    assert not result.ok


# ---------------------------------------------------------------------------
# depth
# ---------------------------------------------------------------------------

def test_a_deeply_nested_path_round_trips(client: BasyxClient, scratch: str) -> None:
    """Five levels, mixing collections and a list index.

    The unit tests parse paths this deep; nothing had ever sent one to a
    repository. Both halves can fail independently -- the path renderer, and
    BaSyx's own parsing of a long percent-encoded segment.
    """
    run_aas("el", "add", scratch, "", "--type", "SubmodelElementCollection",
            "--id-short", "A")
    run_aas("el", "add", scratch, "A", "--type", "SubmodelElementList",
            "--id-short", "B", "--type-value-list-element", "SubmodelElementCollection")
    run_aas("el", "add", scratch, "A.B", "--type", "SubmodelElementCollection")
    run_aas("el", "add", scratch, "A.B[0]", "--type", "SubmodelElementCollection",
            "--id-short", "C")
    run_aas("el", "add", scratch, "A.B[0].C", "--type", "Property",
            "--id-short", "D", "--value", "deep", "--value-type", "xs:string")

    assert value_at(client, scratch, "A.B[0].C.D") == "deep"

    run_aas("set", scratch, "A.B[0].C.D", "deeper")
    assert value_at(client, scratch, "A.B[0].C.D") == "deeper"

    assert run_aas("--json", "validate", scratch).json()["findings"] == []


def test_a_shell_can_be_named_by_its_id_short(client: BasyxClient, scratch_shell: str) -> None:
    """The gap that stopped the wizard.

    Every shell verb took an id and the callers mostly have a label: the wizard
    passes `targetAAS`, which is an idShort, and an agent told "the AAS
    InjectionMoldingMachine_653" cannot learn the id from that sentence.
    Encoding the label as an id gave `404 Element with Id ... does not exist` --
    correct, unhelpful, and a dead end, because nothing in it says the label is
    not an id.
    """
    label = client.get_shell(scratch_shell)["idShort"]

    by_label = run_aas("--json", "shell", "show", label).json()

    assert by_label["id"] == scratch_shell
    assert by_label["idShort"] == label


def test_an_id_is_still_taken_at_face_value(scratch_shell: str) -> None:
    """The scan happens only on a 404, so the common case costs no extra call."""
    assert run_aas("--json", "shell", "show", scratch_shell).json()["id"] == scratch_shell


def test_a_name_that_is_neither_says_so_and_names_the_way_out(scratch_shell: str) -> None:
    result = run_aas("shell", "show", "NoSuchLabelOrId", expect_ok=False)

    assert not result.ok
    assert "NoSuchLabelOrId" in result.stderr
    assert "aas shell list" in result.stderr


def test_an_ambiguous_id_short_is_refused_rather_than_guessed(
    client: BasyxClient
) -> None:
    """AASd-022 forbids duplicate idShorts among *siblings*, not across a
    repository, so two machines may share a label.

    Picking one of them is worse than stopping: the caller asked for a machine
    and would be told it got the one it meant.
    """
    import contextlib

    from a4s.aasctl.client import AasHttpError
    from a4s.aasctl.model import make_shell

    twins = [f"{TEST_PREFIX}aas:twin-{n}" for n in (1, 2)]
    for identifier in twins:
        client.create_shell(make_shell(identifier, "AmbiguousTwin",
                                       global_asset_id=f"{identifier}/asset"))
    try:
        result = run_aas("shell", "show", "AmbiguousTwin", expect_ok=False)

        assert not result.ok
        assert "2 shells" in result.stderr
        assert all(t in result.stderr for t in twins)
    finally:
        for identifier in twins:
            with contextlib.suppress(AasHttpError):
                client.delete_shell(identifier)
