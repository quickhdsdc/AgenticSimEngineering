"""The two verbs that close asymmetries in the command surface.

`shell attach` existed without an inverse, and a submodel holding a duplicate
idShort could be *detected* by `aas validate` and repaired by nothing. Both
gaps were found by surveying the published API against the verbs
(PROGRESS.md 3.4), which is a different way of finding defects from running
the code and finds different ones.

The repair path matters more than it looks. BaSyx does not enforce AASd-022:
the duplicate is appended, answered with 404, and from then on every
path-addressed read *and delete* of that path answers 404 because the lookup
is ambiguous. `el rm` cannot reach either copy. Only a whole-submodel PUT can,
and until now nothing exposed one.
"""

from __future__ import annotations

import pytest

from a4s.aasctl.client import AasHttpError, BasyxClient
from a4s.aasctl.model import make_element
from a4s.aasctl.paths import ElementPath

from .conftest import children_of, run_aas, value_at

pytestmark = pytest.mark.integration


def _corrupt(client: BasyxClient, submodel: str, id_short: str = "Twin") -> None:
    """Create the damage the CLI now refuses to create, by going around it."""
    run_aas("el", "add", submodel, "", "--type", "Property", "--id-short", id_short,
            "--value", "first", "--value-type", "xs:string")
    with pytest.raises(AasHttpError):
        client.add_element(
            submodel, ElementPath.parse(""),
            make_element("Property", id_short, value="second", value_type="xs:string"),
        )


# ---------------------------------------------------------------------------
# detach
# ---------------------------------------------------------------------------

def test_detach_removes_the_reference_and_keeps_the_submodel(
    client: BasyxClient, scratch: str, scratch_shell: str
) -> None:
    """The distinction the verb exists for.

    Without it the only way to undo an attach was `sm delete`, which destroys
    the submodel rather than the binding -- a far larger action than the one
    being reversed, and irreversible.
    """
    run_aas("shell", "attach", scratch_shell, scratch)
    assert client.list_submodel_refs(scratch_shell), "attach did not bind"

    run_aas("shell", "detach", scratch_shell, scratch)

    assert client.list_submodel_refs(scratch_shell) == []
    assert client.get_submodel(scratch)["id"] == scratch, "the submodel was deleted"


def test_detaching_something_never_attached_names_the_command_that_lists(
    scratch: str, scratch_shell: str
) -> None:
    result = run_aas("shell", "detach", scratch_shell, scratch, expect_ok=False)

    assert not result.ok
    assert "shell show" in result.stderr, (
        f"the hint did not name a recovery command: {result.stderr!r}"
    )


def test_detach_dry_run_changes_nothing(
    client: BasyxClient, scratch: str, scratch_shell: str
) -> None:
    run_aas("shell", "attach", scratch_shell, scratch)
    run_aas("shell", "detach", scratch_shell, scratch, "--dry-run")
    assert client.list_submodel_refs(scratch_shell), "--dry-run detached anyway"


# ---------------------------------------------------------------------------
# repair
# ---------------------------------------------------------------------------

def test_repair_reports_a_clean_submodel_as_clean(scratch: str) -> None:
    run_aas("el", "add", scratch, "", "--type", "Property", "--id-short", "Fine",
            "--value", "x", "--value-type", "xs:string")

    payload = run_aas("--json", "sm", "repair", scratch).json()

    assert payload["duplicates"] == []
    assert payload["repaired"] is False


def test_repair_reports_without_changing_anything_by_default(
    client: BasyxClient, scratch: str
) -> None:
    """Overwriting a submodel is not a small action, so it is opt-in.

    The agent asking about damage usually wants to know what it would lose
    before it loses it, and a repair that ran on sight would be a destructive
    default reached by a read-shaped command name.
    """
    _corrupt(client, scratch)

    payload = run_aas("--json", "sm", "repair", scratch).json()

    assert payload["repaired"] is False
    assert [d["path"] for d in payload["duplicates"]] == ["Twin"]
    assert payload["duplicates"][0]["count"] == 2
    assert [c.get("idShort") for c in children_of(client, scratch)].count("Twin") == 2


def test_repair_apply_makes_the_element_readable_again(
    client: BasyxClient, scratch: str
) -> None:
    """The end-to-end claim: damage the path-addressed API cannot undo, undone.

    The surviving copy must be the *first* one. It holds the value that was
    readable before the duplicate arrived, so keeping it restores the state the
    repository was in; keeping the later one would silently substitute a value
    nobody asked for.
    """
    _corrupt(client, scratch)
    with pytest.raises(AasHttpError):
        value_at(client, scratch, "Twin")

    payload = run_aas("--json", "sm", "repair", scratch, "--apply").json()

    assert payload["repaired"] is True
    assert payload["removed"] == 1
    assert [c.get("idShort") for c in children_of(client, scratch)] == ["Twin"]
    assert value_at(client, scratch, "Twin") == "first"


def test_repair_clears_the_validation_finding_it_was_written_for(
    client: BasyxClient, scratch: str
) -> None:
    """`validate` detects AASd-022 and `repair` is the only thing that fixes it.

    Pinning the pair together: a detector with no remedy leaves an agent able
    to describe a problem it cannot act on, which costs it a step and the
    operator a phone call.
    """
    _corrupt(client, scratch)
    before = run_aas("--json", "validate", scratch).json()["findings"]
    assert any(f["rule"] == "AASd-022" for f in before)

    run_aas("sm", "repair", scratch, "--apply")

    after = run_aas("--json", "validate", scratch).json()["findings"]
    assert not any(f["rule"] == "AASd-022" for f in after), after


def test_repair_reaches_a_duplicate_nested_in_a_collection(
    client: BasyxClient, scratch: str
) -> None:
    """The nested case is the one BaSyx accepts with HTTP 200 and no error.

    A repair that only rewrote the root would leave exactly the damage that
    arrives silently.
    """
    run_aas("el", "add", scratch, "", "--type", "SubmodelElementCollection",
            "--id-short", "Box")
    run_aas("el", "add", scratch, "Box", "--type", "Property", "--id-short", "Inner",
            "--value", "first", "--value-type", "xs:string")
    client.add_element(                                    # accepted, no error
        scratch, ElementPath.parse("Box"),
        make_element("Property", "Inner", value="second", value_type="xs:string"),
    )

    run_aas("sm", "repair", scratch, "--apply")

    assert [c.get("idShort") for c in children_of(client, scratch, "Box")] == ["Inner"]
    assert value_at(client, scratch, "Box.Inner") == "first"


def test_repair_leaves_list_children_alone(client: BasyxClient, scratch: str) -> None:
    """An SML child has no idShort, so it cannot collide and must not be pruned.

    A repair that treated a list as a collection would silently delete every
    item after the first, turning a fix for one defect into a much worse one.
    """
    run_aas("el", "add", scratch, "", "--type", "SubmodelElementList",
            "--id-short", "Items", "--type-value-list-element", "Property",
            "--value-type-list-element", "xs:string")
    for value in ("a", "b", "c"):
        run_aas("el", "add", scratch, "Items", "--type", "Property",
                "--value", value, "--value-type", "xs:string")

    payload = run_aas("--json", "sm", "repair", scratch, "--apply").json()

    assert payload["duplicates"] == []
    assert [c.get("value") for c in children_of(client, scratch, "Items")] == ["a", "b", "c"]
