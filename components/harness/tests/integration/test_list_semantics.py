"""What a SubmodelElementList actually does, as opposed to what we assume.

`SubmodelElementList` is the type the CLI exists to reach. The tool set it
replaces addressed elements by idShort only, and an SML child has no idShort
-- it is identified by position -- so the whole type was unreachable. Positional
addressing is the concrete thing `paths.py` bought.

Which makes it the type whose real behaviour is least written down. Three
questions had no answer anywhere in the codebase before this file:

**Does deleting `Markings[1]` renumber `Markings[2]` to `Markings[1]`?** An
agent deleting several items in a loop gets a different, wrong result
depending on the answer, and nothing tells it which. If indices shift, the
only correct loop runs backwards.

**Does `orderRelevant` change that?** `make_element` sets it True and no test
has ever read it back.

**Does AASd-108 hold on the wire?** The unit tests check that `validate`
*reports* a child of the wrong type. Whether BaSyx *accepts* one in the first
place is a different question, and if it does, the repository can hold a
submodel the CLI calls invalid.

The tests below assert the behaviour that makes an agent's loop correct.
Where the answer is genuinely unknown rather than merely untested, the test
records both possibilities and fails only on the combination that would make
a documented command sequence wrong.
"""

from __future__ import annotations

from typing import List

import pytest

from a4s.aasctl.client import BasyxClient

from .conftest import assert_absent, children_of, element_at, run_aas, value_at

pytestmark = pytest.mark.integration


def _list_of_properties(scratch: str, *values: str, id_short: str = "Items") -> None:
    """A SubmodelElementList of Property children holding `values` in order."""
    run_aas(
        "el", "add", scratch, "", "--type", "SubmodelElementList",
        "--id-short", id_short,
        "--type-value-list-element", "Property",
        "--value-type-list-element", "xs:string",
    )
    for value in values:
        run_aas(
            "el", "add", scratch, id_short, "--type", "Property",
            "--value", value, "--value-type", "xs:string",
        )


def _values(client: BasyxClient, scratch: str, id_short: str = "Items") -> List[str]:
    return [child.get("value") for child in children_of(client, scratch, id_short)]


# ---------------------------------------------------------------------------
# appending and addressing
# ---------------------------------------------------------------------------

def test_children_are_appended_in_order(client: BasyxClient, scratch: str) -> None:
    """Position is assigned by the server, and it has to be append-at-end.

    `cmd_el_add` reports `Items[appended]` rather than an index precisely
    because it does not know which one it got. If the server prepended, every
    command sequence in the skills that writes a list then addresses `[0]`
    would be addressing the wrong item.
    """
    _list_of_properties(scratch, "alpha", "beta", "gamma")

    assert _values(client, scratch) == ["alpha", "beta", "gamma"]


def test_an_index_addresses_the_item_at_that_position(
    client: BasyxClient, scratch: str
) -> None:
    """The percent-encoding fix, verified against a repository rather than a stub.

    `[` and `]` are gen-delims and BaSyx rejects them raw with a bare 400 and
    an HTML body. The unit test asserts that `_element_url` encodes them; this
    asserts that the encoding is the one BaSyx accepts. The stub the unit
    tests run against took either form, which is how the defect survived.
    """
    _list_of_properties(scratch, "alpha", "beta", "gamma")

    assert value_at(client, scratch, "Items[1]") == "beta"

    run_aas("set", scratch, "Items[1]", "BETA")
    assert value_at(client, scratch, "Items[1]") == "BETA"
    assert _values(client, scratch) == ["alpha", "BETA", "gamma"]


def test_an_index_past_the_end_is_a_404_not_an_append(
    client: BasyxClient, scratch: str
) -> None:
    """Writing to `Items[9]` on a three-item list must not silently create one."""
    _list_of_properties(scratch, "alpha", "beta", "gamma")

    result = run_aas("set", scratch, "Items[9]", "ghost", expect_ok=False)

    assert not result.ok, "an out-of-range index was accepted"
    assert len(_values(client, scratch)) == 3, "the list grew from an out-of-range write"


# ---------------------------------------------------------------------------
# the question this file exists for
# ---------------------------------------------------------------------------

def test_deleting_an_item_renumbers_the_ones_after_it(
    client: BasyxClient, scratch: str
) -> None:
    """Whether indices shift decides how an agent must write a deletion loop.

    If they shift, deleting `[0]` then `[1]` then `[2]` from a five-item list
    removes items 0, 2 and 4 -- not 0, 1 and 2 -- and the agent has no way to
    notice. The only correct forward loop would be to delete `[0]` n times, or
    to iterate backwards.

    This asserts the shifting behaviour, which is what AAS Part 1 implies for
    an ordered list: position is the identity, so removing one moves the rest.
    A failure here is a real finding either way -- it means BaSyx leaves holes,
    and every skill that iterates a list needs rewriting.
    """
    _list_of_properties(scratch, "a0", "a1", "a2", "a3")

    run_aas("el", "rm", scratch, "Items[1]")

    remaining = _values(client, scratch)
    assert remaining == ["a0", "a2", "a3"], (
        "deletion did not compact the list as expected; "
        f"got {remaining!r}. If this reads ['a0', None, 'a2', 'a3'] the "
        "repository leaves holes and positional addressing is unstable."
    )
    assert value_at(client, scratch, "Items[1]") == "a2", (
        "index 1 does not address the item that moved into it"
    )


def test_a_forward_deletion_loop_removes_the_wrong_items(
    client: BasyxClient, scratch: str
) -> None:
    """The trap, written down as a test so the skills can warn about it.

    This is not asserting a defect in the CLI. It is pinning the consequence
    of the previous test: given that indices compact, the naive loop an agent
    writes first is wrong, and the repository is the only thing that can say
    so. Documenting it in `skills/` is only safe if it is also verified here.
    """
    _list_of_properties(scratch, "a0", "a1", "a2", "a3", "a4")

    for index in (0, 1, 2):                      # the loop an agent writes first
        run_aas("el", "rm", scratch, f"Items[{index}]")

    survivors = _values(client, scratch)
    assert survivors == ["a1", "a3"], (
        f"expected the forward loop to skip items, got {survivors!r}"
    )


def test_a_backward_deletion_loop_removes_exactly_what_was_asked(
    client: BasyxClient, scratch: str
) -> None:
    """The correct form, so the skill has something to prescribe."""
    _list_of_properties(scratch, "a0", "a1", "a2", "a3", "a4")

    for index in (2, 1, 0):                      # highest index first
        run_aas("el", "rm", scratch, f"Items[{index}]")

    assert _values(client, scratch) == ["a3", "a4"]


def test_deleting_the_list_removes_its_children(
    client: BasyxClient, scratch: str
) -> None:
    """A non-empty container must not need emptying first, or leave orphans."""
    _list_of_properties(scratch, "a0", "a1", "a2")

    run_aas("el", "rm", scratch, "Items")

    assert_absent(client, scratch, "Items")
    assert_absent(client, scratch, "Items[0]")


# ---------------------------------------------------------------------------
# constraints on the wire
# ---------------------------------------------------------------------------

def test_a_child_of_the_declared_type_validates_clean(
    client: BasyxClient, scratch: str
) -> None:
    _list_of_properties(scratch, "alpha")

    assert run_aas("--json", "validate", scratch).json()["findings"] == []


def test_a_child_of_the_wrong_type_is_caught_somewhere(
    client: BasyxClient, scratch: str
) -> None:
    """AASd-108: every child must match `typeValueListElement`.

    Two acceptable outcomes, and the test accepts either: the repository
    refuses the write, or it accepts it and `aas validate` reports it. What is
    not acceptable is both accepting it and calling it clean -- that leaves a
    submodel strict consumers cannot read, with nothing to indicate it.
    """
    run_aas(
        "el", "add", scratch, "", "--type", "SubmodelElementList",
        "--id-short", "Items", "--type-value-list-element", "Property",
    )

    wrote = run_aas(
        "el", "add", scratch, "Items", "--type", "SubmodelElementCollection",
        expect_ok=False,
    )
    if not wrote.ok:
        return                                   # refused at the wire: fine

    findings = run_aas("--json", "validate", scratch).json()["findings"]
    rules = {finding["rule"] for finding in findings}
    assert "AASd-108" in rules, (
        "a SubmodelElementCollection was accepted into a list declaring Property, "
        f"and validate did not report it. findings={findings}"
    )


def test_a_child_carrying_an_id_short_is_caught_somewhere(
    client: BasyxClient, scratch: str
) -> None:
    """AASd-120: an SML child must not carry an idShort.

    `make_element` omits it when `--id-short` is not given, so the CLI cannot
    produce this by accident. Passing one explicitly is the way to find out
    whether the repository enforces the constraint or merely tolerates it --
    and the answer decides whether `validate` is a convenience or a necessity.
    """
    run_aas(
        "el", "add", scratch, "", "--type", "SubmodelElementList",
        "--id-short", "Items", "--type-value-list-element", "Property",
    )

    wrote = run_aas(
        "el", "add", scratch, "Items", "--type", "Property",
        "--id-short", "NotAllowed", "--value", "x", "--value-type", "xs:string",
        expect_ok=False,
    )
    if not wrote.ok:
        return

    findings = run_aas("--json", "validate", scratch).json()["findings"]
    rules = {finding["rule"] for finding in findings}
    assert "AASd-120" in rules, (
        f"an idShort on a list child went unreported. findings={findings}"
    )


def test_order_relevant_survives_the_round_trip(
    client: BasyxClient, scratch: str
) -> None:
    """`make_element` sets it; nothing had ever read it back.

    It matters because it is the declaration that makes the index meaningful.
    A repository that drops the flag turns an ordered list into a bag, and
    every positional path becomes a guess.
    """
    run_aas(
        "el", "add", scratch, "", "--type", "SubmodelElementList",
        "--id-short", "Items", "--type-value-list-element", "Property",
    )

    element = element_at(client, scratch, "Items")
    assert element.get("orderRelevant") is True, (
        f"orderRelevant was not preserved: {element.get('orderRelevant')!r}"
    )
