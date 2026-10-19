"""The wizard workspace, and the traversal it refuses.

These routes take a filename out of a URL and turn it into a path. That is the
exact shape of a directory-traversal hole, and the wizard has six of them, so
the check lives in one place and is tested here rather than six times over.
"""

from __future__ import annotations

from pathlib import Path

import pytest

from a4s.server import workspace


@pytest.fixture(autouse=True)
def scratch_workspace(tmp_path: Path, monkeypatch) -> Path:
    monkeypatch.setenv("A4S_WEB_WORKSPACE", str(tmp_path))
    return tmp_path


# ---------------------------------------------------------------------------
# names
# ---------------------------------------------------------------------------

@pytest.mark.parametrize("name", [
    "../secrets.json",
    "..\\secrets.json",
    "sub/dir.json",
    "..",
    ".",
    "",
    "   ",
])
def test_a_name_that_is_not_a_plain_filename_is_refused(name: str) -> None:
    with pytest.raises(workspace.WorkspaceError):
        workspace.safe_name(name)


def test_refusing_beats_sanitising() -> None:
    """Stripping the separators out of `../../etc/passwd` leaves `etcpasswd`.

    The read then succeeds against a file the caller never asked for, or fails
    for a reason that has nothing to do with what they typed. Either way they
    are told the operation worked on the name they gave.
    """
    with pytest.raises(workspace.WorkspaceError) as caught:
        workspace.safe_name("../../etc/passwd")

    assert "not a plain filename" in str(caught.value)


def test_an_ordinary_datasheet_name_survives() -> None:
    assert workspace.safe_name("Type-400.5_datasheet (1).pdf") == \
        "Type-400.5_datasheet (1).pdf"


def test_under_keeps_the_result_inside_the_directory(tmp_path: Path) -> None:
    resolved = workspace.under(workspace.uploads(), "datasheet.pdf")

    assert resolved.parent == (tmp_path / "uploads").resolve()


# ---------------------------------------------------------------------------
# the identifier the stages share
# ---------------------------------------------------------------------------

def test_all_four_source_types_land_on_the_same_key() -> None:
    """The frontend passes whatever each source gave it. The stages after it
    have to agree on one name or the extraction is looked for under a key
    nothing wrote."""
    assert workspace.stem("datasheet.pdf") == "datasheet"
    assert workspace.stem("datasheet") == "datasheet"
    assert workspace.stem("text_input_1a2b3c4d.json") == "text_input_1a2b3c4d"
    assert workspace.stem("parts.csv") == "parts"


def test_a_product_name_keeps_its_dotted_tail() -> None:
    """`Type 400.5` is a product, not a file with a `.5` extension. Stripping
    everything after the last dot would key the extraction under `Type 400`."""
    assert workspace.stem("Type-400.5") == "Type-400.5"


def test_the_match_file_is_named_for_the_document_and_the_shell() -> None:
    """The same datasheet gets matched against several shells. One name per
    document would make the second match overwrite the first."""
    first = workspace.match_file("datasheet.pdf", "AGV")
    second = workspace.match_file("datasheet.pdf", "Press")

    assert first != second
    assert "AGV" in first.name and "Press" in second.name


def test_the_shell_is_not_appended_twice() -> None:
    """The frontend already builds `{base}_{aas}` before it calls save, then
    passes the plain document name to check. Both spellings must name one
    file, or the wizard re-runs a match it has already done."""
    assert workspace.match_file("datasheet_AGV", "AGV") == \
        workspace.match_file("datasheet.pdf", "AGV")
