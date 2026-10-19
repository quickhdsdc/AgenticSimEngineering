"""Reading documents the model has to look at, and delivering them to it.

The capability these cover replaces a nested model call: `read_file` on a PDF
hands the agent the rendered pages instead of calling a vision model on its
behalf. A nested call is invisible to the ledger, does not count against the
step budget, cannot be approved and cannot be interrupted -- and a six-page
datasheet once extracted three entities and reported success because five such
calls returned nothing and nobody could see it.

The delivery is the part worth testing hardest, because the two providers do
not agree and the disagreement is not optional. Anthropic accepts image blocks
inside a `tool_result`. OpenAI does not: a `role: "tool"` message's content
must be a string. So the loop writes attachments as a separate user message and
each adapter does what its API needs -- and a tool never learns which model is
reading it.
"""

from __future__ import annotations

from typing import Any, Dict, List

import pytest

from a4s.kernel.loop import _attachment_message
from a4s.kernel.providers import _for_openai, _to_anthropic
from a4s.tools import ToolResult, _page_range


PNG = bytes([137, 80, 78, 71, 13, 10, 26, 10]) + b"fake"


def tool_call_exchange(images: List[bytes]) -> List[Dict[str, Any]]:
    """The three messages a tool call with attachments produces."""
    return [
        {"role": "assistant", "content": "", "tool_calls": [
            {"id": "c1", "type": "function",
             "function": {"name": "read_file", "arguments": "{}"}}]},
        {"role": "tool", "tool_call_id": "c1", "content": "spec.pdf: 6 pages"},
        _attachment_message(images, ["spec.pdf page 3"]),
    ]


# ---------------------------------------------------------------------------
# selecting pages
# ---------------------------------------------------------------------------

@pytest.mark.parametrize(
    "spec,total,expected",
    [
        ("3", 6, [3]),
        ("1-4", 6, [1, 2, 3, 4]),
        ("1,3,5-7", 8, [1, 3, 5, 6, 7]),
        ("4-1", 6, [1, 2, 3, 4]),          # written backwards, meant forwards
        ("1-99", 3, [1, 2, 3]),            # clamped, not refused
        ("0,7", 6, []),                    # both out of range
        ("nonsense", 6, []),
        ("", 3, [1, 2, 3]),
    ],
)
def test_page_ranges_are_read_the_way_a_person_writes_them(
    spec: str, total: int, expected: List[int]
) -> None:
    """Clamping beats refusing: a model asking for 1-99 of a 3-page file wants
    the three. Refusing costs a step and teaches it nothing."""
    assert _page_range(spec, total) == expected


def test_an_empty_spec_stops_at_the_cap_rather_than_reading_everything() -> None:
    """A page is on the order of 1.5k tokens and stays in context for the rest
    of the run, so an unbounded default would let one call on a 200-page manual
    end the session."""
    from a4s.tools import ToolRegistry

    assert _page_range("", 500) == list(range(1, ToolRegistry.MAX_PAGES + 1))


def test_duplicate_pages_are_asked_for_once() -> None:
    assert _page_range("2,2,1-3", 5) == [2, 1, 3]


# ---------------------------------------------------------------------------
# delivery
# ---------------------------------------------------------------------------

def test_openai_never_sees_an_image_inside_a_tool_result() -> None:
    """The constraint that shaped the design.

    A `role: "tool"` message's content must be a string for this API. If
    attachments ever leaked into one, every request carrying a document would
    fail with a validation error rather than degrading.
    """
    sent = _for_openai(tool_call_exchange([PNG]))

    tool_messages = [m for m in sent if m["role"] == "tool"]
    assert tool_messages and all(isinstance(m["content"], str) for m in tool_messages)

    user = [m for m in sent if m["role"] == "user"]
    assert user, "the attachment message was dropped"
    assert any(part["type"] == "image_url" for part in user[-1]["content"])


def test_the_private_marker_never_reaches_the_wire() -> None:
    """`_attachments` marks the message for the Anthropic adapter. This API
    rejects unknown fields, so it has to be stripped -- but it must survive in
    the loop's own message list, because the conversation is replayed to
    whichever provider is configured on the *next* call."""
    original = tool_call_exchange([PNG])
    sent = _for_openai(original)

    assert not any("_attachments" in m for m in sent)
    assert original[-1]["_attachments"] is True, "the caller's list was mutated"


def test_anthropic_folds_the_pages_into_the_tool_result() -> None:
    """Sending them as their own user turn would put a bare message between a
    tool_use and its result, which that API rejects."""
    sent = _to_anthropic(tool_call_exchange([PNG]))

    results = [
        block
        for message in sent
        for block in (message["content"] if isinstance(message["content"], list) else [])
        if isinstance(block, dict) and block.get("type") == "tool_result"
    ]
    assert len(results) == 1
    blocks = results[0]["content"]
    assert isinstance(blocks, list)
    assert [b["type"] for b in blocks] == ["text", "text", "image"]
    assert blocks[-1]["source"]["media_type"] == "image/png"

    # And no leftover user turn carrying the same pages twice.
    assert len(sent) == 2, [m["role"] for m in sent]


def test_a_tool_result_without_attachments_stays_a_plain_string() -> None:
    """The common case must not grow a block wrapper it does not need."""
    plain = [{"role": "tool", "tool_call_id": "c1", "content": "ok"}]

    block = _to_anthropic(plain)[0]["content"][0]
    assert block["content"] == "ok"


def test_tool_result_stringifies_to_its_text() -> None:
    """The loop puts `str(result)` in the tool message, so a ToolResult has to
    read as its text everywhere a plain string used to."""
    result = ToolResult(text="spec.pdf: 6 pages", images=[PNG], labels=["p3"])

    assert str(result) == "spec.pdf: 6 pages"
    assert f"{result}" == "spec.pdf: 6 pages"
