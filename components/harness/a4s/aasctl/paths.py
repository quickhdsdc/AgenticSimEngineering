"""idShortPath algebra for the AAS metamodel (IEC 63278 / AAS Part 2 API).

A single addressing scheme is the whole reason this CLI can replace a fixed
tool set: every command takes a path, so any element reachable in the
metamodel is reachable from the command line without new code.

Grammar (AAS Part 2, "idShortPath"):

    path       := segment ( "." segment | index )*
    segment    := idShort
    index      := "[" integer "]"

Examples::

    Nameplate.ManufacturerName
    TechnicalData.ProductClassifications[0].ProductClassId
    OperatingData.Measurements[3].Series[0].Value

The bracket form addresses ``SubmodelElementList`` children positionally.
The reference implementation this replaces indexed only by idShort and so
could not descend into an SML at all; here list indices are first-class in
both directions (parse and render), which is what lets an agent write to a
list element without a bespoke tool.
"""

from __future__ import annotations

import base64
import re
from dataclasses import dataclass
from typing import Iterator, List, Union

# An AAS idShort is constrained to ``[a-zA-Z][a-zA-Z0-9_]*`` by the metamodel.
# BaSyx-Java emits violations of this in the wild (see the FINDINGS catalogue
# on Java/Python SDK divergence), so parsing stays permissive and validation
# is a separate, explicit step.
_SEGMENT_RE = re.compile(r"[^.\[\]]+")
_INDEX_RE = re.compile(r"\[(\d+)\]")

Step = Union[str, int]


class PathError(ValueError):
    """Raised for a syntactically invalid idShortPath."""


@dataclass(frozen=True)
class ElementPath:
    """A parsed idShortPath: an ordered list of idShort / list-index steps."""

    steps: tuple[Step, ...]

    # -- construction ----------------------------------------------------

    @classmethod
    def parse(cls, raw: str) -> "ElementPath":
        if raw is None:
            raise PathError("path is required")
        text = raw.strip()
        if not text:
            return cls(steps=())
        # `/` is how a person -- and an agent -- writes "the root of the
        # submodel". It is not AAS idShortPath syntax, which has no notion of
        # a leading separator, and sending it on produced a 400 whose body was
        # a Tomcat error page. Accepting it costs nothing: no idShort may
        # contain a slash (AASd-002), so it can mean nothing else.
        if text in ("/", "."):
            return cls(steps=())

        steps: List[Step] = []
        pos = 0
        expect_segment = True

        while pos < len(text):
            char = text[pos]

            if char == ".":
                if expect_segment:
                    raise PathError(f"empty segment at offset {pos} in {raw!r}")
                expect_segment = True
                pos += 1
                continue

            if char == "[":
                match = _INDEX_RE.match(text, pos)
                if not match:
                    raise PathError(f"malformed list index at offset {pos} in {raw!r}")
                if not steps:
                    raise PathError(f"path may not start with a list index: {raw!r}")
                steps.append(int(match.group(1)))
                pos = match.end()
                expect_segment = False
                continue

            match = _SEGMENT_RE.match(text, pos)
            if not match:
                raise PathError(f"unparsable character {char!r} at offset {pos} in {raw!r}")
            steps.append(match.group(0))
            pos = match.end()
            expect_segment = False

        if expect_segment:
            raise PathError(f"trailing separator in {raw!r}")
        return cls(steps=tuple(steps))

    # -- rendering -------------------------------------------------------

    def render(self) -> str:
        """Render back to the wire form BaSyx expects in a URL segment."""
        out: List[str] = []
        for step in self.steps:
            if isinstance(step, int):
                out.append(f"[{step}]")
            elif out:
                out.append(f".{step}")
            else:
                out.append(step)
        return "".join(out)

    def __str__(self) -> str:  # pragma: no cover - convenience
        return self.render()

    def __len__(self) -> int:
        return len(self.steps)

    def __iter__(self) -> Iterator[Step]:
        return iter(self.steps)

    def __bool__(self) -> bool:
        return bool(self.steps)

    # -- navigation ------------------------------------------------------

    @property
    def parent(self) -> "ElementPath":
        return ElementPath(steps=self.steps[:-1])

    @property
    def leaf(self) -> Step | None:
        return self.steps[-1] if self.steps else None

    @property
    def leaf_id_short(self) -> str | None:
        """The last idShort step, skipping any trailing list indices.

        ``Measurements[2]`` has no idShort of its own -- an SML child is
        identified by position -- so the owning list's idShort is returned.
        """
        for step in reversed(self.steps):
            if isinstance(step, str):
                return step
        return None

    @property
    def addresses_list_item(self) -> bool:
        return isinstance(self.leaf, int)

    def child(self, step: Step) -> "ElementPath":
        return ElementPath(steps=self.steps + (step,))

    def is_ancestor_of(self, other: "ElementPath") -> bool:
        return len(self) < len(other) and other.steps[: len(self)] == self.steps


def encode_id(identifier: str) -> str:
    """Base64url-encode an AAS Identifier for use in a URL path segment.

    AAS Part 2 mandates base64url. Padding is kept: BaSyx accepts it and
    stripping it breaks round-tripping through some proxies.
    """
    return base64.urlsafe_b64encode(identifier.encode("utf-8")).decode("ascii")


def decode_id(encoded: str) -> str:
    padding = "=" * (-len(encoded) % 4)
    return base64.urlsafe_b64decode(encoded + padding).decode("utf-8")
