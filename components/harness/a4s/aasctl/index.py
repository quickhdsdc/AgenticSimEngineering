"""Semantic retrieval over a submodel: build an index, then search it.

The problem this solves is that a submodel in this domain routinely holds more
than a hundred elements, and the one you want is rarely named what you called
it. "Nominal capacity" is stored as `RatedCapacity`; the ConceptDescription is
what says they are the same thing. Substring matching does not find it, and
dumping the whole submodel into the model's context to let it look is what the
step budget cannot afford.

Two halves, deliberately separated into two commands:

**Indexing** walks the submodel, builds one descriptor per element out of its
idShort, its ConceptDescription definition and unit, and embeds the lot. It is
slow and it costs embedding tokens, so its result is cached on disk. The tool
this replaces re-embedded every node on *every* query.

**Search** embeds the query alone, ranks by cosine similarity, and then does
the part that matters: **bounded context expansion**. A bare top-k list of
element names invites a model to combine a hit from one submodel with a sibling
it imagined from another. Instead each hit is returned inside its structural
boundary -- its parent collection and that parent's actual children -- so the
neighbourhood the model reasons over is the one the AAS really has.

Two changes from the source implementation, both deliberate:

*No FAISS.* The original built `faiss.IndexFlatL2`, which is exact brute-force
search -- a full scan, no approximation. A numpy matmul over a few thousand
vectors is the same computation and the same results, without a binary
dependency. Nothing is traded away here; if the corpus ever reaches the scale
where an approximate index earns its keep, that is a change with a measurable
reason behind it.

*Scores are returned.* The source reported ranks with no similarity attached,
so a perfect match and a nonsense one looked alike. `aas match` arbitrates on
thresholds, and a threshold needs a number.
"""

from __future__ import annotations

import hashlib
import json
import os
import re
import time
from dataclasses import asdict, dataclass, field
from pathlib import Path
from typing import Any, Dict, Iterable, List, Optional, Sequence

DEFAULT_EMBEDDING_MODEL = "text-embedding-3-large"

# How much of a stored value reaches the descriptor. Enough to identify a
# marking or a designation; not enough for a free-text note to swamp the
# schema half of the text.
VALUE_CHARS = 120

# Where a built index is cached. Beside the ledger rather than inside the AAS
# repository: an index is a derived artefact of one endpoint at one moment, and
# storing it as a submodel would make it look like an assertion about the asset.
DEFAULT_INDEX_HOME = Path(os.getenv("A4S_HOME", Path.home() / ".a4s")).expanduser() / "index"

_CHILD_KEYS = ("submodelElements", "value", "statements", "annotations")


# Splits CamelCase, ACRONYMWord and snake_case into separate words. AAS
# idShorts are constrained to [A-Za-z][A-Za-z0-9_]* by AASd-002, so a
# multi-word name has no choice but to run together.
_WORD_BOUNDARY = re.compile(r"(?<=[a-z0-9])(?=[A-Z])|(?<=[A-Z])(?=[A-Z][a-z])|_+")

# "DIN DKE Spec 99100 chapter reference: 6.7.2.2" and its relatives. Matched by
# shape -- <anything> <chapter|section|clause|paragraph> reference: <rest> --
# so no single standard's name is baked in.
_CITATION = re.compile(
    r"\s*\b[\w .\-]{0,40}?(chapter|section|clause|paragraph)\s+reference\s*:?[^\n]*",
    re.IGNORECASE,
)


def _split_words(raw: str) -> str:
    return _WORD_BOUNDARY.sub(" ", raw or "").strip()


class IndexError_(RuntimeError):
    """The index cannot be built or read as asked."""


# ---------------------------------------------------------------------------
# nodes
# ---------------------------------------------------------------------------

@dataclass
class Node:
    """One addressable element, with everything that makes it findable."""

    path: str                      # idShortPath, list items included: Markings[0].MarkingName
    id_short: Optional[str]
    model_type: str
    parent: Optional[str]
    description: str = ""
    semantic_id: Optional[str] = None
    definition: str = ""           # from the ConceptDescription, IEC 61360
    preferred_name: str = ""
    unit: str = ""
    value: Optional[str] = None
    # Which submodel this element lives in. Defaulted so an index cached before
    # repository-wide indexing existed still loads, with the field filled in
    # from the index's own submodel_id.
    submodel: Optional[str] = None
    submodel_id_short: Optional[str] = None
    # The semanticId of the *submodel*, i.e. which template it instantiates.
    # Carried on every node so "only elements in a TechnicalData submodel" is a
    # filter over rows rather than a second lookup per candidate.
    submodel_semantic_id: Optional[str] = None
    value_type: Optional[str] = None
    # Only a SubmodelElementList carries this: the modelType every child of it
    # must have (AASd-108). Recorded because it is the one place where the type
    # of a *new* element is not a judgement but a constraint -- `match.decide`
    # reads it rather than asking a model a question the metamodel already
    # answers. Defaulted, so an index cached before this field existed loads.
    type_value_list_element: Optional[str] = None

    @property
    def in_list(self) -> bool:
        """Whether any ancestor is a SubmodelElementList, i.e. the path is positional."""
        return "[" in self.path

    def descriptor(
        self,
        with_values: bool = False,
        *,
        parent_depth: int = 1,
        include_submodel: bool = False,
    ) -> str:
        """The text that gets embedded.

        Three decisions here, each measured on 10 query/target pairs drawn from
        a real battery-passport submodel rather than argued from taste. The
        baseline is the descriptor the source implementation built,
        ``title: <idShort>, description: <definition>, unit: <unit>``:

        | descriptor                        |   MRR | top-1 | top-3 |
        |-----------------------------------|------:|------:|------:|
        | source baseline                   | 0.657 |  5/10 |  8/10 |
        | + idShort split into words        | 0.704 |  6/10 |  7/10 |
        | + citation boilerplate stripped   | 0.729 |  6/10 |  8/10 |
        | + parent collection name          | 0.803 |  7/10 |  9/10 |

        **The idShort is split into words.** ``RatedCapacity`` and
        ``InitialSelfDischargingRate`` are how the metamodel spells things and
        not how anyone asks for them; an embedding of the run-together form is
        further from "nominal capacity" than it should be.

        **The parent collection name is included**, and it is the largest single
        gain. ``MinVoltage`` inside ``CapacityEnergyVoltage`` means something
        that ``MinVoltage`` alone does not, and the same short idShort recurs
        under different parents across a submodel.

        **Citation boilerplate is stripped.** Definitions in this corpus end
        with a standards chapter reference — 31 of 46 elements carry one — which
        is a pointer, not a meaning. Honest note: on its own this does *not*
        improve ranking (0.803 either way); it only widens the gap between the
        first and fifth hit by about 8%, which matters only because ``aas match``
        arbitrates on score thresholds. The pattern matches the *shape* of a
        citation rather than any one standard's name, so it does not encode this
        dataset.

        Values are excluded by default. Search here answers "where does this
        belong", which is what ``aas match`` consumes, and a slot's meaning does
        not change when someone writes 36.8 into it. Pass ``with_values`` to
        search stored content instead.

        The one exception is an element inside a SubmodelElementList, whose
        value is always included: list items have, by construction, identical
        schemas — three ``MarkingName`` properties differing only in position —
        so the value is the only thing that can tell them apart. Without it a
        query for "British conformity mark" cannot distinguish ``Markings[1]``
        (UKCA) from ``Markings[2]`` (RoHS).
        """
        title = _split_words(self.preferred_name or self.id_short or self.path)
        text = _CITATION.sub("", self.definition or self.description or "").strip()
        out = f"title: {title}, description: {text}"
        if self.unit:
            out += f", unit: {self.unit}"
        within = self.ancestry(parent_depth)
        if within:
            out += f", within: {within}"
        if include_submodel and self.submodel_id_short:
            out += f", submodel: {_split_words(self.submodel_id_short)}"
        if self.id_short and self.preferred_name and self.id_short != self.preferred_name:
            out += f", idShort: {self.id_short}"
        if (with_values or self.in_list) and self.value:
            # Truncated so a long free-text value cannot drown the schema half.
            out += f", value: {self.value[:VALUE_CHARS]}"
        return out

    def ancestry(self, depth: int = 1) -> str:
        """The last `depth` ancestor names, outermost first; all of them if < 0.

        Parameterised because how much of the path belongs in the descriptor is
        an empirical question, not a settled one. One level was chosen when the
        corpus was a single submodel of 46 elements; across 3,789 elements in
        116 submodels the same short name recurs under deeper structures --
        seven `Payload` properties differing only by which link they hang from
        -- and one level cannot separate them.

        List indices are dropped: `Markings[0]` and `Markings[2]` share a
        meaning, and the position is already in `path`.
        """
        if not self.parent or depth == 0:
            return ""
        segments = [s for s in re.split(r"\.|\[\d+\]", self.parent) if s]
        if depth > 0:
            segments = segments[-depth:]
        return " ".join(_split_words(s) for s in segments)


def walk(submodel: Dict[str, Any]) -> List[Node]:
    """Flatten a submodel into addressable nodes, list items included.

    The source implementation composed paths as ``parent.idShort``. A
    SubmodelElementList's children have no idShort -- AASd-120 forbids one --
    so every item in a list produced the path ``Markings.None`` and collided
    with its siblings: not merely unfindable but silently overwritten. Here a
    list child is addressed by position, which is what ``paths.ElementPath``
    has always supported.
    """
    nodes: List[Node] = []
    owner = submodel.get("id")
    owner_short = submodel.get("idShort")
    owner_semantic = _semantic_id(submodel)

    def visit(node: Dict[str, Any], path: str, parent: Optional[str]) -> None:
        in_list = node.get("modelType") == "SubmodelElementList"
        for key in _CHILD_KEYS:
            children = node.get(key)
            if not isinstance(children, list):
                continue
            for position, child in enumerate(children):
                if not isinstance(child, dict) or "modelType" not in child:
                    continue
                if in_list:
                    child_path = f"{path}[{position}]"
                else:
                    id_short = child.get("idShort")
                    if not id_short:
                        continue
                    child_path = f"{path}.{id_short}" if path else id_short
                nodes.append(
                    Node(
                        path=child_path,
                        id_short=child.get("idShort"),
                        model_type=child.get("modelType", "?"),
                        parent=path or None,
                        description=_description(child),
                        semantic_id=_semantic_id(child),
                        value=_value_text(child),
                        submodel=owner,
                        submodel_id_short=owner_short,
                        submodel_semantic_id=owner_semantic,
                        value_type=child.get("valueType"),
                        type_value_list_element=child.get("typeValueListElement"),
                    )
                )
                visit(child, child_path, path or None)

    visit(submodel, "", None)
    return nodes


def _description(node: Dict[str, Any]) -> str:
    for entry in node.get("description") or []:
        if isinstance(entry, dict) and entry.get("text"):
            return str(entry["text"])
    return ""


def _semantic_id(node: Dict[str, Any]) -> Optional[str]:
    keys = (node.get("semanticId") or {}).get("keys") or []
    return keys[0].get("value") if keys else None


def _value_text(node: Dict[str, Any]) -> Optional[str]:
    model_type = node.get("modelType")
    if model_type == "MultiLanguageProperty":
        parts = node.get("value") or []
        return "; ".join(
            f"{k}: {v}" for entry in parts if isinstance(entry, dict) for k, v in entry.items()
        ) or None
    if model_type == "Range":
        return f"min={node.get('min')}, max={node.get('max')}"
    if model_type in ("SubmodelElementCollection", "SubmodelElementList"):
        return None
    value = node.get("value")
    return None if value is None else str(value)


# ---------------------------------------------------------------------------
# concept descriptions
# ---------------------------------------------------------------------------

def enrich(nodes: Sequence[Node], concepts: Dict[str, Dict[str, str]]) -> int:
    """Attach IEC 61360 definition, preferred name and unit by semanticId.

    Returns how many nodes were enriched, which is worth reporting: a
    repository with no ConceptDescriptions loaded still indexes, but the
    retrieval quality is a different thing entirely and the caller should know
    which of the two they have.
    """
    hits = 0
    for node in nodes:
        concept = concepts.get((node.semantic_id or "").strip())
        if not concept:
            continue
        node.definition = concept.get("definition", "")
        node.preferred_name = concept.get("preferred_name", "")
        node.unit = concept.get("unit", "")
        hits += 1
    return hits


def concept_map(concept_descriptions: Iterable[Dict[str, Any]]) -> Dict[str, Dict[str, str]]:
    """Index ConceptDescriptions by id, pulling out the IEC 61360 content.

    Ids are stripped: the reference repository stores several with a leading
    space (`" 0173-1#02-AAO220#003"`), which never matches a semanticId as
    written by anything else.
    """
    out: Dict[str, Dict[str, str]] = {}
    for concept in concept_descriptions:
        identifier = (concept.get("id") or "").strip()
        if not identifier:
            continue
        content: Dict[str, Any] = {}
        for spec in concept.get("embeddedDataSpecifications") or []:
            candidate = spec.get("dataSpecificationContent") or {}
            if candidate.get("modelType") == "DataSpecificationIec61360":
                content = candidate
                break
        out[identifier] = {
            "definition": _lang_text(content.get("definition")),
            "preferred_name": _lang_text(content.get("preferredName")),
            "unit": str(content.get("unit") or ""),
        }
    return out


def _lang_text(entries: Any, language: str = "en") -> str:
    if not isinstance(entries, list):
        return ""
    for entry in entries:
        if isinstance(entry, dict) and entry.get("language") == language and entry.get("text"):
            return str(entry["text"])
    for entry in entries:
        if isinstance(entry, dict) and entry.get("text"):
            return str(entry["text"])
    return ""


# ---------------------------------------------------------------------------
# the index itself
# ---------------------------------------------------------------------------

@dataclass
class Index:
    submodel_id: str
    id_short: Optional[str]
    model: str
    nodes: List[Node] = field(default_factory=list)
    vectors: Any = None            # numpy array, rows aligned with `nodes`
    enriched: int = 0
    with_values: bool = False      # whether stored values were embedded
    # An index is a snapshot: nothing invalidates it when the submodel changes,
    # so a search can return a path that has since been deleted. Rebuilding is
    # seconds, so the timestamp is reported rather than a watch being kept.
    built_at: float = field(default_factory=time.time)

    def save(self, home: Path = DEFAULT_INDEX_HOME) -> Path:
        import numpy as np

        home.mkdir(parents=True, exist_ok=True)
        stem = home / _slug(self.submodel_id)
        stem.with_suffix(".json").write_text(
            json.dumps(
                {
                    "submodel_id": self.submodel_id,
                    "idShort": self.id_short,
                    "model": self.model,
                    "enriched": self.enriched,
                    "with_values": self.with_values,
                    "built_at": self.built_at,
                    "nodes": [asdict(n) for n in self.nodes],
                },
                ensure_ascii=False,
            ),
            encoding="utf-8",
        )
        np.save(stem.with_suffix(".npy"), self.vectors)
        return stem.with_suffix(".json")

    @classmethod
    def load(cls, submodel_id: str, home: Path = DEFAULT_INDEX_HOME) -> "Index":
        import numpy as np

        stem = home / _slug(submodel_id)
        meta_path = stem.with_suffix(".json")
        if not meta_path.exists():
            raise IndexError_(
                f"no index for {submodel_id!r}. Build one first: aas index {submodel_id}"
            )
        meta = json.loads(meta_path.read_text(encoding="utf-8"))
        owner = meta["submodel_id"]
        known = set(Node.__dataclass_fields__)
        nodes: List[Node] = []
        for raw in meta["nodes"]:
            # Unknown keys are dropped rather than raising: an index written by
            # a newer build should degrade to what this one understands, not
            # refuse to load. Missing keys take their defaults, except
            # `submodel`, which is filled from the index's own id -- a cache
            # from before repository-wide indexing has exactly one owner, and
            # leaving it None would make every structural filter reject it.
            node = Node(**{k: v for k, v in raw.items() if k in known})
            if node.submodel is None and owner != ALL:
                node.submodel = owner
                node.submodel_id_short = node.submodel_id_short or meta.get("idShort")
            nodes.append(node)

        return cls(
            submodel_id=owner,
            id_short=meta.get("idShort"),
            model=meta.get("model", DEFAULT_EMBEDDING_MODEL),
            nodes=nodes,
            vectors=np.load(stem.with_suffix(".npy")),
            enriched=meta.get("enriched", 0),
            with_values=meta.get("with_values", False),
            built_at=meta.get("built_at", 0.0),
        )

    @staticmethod
    def drop(submodel_id: str, home: Path = DEFAULT_INDEX_HOME) -> bool:
        stem = home / _slug(submodel_id)
        removed = False
        for suffix in (".json", ".npy"):
            target = stem.with_suffix(suffix)
            if target.exists():
                target.unlink()
                removed = True
        return removed

    @staticmethod
    def cached(home: Path = DEFAULT_INDEX_HOME) -> List[Dict[str, Any]]:
        rows: List[Dict[str, Any]] = []
        if not home.exists():
            return rows
        for meta_path in sorted(home.glob("*.json")):
            meta = json.loads(meta_path.read_text(encoding="utf-8"))
            rows.append(
                {
                    "submodel_id": meta["submodel_id"],
                    "idShort": meta.get("idShort"),
                    "nodes": len(meta.get("nodes") or []),
                    "enriched": meta.get("enriched", 0),
                    "with_values": meta.get("with_values", False),
                    "built_at": meta.get("built_at", 0.0),
                    "model": meta.get("model"),
                    "path": str(meta_path),
                }
            )
        return rows


def _slug(submodel_id: str) -> str:
    """A filesystem-safe name that still says which submodel it is.

    The hash carries uniqueness -- ids are URIs and two can differ only in a
    path segment -- while the readable prefix means a stale cache directory can
    be understood without opening every file.
    """
    digest = hashlib.sha256(submodel_id.encode("utf-8")).hexdigest()[:12]
    readable = re.sub(r"[^A-Za-z0-9]+", "-", submodel_id).strip("-")[-40:]
    return f"{readable}-{digest}" if readable else digest


# ---------------------------------------------------------------------------
# search
# ---------------------------------------------------------------------------

@dataclass
class Hit:
    node: Node
    score: float
    parent: Optional[Node] = None
    siblings: List[str] = field(default_factory=list)
    # Other submodels holding the same element at the same path. A plant model
    # is full of these -- three copies of one machine's TechnicalData, three of
    # a robot's KinematicModel -- and they are the same fact, not competing
    # answers.
    also_in: List[str] = field(default_factory=list)

    def block(self) -> List[str]:
        """The bounded context block: a hit inside the boundary it really lives in."""
        lines = [
            f"{self.score:.3f}  {self.node.path}  ({self.node.model_type})",
        ]
        label = self.node.preferred_name or self.node.id_short
        if label and label != self.node.path:
            lines.append(f"        name:     {label}")
        if self.node.definition or self.node.description:
            lines.append(f"        meaning:  {(self.node.definition or self.node.description)[:150]}")
        if self.node.unit:
            lines.append(f"        unit:     {self.node.unit}")
        if self.node.semantic_id:
            lines.append(f"        semantic: {self.node.semantic_id}")
        if self.node.value is not None:
            lines.append(f"        value:    {self.node.value[:80]}")
        if self.parent is not None:
            lines.append(
                f"        within:   {self.parent.path or '<submodel root>'} "
                f"({self.parent.model_type})"
            )
        if self.also_in:
            shown = ", ".join(self.also_in[:3])
            more = f" (+{len(self.also_in) - 3} more)" if len(self.also_in) > 3 else ""
            lines.append(f"        also in:  {shown}{more}")
        if self.siblings:
            shown = ", ".join(self.siblings[:12])
            more = f" (+{len(self.siblings) - 12} more)" if len(self.siblings) > 12 else ""
            lines.append(f"        beside:   {shown}{more}")
        return lines


@dataclass
class Filter:
    """Structural constraints applied *before* similarity is computed.

    The reason this exists is a measurement. Over 3,969 elements in the
    reference repository, an exact `semanticId` narrows the candidate set to a
    median of 3 rows -- 0.08% -- and knowing which template a submodel
    instantiates narrows it to 69. Similarity has no notion of precision: it
    ranks whatever it is given and always returns something, so it is a poor
    way to express "not this kind of thing". A filter says that exactly, for
    free, and leaves ranking the job it is actually good at.

    This is the shape of searching a code repository, where the first tool is
    `grep` and not a similarity model. `grep` is not a ranker -- measured on
    the labelled set, lexical matching alone reaches recall@5 0.81 and recall@1
    0.19, which is precisely a filter's profile: the answer is in the set and
    the set is unordered. Fusing the two rankings makes it worse (recall@1
    falls from 0.81 to 0.25); filtering then ranking does not.

    Every field is a conjunction, and an unset field constrains nothing.
    """

    semantic_id: Optional[str] = None            # exact, the strongest key here
    model_type: Optional[str] = None
    value_type: Optional[str] = None
    unit: Optional[str] = None
    submodel: Optional[str] = None
    in_template: Optional[str] = None            # the submodel's semanticId
    id_short_like: Optional[str] = None          # substring, case-insensitive
    has_value: Optional[bool] = None
    under: Optional[str] = None                  # path prefix

    def empty(self) -> bool:
        return all(getattr(self, f) is None for f in self.__dataclass_fields__)

    def keeps(self, node: Node) -> bool:
        if self.semantic_id and (node.semantic_id or "") != self.semantic_id:
            return False
        if self.model_type and node.model_type != self.model_type:
            return False
        if self.value_type and (node.value_type or "") != self.value_type:
            return False
        if self.unit and self.unit.lower() not in (node.unit or "").lower():
            return False
        if self.submodel and (node.submodel or "") != self.submodel:
            return False
        if self.in_template and (node.submodel_semantic_id or "") != self.in_template:
            return False
        if self.id_short_like and self.id_short_like.lower() not in (node.id_short or "").lower():
            return False
        if self.has_value is not None and bool(node.value) != self.has_value:
            return False
        if self.under and not (node.path == self.under or node.path.startswith(f"{self.under}.")
                               or node.path.startswith(f"{self.under}[")):
            return False
        return True

    def describe(self) -> str:
        parts = [
            f"{name}={value!r}"
            for name, value in ((f, getattr(self, f)) for f in self.__dataclass_fields__)
            if value is not None
        ]
        return ", ".join(parts) or "(none)"


def _dedupe_key(node: Node) -> Any:
    """What makes two hits the same hit: the full path and the modelType.

    `semanticId` was in this key and has been taken out, for a reason the
    ablation in `a4s/experiment/ablation.py` settles and an argument decides.

    The measurement first: over the repository-wide labelled set, `idShort`,
    `path`, `path + type` and `path + semanticId + type` score *identically*
    (paraphrase recall@5 0.700 against 0.500 for no deduplication at all).
    **Whether deduplication happens matters; which key it uses does not** --
    at least not on 21 cases, where one case is 0.05 and the spread between
    keys is smaller than that.

    Given no measured difference, the tie breaks on what the field can be
    relied on to contain, and `semanticId` cannot. 31% of the elements here
    carry none, so a key including it never matches for them and they silently
    stop collapsing -- reintroducing duplicate rows for exactly the elements
    with the least other information to identify them. Another 281 carry an
    ECLASS IRDI, `0173-1#02-AAO677#002`, which is a correct identifier and an
    opaque one.

    The path is available for every element by construction, and it is what an
    agent reads anyway. `modelType` stays because two different kinds of
    element at the same path -- possible across unrelated templates -- are not
    the same fact.

    A module-level function so the ablation can substitute one and measure the
    difference without threading a parameter through every caller.
    """
    return (node.path, node.model_type)


def rows_matching(index: Index, where: Optional[Filter]) -> List[int]:
    """Row positions that survive the filter. All of them when it is empty."""
    if where is None or where.empty():
        return list(range(len(index.nodes)))
    return [i for i, node in enumerate(index.nodes) if where.keeps(node)]


def search(
    index: Index,
    query_vector: Any,
    top_k: int = 5,
    min_score: float = 0.0,
    where: Optional[Filter] = None,
    dedupe: bool = True,
) -> List[Hit]:
    """Filter, then rank by cosine similarity, then expand into boundaries.

    The filter is a row mask applied before the dot product rather than a
    post-hoc discard of low-ranked hits. That ordering is what makes it useful:
    discarding after ranking still asks the model to compete against 3,969
    irrelevant rows for the top five places, and the right answer loses those
    contests routinely -- the elements whose idShort carries least meaning
    (`X`, `Y`, `Z`, `Roll`, `Pitch`, recurring 57 to 72 times) are exactly the
    ones a filter removes for free.

    Masking is nearly free. The vectors are already resident, so this is a
    numpy fancy-index over a few thousand rows.
    """
    import numpy as np

    if index.vectors is None or len(index.nodes) == 0:
        return []
    rows = rows_matching(index, where)
    if not rows:
        return []

    corpus = _normalise(np.asarray(index.vectors, dtype="float32")[rows])
    query = _normalise(np.asarray(query_vector, dtype="float32").reshape(1, -1))[0]
    scores = corpus @ query
    order = np.argsort(-scores)

    by_path = {(n.submodel, n.path): n for n in index.nodes}
    wanted = max(top_k, 0)
    # How far past `top_k` to keep scanning, once full, purely to attribute
    # duplicates to the hits already kept. Bounded so a repository-wide search
    # does not walk every row for a cosmetic field.
    horizon = max(wanted * 20, 200)

    hits: List[Hit] = []
    seen: Dict[tuple, int] = {}                       # collapsed key -> position in hits

    for scanned, position in enumerate(order):
        if len(hits) >= wanted and (not dedupe or scanned >= horizon):
            break
        score = float(scores[position])
        if score < min_score:
            break
        node = index.nodes[rows[int(position)]]

        if dedupe:
            # Collapse copies of one element across duplicated submodels.
            #
            # Measured, not anticipated. On the repository-wide labelled set
            # four of the five near-misses landed at exactly rank 4: the first
            # three slots held one wrong element repeated across three
            # duplicate submodels, and the right answer came fourth. Returning
            # k identical rows spends the caller's whole budget saying one
            # thing, and the caller here is an agent with a step budget.
            #
            # Keyed on path *and* concept id, so two genuinely different
            # elements that happen to share a path -- possible across unrelated
            # templates -- stay separate.
            key = _dedupe_key(node)
            if key in seen:
                kept = hits[seen[key]]
                if node.submodel and node.submodel not in kept.also_in:
                    kept.also_in.append(node.submodel)
                continue
            if len(hits) >= wanted:
                continue                              # full, but still attributing
            seen[key] = len(hits)

        parent = by_path.get((node.submodel, node.parent)) if node.parent else None
        hits.append(
            Hit(node=node, score=score, parent=parent, siblings=_siblings(index, node))
        )

    return hits[:wanted]


def _siblings(index: Index, node: Node) -> List[str]:
    """The parent's other children, as the repository actually holds them.

    This is the part that keeps a model honest. Given a bare list of top-k
    names it will happily pair a hit from one collection with a field it
    remembers from another template; given the boundary and its real contents
    it has the structure in front of it.
    """
    out: List[str] = []
    for other in index.nodes:
        # The submodel has to match too. In a repository-wide index the same
        # parent path recurs across submodels -- `GeneralInformation` exists in
        # dozens -- and without this the boundary would be assembled from
        # elements that never sit beside each other, which is precisely the
        # invention the boundary exists to prevent.
        if (
            other.submodel == node.submodel
            and other.parent == node.parent
            and other.path != node.path
        ):
            out.append(other.id_short or other.path.rsplit(".", 1)[-1])
    return out


def _normalise(matrix: Any) -> Any:
    import numpy as np

    norms = np.linalg.norm(matrix, axis=-1, keepdims=True)
    return matrix / np.where(norms == 0, 1.0, norms)


# ---------------------------------------------------------------------------
# embeddings
# ---------------------------------------------------------------------------

def embed(texts: Sequence[str], *, model: Optional[str] = None, batch: int = 256) -> Any:
    """Embed a list of texts with whichever provider is configured.

    Batched because a submodel of 130 elements is one request but a repository
    crawl is not, and an embeddings endpoint has an input-count ceiling.
    """
    import numpy as np

    from ..kernel.config import first
    from ..kernel.providers import DEFAULT_AZURE_API_VERSION, resolve_provider

    if not texts:
        return np.zeros((0, 0), dtype="float32")

    name = model or first("A4S_EMBEDDING_MODEL", "AZURE_OPENAI_EMBEDDING_DEPLOYMENT") \
        or DEFAULT_EMBEDDING_MODEL
    provider = resolve_provider()

    if provider == "azure":
        from openai import AzureOpenAI

        endpoint = first("A4S_BASE_URL", "AZURE_OPENAI_ENDPOINT")
        if not endpoint:
            raise IndexError_("Azure needs AZURE_OPENAI_ENDPOINT to embed.")
        client: Any = AzureOpenAI(
            api_key=first("A4S_API_KEY", "AZURE_OPENAI_API_KEY", "AZURE_OPENAI_KEY"),
            azure_endpoint=endpoint,
            api_version=first("A4S_AZURE_API_VERSION", "AZURE_OPENAI_API_VERSION")
            or DEFAULT_AZURE_API_VERSION,
        )
    elif provider == "anthropic":
        # Anthropic serves no embeddings endpoint. Saying so beats a 404.
        raise IndexError_(
            "the anthropic provider has no embeddings API. Set A4S_PROVIDER=azure "
            "(or openai) for indexing; the run itself can still use Claude."
        )
    else:
        from openai import OpenAI

        client = OpenAI(
            api_key=first("A4S_API_KEY", "OPENAI_API_KEY"),
            base_url=first("A4S_BASE_URL", "OPENAI_BASE_URL"),
        )

    vectors: List[List[float]] = []
    for start in range(0, len(texts), batch):
        chunk = list(texts[start : start + batch])
        response = client.embeddings.create(model=name, input=chunk)
        vectors.extend(item.embedding for item in response.data)
    return np.asarray(vectors, dtype="float32")


def build(
    submodel: Dict[str, Any],
    concepts: Dict[str, Dict[str, str]],
    *,
    model: Optional[str] = None,
    with_values: bool = False,
) -> Index:
    """Walk, enrich, embed. The slow half, run once and cached."""
    nodes = walk(submodel)
    if not nodes:
        raise IndexError_(
            f"submodel {submodel.get('id')!r} has no elements to index"
        )
    enriched = enrich(nodes, concepts)
    resolved = model or DEFAULT_EMBEDDING_MODEL
    vectors = embed([n.descriptor(with_values) for n in nodes], model=model)
    return Index(
        submodel_id=submodel.get("id", ""),
        id_short=submodel.get("idShort"),
        model=resolved,
        nodes=nodes,
        vectors=vectors,
        enriched=enriched,
        with_values=with_values,
    )


# The id a repository-wide index is cached under. Not a legal AAS Identifier,
# which is the point: it cannot collide with a real submodel.
ALL = "*"


def build_all(
    submodels: Sequence[Dict[str, Any]],
    concepts: Dict[str, Dict[str, str]],
    *,
    model: Optional[str] = None,
    with_values: bool = False,
    on_progress: Optional[Any] = None,
) -> Index:
    """One index over every submodel in the repository.

    Preferred to the per-submodel form for any question that does not already
    name its submodel, which is most of them: an operator asking where a
    machine's barrel temperature lives does not know, and cannot be expected
    to, which of 116 submodels holds it.

    A separate shell-level index was the obvious alternative and is the wrong
    shape. Two stages mean two ranking decisions, and the first one -- picking
    the right submodel from its idShort and description alone -- is made with
    less information than the second. Indexing every element once and
    constraining with `Filter` reaches the same place with one ranking and no
    cascade of errors.

    The cost argument for doing it per-submodel does not survive contact with
    the numbers: 3,969 elements at roughly 50 tokens each is about 200k
    embedding tokens, a few cents, once. The vectors occupy ~49 MB and one
    matmul over them is microseconds. **Speed was never the constraint here --
    precision is**, which is what `Filter` addresses.
    """
    nodes: List[Node] = []
    for submodel in submodels:
        nodes.extend(walk(submodel))
        if on_progress is not None:
            on_progress(submodel.get("idShort") or submodel.get("id"), len(nodes))
    if not nodes:
        raise IndexError_("the repository served no submodel elements to index")

    enriched = enrich(nodes, concepts)
    resolved = model or DEFAULT_EMBEDDING_MODEL
    vectors = embed([n.descriptor(with_values) for n in nodes], model=model)
    return Index(
        submodel_id=ALL,
        id_short=f"{len(submodels)} submodels",
        model=resolved,
        nodes=nodes,
        vectors=vectors,
        enriched=enriched,
        with_values=with_values,
    )
