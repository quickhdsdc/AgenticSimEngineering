"""Matching extracted entities to submodel elements, and applying the result.

This is the substantive algorithm of the system, and the one whose numbers are
reported: a three-way decision per entity, on the cosine similarity between the
entity and its best candidate element.

    similarity >= theta_high (0.8)   accept outright
    theta_low <= s < theta_high      hand to a model to arbitrate
    similarity <  theta_low  (0.5)   reject outright

The middle band is the point. Above it retrieval is reliable enough that asking
costs tokens and adds noise; below it no candidate is worth arbitrating over.
Only the band between is genuinely uncertain, and that is where a model earns
its place. Arbitration sees each candidate inside its **parent boundary** --
the collection that contains it -- because "MinVoltage under
CapacityEnergyVoltage" is a different proposition from "MinVoltage" alone.

An entity with no element to write into is not dropped: a second pass asks
which collection should *host* it, and the plan records a creation rather than
a write.

Matching and applying are separate on purpose. ``match`` emits a plan and
touches nothing; ``populate`` applies a plan. The plan is a file, so it can be
read, edited, diffed and re-run, and the ledger records it beside the writes it
produced -- which is what makes a middle-band arbitration auditable after the
fact rather than a decision that happened once inside a process.

Departures from the source implementation, each with its reason:

*Thresholds are parameters, not constants.* They are reported in the paper; a
number that appears in a paper and is buried in a function body is a number
that will silently drift.

*Units are checked.* The source matched on name and description and ignored the
unit entirely. Writing a value in kWh into a property declared in Wh is wrong
in a way nothing downstream catches, so a mismatch becomes a warning carried in
the plan.

*The corpus is not re-embedded.* The source embedded every element of the
submodel on every call. ``aas index`` already did that once, so matching reads
the cache.

*No hardcoded destination.* The source routed low-similarity entities to a
collection literally named ``TechnicalPropertyAreas`` inside a submodel named
``TechnicalData``. That is a fact about one battery-passport deployment, not
about the metamodel; here the caller names the fallback with ``--create-in`` or
gets none.

*Value types are inferred.* The source created every new element as
``xs:string``. A capacity written as a string is not comparable, not
summable, and not valid against a template that declares a decimal.

*Element types are decided, not assumed.* Creation used to emit ``Property``
unconditionally, so ``250-550 mm`` became a string and a name given in three
languages became one. ``aas el add`` can build every authorable ``modelType``
already, so the constraint was never the CLI -- it was that nothing chose. The
choice now happens here, in three layers, and only the last of them is a guess:

1. **A constraint, when the metamodel gives one.** A child of a
   ``SubmodelElementList`` must have that list's ``typeValueListElement``
   (AASd-108) and must carry no ``idShort`` (AASd-120). Nothing is asked.
2. **The model, in the call that already happens.** ``HOST_PROMPT`` was already
   spending one round trip per unplaced entity to choose *where*; it now
   answers *what* in the same breath, choosing from `CREATABLE_TYPES`. No extra
   token is spent to gain the type.
3. **The value's own shape, when no model is available.** ``--no-arbitration``
   and ``--create-in`` still reach creation, and there the value is read for an
   interval, a language tag or a document path. Deliberately conservative:
   `1234-5678` with no unit is an order number, not a range.

Layer 3 also seeds layer 2, so a model that answers only ``index`` still gets
the shape-inferred type rather than falling back to ``Property``.
"""

from __future__ import annotations

import json
import re
from dataclasses import asdict, dataclass, field
from typing import Any, Dict, List, Optional, Sequence, Tuple

from .index import Filter, Index, search
from .model import external_reference, lang_strings, make_element

# The thresholds the paper reports. Named here so the defaults and the text
# that describes them cannot drift apart.
THETA_HIGH = 0.8
THETA_LOW = 0.5

# Candidates of these types are never a *write* target -- they hold other
# elements rather than a value -- but they are the candidates for hosting a
# newly created one.
CONTAINER_TYPES = ("SubmodelElementCollection", "SubmodelElementList")

# Never a write target, because none of these holds a scalar `$value`. The
# retrieval index returns them like anything else -- they have idShorts and
# descriptions and embed perfectly well -- and a plan that proposes writing
# "1500 kN" onto a Capability is accepted by nothing and refused by BaSyx with
# an HTTP 500 that names no reason.
#
# Measured before it was fixed: of 396 decisions applied against a real shell,
# 295 failed, every one of them a 500, and every one of them a write onto a
# `Capability`, a `SubmodelElementCollection` or an `Operation`. Only
# `TechnicalData` -- the submodel whose elements are actually Properties --
# applied anything at all.
#
# `Operation` and `Capability` are the ones worth naming: an Operation is
# *invoked*, and the harness has `aas op invoke` for that, gated. Writing to
# one is not a smaller version of calling it.
NON_CANDIDATE_TYPES = (
    "AssetAdministrationShell",
    "Submodel",
    "Capability",
    "Operation",
    "Entity",
    "BasicEventElement",
    "RelationshipElement",
    "AnnotatedRelationshipElement",
)

# The modelTypes an *extracted entity* may be created as. A strict subset of
# `model.ELEMENT_TYPES`, and the omissions are the argument:
#
# `SubmodelElementCollection` and `SubmodelElementList` hold other elements and
# no value, so creating one from a single extracted fact produces an empty
# container and loses the fact. `Operation`, `Entity` and `RelationshipElement`
# are not things a datasheet line is: an Operation is authored against a
# service binding, an Entity against an asset. `Blob` is excluded because an
# extractor emits text, not embedded binary -- a document arrives as a path,
# which is a `File`.
#
# This tuple is the enum handed to the model. Keeping it narrow is what makes
# the question answerable: five types with visibly different value shapes, not
# eleven with overlapping ones.
CREATABLE_TYPES = (
    "Property",
    "MultiLanguageProperty",
    "Range",
    "File",
    "ReferenceElement",
)

_IDSHORT_SAFE = re.compile(r"[^A-Za-z0-9_]+")


class MatchError(ValueError):
    """The match cannot be run or the plan cannot be read as asked."""


# ---------------------------------------------------------------------------
# entities
# ---------------------------------------------------------------------------

@dataclass
class Entity:
    """One fact extracted from a source document, with where it came from."""

    entity: str
    value: Optional[str] = None
    description: str = ""
    unit: str = ""
    source: str = ""               # document location; provenance, not decoration

    @classmethod
    def parse(cls, raw: Dict[str, Any]) -> "Entity":
        # `name` and `entity` both appear in extractor output in the wild.
        name = raw.get("entity") or raw.get("name") or ""
        if not name:
            raise MatchError(f"entity has no name: {json.dumps(raw)[:120]}")
        value = raw.get("value")
        return cls(
            entity=str(name),
            value=None if value is None else str(value),
            description=str(raw.get("description") or ""),
            unit=str(raw.get("unit") or ""),
            source=str(raw.get("source") or raw.get("location") or ""),
        )

    def query(self) -> str:
        """The text embedded to find candidates.

        Deliberately name and description only. The unit is checked against the
        target afterwards rather than embedded: it is a constraint on
        correctness, not a hint about meaning, and mixing it into the query
        pulls every dimensionless entity towards every unit-bearing element.
        """
        return f"name: {self.entity}; description: {self.description}"


def load_entities(raw: Any) -> List[Entity]:
    if isinstance(raw, dict):
        raw = raw.get("entities", raw.get("result", []))
    if not isinstance(raw, list):
        raise MatchError("entities file must be a JSON list, or an object with an 'entities' list")
    return [Entity.parse(item) for item in raw if isinstance(item, dict)]


# ---------------------------------------------------------------------------
# decisions
# ---------------------------------------------------------------------------

@dataclass
class Candidate:
    path: str
    id_short: Optional[str]
    model_type: str
    score: float
    parent: Optional[str]
    unit: str = ""
    definition: str = ""
    # Set only when this candidate is a SubmodelElementList: the modelType its
    # children are required to have. Carried through so a create hosted into a
    # list is typed by the list rather than by a guess.
    type_value_list_element: Optional[str] = None


@dataclass
class Decision:
    """What the plan says to do about one entity, and why."""

    entity: str
    value: Optional[str]
    unit: str
    source: str
    action: str                    # "write" | "create" | "skip"
    band: str                      # "auto" | "arbitrated" | "below" | "hosted" | "none"
    reason: str
    path: Optional[str] = None
    model_type: Optional[str] = None
    value_type: Optional[str] = None
    id_short: Optional[str] = None
    # The type-specific half of a create. Empty for a Property, which is why
    # they are defaulted -- and why a plan written before `model_type` was a
    # decision still loads and still applies, as a Property.
    min_value: Optional[str] = None
    max_value: Optional[str] = None
    content_type: Optional[str] = None
    # For a `create`, the modelType of the element at `path` -- the container
    # the new element goes into. `model_type` is the thing being created, and
    # for a create the two are different: a Property created inside a
    # SubmodelElementCollection. Recorded because a caller that has only the
    # plan cannot otherwise tell whether `path` names a slot or a home, and one
    # that guesses turns a create into a write onto a collection, which BaSyx
    # answers with HTTP 500.
    container_type: Optional[str] = None
    score: float = 0.0
    # Distance from the best leaf candidate to the runner-up. Recorded, not
    # acted on: on the first live run it separated entities that had a correct
    # target (margins 0.098-0.200) from ones the submodel simply cannot hold
    # (0.003-0.029) far more sharply than the absolute score did. That is an
    # observation from n=7 on one submodel, not a decision rule, and it is kept
    # in the plan so the question can be studied from real runs rather than
    # needing new instrumentation.
    margin: float = 0.0
    warnings: List[str] = field(default_factory=list)
    candidates: List[Dict[str, Any]] = field(default_factory=list)

    def line(self) -> str:
        mark = {"write": "write ", "create": "create", "skip": "skip  "}[self.action]
        target = self.path or "-"
        warn = f"  !! {'; '.join(self.warnings)}" if self.warnings else ""
        return (
            f"{mark} {self.score:.3f} +{self.margin:.3f} {self.band:<10} "
            f"{self.entity[:26]:<26} -> {target}{warn}"
        )


@dataclass
class Plan:
    submodel_id: str
    theta_high: float
    theta_low: float
    model: Optional[str]
    decisions: List[Decision] = field(default_factory=list)

    def counts(self) -> Dict[str, int]:
        out = {"write": 0, "create": 0, "skip": 0, "auto": 0, "arbitrated": 0, "warnings": 0}
        for decision in self.decisions:
            out[decision.action] += 1
            if decision.band in ("auto", "arbitrated"):
                out[decision.band] += 1
            out["warnings"] += len(decision.warnings)
        return out

    def calibration_note(self) -> Optional[str]:
        """Say so when the auto-accept band never fires.

        theta_high is calibrated against a particular descriptor and a
        particular embedding model; change either and the similarity
        distribution moves under it. If nothing clears the bar, every entity
        goes to arbitration -- correct answers included -- and the caller pays
        one model call each without being told the band is dead.
        """
        if not self.decisions:
            return None
        best = max(d.score for d in self.decisions)
        if best >= self.theta_high:
            return None
        return (
            f"no entity reached theta_high={self.theta_high} (best was {best:.3f}), so the "
            "auto-accept band never fired and every decision above theta_low went to "
            "arbitration. Check the threshold against this descriptor and embedding model "
            "before reading the band counts as a result."
        )

    def to_json(self) -> Dict[str, Any]:
        return {
            "submodel": self.submodel_id,
            "thresholds": {"high": self.theta_high, "low": self.theta_low},
            "arbitration_model": self.model,
            "counts": self.counts(),
            "decisions": [asdict(d) for d in self.decisions],
        }

    @classmethod
    def from_json(cls, raw: Dict[str, Any]) -> "Plan":
        try:
            thresholds = raw["thresholds"]
            plan = cls(
                submodel_id=raw["submodel"],
                theta_high=float(thresholds["high"]),
                theta_low=float(thresholds["low"]),
                model=raw.get("arbitration_model"),
            )
            for item in raw["decisions"]:
                known = {k: v for k, v in item.items() if k in Decision.__annotations__}
                plan.decisions.append(Decision(**known))
        except (KeyError, TypeError, ValueError) as exc:
            raise MatchError(f"not a match plan: {exc}") from exc
        return plan


# ---------------------------------------------------------------------------
# value typing
# ---------------------------------------------------------------------------

_DATE = re.compile(r"^\d{4}-\d{2}-\d{2}$")
_DATETIME = re.compile(r"^\d{4}-\d{2}-\d{2}[T ]\d{2}:\d{2}")

# Digit runs at least this long, with no unit, are read as identifiers rather
# than quantities. Ten covers GTIN-12/13, ISBN and order numbers while leaving
# any plausible measured integer -- a cycle count, a voltage, a year -- alone.
_IDENTIFIER_DIGITS = 10


def infer_value_type(value: Optional[str], unit: str = "") -> str:
    """Guess the xsd type of an extracted value.

    The source created every element as ``xs:string``. A capacity stored as a
    string is not comparable, not summable, and fails a template that declares a
    decimal -- and nothing reports it, because a string is valid JSON in the
    slot. So inference is worth doing, but only where it cannot lose anything:

    * **A leading zero forces a string.** ``"007"`` as an integer is ``7``, and
      the difference is not recoverable. This one is not a heuristic -- the
      conversion is lossy, full stop.
    * **A long unitless digit run is an identifier, not a quantity.** A barcode,
      a GTIN, an order number. The signal is the unit: a measured quantity
      almost always carries one and an identifier never does. This *is* a
      heuristic, and it is why ``unit`` is a parameter rather than the value
      being judged alone.

    Anything else that is not clearly a number, boolean or date stays a string.
    """
    if value is None:
        return "xs:string"
    text = value.strip()
    if not text:
        return "xs:string"
    if text.lower() in ("true", "false"):
        return "xs:boolean"
    if _DATETIME.match(text):
        return "xs:dateTime"
    if _DATE.match(text):
        return "xs:date"

    digits = text.lstrip("+-")
    if digits.isdigit():
        if len(digits) > 1 and digits[0] == "0":
            return "xs:string"
        if not unit.strip() and len(digits) >= _IDENTIFIER_DIGITS:
            return "xs:string"
        return "xs:integer"
    try:
        float(text)
        return "xs:double"
    except ValueError:
        return "xs:string"


# ---------------------------------------------------------------------------
# element typing
# ---------------------------------------------------------------------------

@dataclass
class ElementSpec:
    """What to create, once something has decided where.

    Separate from `Decision` because it is produced by three different things
    -- a list's declared child type, the hosting model, the value's own shape
    -- and merged before any of it reaches a plan.
    """

    model_type: str = "Property"
    value_type: Optional[str] = None
    min_value: Optional[str] = None
    max_value: Optional[str] = None
    content_type: Optional[str] = None
    reason: str = ""

    def merged_with(self, other: Optional["ElementSpec"]) -> "ElementSpec":
        """`other` wins where it says anything; this fills the rest.

        The asymmetry is the point: the model answers `modelType` and often
        nothing else, and the shape inference has already worked out the min,
        the max and the content type from the same value. Discarding those
        because a model was available would make the model call a downgrade.
        """
        if other is None:
            return self
        return ElementSpec(
            model_type=other.model_type or self.model_type,
            value_type=other.value_type or self.value_type,
            min_value=other.min_value if other.min_value is not None else self.min_value,
            max_value=other.max_value if other.max_value is not None else self.max_value,
            content_type=other.content_type or self.content_type,
            reason=other.reason or self.reason,
        )


# An interval, written the several ways datasheets write one. The separator
# alternatives are ordered longest-first so `...` is not read as `.` twice.
_RANGE_SEPARATOR = r"(?:\.\.\.|…|\.\.|--|–|—|~|-|\bto\b|\bbis\b)"
_NUMBER = r"[+-]?\d+(?:[.,]\d+)?"
_RANGE = re.compile(
    rf"^\s*({_NUMBER})\s*{_RANGE_SEPARATOR}\s*({_NUMBER})\s*(.*?)\s*$",
    re.IGNORECASE,
)

# A unit is letters and symbols; crucially it carries no digits. That is what
# keeps `2026-08-29` out: the tail after the second number is `-29`, which is
# not a unit, so a date is never read as an interval.
_UNIT_SHAPED = re.compile(r"^[A-Za-z°µΩ%/\s.^*-]{1,12}$")

# Words that say "this is an interval" when the numbers alone are ambiguous.
_RANGE_WORDS = re.compile(
    r"\b(range|interval|span|limits?|min(?:imum)?|max(?:imum)?|between|"
    r"bereich|spanne|grenzwerte?)\b",
    re.IGNORECASE,
)

# The `text@lang` shorthand this CLI already uses for MultiLanguageProperty.
_LANG_TAGGED = re.compile(r"@[a-z]{2}(?:[-_][A-Za-z]{2,4})?\s*(?:\||$)")

_URLISH = re.compile(r"^(?:https?://|ftp://|file://|/|\./|\.\./|[A-Za-z]:[\\/])")
_DOCUMENT_SUFFIX = re.compile(
    r"\.(pdf|png|jpe?g|gif|svg|tiff?|dwg|dxf|stp|step|igs|iges|"
    r"xlsx?|docx?|pptx?|csv|txt|xml|json|zip|aasx)$",
    re.IGNORECASE,
)

_CONTENT_TYPES = {
    "pdf": "application/pdf", "png": "image/png", "jpg": "image/jpeg",
    "jpeg": "image/jpeg", "gif": "image/gif", "svg": "image/svg+xml",
    "tif": "image/tiff", "tiff": "image/tiff", "csv": "text/csv",
    "txt": "text/plain", "xml": "application/xml", "json": "application/json",
    "zip": "application/zip", "aasx": "application/asset-administration-shell-package",
    "xls": "application/vnd.ms-excel",
    "xlsx": "application/vnd.openxmlformats-officedocument.spreadsheetml.sheet",
    "doc": "application/msword",
    "docx": "application/vnd.openxmlformats-officedocument.wordprocessingml.document",
}


def infer_element_type(entity: Entity) -> ElementSpec:
    """Read an entity's value for the kind of element that can hold it.

    The deterministic layer, and the floor under the model: it runs first, its
    answer seeds the prompt's default, and it is the whole answer when
    ``--no-arbitration`` is in force. So it is deliberately timid -- every rule
    here demands a second signal before it fires, because a wrong type is worse
    than a Property. A Property holding ``"250-550"`` is at least readable; a
    Range built from an order number is a defect in the model of a machine.

    * **Interval -> Range**, but only with corroboration. Two numbers around a
      dash is also a GTIN block, an order number and a date. It counts as a
      range when a unit is present -- on the entity or trailing the numbers --
      or when the name or description says so.
    * **Language tags -> MultiLanguageProperty.** Only the ``text@lang``
      shorthand, which is already this CLI's convention and which nothing else
      produces by accident.
    * **A path or URL -> File**, with the content type taken from the suffix.
    * **Everything else -> Property.**

    `ReferenceElement` is never inferred. Its value is an IRI or an IRDI, and
    those are indistinguishable from an identifier stored as a string without
    knowing what the value is *for* -- which is a judgement, so it is left to
    the model rather than guessed here.
    """
    text = (entity.value or "").strip()
    context = f"{entity.entity} {entity.description}"
    if not text:
        return ElementSpec(model_type="Property", value_type="xs:string",
                           reason="no value to read a shape from")

    if _LANG_TAGGED.search(text):
        return ElementSpec(model_type="MultiLanguageProperty",
                           reason="value carries language tags")

    found = _RANGE.match(text)
    if found:
        low, high, tail = found.group(1), found.group(2), found.group(3)
        tail_is_unit = bool(tail) and bool(_UNIT_SHAPED.match(tail))
        corroborated = bool(entity.unit.strip()) or bool(_RANGE_WORDS.search(context))
        if (not tail or tail_is_unit) and (corroborated or tail_is_unit):
            # The bounds are typed together: a range whose min reads as an
            # integer and whose max reads as a double has one valueType, and it
            # is the wider of the two.
            low_type = infer_value_type(low, entity.unit)
            high_type = infer_value_type(high, entity.unit)
            value_type = "xs:double" if "xs:double" in (low_type, high_type) else low_type
            return ElementSpec(
                model_type="Range", value_type=value_type,
                min_value=low, max_value=high,
                reason=f"value reads as the interval {low}..{high}",
            )

    if _URLISH.match(text) or _DOCUMENT_SUFFIX.search(text):
        suffix = text.rsplit(".", 1)[-1].lower() if "." in text else ""
        return ElementSpec(
            model_type="File",
            content_type=_CONTENT_TYPES.get(suffix, "application/octet-stream"),
            reason="value reads as a document path or URL",
        )

    return ElementSpec(model_type="Property",
                       value_type=infer_value_type(entity.value, entity.unit))


def resolve_spec(
    entity: Entity,
    container: Optional[Candidate],
    chosen: Optional[ElementSpec],
) -> Tuple[ElementSpec, List[str]]:
    """Settle the three layers into one spec, and say where it had to give way.

    Returns the spec and any warnings, because both ways this can degrade -- a
    list that dictates a type nobody can build, a Range with no bounds -- are
    things the person reviewing the plan needs to see in the table rather than
    discover afterwards from an HTTP 500.
    """
    warnings: List[str] = []
    spec = infer_element_type(entity).merged_with(chosen)

    # A constraint outranks both judgements. AASd-108: every child of a
    # SubmodelElementList has the list's declared type, whatever anyone thinks.
    if container is not None and container.model_type == "SubmodelElementList":
        declared = container.type_value_list_element
        if declared and declared != spec.model_type:
            if declared in CREATABLE_TYPES:
                warnings.append(
                    f"{container.path} is a SubmodelElementList of {declared}; "
                    f"creating a {declared} rather than the {spec.model_type} chosen"
                )
                spec = ElementSpec(
                    model_type=declared, value_type=spec.value_type,
                    min_value=spec.min_value, max_value=spec.max_value,
                    content_type=spec.content_type, reason=spec.reason,
                )
            else:
                warnings.append(
                    f"{container.path} declares children of type {declared}, "
                    "which no extracted value can be created as"
                )

    if spec.model_type not in CREATABLE_TYPES:
        warnings.append(
            f"{spec.model_type!r} is not a modelType an extracted value can be "
            "created as; creating a Property instead"
        )
        spec = ElementSpec(model_type="Property",
                           value_type=infer_value_type(entity.value, entity.unit))

    # A Range with one bound is not a Range. Better a Property holding the text
    # than an element whose min is null and whose max a reader has to guess.
    if spec.model_type == "Range" and (spec.min_value is None or spec.max_value is None):
        warnings.append(
            "Range was chosen but the value gives no min and max; "
            "creating a Property instead"
        )
        spec = ElementSpec(model_type="Property",
                           value_type=infer_value_type(entity.value, entity.unit))

    if spec.model_type in ("Property", "Range") and not spec.value_type:
        spec.value_type = infer_value_type(entity.value, entity.unit)
    if spec.model_type == "File" and not spec.content_type:
        spec.content_type = "application/octet-stream"
    return spec, warnings


def safe_id_short(raw: str, fallback: str = "Entity") -> str:
    """Turn an arbitrary label into an idShort AASd-002 accepts."""
    import unicodedata

    text = unicodedata.normalize("NFKD", str(raw or "")).encode("ascii", "ignore").decode()
    text = _IDSHORT_SAFE.sub("_", text).strip("_")
    text = re.sub(r"_+", "_", text)
    if not text or not text[0].isalpha():
        text = f"{fallback}_{text}".strip("_")
    return text[:128]


# ---------------------------------------------------------------------------
# unit comparison
# ---------------------------------------------------------------------------

# The same unit is spelled two ways in the reference repository: IEC 61360
# long form in the ConceptDescriptions (`kilogram`, `percent`, `ampereHour`,
# `minuteUnitOfTime`) and the symbol in documents and in other CDs (`kg`, `V`,
# `mm`). Comparing them textually reports a mismatch between `kg` and
# `kilogram`, and a false warning costs a correct write when the caller runs
# `aas populate --skip-warnings`.
#
# This is an alias table, not a unit library. It normalises spellings; it does
# not convert, so `kWh` against `Wh` still warns -- which is the case that
# matters.
_UNIT_ALIASES = {
    # mass
    "kilogram": "kg", "gram": "g", "milligram": "mg", "tonne": "t",
    # electrical
    "volt": "v", "millivolt": "mv", "kilovolt": "kv",
    "ampere": "a", "milliampere": "ma",
    "watt": "w", "milliwatt": "mw", "kilowatt": "kw",
    "amperehour": "ah", "milliamperehour": "mah",
    "watthour": "wh", "kilowatthour": "kwh",
    "ohm": "ohm", "Ω": "ohm", "Ω": "ohm",
    "henry": "h_ind", "millihenry": "mh", "farad": "f_cap", "microfarad": "µf",
    # time -- `a` is the SI symbol for a year and appears in datasheets
    "second": "s", "minute": "min", "minuteunitoftime": "min",
    "hour": "h", "day": "d", "week": "wk", "month": "mo", "months": "mo",
    "year": "y", "years": "y", "annum": "y", "a": "y",
    # geometry
    "metre": "m", "meter": "m", "millimetre": "mm", "millimeter": "mm",
    "centimetre": "cm", "centimeter": "cm", "kilometre": "km", "kilometer": "km",
    # dimensionless and temperature
    "percent": "%", "percentage": "%", "cycle": "cycles", "piece": "pcs", "pieces": "pcs",
    "degreecelsius": "°c", "celsius": "°c", "degc": "°c", "c": "°c",
    "degreefahrenheit": "°f", "fahrenheit": "°f", "degf": "°f",
    "kelvin": "k",
}

# Characters that render identically but are different code points. The
# repository contains both U+00BA MASCULINE ORDINAL and U+00B0 DEGREE SIGN in
# what are meant to be the same unit.
_UNIT_LOOKALIKES = str.maketrans({
    "º": "°",   # MASCULINE ORDINAL -> DEGREE SIGN
    "Ω": "Ω",   # OHM SIGN -> GREEK CAPITAL OMEGA
    "μ": "µ",   # GREEK SMALL MU -> MICRO SIGN
})


def _normalise_unit(unit: str) -> str:
    """Reduce a unit to a canonical spelling. Does not convert magnitudes."""
    import unicodedata

    text = unicodedata.normalize("NFC", (unit or "").strip())
    text = text.translate(_UNIT_LOOKALIKES)
    text = re.sub(r"[\s._]", "", text).lower()
    if not text:
        return ""
    if text in _UNIT_ALIASES:
        return _UNIT_ALIASES[text]
    # `degC`, `deg C`, `°C` all reach here as some form of degree-c.
    stripped = text.lstrip("°")
    return _UNIT_ALIASES.get(stripped, text)


def unit_warning(entity_unit: str, target_unit: str) -> Optional[str]:
    """Report a unit disagreement between the source document and the element.

    Only when both sides declare one: an absent unit is unknown, not zero, and
    warning about silence would bury the warnings that matter. Comparison is
    textual and deliberately shallow -- this flags "kWh into Wh" for a human,
    it does not convert.
    """
    left, right = _normalise_unit(entity_unit), _normalise_unit(target_unit)
    if not left or not right or left == right:
        return None
    return f"unit mismatch: document says {entity_unit!r}, element declares {target_unit!r}"


# ---------------------------------------------------------------------------
# the match
# ---------------------------------------------------------------------------

def partition(candidates: Sequence[Candidate]) -> Tuple[List[Candidate], List[Candidate]]:
    """Split retrieval hits into things to write into and things to create in."""
    leaves = [c for c in candidates if c.model_type not in CONTAINER_TYPES + NON_CANDIDATE_TYPES]
    containers = [c for c in candidates if c.model_type in CONTAINER_TYPES]
    return leaves, containers


def band_for(score: float, theta_high: float, theta_low: float) -> str:
    if score >= theta_high:
        return "auto"
    if score < theta_low:
        return "below"
    return "arbitrated"


def _as_candidate(hit: Any) -> Candidate:
    return Candidate(
        path=hit.node.path,
        id_short=hit.node.id_short,
        model_type=hit.node.model_type,
        score=hit.score,
        parent=hit.parent.path if hit.parent else None,
        unit=hit.node.unit,
        definition=hit.node.definition or hit.node.description,
        type_value_list_element=hit.node.type_value_list_element,
    )


def candidates_for(
    index: Index, query_vector: Any, top_k: int, *, host_k: int = 3
) -> List[Candidate]:
    """Candidates for both questions the decision asks, not just the first.

    `decide` asks two things of one ranking: which element to write into, and
    -- when nothing fits -- which collection should host a new one. A single
    top-k answers only the first. The nearest neighbours of an entity are
    leaves, because an entity *is* a leaf; a collection's descriptor is its
    name and its parent, which no value-shaped query ranks highly.

    Measured on a live run rather than argued: of 72 skipped decisions, most
    carried five candidates and **zero containers**, so the hosting fallback
    could not fire even when the model would have placed the entity happily.
    The create path was starved, not declining.

    So containers are retrieved under their own filter and appended. This costs
    no model call and no request -- `search` is a masked dot product over
    vectors already resident -- and it leaves the leaf ranking untouched:
    `partition` splits by type, `leaves[0]` is still the best leaf, and
    arbitration still sees leaves only.
    """
    hits = search(index, query_vector, top_k=top_k)
    out = [_as_candidate(h) for h in hits]

    if host_k > 0:
        seen = {c.path for c in out}
        # One pass per container type: `Filter.model_type` is a single value,
        # and two masked dot products are cheaper than making it a set.
        for kind in CONTAINER_TYPES:
            for hit in search(
                index, query_vector, top_k=host_k, where=Filter(model_type=kind)
            ):
                if hit.node.path in seen:
                    continue
                seen.add(hit.node.path)
                out.append(_as_candidate(hit))
    return out


def decide(
    entity: Entity,
    candidates: Sequence[Candidate],
    *,
    theta_high: float = THETA_HIGH,
    theta_low: float = THETA_LOW,
    arbitrate: Optional[Any] = None,
    create_in: Optional[str] = None,
    host: Optional[Any] = None,
) -> Decision:
    """Turn one entity's candidates into one plan entry.

    ``arbitrate`` and ``host`` are callables so the decision logic is testable
    without a model: the bands, the fallbacks and the warnings are the part that
    has to be right, and none of them should depend on a network call.
    """
    leaves, containers = partition(candidates)
    shown = [
        {"path": c.path, "score": round(c.score, 4), "modelType": c.model_type, "unit": c.unit}
        for c in candidates
    ]

    def skip(reason: str, band: str, score: float = 0.0) -> Decision:
        return Decision(
            entity=entity.entity, value=entity.value, unit=entity.unit, source=entity.source,
            action="skip", band=band, reason=reason, score=score,
            margin=margin_of(leaves), candidates=shown,
        )

    def write(target: Candidate, band: str, reason: str) -> Decision:
        warnings = []
        note = unit_warning(entity.unit, target.unit)
        if note:
            warnings.append(note)
        if entity.value is None:
            warnings.append("entity carries no value; nothing would be written")
        if not entity.source:
            warnings.append("no source recorded for this value")

        # Four types have a structured `$value` rather than a scalar, and a
        # scalar sent to one draws HTTP 500 from BaSyx with no reason given.
        # The parts are worked out *here* rather than in `populate` for the
        # same reason the type is: a plan is what a person reviews, and a
        # judgement made after they approved the table is one they did not see.
        shape = infer_element_type(entity)
        min_value = max_value = content_type = None
        if target.model_type == "Range":
            if shape.model_type == "Range":
                min_value, max_value = shape.min_value, shape.max_value
            else:
                warnings.append(
                    f"{target.path} is a Range, but {entity.value!r} gives no "
                    "min and max; nothing can be written"
                )
        elif target.model_type in ("File", "Blob"):
            content_type = shape.content_type or "application/octet-stream"

        return Decision(
            entity=entity.entity, value=entity.value, unit=entity.unit, source=entity.source,
            action="write", band=band, reason=reason, path=target.path,
            model_type=target.model_type, id_short=target.id_short,
            value_type=infer_value_type(entity.value, entity.unit),
            min_value=min_value, max_value=max_value, content_type=content_type,
            score=target.score,
            margin=margin_of(leaves), warnings=warnings, candidates=shown,
        )

    chosen: Optional[Candidate] = None
    band = "none"
    reason = "no candidate element"

    if leaves:
        best = leaves[0]
        band = band_for(best.score, theta_high, theta_low)
        if band == "auto":
            return write(best, band, f"similarity {best.score:.4f} >= {theta_high}")
        if band == "arbitrated" and arbitrate is not None:
            picked = arbitrate(entity, leaves)
            if picked is not None:
                chosen, reason = picked
                if chosen is not None:
                    return write(chosen, "arbitrated", reason)
            else:
                reason = "arbitration returned no choice"
        elif band == "arbitrated":
            # No arbitrator available: the honest outcome is an unresolved
            # entity, not a coin flip dressed as a decision.
            return skip(
                f"similarity {best.score:.4f} is in the arbitration band "
                f"[{theta_low}, {theta_high}) and no arbitration model was available",
                "arbitrated",
                best.score,
            )
        else:
            reason = f"similarity {best.score:.4f} < {theta_low}"

    # Nothing to write into. Ask where it could be created instead -- and, in
    # the same round trip, what kind of element it should be.
    target_container: Optional[Candidate] = None
    host_reason = ""
    chosen: Optional[ElementSpec] = None
    if host is not None and containers:
        picked = host(entity, containers)
        if picked is not None:
            # Two shapes are accepted: `(container, reason)` and
            # `(container, reason, spec)`. The type is an addition to the
            # question, not a new obligation on everything that answers it, and
            # keeping the short form working is what lets the bands, the
            # fallbacks and the warnings stay testable with a two-line stub.
            target_container, host_reason = picked[0], picked[1]
            if len(picked) > 2:
                chosen = picked[2]
    if target_container is None and create_in:
        target_container = Candidate(
            path=create_in, id_short=create_in.rsplit(".", 1)[-1],
            model_type="SubmodelElementCollection", score=0.0, parent=None,
        )
        host_reason = f"no element matched; --create-in {create_in}"

    if target_container is None:
        return skip(reason, band if band != "none" else "none",
                    leaves[0].score if leaves else 0.0)

    spec, warnings = resolve_spec(entity, target_container, chosen)
    if not entity.source:
        warnings.append("no source recorded for this value")

    # AASd-120: a child of a SubmodelElementList has no idShort -- it is
    # addressed by position. Emitting one there is among the violations that
    # make a submodel BaSyx accepted unreadable to the strict Python SDK.
    in_list = target_container.model_type == "SubmodelElementList"

    placement = host_reason or reason
    return Decision(
        entity=entity.entity, value=entity.value, unit=entity.unit, source=entity.source,
        action="create", band="hosted",
        reason=f"{placement}; {spec.reason}" if spec.reason else placement,
        path=target_container.path, model_type=spec.model_type,
        container_type=target_container.model_type,
        id_short=None if in_list else safe_id_short(entity.entity),
        value_type=spec.value_type,
        min_value=spec.min_value, max_value=spec.max_value,
        content_type=spec.content_type,
        score=leaves[0].score if leaves else 0.0, margin=margin_of(leaves),
        warnings=warnings, candidates=shown,
    )


# ---------------------------------------------------------------------------
# applying a decision
# ---------------------------------------------------------------------------

# Element types that hold no `$value` at all. A write onto one is refused
# before it is sent, because BaSyx answers it with a bare HTTP 500 that names
# no reason -- and an agent reading that 500 has nothing to act on. The wizard
# route has refused these since 3.21; the CLI, which is what the agent actually
# runs, did not, and a live run spent three writes and three 500s finding out.
NOT_WRITABLE = CONTAINER_TYPES + NON_CANDIDATE_TYPES


def value_for_write(decision: Decision) -> Any:
    """The ``$value`` payload the target element's modelType requires.

    Five of the writable types serialise their value as an object under AAS
    Part 2 ValueOnly, not as a scalar, and BaSyx answers a scalar sent to one
    with HTTP 500 and no reason. Both apply paths had this half-right and
    differently: the CLI handled MultiLanguageProperty and nothing else, the
    wizard handled it in the *element* shape rather than the ValueOnly shape --
    which is the exact 500 `lang_strings` exists to prevent. One function now,
    so there is one place for the next type to be got right in.
    """
    value = decision.value
    model_type = decision.model_type or "Property"

    #  is the band the create branch sets and nothing else does, so a
    # *write* carrying it is a plan that was mangled between being made and
    # being applied -- which is exactly what the wizard's table round trip did
    # until the field collision behind it was fixed. Caught by the band rather
    # than the type, because the mangling also rewrote the type to Property.
    if decision.action == "write" and decision.band == "hosted":
        raise MatchError(
            f"{decision.path} was chosen as a *home* for a new element, not as "
            "a slot to write into, so this decision has to be a create. "
            "Re-run the match to rebuild the plan."
        )

    if model_type in NOT_WRITABLE:
        raise MatchError(
            f"{decision.path} is a {model_type}, which holds no value. "
            + ("An Operation is invoked, not written to."
               if model_type == "Operation"
               else "To put a value in here the decision has to be a create, "
                    "not a write.")
        )

    if model_type == "MultiLanguageProperty":
        return lang_strings(str(value))
    if model_type == "Range":
        if decision.min_value is None or decision.max_value is None:
            raise MatchError(
                f"{decision.path} is a Range and the plan carries no min and max "
                f"for {value!r}; edit the plan or write it with `aas set --json-value`"
            )
        return {"min": decision.min_value, "max": decision.max_value}
    if model_type in ("File", "Blob"):
        return {
            "contentType": decision.content_type or "application/octet-stream",
            "value": value,
        }
    if model_type == "ReferenceElement":
        return external_reference(str(value))
    return value


def element_for_create(decision: Decision) -> Dict[str, Any]:
    """Build the element a `create` decision describes.

    Everything type-specific was settled in `decide` and is in the plan, so
    this is assembly, not judgement -- which is the property that makes the
    reviewed table an honest record of what will happen.
    """
    model_type = decision.model_type or "Property"
    value: Any = decision.value

    if model_type == "MultiLanguageProperty" and value is not None:
        # The *element* shape here, not ValueOnly: this is POSTed whole, not
        # PATCHed into a `$value`. The two are different serialisations of the
        # same data and each endpoint accepts exactly one.
        value = [
            {"language": language, "text": text}
            for entry in lang_strings(str(value))
            for language, text in entry.items()
        ]

    return make_element(
        model_type,
        id_short=decision.id_short,
        value=value,
        value_type=decision.value_type or "xs:string",
        description=decision.source or None,
        min_value=decision.min_value,
        max_value=decision.max_value,
        content_type=decision.content_type,
    )


# ---------------------------------------------------------------------------
# arbitration
# ---------------------------------------------------------------------------

ARBITRATE_PROMPT = """You map entities extracted from a supplier document onto \
elements of an Asset Administration Shell.

Given a QUERY entity and a list of CANDIDATE elements, choose the single \
candidate that is the correct semantic match. The query's value will be written \
into whichever you choose, so a wrong choice writes wrong data into a machine's \
model. Be strict: if no candidate is right, answer -1.

Weigh `within` heavily. It names the collection that contains the candidate, \
which is the semantic boundary the element lives in. An entity that belongs to \
that boundary is far more likely to be the right match than one that merely \
shares a word with the candidate's name.

Return ONLY JSON:
{"index": <int, or -1>, "reason": "<one short sentence>"}"""

HOST_PROMPT = """You map entities extracted from a supplier document onto an \
Asset Administration Shell.

No existing element matches this QUERY entity, so it has to be created. Answer \
two things about it.

**Where.** Choose the single CANDIDATE collection that should contain it. Pick \
the collection whose subject the entity belongs to -- a charging power belongs \
among technical properties, a contact address does not. If none of them is a \
defensible home, answer index -1: creating an element in the wrong collection \
is worse than reporting that there is no place for it.

**What.** Choose the modelType that can actually hold this value:

* `Property` -- one scalar. The default; choose it whenever the others do not \
  clearly apply.
* `Range` -- the value is an interval with two ends ("250-550", "-20 to +60"). \
  Give `min` and `max` separately, as bare numbers with no unit.
* `MultiLanguageProperty` -- the value is human-readable text that is or should \
  be given per language. A name, a designation, a description.
* `File` -- the value is a path or URL to a document. Give `contentType`.
* `ReferenceElement` -- the value identifies another element or a global \
  concept, and is meant to be followed rather than read.

SHAPE is what reading the value alone suggests. It has no idea what the entity \
*means*, which you do -- correct it when the meaning says otherwise, and keep \
it when it does not.

If a candidate reports `holds`, it is a list and every child of it must have \
that modelType. Choosing it means choosing that type.

Return ONLY JSON:
{"index": <int, or -1>, "modelType": "<one of the five above>", \
"valueType": "<xsd type, only for Property or Range>", \
"min": "<only for Range>", "max": "<only for Range>", \
"contentType": "<only for File>", "reason": "<one short sentence>"}"""


# What an arbitration or a hosting answer must look like. `index` is the whole
# answer for arbitration; the rest is what hosting adds, and is nullable rather
# than absent because `strict` mode requires every property to be listed.
CHOICE_SCHEMA = {
    "type": "object",
    "additionalProperties": False,
    "required": ["index", "reason", "modelType", "valueType", "min", "max",
                 "contentType"],
    "properties": {
        "index": {"type": "integer"},
        "reason": {"type": "string"},
        "modelType": {"type": ["string", "null"], "enum": [*CREATABLE_TYPES, None]},
        "valueType": {"type": ["string", "null"]},
        "min": {"type": ["string", "null"]},
        "max": {"type": ["string", "null"]},
        "contentType": {"type": ["string", "null"]},
    },
}


def _ask(provider: Any, config: Any, system: str, payload: Dict[str, Any]) -> Dict[str, Any]:
    """One JSON question to the model, tolerant of the shapes models return.

    The schema is enforced where the endpoint supports it and the tolerant
    parse below stays regardless -- see `providers.ask_json` for why keeping
    both is the point rather than a hedge.
    """
    from ..kernel.providers import ask_json

    completion = ask_json(
        provider,
        [
            {"role": "system", "content": system},
            {"role": "user", "content": json.dumps(payload, ensure_ascii=False, indent=2)},
        ],
        config,
        schema=CHOICE_SCHEMA,
        name="choice",
    )
    text = (completion.text or "").strip()
    try:
        return json.loads(text)
    except json.JSONDecodeError:
        # Models fence JSON, or prefix it with a sentence. Take the outermost
        # object rather than failing the whole entity over formatting.
        found = re.search(r"\{[\s\S]*\}", text)
        if not found:
            raise MatchError(f"arbitration returned no JSON: {text[:160]!r}")
        return json.loads(found.group(0))


def margin_of(leaves: Sequence[Candidate]) -> float:
    return round(leaves[0].score - leaves[1].score, 4) if len(leaves) > 1 else 0.0


def _payload(entity: Entity, candidates: Sequence[Candidate]) -> Dict[str, Any]:
    return {
        "QUERY": {
            "name": entity.entity,
            "description": entity.description[:400],
            "unit": entity.unit,
        },
        "CANDIDATES": [
            {
                "index": i,
                "idShort": c.id_short,
                "path": c.path,
                "within": c.parent or "<submodel root>",
                "meaning": (c.definition or "")[:300],
                "unit": c.unit,
                "similarity": round(c.score, 4),
                # Only a SubmodelElementList has one, and where it does the
                # child's type is settled by AASd-108 rather than by choice.
                # Shown so the model does not propose a type the list forbids.
                **({"holds": c.type_value_list_element}
                   if c.type_value_list_element else {}),
            }
            for i, c in enumerate(candidates)
        ],
    }


def make_host_chooser(provider: Any, config: Any, prompt: str = HOST_PROMPT):
    """Choose the container *and* the element type, in one round trip.

    Separate from `make_arbitrator` because the two questions have different
    answers, not because they are asked differently. Arbitration picks an
    element that already exists, so its whole answer is an index; hosting
    creates one, so it has to say what is being created. Folding the second
    question into the call that was already being made is why typing costs no
    extra tokens: the middle band is one model call per unplaced entity either
    way.

    The model's answer is filtered against `CREATABLE_TYPES` rather than
    trusted. A model that replies `SubmodelElementCollection` -- which happens,
    because a collection is a plausible-sounding container for anything -- would
    otherwise author an empty collection and drop the value it was given.
    """

    def host(
        entity: Entity, candidates: Sequence[Candidate]
    ) -> Optional[Tuple[Optional[Candidate], str, Optional[ElementSpec]]]:
        payload = _payload(entity, candidates)
        # The deterministic read is shown, not hidden. It grounds the answer in
        # the value the model is typing, and it means a model that replies with
        # an index and nothing else still gets the inferred type rather than
        # silently reverting to Property.
        shape = infer_element_type(entity)
        payload["SHAPE"] = {
            "modelType": shape.model_type,
            "valueType": shape.value_type,
            "min": shape.min_value,
            "max": shape.max_value,
            "contentType": shape.content_type,
            "why": shape.reason or "value reads as a plain scalar",
        }
        try:
            answer = _ask(provider, config, prompt, payload)
        except Exception as exc:  # noqa: BLE001 - any failure is "did not decide"
            return None, f"hosting failed: {exc}", None
        try:
            index = int(answer.get("index", -1))
        except (TypeError, ValueError):
            index = -1
        reason = str(answer.get("reason") or "").strip()[:200]
        if not (0 <= index < len(candidates)):
            return None, reason or "no collection is a defensible home", None

        model_type = str(answer.get("modelType") or "").strip()
        notes = []
        if model_type and model_type not in CREATABLE_TYPES:
            notes.append(f"model asked for {model_type!r}, which is not creatable here")
            model_type = ""
        spec = ElementSpec(
            model_type=model_type,
            value_type=(str(answer["valueType"]).strip()
                        if answer.get("valueType") else None),
            min_value=None if answer.get("min") is None else str(answer["min"]).strip(),
            max_value=None if answer.get("max") is None else str(answer["max"]).strip(),
            content_type=(str(answer["contentType"]).strip()
                          if answer.get("contentType") else None),
            reason="; ".join(notes),
        )
        return candidates[index], reason or "chosen by hosting", spec

    return host


def make_arbitrator(provider: Any, config: Any, prompt: str = ARBITRATE_PROMPT):
    """A callable the decision logic can use, or not, without knowing about models."""

    def arbitrate(
        entity: Entity, candidates: Sequence[Candidate]
    ) -> Optional[Tuple[Optional[Candidate], str]]:
        try:
            answer = _ask(provider, config, prompt, _payload(entity, candidates))
        except Exception as exc:  # noqa: BLE001 - any failure is "did not decide"
            return None, f"arbitration failed: {exc}"
        try:
            index = int(answer.get("index", -1))
        except (TypeError, ValueError):
            index = -1
        reason = str(answer.get("reason") or "").strip()[:200]
        if 0 <= index < len(candidates):
            return candidates[index], reason or "chosen by arbitration"
        return None, reason or "arbitration found no suitable candidate"

    return arbitrate
