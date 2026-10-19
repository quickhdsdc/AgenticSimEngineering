"""Constructors for the AAS metamodel's submodel-element types.

This module is the reason ``aas el add`` needs no per-template code. The
reference implementation shipped one tool per authoring intent
(``aas_from_smt``, ``aas_populate_inputs``, ...), so a submodel template it
had never seen was unreachable. Here the *metamodel* is the API surface:
every ``modelType`` in IEC 63278 Part 1 that an agent can legitimately author
has a factory, and templates become data rather than code.

Value serialisation follows AAS Part 2 ``$value`` semantics, which is what
BaSyx accepts on PATCH.
"""

from __future__ import annotations

from typing import Any, Dict, List, Optional

# xsd types an agent may assign to a Property. Kept explicit rather than
# free-form: this list is the natural enum to inject into a constrained
# decoder, so an invalid valueType is unrepresentable rather than merely
# rejected downstream.
# The complete DataTypeDefXsd enumeration from AAS Part 1. All thirty, not a
# convenient subset: an incomplete list makes `aas validate` cry wolf on a
# valid IDTA template and makes `aas el add` refuse a legal type. Found by
# running the validator over the ZVEI TechnicalData template, which declares
# `xs:gYearMonth` for a warranty period.
VALUE_TYPES = (
    "xs:anyURI", "xs:base64Binary", "xs:boolean", "xs:byte",
    "xs:date", "xs:dateTime", "xs:decimal", "xs:double", "xs:duration",
    "xs:float", "xs:gDay", "xs:gMonth", "xs:gMonthDay", "xs:gYear", "xs:gYearMonth",
    "xs:hexBinary", "xs:int", "xs:integer", "xs:long",
    "xs:negativeInteger", "xs:nonNegativeInteger", "xs:nonPositiveInteger",
    "xs:positiveInteger", "xs:short", "xs:string", "xs:time",
    "xs:unsignedByte", "xs:unsignedInt", "xs:unsignedLong", "xs:unsignedShort",
)

# modelType values this CLI can construct. Deliberately a subset of the
# metamodel: types an agent has no business authoring by hand
# (AnnotatedRelationshipElement, Capability) are omitted until a use case
# forces them in.
ELEMENT_TYPES = (
    "Property",
    "MultiLanguageProperty",
    "Range",
    "Blob",
    "File",
    "ReferenceElement",
    "RelationshipElement",
    "SubmodelElementCollection",
    "SubmodelElementList",
    "Operation",
    "Entity",
)

KIND_VALUES = ("Instance", "Template")


class ModelError(ValueError):
    """Raised when the requested element cannot be constructed."""


# ---------------------------------------------------------------------------
# references
# ---------------------------------------------------------------------------

def external_reference(value: str) -> Dict[str, Any]:
    """An ExternalReference, the form a semanticId normally takes.

    BaSyx-Java has been observed emitting ExternalReference where the
    metamodel requires ModelReference; this helper only ever produces the
    form that is correct for a semanticId, so round-tripping through this
    CLI normalises rather than propagates that defect.
    """
    return {
        "type": "ExternalReference",
        "keys": [{"type": "GlobalReference", "value": value}],
    }


def model_reference(keys: List[Dict[str, str]]) -> Dict[str, Any]:
    return {"type": "ModelReference", "keys": keys}


def submodel_reference(submodel_id: str) -> Dict[str, Any]:
    return model_reference([{"type": "Submodel", "value": submodel_id}])


# ---------------------------------------------------------------------------
# qualifiers
# ---------------------------------------------------------------------------

def qualifier(kind: str, value: str, value_type: str = "xs:string") -> Dict[str, Any]:
    return {"type": kind, "valueType": value_type, "value": value}


def cardinality_qualifier(value: str) -> Dict[str, Any]:
    """The SMT cardinality declaration, in the spelling instantiation reads back.

    `kind: TemplateQualifier` is not decoration. AASd-129 says a qualifier with
    that kind may appear *only* inside a submodel whose own kind is `Template`,
    which is what makes it the marker `smt.instantiate` strips when it produces
    an instance. Emitting the cardinality without it would author a template
    whose slots look mandatory to a reader and carry nothing an instantiation
    can act on -- and 846 findings in the reference repository say the opposite
    mistake, leaving these on an instance, is the one people actually make.

    `SMT/Cardinality` is the current IDTA spelling. `smt.cardinality` reads
    three others besides, but a template written here should be written in the
    one that is not a legacy.
    """
    out = qualifier("SMT/Cardinality", value)
    out["kind"] = "TemplateQualifier"
    return out


def delegation_qualifier(url: str) -> Dict[str, Any]:
    """The ``invocationDelegation`` qualifier that binds an Operation to a service.

    This is the CSS *Service* binding in the A4S architecture: a Capability
    submodel advertises what an asset can do, an Operation is the Skill, and
    this qualifier is the endpoint the Skill actually resolves to.
    """
    return qualifier("invocationDelegation", url)


# ---------------------------------------------------------------------------
# elements
# ---------------------------------------------------------------------------

def _base(
    id_short: Optional[str],
    model_type: str,
    *,
    semantic_id: Optional[str] = None,
    description: Optional[str] = None,
    display_name: Optional[str] = None,
    qualifiers: Optional[List[Dict[str, Any]]] = None,
) -> Dict[str, Any]:
    element: Dict[str, Any] = {"modelType": model_type}
    # An SML child carries no idShort -- it is addressed by position. Emitting
    # an empty idShort there is one of the constraint violations that makes a
    # submodel unreadable by the strict Python SDK.
    if id_short:
        element["idShort"] = id_short
    if semantic_id:
        element["semanticId"] = external_reference(semantic_id)
    if description:
        element["description"] = [{"language": "en", "text": description}]
    if display_name:
        element["displayName"] = [{"language": "en", "text": display_name}]
    if qualifiers:
        element["qualifiers"] = qualifiers
    return element


def make_element(
    model_type: str,
    id_short: Optional[str] = None,
    *,
    value: Any = None,
    value_type: str = "xs:string",
    semantic_id: Optional[str] = None,
    description: Optional[str] = None,
    display_name: Optional[str] = None,
    qualifiers: Optional[List[Dict[str, Any]]] = None,
    # type-specific
    min_value: Optional[str] = None,
    max_value: Optional[str] = None,
    content_type: Optional[str] = None,
    first: Optional[Dict[str, Any]] = None,
    second: Optional[Dict[str, Any]] = None,
    type_value_list_element: Optional[str] = None,
    value_type_list_element: Optional[str] = None,
    order_relevant: bool = True,
    entity_type: str = "SelfManagedEntity",
    global_asset_id: Optional[str] = None,
    input_variables: Optional[List[Dict[str, Any]]] = None,
    output_variables: Optional[List[Dict[str, Any]]] = None,
) -> Dict[str, Any]:
    """Build one submodel element of ``model_type``.

    Every parameter maps to a metamodel attribute; nothing here is specific
    to any submodel template. Adding support for a new IDTA template is a
    matter of issuing different commands, not of extending this function.
    """
    if model_type not in ELEMENT_TYPES:
        raise ModelError(
            f"unknown modelType {model_type!r}; expected one of {', '.join(ELEMENT_TYPES)}"
        )

    element = _base(
        id_short,
        model_type,
        semantic_id=semantic_id,
        description=description,
        display_name=display_name,
        qualifiers=qualifiers,
    )

    if model_type == "Property":
        if value_type not in VALUE_TYPES:
            raise ModelError(
                f"unknown valueType {value_type!r}; expected one of {', '.join(VALUE_TYPES)}"
            )
        element["valueType"] = value_type
        element["value"] = None if value is None else str(value)

    elif model_type == "MultiLanguageProperty":
        if value is None:
            element["value"] = []
        elif isinstance(value, list):
            element["value"] = value
        else:
            element["value"] = [{"language": "en", "text": str(value)}]

    elif model_type == "Range":
        element["valueType"] = value_type
        element["min"] = min_value
        element["max"] = max_value

    elif model_type == "Blob":
        element["contentType"] = content_type or "application/octet-stream"
        if value is not None:
            element["value"] = value

    elif model_type == "File":
        element["contentType"] = content_type or "application/octet-stream"
        element["value"] = value

    elif model_type == "ReferenceElement":
        if value is None:
            raise ModelError("ReferenceElement requires --value (the target reference)")
        element["value"] = (
            value if isinstance(value, dict) else external_reference(str(value))
        )

    elif model_type == "RelationshipElement":
        if not first or not second:
            raise ModelError("RelationshipElement requires --first and --second references")
        element["first"] = first
        element["second"] = second

    elif model_type == "SubmodelElementCollection":
        element["value"] = []

    elif model_type == "SubmodelElementList":
        if not type_value_list_element:
            raise ModelError(
                "SubmodelElementList requires --type-value-list-element "
                "(the modelType every child must have)"
            )
        element["typeValueListElement"] = type_value_list_element
        element["orderRelevant"] = order_relevant
        if value_type_list_element:
            element["valueTypeListElement"] = value_type_list_element
        element["value"] = []

    elif model_type == "Operation":
        element["inputVariables"] = input_variables or []
        element["outputVariables"] = output_variables or []

    elif model_type == "Entity":
        element["entityType"] = entity_type
        element["statements"] = []
        if global_asset_id:
            element["globalAssetId"] = global_asset_id

    return element


def lang_strings(raw: str) -> List[Dict[str, str]]:
    """Parse the ``text@lang`` form into MultiLanguageProperty ValueOnly.

    Two shapes exist for the same data and only one works per endpoint. A
    *element* carries ``[{"language": "en", "text": "..."}]``; its ``$value``
    under ValueOnly serialisation carries ``[{"en": "..."}]``. BaSyx answers
    500, not 400, to the wrong one, which says nothing about what is wrong.

    A live run lost four steps here -- the shell ate the braces of the
    hand-written JSON, then the corrected JSON was the element shape and drew
    the 500. ``text@lang`` is the shorthand IDTA templates already use in their
    ``ExampleValue`` qualifiers, so this is one convention rather than a new
    one. Several translations separate with ``|``; raw JSON stays available
    through --json-value.

    It lives here rather than in the CLI because ``populate`` needs it too, and
    the two had drifted: the wizard's apply path sent the *element* shape into
    ``$value`` and drew the 500 this function exists to prevent.
    """
    out: List[Dict[str, str]] = []
    for part in raw.split("|"):
        text, sep, lang = part.rpartition("@")
        if not sep or not lang or " " in lang:
            # No usable tag -- the whole fragment is the text, in English.
            text, lang = part, "en"
        out.append({lang.strip(): text.strip()})
    return out


def operation_variable(element: Dict[str, Any]) -> Dict[str, Any]:
    """Wrap a submodel element as an Operation input/output variable."""
    return {"value": element}


# ---------------------------------------------------------------------------
# containers
# ---------------------------------------------------------------------------

def make_submodel(
    submodel_id: str,
    id_short: str,
    *,
    semantic_id: Optional[str] = None,
    kind: str = "Instance",
    description: Optional[str] = None,
) -> Dict[str, Any]:
    if kind not in KIND_VALUES:
        raise ModelError(f"kind must be one of {', '.join(KIND_VALUES)}")
    submodel = _base(id_short, "Submodel", semantic_id=semantic_id, description=description)
    submodel["id"] = submodel_id
    submodel["kind"] = kind
    submodel["submodelElements"] = []
    return submodel


def make_shell(
    shell_id: str,
    id_short: str,
    *,
    global_asset_id: Optional[str] = None,
    asset_kind: str = "Instance",
    description: Optional[str] = None,
) -> Dict[str, Any]:
    shell = _base(id_short, "AssetAdministrationShell", description=description)
    shell["id"] = shell_id
    # assetInformation is mandatory; BaSyx-Java tolerates null here but the
    # Python SDK rejects it on read, so it is always populated.
    shell["assetInformation"] = {
        "assetKind": asset_kind,
        "globalAssetId": global_asset_id or shell_id,
    }
    shell["submodels"] = []
    return shell
