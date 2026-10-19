"""Explicit project mapping checks, separate from official AAS conformance."""
import math

UNITS = {"Nm": ("torque", 1.), "Ncm": ("torque", .01), "s": ("time", 1.),
         "ms": ("time", .001), "rad": ("angle", 1.), "deg": ("angle", math.pi / 180.)}
CANONICAL = {"torque": "N*m", "time": "s", "angle": "rad"}


def mapping_findings(case):
    """Return reject/clarify findings against supplied source facts, not truth inferred by an LLM."""
    findings = []

    def add(code, decision="reject"):
        if not any(f["code"] == code for f in findings):
            findings.append({"code": code, "decision": decision})

    document, source = case["aas"], case["source_facts"]
    shell = document["assetAdministrationShells"][0]
    if shell["assetInformation"]["globalAssetId"] != source["asset_id"]:
        add("ASSET_ID")
    submodel = document["submodels"][0]
    expected_ref = [{"type": "Submodel", "value": submodel["id"]}]
    if shell["submodels"][0].get("keys") != expected_ref:
        add("REFERENCE")
    if not source.get("provenance"):
        add("PROVENANCE_MISSING", "clarify")
    if source.get("quality") != "Good":
        add("QUALITY_BAD")
    if source.get("age_seconds", float("inf")) > source["maximum_age_seconds"]:
        add("STALE")
    if case["mapping"].get("utc_timestamp_required") and not source.get("timestamp_timezone"):
        add("TIMEZONE_UNKNOWN", "clarify")
    elements = {e["idShort"]: e for e in submodel["submodelElements"]}
    for row in case["mapping"]["signals"]:
        element = elements.get(row["target"])
        if element is None:
            add("TARGET_MISSING")
            continue
        qualifiers = {q["type"]: q.get("value") for q in element.get("qualifiers", [])}
        keys = element.get("semanticId", {}).get("keys", [])
        if keys != [{"type": "GlobalReference", "value": row["semantic_id"]}]:
            add("SEMANTIC_ID")
        if qualifiers.get("SourceKind") != source["kind"]:
            add("SOURCE_CLASS")
        if qualifiers.get("Frame") != row["source_frame"]:
            add("FRAME")
        kind, factor = UNITS.get(row["source_unit"], (None, None))
        if kind != row["quantity"] or qualifiers.get("Unit") != CANONICAL.get(row["quantity"]):
            add("UNIT")
            continue
        if not math.isclose(row["scale"], factor, rel_tol=1e-12, abs_tol=1e-12) or row["offset"] != 0:
            add("UNIT_CONVERSION")
        expected = row["source_value"] * factor
        try:
            value = float(element["value"])
        except (KeyError, TypeError, ValueError):
            add("VALUE")
            continue
        if not math.isfinite(value) or not math.isclose(value, expected, rel_tol=1e-9, abs_tol=1e-9):
            add("VALUE")
        lower, upper = row["valid_range"]
        if not lower <= value <= upper:
            add("RANGE")
    return findings


def assess(case, schema_ok):
    if not schema_ok:
        return {"decision": "reject", "codes": ["AAS_SCHEMA"]}
    findings = mapping_findings(case)
    decision = "reject" if any(f["decision"] == "reject" for f in findings) else "clarify" if findings else "accept"
    return {"decision": decision, "codes": sorted(f["code"] for f in findings)}
