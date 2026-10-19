"""Synthetic semantic challenge fixtures built with the vendored Harness factories."""
import copy
import math


def base_case():
    # Optional Harness imports only while authoring; no new Agent loop.
    from a4s.aasctl.model import make_element, make_shell, make_submodel, qualifier, submodel_reference
    asset = "urn:ase:fixture:screw-station-A"
    shell = make_shell("urn:ase:fixture:aas:screw", "ScrewStation", global_asset_id=asset)
    submodel = make_submodel("urn:ase:fixture:submodel:signals", "OperationalSignals",
                            semantic_id="urn:ase:project:OperationalSignals")
    rows = []
    for name, quantity, unit, value, canonical, frame, bounds in [
        ("Torque", "torque", "Nm", 1.4, "N*m", "shaft-clockwise-positive", [0, 2]),
        ("Elapsed", "time", "ms", 1000, "s", "cycle-relative", [0, 10]),
        ("Angle", "angle", "deg", 30, "rad", "shaft-clockwise-positive", [0, 2 * math.pi]),
    ]:
        factor = {"Nm": 1., "ms": .001, "deg": math.pi / 180.}[unit]
        semantic = "urn:ase:project:" + quantity
        submodel["submodelElements"].append(make_element("Property", name, value=value * factor,
            value_type="xs:double", semantic_id=semantic,
            qualifiers=[qualifier("Unit", canonical), qualifier("Frame", frame), qualifier("SourceKind", "synthetic_fixture")]))
        rows.append({"target": name, "quantity": quantity, "semantic_id": semantic,
                     "source_unit": unit, "source_value": value, "scale": factor, "offset": 0.,
                     "source_frame": frame, "valid_range": bounds})
    shell["submodels"] = [submodel_reference(submodel["id"])]
    # Optional AAS arrays must be omitted when empty; the official schema
    # rejects conceptDescriptions: [] even though a permissive JSON client may accept it.
    return {"aas": {"assetAdministrationShells": [shell], "submodels": [submodel]},
            "source_facts": {"asset_id": asset, "kind": "synthetic_fixture", "quality": "Good",
                             "age_seconds": 1, "maximum_age_seconds": 60, "timestamp_timezone": None,
                             "provenance": "fixture definition; source units inspired by screw-s01 README; values are synthetic"},
            "mapping": {"utc_timestamp_required": False, "signals": rows}}


def cases():
    base = base_case()
    inputs, gold = [], {}

    def add(change=None, decision="accept", codes=()):
        case = copy.deepcopy(base)
        if change:
            change(case)
        case["case_id"] = f"case-{len(inputs) + 1:03}"
        inputs.append(case)
        gold[case["case_id"]] = {"decision": decision, "codes": sorted(codes)}

    def element(case, index=0):
        return case["aas"]["submodels"][0]["submodelElements"][index]

    add()
    def centimeters(case):
        row = case["mapping"]["signals"][0]
        row.update(source_unit="Ncm", source_value=140, scale=.01)
    add(centimeters)
    add(lambda c: element(c).update(valueType="xs:banana"), "reject", ["AAS_SCHEMA"])
    add(lambda c: element(c).update(idShort="Torque invalid"), "reject", ["AAS_SCHEMA"])
    add(lambda c: c["aas"]["assetAdministrationShells"][0]["assetInformation"].update(
        globalAssetId="urn:ase:fixture:other-machine"), "reject", ["ASSET_ID"])
    add(lambda c: element(c)["semanticId"]["keys"][0].update(value="urn:ase:project:temperature"),
        "reject", ["SEMANTIC_ID"])
    add(lambda c: c["mapping"]["signals"][1].update(scale=1), "reject", ["UNIT_CONVERSION"])
    add(lambda c: element(c)["qualifiers"][0].update(value="Ncm"), "reject", ["UNIT"])
    add(lambda c: c["source_facts"].update(provenance=None), "clarify", ["PROVENANCE_MISSING"])
    add(lambda c: c["source_facts"].update(quality="Bad"), "reject", ["QUALITY_BAD"])
    add(lambda c: c["source_facts"].update(age_seconds=3600), "reject", ["STALE"])
    add(lambda c: element(c)["qualifiers"][2].update(value="measurement"), "reject", ["SOURCE_CLASS"])
    add(lambda c: c["mapping"].update(utc_timestamp_required=True), "clarify", ["TIMEZONE_UNKNOWN"])
    add(lambda c: element(c)["qualifiers"][1].update(value="shaft-counterclockwise-positive"), "reject", ["FRAME"])
    add(lambda c: c["mapping"]["signals"][0].update(valid_range=[0, 1]), "reject", ["RANGE"])
    add(lambda c: c["aas"]["assetAdministrationShells"][0]["submodels"][0]["keys"][0].update(
        value="urn:ase:fixture:submodel:missing"), "reject", ["REFERENCE"])
    return inputs, gold
