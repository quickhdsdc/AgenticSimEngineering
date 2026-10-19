"""A source-labelled maintenance application profile for the existing instance.

Reuses IDTA 02018 element semantics, but does not claim full SMT conformance or
invent manufacturer maintenance intervals / RUL predictions. Never touches OPC.
"""

import copy
import json
from pathlib import Path

from ...aasctl import smt

SHELL_ID = "https://fraunhoferIPA/ids/aas/FestoWorkCenterStamping_381"
MAINTENANCE_ID = SHELL_ID + "/sm/MaintenanceInstructions"
CONTEXT_ID = SHELL_ID + "/sm/ProductionContext"
PROFILE = "urn:a4s:submodel:maintenance-instructions-profile:1:0"
SOURCE = "Anleitung - Festo CP Lab.pptx, slides 4-9, 17-18; Festo CP & Module.pptx, slide 1"
STEPS = (
    ("CaptureEvidence", "Capture evidence without operating equipment",
     "Record the station fault word, signal timestamps and communication errors. "
     "An application xError=false is not proof that the workstation is healthy. "
     "A simulated fault is not an OPC observation."),
    ("OperatorAssessment", "Request qualified operator assessment",
     "DEMO FICTION: a pneumatic connector leak lowers synthetic pressure from 6.0 bar; "
     "below 4.8 bar for 3 seconds raises the scenario alarm. These are not calibrated PLC limits. "
     "The reference describes a press controlled through RevPi and a turning-over module "
     "controlled through a microcontroller, each communicating with a Siemens PLC. "
     "Ask the qualified operator to assess readiness and communication using the approved "
     "site procedure. Do not infer a root cause or send reset/start commands."),
    ("VerifyRecovery", "Separate physical recovery from business recovery",
     "After the operator confirms the physical condition, collect fresh independent evidence. "
     "Resolve the MES issue only with an explicit explanation and read back the order. "
     "A closed issue or changed schedule does not prove that the machine is repaired."),
)


def _pick(parent, name):
    return next(el for el in parent.get("value", parent.get("submodelElements", []))
                if el.get("idShort") == name)


def _value(element, value):
    el = copy.deepcopy(element)
    el["value"] = ([{"language": "en", "text": value}]
                   if el["modelType"] == "MultiLanguageProperty" else str(value))
    return el


def prop(name, value):
    return {"modelType": "Property", "idShort": name, "valueType": "xs:string", "value": value}


def build(template_path: Path):
    document = json.loads(template_path.read_text(encoding="utf-8"))
    template = next(sm for sm in document["submodels"] if sm["idShort"] == "MaintenanceInstructions")
    result = smt.instantiate(template, submodel_id=MAINTENANCE_ID,
                             id_short="MaintenanceInstructions", include_optional=True).submodel
    interval = _pick(result, "MaintenanceInstructionsForSpecificInterval__00__")
    basic = _pick(interval, "BasicMaintenanceInformation")
    basic["value"] = [
        _value(_pick(basic, "MaintenanceID"), "A4S-FESTO-381-CONDITION-REVIEW"),
        _value(_pick(basic, "NameOfMaintenance"), "Condition-triggered production readiness review"),
        _value(_pick(basic, "SourceOfMaintenanceInstructions"),
               SOURCE + ". A4S-authored review workflow, not manufacturer-approved maintenance instructions."),
        _value(_pick(basic, "SafetyRegulationsToBeObserved"),
               "Qualified operator and approved site procedures required for physical intervention. "
               "This demo authorizes MES business actions only; no PLC reset, start or parameter write."),
    ]
    steps = _pick(interval, "ListMaintenanceSteps")
    exemplar = _pick(steps, "MaintenanceStep__00__")
    steps["value"] = []
    for number, (code, title, instruction) in enumerate(STEPS, 1):
        step = copy.deepcopy(exemplar)
        step["idShort"] = f"MaintenanceStep{number:02}"
        step["value"] = [
            _value(_pick(exemplar, "MaintenanceStepID"), code),
            _value(_pick(exemplar, "MaintenanceStepName"), title),
            _value(_pick(exemplar, "InstructionMaintenanceStep"), instruction),
        ]
        steps["value"].append(step)
    interval["idShort"] = "ConditionTriggeredReview"
    interval["value"] = [basic, steps]
    result["semanticId"] = {"type": "ExternalReference", "keys": [
        {"type": "GlobalReference", "value": PROFILE}]}
    result["qualifiers"] = [{"type": "basedOnTemplate", "valueType": "xs:anyURI",
                              "value": template["id"]}]
    result["description"] = [{"language": "en", "text":
        "A4S application profile based on IDTA 02018, with reused element semantics. "
        "Not a claim of complete IDTA template conformance, manufacturer approval, "
        "predictive maintenance, or a validated physical recovery procedure. "
        "No manufacturer interval or measured RUL is asserted. Scenario thresholds "
        "are explicitly fictional, not calibrated equipment limits."}]
    result["submodelElements"] = [interval, prop("ProfileStatus", "EngineeringReviewRequired"),
                                    prop("SourceReferences", SOURCE)]
    return result


def production_context(line_code, backup_code, product_code, endpoint):
    """Store equipment identity and service location, never order IDs or state."""
    return {
        "modelType": "Submodel", "kind": "Instance", "id": CONTEXT_ID,
        "idShort": "ProductionContext",
        "description": [{"language": "en", "text":
            "Demo business references. Query OpenMES for current state. Backup capacity "
            "and the fault scenario are simulated; neither implies live PLC mapping."}],
        "submodelElements": [prop(name, str(value)) for name, value in {
            "MESBaseUrl": endpoint, "MESLineCode": line_code,
            "MESBackupLineCode": backup_code, "MESProductTypeCode": product_code,
            "EvidenceMode": "SIMULATION",
            "MaintenanceSubmodelId": MAINTENANCE_ID,
            "DemoProduct": "Smartphone housing (fictional use of A4S-DEMO-PART)",
            "DemoRoute": "Stamp / press, turn over 180 degrees, fictional inspection handoff",
            "KnowledgeCommand": "python -m a4s.machines.festo_cp_lab.knowledge --query 'pressure leakage maintenance'",
        }.items()],
    }
