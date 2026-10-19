"""In-memory simulation profile for the selected instance; never provisions an AAS."""
import copy
from .maintenance import prop
from .opc_simulator import ENDPOINT, signal_values

DESCRIPTIONS = {
    "Press.Fault": "Collective press alarm. Inspect individual alarm, pressure, flow and safety state.",
    "Turning.Fault": "Collective turning alarm; inspect safety and module state separately.",
    "Press.Ready": "Application readiness; not permission to operate equipment.",
    "Turning.Ready": "Turning application readiness; independent of press pressure alarm.",
    "Press.Fault01": "Pneumatic supply alarm for this commissioned profile: pressure below 4.8 bar. Root cause requires pressure/flow and safety evidence. This mapping is not the physical 381 xSF01 mapping.",
    "Press.PressureBar": "Pneumatic supply pressure in bar; expected 6.0, low-pressure threshold 4.8 for this profile.",
    "Press.FlowLMin": "Supply flow in litres/minute; normal idle 12, investigate sustained values over 30 with low pressure.",
    "Safety.EmergencyStop": "True means emergency stop active. Never reset automatically; require onsite assessment.",
}


def models(context, asset_id):
    groups = {}
    for path, value in signal_values("normal").items():
        group, name = path.split(".")
        groups.setdefault(group, []).append({"modelType": "Property", "idShort": name,
            "valueType": "xs:boolean" if isinstance(value, bool) else "xs:double",
            "description": [{"language": "en", "text": DESCRIPTIONS[path]}],
            "qualifiers": [{"type": "valueSource", "valueType": "xs:string",
                            "value": ENDPOINT + "#ns=2;s=" + path}]})
    operational = {"modelType": "Submodel", "id": asset_id + "/sm/OperationalData",
        "idShort": "OperationalData", "kind": "Instance",
        "description": [{"language": "en", "text": "Authored demonstration profile; not manufacturer-validated physical PLC semantics."}],
        "submodelElements": [{"modelType": "SubmodelElementCollection", "idShort": k, "value": v}
                             for k, v in groups.items()]}
    maintenance = {"modelType": "Submodel", "kind": "Instance",
        "id": asset_id + "/sm/MaintenanceInstructions", "idShort": "MaintenanceInstructions",
        "submodelElements": [prop("Manual", "Festo-Pneumatic-Fault-Service-Guide.pdf"),
            prop("KnowledgeCommand", "python -m a4s.machines.festo_cp_lab.knowledge --query 'pneumatic pressure flow Fault01 emergency stop'"),
            prop("Scope", "Pressure loss/high flow or emergency-stop investigation; retrieve manual before determining cause."),
            prop("InspectionAllowanceMinutes", "30"),
            prop("PlanningBasis", "Engineering planning allowance, not a guaranteed repair duration."),
            prop("Safety", "Qualified onsite personnel only. No automatic PLC start/reset or physical repair.")]}
    business = copy.deepcopy(context)
    business["id"] = asset_id + "/sm/ProductionContext"
    for field in business.get("submodelElements", []):
        if field.get("idShort") == "MaintenanceSubmodelId":
            field["value"] = maintenance["id"]
    return [operational, maintenance, business]

