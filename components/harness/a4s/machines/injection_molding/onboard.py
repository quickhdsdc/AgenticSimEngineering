"""Putting the machine into the AAS repository.

The machine is only reachable to an agent once its capabilities are modelled.
This authors two submodels on the machine's shell:

``OperationalData`` -- one Property per published reading, with its unit, so a
value can be read through the AAS rather than through the service's own API.

``Capability`` -- one Operation per endpoint, each carrying an
``invocationDelegation`` qualifier that names the endpoint. That qualifier is
the whole integration: ``aas op ls`` then discovers what the machine can do,
and ``aas op invoke`` does it, with no code in the harness that knows anything
about injection moulding.

The Operation's declared input variables are taken from the service's own
published contract rather than written here. A declared signature that drifts
from the service it delegates to is a defect this repository already contains
-- ``ProcessClassifyDefect`` declares ``laser_power`` where its service wants
``power`` -- and generating one from the other is how that is not repeated.

Operations that move the machine are marked. `Qualifier/type = "requiresSafetyApproval"`
is not part of any IDTA template; it is carried so an approval gate has
something to read that is a property of the *modelled capability* rather than a
guess from its name.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Any, Dict, List, Optional

from ...aasctl.client import BasyxClient
from ...aasctl.model import (
    delegation_qualifier,
    make_element,
    make_shell,
    make_submodel,
    operation_variable,
    qualifier,
    submodel_reference,
)
from ...aasctl.paths import ElementPath
# The qualifier name and the classification behind it describe any Operation in
# any AAS, not this machine, so they live in `approval` -- imported rather than
# redefined, so a gate and the model it reads cannot disagree about the
# spelling of the string.
from ...kernel.approval import SAFETY_APPROVAL, moves_machine  # noqa: F401
from .machine import SETTABLE, UNITS
from .service import ENDPOINTS

SHELL_ID = "https://fraunhoferIPA/ids/aas/injectionMoldingMachine_sim"
OPERATIONAL_SM = "https://fraunhoferIPA/ids/sm/imm_sim/OperationalData"
CAPABILITY_SM = "https://fraunhoferIPA/ids/sm/imm_sim/Capability"


@dataclass
class Onboarding:
    shell_id: str
    submodels: List[str]
    operations: List[str]
    readings: int
    moving: List[str]

    def summary(self) -> Dict[str, Any]:
        return {
            "shell": self.shell_id,
            "submodels": self.submodels,
            "readings": self.readings,
            "operations": self.operations,
            "moves_machine": self.moving,
        }


def _value_type(name: str) -> str:
    return "xs:integer" if name in ("CycleCounter",) else "xs:double"


def _operation(path: str, spec: Dict[str, Any], base_url: str) -> Dict[str, Any]:
    """One Operation, its inputs taken from the service's own contract."""
    # `/predict-quality` -> `PredictQuality`. Composed rather than transliterated
    # because AASd-002 forbids the hyphen, and the first attempt at this shipped
    # `Predict-quality` -- caught by `aas validate` against its own output.
    import re

    words = [w for w in re.split(r"[^A-Za-z0-9]+", path) if w]
    id_short = "".join(w[:1].upper() + w[1:] for w in words) or "Operation"
    quals = [delegation_qualifier(f"{base_url.rstrip('/')}{path}")]
    # Written on *every* operation, including the ones that answer "false".
    # Only the true ones used to carry it, which was indistinguishable from an
    # operation nobody had classified -- and since `moves_machine` now treats
    # "undeclared" as "assume it does", a read with no qualifier became a false
    # stop. The service publishes the flag for every endpoint; the AAS should
    # record what it published, not only half of it.
    quals.append(
        qualifier(
            SAFETY_APPROVAL,
            "true" if spec.get("moves_machine") else "false",
            "xs:boolean",
        )
    )

    element = make_element(
        "Operation",
        id_short,
        description=spec.get("summary"),
        qualifiers=quals,
    )
    element["inputVariables"] = [
        operation_variable(
            make_element("Property", name, value_type="xs:string", description=str(description))
        )
        for name, description in (spec.get("inputs") or {}).items()
    ]
    return element


def onboard(
    client: BasyxClient,
    *,
    base_url: str,
    readings: Dict[str, float],
    shell_id: str = SHELL_ID,
    id_short: str = "InjectionMoldingMachineSim",
    replace: bool = False,
) -> Onboarding:
    """Create the shell and its two submodels. Idempotent under ``replace``."""
    if replace:
        for identifier in (OPERATIONAL_SM, CAPABILITY_SM):
            try:
                client.delete_submodel(identifier)
            except Exception:  # noqa: BLE001 - absent is the state we want
                pass
        try:
            client.delete_shell(shell_id)
        except Exception:  # noqa: BLE001
            pass

    try:
        client.get_shell(shell_id)
    except Exception:  # noqa: BLE001 - it does not exist yet
        client.create_shell(
            make_shell(
                shell_id,
                id_short,
                global_asset_id=shell_id.replace("/aas/", "/asset/"),
                description="Simulated injection moulding machine replaying a published run",
            )
        )

    # -- operational data ------------------------------------------------
    operational = make_submodel(
        OPERATIONAL_SM, "OperationalData",
        description="Live readings published by the machine",
    )
    operational["submodelElements"] = [
        make_element(
            "Property", name,
            value=None if value is None else str(value),
            value_type=_value_type(name),
            description=f"{name} in {UNITS[name]}" if UNITS.get(name) else name,
            qualifiers=[qualifier("Unit", UNITS[name])] if UNITS.get(name) else None,
        )
        for name, value in sorted(readings.items())
    ]
    client.create_submodel(operational)
    client.add_submodel_ref(shell_id, submodel_reference(OPERATIONAL_SM))

    # -- capability ------------------------------------------------------
    capability = make_submodel(
        CAPABILITY_SM, "Capability",
        description="What the machine can be asked to do, and where each request goes",
    )
    operations = [_operation(path, spec, base_url) for path, spec in ENDPOINTS.items()]
    capability["submodelElements"] = operations
    client.create_submodel(capability)
    client.add_submodel_ref(shell_id, submodel_reference(CAPABILITY_SM))

    return Onboarding(
        shell_id=shell_id,
        submodels=[OPERATIONAL_SM, CAPABILITY_SM],
        operations=[str(o.get("idShort")) for o in operations],
        readings=len(operational["submodelElements"]),
        moving=[
            str(o.get("idShort"))
            for o, spec in zip(operations, ENDPOINTS.values())
            if spec.get("moves_machine")
        ],
    )


def sync_readings(client: BasyxClient, readings: Dict[str, float]) -> int:
    """Write the machine's current readings into its OperationalData submodel.

    A telemetry value written into the AAS is a snapshot with a timestamp
    implied by when it was written -- not a live feed. Anything that needs the
    current value should invoke `Status` and read the answer; this exists so a
    reading can be *recorded*, which is a different thing.
    """
    written = 0
    for name, value in readings.items():
        if value is None:
            continue
        try:
            client.set_value(OPERATIONAL_SM, ElementPath.parse(name), str(value))
            written += 1
        except Exception:  # noqa: BLE001 - a missing slot is reported by the count
            continue
    return written


def settable_readings() -> List[str]:
    return list(SETTABLE)


def main(argv: Optional[list] = None) -> int:
    import argparse
    import json

    import httpx

    from ...aasctl.client import Endpoint

    parser = argparse.ArgumentParser(
        prog="a4s-imm-onboard",
        description="Model the simulated machine in the AAS repository.",
    )
    parser.add_argument("--service", default="http://localhost:5080",
                        help="where the machine service listens")
    parser.add_argument("--endpoint", help="AAS repository (else $A4S_AAS_ENDPOINT)")
    parser.add_argument("--shell-id", default=SHELL_ID)
    parser.add_argument("--replace", action="store_true", help="delete and recreate")
    parser.add_argument("--sync", action="store_true",
                        help="also write the current readings into OperationalData")
    args = parser.parse_args(argv)

    try:
        telemetry = httpx.get(f"{args.service.rstrip('/')}/status", timeout=15).json()
    except Exception as exc:  # noqa: BLE001
        print(f"error: cannot reach the machine at {args.service}: {exc}")
        print("hint: start it with `python -m a4s.machines.injection_molding.service`")
        return 1
    readings = {name: entry["value"] for name, entry in telemetry["readings"].items()}

    with BasyxClient(Endpoint.from_env(args.endpoint)) as client:
        result = onboard(
            client, base_url=args.service, readings=readings,
            shell_id=args.shell_id, replace=args.replace,
        )
        if args.sync:
            written = sync_readings(client, readings)
            print(f"synced {written} readings")
    print(json.dumps(result.summary(), indent=2))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
