"""The Festo CP Lab as one Asset Administration Shell.

**One shell, because it is one asset.** The cell is a demo work center for
metal stamping with two work cells -- press tending and turning over -- and an
operator buys, installs, moves and decommissions the whole thing. Modelling
each work cell as its own shell would say they are separately owned assets,
which is a claim about the plant that is not true, and it would leave nothing
holding the nameplate.

The three OPC UA servers are three *submodels* of that one shell, each an
instance of the `UAServerDataSheet` template already in the repository. A
server is an interface an asset offers, not an asset.

Built with the harness's own `model.py` constructors -- the same code
`aas el add` uses -- so `aas validate` and `aas smt diff` have something to
agree with rather than something to forgive.

**What is known and what is not.** The addresses come from the commissioning
slides, the server identities and data types from probing the live servers,
the product naming from Festo's own description of the cell. Serial number,
article number and date of manufacture are not recorded anywhere available, so
they are left empty. An earlier pass invented a scale for the pressure
readings and the agent caught it reading the value back as implausible; absent
is what this harness means by unknown, and it is the honest state until
somebody reads the number off the sensor.
"""

from __future__ import annotations

from typing import Any, Dict, List, Optional, Tuple

from ...aasctl.model import (
    delegation_qualifier,
    make_element,
    make_shell,
    make_submodel,
    operation_variable,
    qualifier,
    submodel_reference,
)
from ...aasctl.opc import VALUE_SOURCE
from ...kernel.approval import SAFETY_APPROVAL

BASE = "https://fraunhoferIPA/ids"
SHELL_ID = f"{BASE}/aas/FestoWorkCenterStamping"
ASSET_ID = f"{BASE}/asset/FestoWorkCenterStamping"

NAMEPLATE_SMT = "https://admin-shell.io/idta/SubmodelTemplate/DigitalNameplate/2/0"
UA_DATASHEET_SMT = f"{BASE}/sm/UAServerDataSheet"

# Probed on 2026-08-30 over the IPA guest network. `internal` is the address
# inside the cell: it is the one that works from the plant network and not from
# anywhere else, and a model carrying only the reachable-today address is wrong
# the moment the demo moves.
SERVERS: Dict[str, Dict[str, str]] = {
    "TurningPLC": {
        "cell": "Turning",
        "device": "PLC",
        "endpoint": "opc.tcp://192.0.2.10:4041",
        "internal": "opc.tcp://198.51.100.1:4840",
        "product": "SIMATIC S7-1500 OPC UA",
        "manufacturer": "SIEMENS AG",
        "version": "V02.06.01",
        "application_uri": "urn:SIMATIC.S7-1500.OPC-UAServer:plcTurning",
    },
    "PressPLC": {
        "cell": "Press",
        "device": "PLC",
        "endpoint": "opc.tcp://192.0.2.10:4042",
        "internal": "opc.tcp://198.51.100.2:4840",
        "product": "SIMATIC S7-1500 OPC UA",
        "manufacturer": "SIEMENS AG",
        "version": "V02.06.01",
        "application_uri": "urn:SIMATIC.S7-1500.OPC-UAServer:plcPress",
    },
    "PressRevPi": {
        "cell": "Press",
        "device": "RevolutionPi",
        "endpoint": "opc.tcp://192.0.2.10:4043",
        "internal": "opc.tcp://198.51.100.60:4840",
        "product": "open62541 OPC UA Server",
        "manufacturer": "open62541",
        "version": "0.3.0",
        "application_uri": "urn:unconfigured:application",
    },
}

# Two interfaces, and the difference is why nothing started.
#
# `ApplicationModule_DB` is the module's own logic interface -- what its
# program reads and writes internally. Its outputs are trustworthy and are
# where the status comes from: `xReady` went true the moment the Press left
# manual mode, so the block is live and reporting.
#
# Its *inputs* are a different matter. Four writes to `xStart` were accepted by
# the server and nothing moved. That is not what a remote caller is supposed to
# use.
#
# `dbVar` is the station's remote-control surface, found in an older
# configuration rather than inferred: `RemoteControlAutomatic_Start` and
# `_Stop`, both Boolean, both writable, present on both PLCs. The name says
# what it is for, which is more than any amount of guessing at `xAuto` was
# going to produce.
APPLICATION_DB = 'ns=3;s="ApplicationModule_DB"'
REMOTE_DB = 'ns=3;s="dbVar"'

# What an application module actually exposes, read off the PLC rather than off
# the commissioning slides.
#
# The slides document four nodes: `xStart`, `xReady`, `xBusy`, `xError`. The
# data block holds eleven, and the difference is why four start attempts wrote
# `xStart` successfully and nothing moved: `Inputs` also carries `xInit`,
# `xAuto`, `xManual` and `xReset`, and a module sitting in manual mode with no
# mode bit set has nowhere to put a start.
#
# Browsed, not guessed: `aas opc browse <endpoint> --node 'ns=3;s=...'`. The
# slides are an introduction, not an interface specification, and treating them
# as the full set is the mistake this list exists to correct.
#
# Every one of them is *writable* on the S7-1500 -- the PLC publishes the whole
# data block and does not distinguish a status bit from a command bit. Which
# may be written is a fact about the model, not one the server will enforce.
STATUS_BITS: List[Tuple[str, str, str]] = [
    ("Ready", "xReady", "the module reports itself ready to run"),
    ("Busy", "xBusy",
     "the module's own activity flag. **Do not use it to decide whether the "
     "station is running.** Measured on 2026-08-31: the conveyor ran for 5.6 "
     "seconds under a remote start and this bit stayed false the whole time. "
     "`ConveyorForward` is the signal that moved."),
    ("Error", "xError",
     "the module's own error flag. **Not the station's fault word.** It stayed "
     "false through a start that faulted; the fault appeared in `Fault` "
     "(`dbVar.Err.xSF`) instead."),
    ("InInitPosition", "xInitPos", "the module is at its initialisation position"),
]

# The station's real fault word. `ApplicationModule_DB.xError` is the module's
# own flag and stayed false through a fault that stopped the cell; these are the
# bits that moved. What each individual bit means is not documented to us -- the
# question is out with Festo -- so they are carried under their own names rather
# than given invented ones.
FAULT_BITS: List[Tuple[str, str, str]] = [
    ("Fault", 'Err"."xSF', "collective fault (Sammelstörung). Set 2.8 s after a "
                           "remote start, and what stops the cell."),
    ("Fault0", 'Err"."xSF0', "individual fault bit 0. Observed pulsing every few "
                             "seconds after a failed start."),
    ("Fault1", 'Err"."xSF1', "individual fault bit 1"),
    ("Fault2", 'Err"."xSF2', "individual fault bit 2"),
    ("Fault01", 'Err"."xSF01', "individual fault bit 01. Set together with the "
                               "collective fault."),
    ("FaultHint", 'Err"."xHint', "hint rather than fault"),
    ("FaultAcknowledged", 'Err"."xQuit',
     "the acknowledge input, as the PLC currently holds it"),
]

# The process image: real switches and real actuators. Named from the
# configuration the earlier MSB demo used, which is where the German names come
# from. These say what the station is *doing*, which the module's own flags did
# not.
PROCESS_SIGNALS: List[Tuple[str, str, str, str]] = [
    ("ConveyorForward", "xQA1_A1", "xs:boolean",
     "Band Rechtslauf -- the conveyor is running forward. This is the reliable "
     "indication that the station is moving."),
    ("ConveyorReverse", "xQA1_A2", "xs:boolean", "Band Linkslauf"),
    ("CarrierAtModule", "xBG1", "xs:boolean", "WT in Modulposition"),
    ("CarrierAtBeltStart", "xBG5", "xs:boolean", "WT am Bandanfang"),
    ("CarrierAtBeltEnd", "xBG6", "xs:boolean", "WT am Bandende"),
    ("EmergencyStopCircuit", "xSF5", "xs:boolean",
     "Notschalter. Polarity not confirmed: true on both stations while the "
     "cell was neither stopped nor running, so it may mean the safety circuit "
     "is closed rather than that the button is pressed."),
    ("AnalogueIn1", "byAnaIn_8Bit", "xs:int",
     "8-bit analogue input on the PLC itself, separate from the Revolution "
     "Pi's sensors"),
    ("AnalogueIn2", "byAnaIn2_8Bit", "xs:int", "second 8-bit analogue input"),
]

RFID_SIGNALS: List[Tuple[str, str, str]] = [
    ("RfidCode", 'dbRfidData"."ID1"."iCode', "the code read from the carrier"),
    ("RfidCarrierId", 'dbRfidData"."ID1"."iCarrierID', "the carrier's identifier"),
]

# Also an output, and an integer rather than a bit. Observed at 7 on Turning
# and 0 on Press while both were idle, which is information even before anyone
# has the table that decodes it.
RETURN_CODE = ("ReturnCode", "iRetCode", "the module's return code")

# The mode and command inputs, named for what asking for them means rather than
# for the bit that carries it. Modelled as Operations because that is what they
# are to a caller -- a request that the module do something -- and because an
# Operation is the only shape in this repository that carries a delegation and
# a safety declaration together.
#
# All declared as needing approval. Selecting a mode is not a reading, and on a
# cell that can move, a mode change is the precondition for movement rather
# than something separate from it.
# The remote-control surface: what an external caller is meant to use. Taken
# from the station's own configuration, not inferred.
REMOTE_CONTROL: List[Tuple[str, str, str]] = [
    ("AcknowledgeFault", 'Err"."xQuit',
     "acknowledge the station's fault so it can be started again. Writes "
     "`dbVar.Err.xQuit`, which is the input the station's own configuration "
     "names -- not the module's `xReset`."),
    ("Start", "RemoteControlAutomatic_Start",
     "start the station in automatic mode. This is the remote-control input; "
     "it is not the module's internal `xStart`, which a remote caller is not "
     "meant to drive and which four writes moved nothing with."),
    ("Stop", "RemoteControlAutomatic_Stop",
     "stop the station running in automatic mode."),
]

# The module's own inputs. Kept because they are real and readable and because
# their state explains a station that will not start -- but demoted out of the
# way, prefixed `Module`, so nobody reaches for `ModuleStart` when they want
# `Start`.
CONTROL_BITS: List[Tuple[str, str, str]] = [
    ("ModuleStart", "xStart",
     "set the module's internal start bit. **Not the remote start** -- use "
     "`Start`. Four writes here were accepted by the server and nothing moved."),
    ("ModuleInitialise", "xInit",
     "home the module to its initialisation position. `InInitPosition` "
     "reports when it has arrived."),
    ("ModuleSelectAutoMode", "xAuto",
     "set the module's auto-mode bit. Its relationship to the remote start is "
     "not documented; the station has its own `OpMode` input, which is "
     "write-only and not modelled here."),
    ("ModuleSelectManualMode", "xManual",
     "set the module's manual-mode bit. Observed: while this was true on the "
     "Press, `Ready` was false; when it cleared, `Ready` went true."),
    ("ModuleResetError", "xReset",
     "the module's own reset input. **Not the station's fault acknowledge** -- "
     "that is `AcknowledgeFault`, which writes `dbVar.Err.xQuit`."),
]

_RAW = "raw Int16 count; the scale to a physical unit is not established"
REVPI_READINGS: List[Tuple[str, str, str]] = [
    ("Pressure1", "Pressure_1", f"pressure at sensor 1 -- {_RAW}"),
    ("Pressure2", "Pressure_2", f"pressure at sensor 2 -- {_RAW}"),
    ("Flow1", "Flow_1", f"flow at sensor 1 -- {_RAW}"),
    ("Flow2", "Flow_2", f"flow at sensor 2 -- {_RAW}"),
]

WORK_CELLS = {
    "Press": "work cell 1: press tending",
    "Turning": "work cell 2: turning over",
}


def target(endpoint: str, node: str) -> str:
    """A server and a node in one string, for a qualifier that holds one."""
    return f"{endpoint}#{node}"


def _property(id_short: str, value: Optional[str], value_type: str = "xs:string",
              **kw: Any) -> Dict[str, Any]:
    return make_element("Property", id_short, value=value,
                        value_type=value_type, **kw)


def _collection(id_short: str, children: List[Dict[str, Any]],
                **kw: Any) -> Dict[str, Any]:
    element = make_element("SubmodelElementCollection", id_short, **kw)
    element["value"] = children
    return element


# ---------------------------------------------------------------------------
# the meta submodels: what the asset *is*
# ---------------------------------------------------------------------------

def nameplate() -> Dict[str, Any]:
    """Who made it and which one it is.

    Slots the sources do not answer are present and empty rather than absent.
    The template declares them 1..1, so leaving them out would make
    `aas smt diff` report a structure that is missing rather than a value that
    is unknown, and those need different work from whoever reads it.
    """
    submodel = make_submodel(
        f"{BASE}/sm/Nameplate_FestoWorkCenterStamping",
        "Nameplate",
        semantic_id=NAMEPLATE_SMT,
        description="Digital nameplate of the Festo demo work center",
    )
    submodel["submodelElements"] = [
        _property("URIOfTheProduct", ASSET_ID),
        _property("SerialNumber", None,
                  description="not recorded in any available source"),
        _property("DateOfManufacture", None, value_type="xs:date",
                  description="not recorded in any available source"),
        _property("UniqueFacilityIdentifier", "Fraunhofer IPA, Stuttgart"),
        _property("LifeCycleStage", "InUse"),
        _property("ManufacturerIdentifier", "Festo"),
        make_element("MultiLanguageProperty", "ManufacturerName",
                     value=[{"language": "en", "text": "Festo"}]),
        make_element("MultiLanguageProperty", "ManufacturerProductDesignation",
                     value=[{"language": "en",
                             "text": "Demo work center for metal stamping"}]),
        _collection("AddressInformation", [
            _property("Street", None),
            _property("Zipcode", None),
            _property("CityTown", "Stuttgart"),
            _property("NationalCode", "DE"),
        ]),
    ]
    return submodel


def technical_data() -> Dict[str, Any]:
    """What it is made of and how it is reached.

    The network addresses live here rather than only in the server data sheets
    because they describe the cell's installation -- which subnet each work
    cell sits on -- and that outlives any one server.
    """
    submodel = make_submodel(
        f"{BASE}/sm/TechnicalData_FestoWorkCenterStamping",
        "TechnicalData",
        description="Technical data of the Festo demo work center",
    )
    submodel["submodelElements"] = [
        _collection("GeneralInformation", [
            _property("ManufacturerName", "Festo"),
            _property("ManufacturerProductDesignation",
                      "Demo work center for metal stamping"),
            _property("ManufacturerArticleNumber", None,
                      description="not recorded in any available source"),
            _property("ManufacturerOrderCode", None,
                      description="not recorded in any available source"),
        ]),
        _collection("ProductClassifications", [
            _property("ProductClassificationSystem", "Festo CP Lab"),
            _property("ProductClass", "Cyber-physical laboratory work center"),
        ]),
        _collection("TechnicalProperties", [
            _property("NumberOfWorkCells", "2", value_type="xs:int"),
            _property("NumberOfOpcUaServers", str(len(SERVERS)), value_type="xs:int"),
            _collection("Networks", [
                _collection(name, [
                    _property("Subnet", spec["internal"].split("//", 1)[1].rsplit(":", 1)[0]
                              .rsplit(".", 1)[0] + ".0/24"),
                    _property("InternalEndpoint", spec["internal"]),
                    _property("GuestNetworkEndpoint", spec["endpoint"]),
                ])
                for name, spec in SERVERS.items()
            ]),
        ]),
    ]
    return submodel


def bill_of_material() -> Dict[str, Any]:
    """The components the work center is assembled from.

    Hand-built: no bill-of-material template is loaded in this repository, and
    instantiating one that is not there would be inventing a semanticId. The
    structure is the obvious one -- an entry per component, grouped by work
    cell -- and it can be re-pointed at a template the day one arrives.
    """
    submodel = make_submodel(
        f"{BASE}/sm/BillOfMaterial_FestoWorkCenterStamping",
        "BillOfMaterial",
        description="Components of the Festo demo work center, by work cell",
    )

    def component(name: str, kind: str, note: str,
                  endpoint: Optional[str] = None) -> Dict[str, Any]:
        children = [
            _property("ComponentName", name),
            _property("ComponentType", kind),
            _property("Note", note),
        ]
        if endpoint:
            children.append(_property("OpcUaEndpoint", endpoint))
        return _collection(name, children)

    submodel["submodelElements"] = [
        _collection("Press", [
            component("PressPLC", "PLC", "SIMATIC S7-1500",
                      SERVERS["PressPLC"]["endpoint"]),
            component("PressRevolutionPi", "EdgeController",
                      "Revolution Pi, open62541 server for the sensors",
                      SERVERS["PressRevPi"]["endpoint"]),
        ], description=WORK_CELLS["Press"]),
        _collection("Turning", [
            component("TurningPLC", "PLC", "SIMATIC S7-1500",
                      SERVERS["TurningPLC"]["endpoint"]),
            component("TurningRevolutionPi", "EdgeController",
                      "Revolution Pi; publishes no OPC UA server"),
        ], description=WORK_CELLS["Turning"]),
    ]
    return submodel


# ---------------------------------------------------------------------------
# one submodel per OPC UA server, on the template the repository already has
# ---------------------------------------------------------------------------

def ua_server_datasheet(name: str, spec: Dict[str, str]) -> Dict[str, Any]:
    # The idShort names *which* server; the semanticId says it is an instance
    # of the `UAServerDataSheet` template. Three submodels all called
    # `UAServerDataSheet` would be indistinguishable in a listing, and the
    # instantiation route -- which derives a new submodel id from the shell id
    # and the idShort -- collides on them outright.
    submodel = make_submodel(
        f"{BASE}/sm/UAServerDataSheet_{name}",
        f"UAServerDataSheet_{name}",
        semantic_id=UA_DATASHEET_SMT,
        description=f"OPC UA server of {name} ({WORK_CELLS[spec['cell']]})",
    )
    submodel["submodelElements"] = [
        # Probed: the session was accepted anonymously with no security policy.
        _collection("Configuration", [
            _property("AllowAnonymousUser", "true", "xs:boolean"),
            _property("SupportSecurityModeNone", "true", "xs:boolean"),
            _property("SupportSecurityModeSign", "false", "xs:boolean"),
            _property("SupportSecurityModeSignEncrypt", "false", "xs:boolean"),
            _property("SupportSecurityPolicyBasic256Sha256", "false", "xs:boolean"),
        ]),
        _collection("Identification", [
            _property("ProductUri", spec["application_uri"]),
            _property("ManufacturerName", spec["manufacturer"]),
            _property("ProductName", spec["product"]),
            _property("SoftwareVersion", spec["version"]),
            _property("BuildNumber", None),
            _property("BuildDate", None),
        ]),
        _collection("EndpointDescription", [
            _property("EndpointUri", spec["endpoint"]),
            _property("securityLevel", "0", "xs:int"),
            _property("securityMode", "None"),
            _property("transportProfileUri",
                      "http://opcfoundation.org/UA-Profile/Transport/uatcp-uasc-uabinary"),
            _property("securityPolicyUri",
                      "http://opcfoundation.org/UA/SecurityPolicy#None"),
            # Anonymous sessions were accepted, so no token type was needed and
            # none was observed. Present and empty rather than absent: the
            # template declares it 1..1, and `aas smt diff` should report a
            # value nobody has filled in, not a structure nobody built.
            _property("userIdentityTokens", None,
                      description="anonymous access; no token type observed"),
            _property("InternalEndpointUri", spec["internal"],
                      description="the address inside the cell's own subnet"),
            _collection("Server", [
                _property("ApplicationUri", spec["application_uri"]),
                _property("ProductUri", spec["application_uri"]),
                _property("ApplicationName", f"{spec['product']} ({name})"),
                _property("ApplicationType", "Server"),
                _property("GatewayServerUri", None,
                          description="no gateway; the server is reached directly"),
                _property("DiscoveryProfileUri", None),
                _property("DiscoveryUrl", spec["endpoint"]),
            ]),
            # No certificate: these endpoints run SecurityMode None, so the
            # server presents none. Declared empty so the slot is visibly
            # unused rather than overlooked.
            make_element("Blob", "serverCertificate",
                         content_type="application/pkix-cert",
                         description="none; the endpoint runs SecurityMode None"),
        ]),
        _property("DiscoveryUrl", spec["endpoint"]),
    ]
    return submodel


# ---------------------------------------------------------------------------
# what it can be asked to do, and what can be read off it
# ---------------------------------------------------------------------------

def services_delegated() -> Dict[str, Any]:
    submodel = make_submodel(
        f"{BASE}/sm/services_delegated_FestoWorkCenterStamping",
        "services_delegated",
        description="What the work center can be asked to do, over OPC UA",
    )
    services: List[Dict[str, Any]] = []
    for cell, server in (("Press", "PressPLC"), ("Turning", "TurningPLC")):
        endpoint = SERVERS[server]["endpoint"]
        for name, node, text, block in (
            [(n, nd, t, REMOTE_DB) for n, nd, t in REMOTE_CONTROL]
            + [(n, nd, t, APPLICATION_DB) for n, nd, t in CONTROL_BITS]
        ):
            action = make_element(
                "Operation",
                f"{name}{cell}",
                description=f'Set {block}."{node}" true on {server} '
                            f"({WORK_CELLS[cell]}). {text}",
                qualifiers=[
                    delegation_qualifier(target(endpoint, f'{block}."{node}"')),
                    qualifier(SAFETY_APPROVAL, "true", "xs:boolean"),
                ],
            )
            # The value is an input because these are not all triggers. A mode
            # bit is a level -- deselecting auto means writing false -- and an
            # Operation that could only ever write true would model half of it.
            action["inputVariables"] = [
                operation_variable(_property(
                    "value", None, "xs:boolean",
                    description="true unless the bit is being cleared; "
                                "defaults to true"))
            ]
            action["outputVariables"] = [
                operation_variable(_property(
                    "accepted", None, "xs:boolean",
                    description="whether the write was acknowledged"))
            ]
            services.append(action)

    submodel["submodelElements"] = [
        _property("ServiceEndpointBase", SERVERS["PressPLC"]["endpoint"]),
        _property("ServiceStatus", ""),
        _property("ProtocolType", "OPC UA"),
        _collection("DelegatedServices", services),
    ]
    return submodel


def operational_data() -> Dict[str, Any]:
    """Readable values, one collection per work cell.

    Properties rather than Operations, because reading a pressure is not an
    action and modelling it as one would put every measurement through the
    approval gate. The stored value is the last one recorded; the
    `valueSource` qualifier is how to fetch a fresh one, and `aas opc ls`
    lists them.
    """
    submodel = make_submodel(
        f"{BASE}/sm/OperationalData_FestoWorkCenterStamping",
        "OperationalData",
        description="Process values of the work center, read over OPC UA",
    )

    def reading(id_short: str, source: str, description: str,
                value_type: str) -> Dict[str, Any]:
        return _property(
            id_short, None, value_type, description=description,
            qualifiers=[
                qualifier(VALUE_SOURCE, source),
                # Said rather than inferred: without it the gate would fall
                # back to reading the name, and `Error` reads like a verb.
                qualifier(SAFETY_APPROVAL, "false", "xs:boolean"),
            ],
        )

    cells: List[Dict[str, Any]] = []
    for cell, server in (("Press", "PressPLC"), ("Turning", "TurningPLC")):
        endpoint = SERVERS[server]["endpoint"]
        children = [
            reading(name, target(endpoint, f'{APPLICATION_DB}."{node}"'),
                    text, "xs:boolean")
            for name, node, text in STATUS_BITS
        ]
        children.append(reading(
            RETURN_CODE[0], target(endpoint, f'{APPLICATION_DB}."{RETURN_CODE[1]}"'),
            RETURN_CODE[2], "xs:int",
        ))
        # The mode inputs read back as well as write. A start that does nothing
        # is explained by these and by nothing else in the model, so they are
        # readable rather than write-only.
        children += [
            reading(f"{name}Requested",
                    target(endpoint, f'{APPLICATION_DB}."{node}"'),
                    f"the {node} input, as the PLC currently holds it",
                    "xs:boolean")
            for name, node, _ in CONTROL_BITS
        ]
        children += [
            reading(f"{name}Requested",
                    target(endpoint, f'{REMOTE_DB}."{node}"'),
                    f"the {node} remote-control input, as the PLC holds it",
                    "xs:boolean")
            for name, node, _ in REMOTE_CONTROL
        ]
        children += [
            reading(name, target(endpoint, f'{REMOTE_DB}."{node}"'),
                    text, "xs:boolean")
            for name, node, text in FAULT_BITS
        ]
        children += [
            reading(name, target(endpoint, f'ns=3;s="{node}"'), text, vtype)
            for name, node, vtype, text in PROCESS_SIGNALS
        ]
        children += [
            reading(name, target(endpoint, f'ns=3;s="{node}"'), text, "xs:int")
            for name, node, text in RFID_SIGNALS
        ]
        if cell == "Press":
            revpi = SERVERS["PressRevPi"]["endpoint"]
            children += [
                reading(name, target(revpi, f"ns=1;s={node}"), text, "xs:int")
                for name, node, text in REVPI_READINGS
            ]
        cells.append(_collection(cell, children, description=WORK_CELLS[cell]))

    submodel["submodelElements"] = cells
    return submodel


# ---------------------------------------------------------------------------

def build() -> Dict[str, Any]:
    """The complete environment: one shell and the submodels it references."""
    submodels = [
        nameplate(),
        technical_data(),
        bill_of_material(),
        *(ua_server_datasheet(name, spec) for name, spec in SERVERS.items()),
        services_delegated(),
        operational_data(),
    ]
    shell = make_shell(
        SHELL_ID,
        "FestoWorkCenterStamping",
        global_asset_id=ASSET_ID,
        description="Festo demo work center for metal stamping: "
                    "press tending and turning over, at Fraunhofer IPA",
    )
    shell["submodels"] = [submodel_reference(s["id"]) for s in submodels]
    return {
        "assetAdministrationShells": [shell],
        "submodels": submodels,
        "conceptDescriptions": [],
    }
