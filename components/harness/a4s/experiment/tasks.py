"""The task suite, and how each task is judged.

Every task carries a **programmatic verifier**. A run that finished is not a
run that succeeded: the interesting failure mode is an agent that reports
having written a value it never wrote, and only the repository can settle that.
Where a task's product is a report rather than a change, the verifier checks
that the report names the paths a correct answer has to name -- crude, and said
so plainly, but it is checkable and it does not depend on a second model's
opinion.

Tasks are ordered by what they exercise, not by difficulty:

    read      one value, and where it sits
    explore   the shape of a curated template collection
    author    a submodel instantiated from a template
    write     values into it, including the MultiLanguageProperty trap
    retrieve  where something belongs, by meaning rather than name
    complete  which mandatory slots are still empty
    validate  what the metamodel says is wrong
    chain     all of it, from a datasheet
    batch     thirty values from a file, in one pass
    prune     delete the empty slots and only those
    locate    find an element without being told which submodel holds it

The last three were added after an audit found the first eight shared three
blind spots, each of which the suite was quietly flattered by.

**`batch` exists because nothing wrote thirty values.** Open question 5 records
that a composed pipeline only pays for itself when there is something to batch,
so a suite whose largest task writes two values measured the condition least
favourable to its own hypothesis. Thirty rows in a file rather than in the
instruction: inlining them would make it a prompt-length experiment.

**`prune` exists because nothing deleted.** Deletion is the only irreversible
operation the CLI offers and it had no agent-level coverage at all. Its
verifier reports over-deletion and under-deletion separately, because an agent
that removes a populated slot has failed differently from one that leaves an
empty slot behind.

**`locate` exists because every other task was handed its submodel id.** That
is not what an operator knows. It is also the only task whose difficulty scales
with the repository -- 3,789 elements across 116 submodels -- and the one that
exercises `find --across`.

`setup` runs before each attempt and `teardown` after, so an arm never
inherits state from the other arm or from its own previous run. A task that
depends on artefacts it did not create is not measuring what it claims to.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from pathlib import Path
from typing import Any, Callable, Dict, List, Optional

from ..aasctl.client import BasyxClient, Endpoint
from ..aasctl.model import make_element, make_submodel
from ..aasctl.paths import ElementPath

# Fixtures in the live repository. Read-only: no task writes to these.
TECHNICAL_DATA_527 = (
    "https://admin-shell.io/ZVEI/TechnicalData/Submodel/1/2_DigitalBatteryPassport_527"
)
TECHNICAL_DATA_TEMPLATE = "https://admin-shell.io/ZVEI/TechnicalData/Submodel/1/2"
NAMEPLATE_TEMPLATE = "https://admin-shell.io/idta/SubmodelTemplate/DigitalNameplate/2/0"
SMT_COLLECTION = "https://fraunhoferIPA/ids/aas/CollectionofSMTs"

# Everything a task creates lives under this prefix, so teardown is exhaustive
# rather than a list someone has to remember to extend.
SCRATCH_PREFIX = "urn:a4s:exp:"


@dataclass
class Verdict:
    passed: bool
    detail: str


@dataclass
class Task:
    name: str
    instruction: str
    check: Callable[["Context"], Verdict]
    max_steps: int = 14
    setup: Optional[Callable[["Context"], None]] = None
    teardown: Optional[Callable[["Context"], None]] = None
    # Files the arm needs in its working directory before it starts.
    fixtures: List[str] = field(default_factory=list)


@dataclass
class Context:
    """What a verifier gets: the repository, the workdir, and what was said."""

    endpoint: str
    workdir: Path
    summary: str = ""
    status: str = ""

    def client(self) -> BasyxClient:
        return BasyxClient(Endpoint.from_env(self.endpoint))

    # -- repository queries ---------------------------------------------

    def value_at(self, submodel_id: str, path: str) -> Any:
        with self.client() as client:
            try:
                return client.get_value(submodel_id, ElementPath.parse(path))
            except Exception:
                return None

    def submodel(self, submodel_id: str) -> Optional[Dict[str, Any]]:
        with self.client() as client:
            try:
                return client.get_submodel(submodel_id)
            except Exception:
                return None

    def shell_exists(self, shell_id: str) -> bool:
        with self.client() as client:
            try:
                client.get_shell(shell_id)
                return True
            except Exception:
                return False

    # -- report checks ---------------------------------------------------

    def said(self, *needles: str) -> bool:
        """Whether the report names all of these, case-insensitively."""
        lowered = self.summary.lower()
        return all(n.lower() in lowered for n in needles)

    def said_any(self, *needles: str) -> bool:
        lowered = self.summary.lower()
        return any(n.lower() in lowered for n in needles)


def _missing(context: Context, *needles: str) -> str:
    lowered = context.summary.lower()
    return ", ".join(repr(n) for n in needles if n.lower() not in lowered)


# ---------------------------------------------------------------------------
# teardown
# ---------------------------------------------------------------------------

def purge_scratch(context: Context) -> None:
    """Delete every shell and submodel a task could have created.

    By prefix, not by a hand-kept list: an agent that invents its own id for a
    submodel it was told to create would otherwise leave it behind and the next
    arm would find the work already done.
    """
    from ..aasctl import index as index_mod

    with context.client() as client:
        for submodel in client.list_submodels(limit=400):
            identifier = submodel.get("id") or ""
            if identifier.startswith(SCRATCH_PREFIX):
                try:
                    client.delete_submodel(identifier)
                    index_mod.Index.drop(identifier)
                except Exception:
                    pass
        for shell in client.list_shells(limit=200):
            identifier = shell.get("id") or ""
            if identifier.startswith(SCRATCH_PREFIX):
                try:
                    client.delete_shell(identifier)
                except Exception:
                    pass


# ---------------------------------------------------------------------------
# the tasks
# ---------------------------------------------------------------------------

def _check_read(context: Context) -> Verdict:
    ok = context.said("RatedCapacity") and context.said_any(
        "CapacityEnergyVoltage", "Capacity Energy Voltage"
    )
    return Verdict(ok, "report names the path and its collection" if ok
                   else f"missing from report: {_missing(context, 'RatedCapacity', 'CapacityEnergyVoltage')}")


def _check_explore(context: Context) -> Verdict:
    ok = context.said("DigitalNameplate") and context.said("8")
    return Verdict(ok, "names DigitalNameplate with 8 mandatory slots" if ok
                   else "did not name DigitalNameplate and the count 8")


def _check_author(context: Context) -> Verdict:
    submodel = context.submodel(f"{SCRATCH_PREFIX}sm:nameplate")
    if submodel is None:
        return Verdict(False, "the submodel was not created")
    if submodel.get("kind") != "Instance":
        return Verdict(False, f"kind is {submodel.get('kind')!r}, not Instance")
    top = submodel.get("submodelElements") or []
    names = {e.get("idShort") for e in top}
    # Mandatory only: the three optional slots must be absent.
    if names & {"OperatorIdentifier", "Markings", "AssetSpecificProperties"}:
        return Verdict(False, "optional slots were created although only mandatory were asked for")
    if "SerialNumber" not in names:
        return Verdict(False, "the mandatory SerialNumber slot is missing")

    def template_qualifiers(node: Dict[str, Any]) -> int:
        count = sum(
            1 for q in node.get("qualifiers") or [] if q.get("kind") == "TemplateQualifier"
        )
        for key in ("submodelElements", "value"):
            children = node.get(key)
            if isinstance(children, list):
                for child in children:
                    if isinstance(child, dict):
                        count += template_qualifiers(child)
        return count

    leftover = template_qualifiers(submodel)
    if leftover:
        return Verdict(False, f"{leftover} TemplateQualifiers survived (AASd-129)")
    if not context.shell_exists(f"{SCRATCH_PREFIX}aas:cell"):
        return Verdict(False, "the shell was not created")
    return Verdict(True, f"{len(top)} mandatory slots, kind=Instance, no TemplateQualifiers")


def _setup_write(context: Context) -> None:
    """Give the arm a submodel to write into, so the task measures writing."""
    from ..aasctl import smt as smt_mod

    with context.client() as client:
        template = client.get_submodel(NAMEPLATE_TEMPLATE)
        result = smt_mod.instantiate(
            template, submodel_id=f"{SCRATCH_PREFIX}sm:write", id_short="Nameplate"
        )
        client.create_submodel(result.submodel)


def _check_write(context: Context) -> Verdict:
    serial = context.value_at(f"{SCRATCH_PREFIX}sm:write", "SerialNumber")
    name = context.value_at(f"{SCRATCH_PREFIX}sm:write", "ManufacturerName")
    if serial != "SN-4711":
        return Verdict(False, f"SerialNumber is {serial!r}, expected 'SN-4711'")
    # ValueOnly for a MultiLanguageProperty is [{"<lang>": "<text>"}].
    if not isinstance(name, list) or not name:
        return Verdict(False, f"ManufacturerName is {name!r}, expected a langString list")
    texts = [v for entry in name if isinstance(entry, dict) for v in entry.values()]
    if not any("fraunhofer" in str(t).lower() for t in texts):
        return Verdict(False, f"ManufacturerName does not carry the value: {name!r}")
    return Verdict(True, "scalar and MultiLanguageProperty both written correctly")


def _check_retrieve(context: Context) -> Verdict:
    ok = context.said("RatedCapacity") and context.said("MinVoltage")
    return Verdict(ok, "names both target paths" if ok
                   else f"missing from report: {_missing(context, 'RatedCapacity', 'MinVoltage')}")


def _setup_complete(context: Context) -> None:
    from ..aasctl import smt as smt_mod

    with context.client() as client:
        template = client.get_submodel(TECHNICAL_DATA_TEMPLATE)
        result = smt_mod.instantiate(
            template,
            submodel_id=f"{SCRATCH_PREFIX}sm:partial",
            id_short="TechnicalData",
            include_optional=True,
        )
        client.create_submodel(result.submodel)
        # A plain string: in *this* template ManufacturerName is a Property.
        # It is a MultiLanguageProperty in DigitalNameplate, and writing the
        # langString shape here draws a BaSyx 500 -- the same trap the `write`
        # task exists to exercise, met in a fixture.
        client.set_value(
            f"{SCRATCH_PREFIX}sm:partial",
            ElementPath.parse("GeneralInformation.ManufacturerName"),
            "Fraunhofer IPA",
        )


def _check_complete(context: Context) -> Verdict:
    # BatteryMass and WarrantyPeriod are mandatory and were left empty on
    # purpose; ManufacturerName was filled and must not be reported as a gap.
    named = context.said("BatteryMass") and context.said("WarrantyPeriod")
    false_positive = "manufacturername" in context.summary.lower()
    if not named:
        return Verdict(False, f"missing from report: {_missing(context, 'BatteryMass', 'WarrantyPeriod')}")
    if false_positive:
        return Verdict(False, "reported ManufacturerName as unfilled although it has a value")
    return Verdict(True, "named the empty mandatory slots and not the filled one")


def _setup_validate(context: Context) -> None:
    from ..aasctl.model import make_submodel

    with context.client() as client:
        submodel = make_submodel(f"{SCRATCH_PREFIX}sm:broken", "Broken")
        # AASd-002: an idShort that is not [A-Za-z][A-Za-z0-9_]*. Written
        # through raw JSON because `aas el add` would refuse it -- which is
        # the point: the repository accepts what the CLI will not author.
        submodel["submodelElements"] = [
            {"modelType": "Property", "idShort": "Serial Number",
             "valueType": "xs:string", "value": "SN-1"},
            {"modelType": "Property", "idShort": "Good", "valueType": "xs:string", "value": "1"},
        ]
        client.create_submodel(submodel)


def _check_validate(context: Context) -> Verdict:
    ok = context.said_any("AASd-002", "aasd002", "idshort") and context.said_any(
        "Serial Number", "Serial_Number", "SerialNumber"
    )
    return Verdict(ok, "identified the illegal idShort" if ok
                   else "did not identify the AASd-002 violation on 'Serial Number'")


def _check_chain(context: Context) -> Verdict:
    submodel_id = f"{SCRATCH_PREFIX}sm:chain"
    submodel = context.submodel(submodel_id)
    if submodel is None:
        return Verdict(False, "the submodel was not created")
    checks = {
        "TechnicalPropertyAreas.CapacityEnergyVoltage.RatedCapacity": "68",
        "TechnicalPropertyAreas.CapacityEnergyVoltage.NominalVoltage": "355.2",
        "GeneralInformation.ManufacturerArticleNumber": "IPA-C9000-355",
    }
    wrong = []
    for path, expected in checks.items():
        actual = context.value_at(submodel_id, path)
        if actual is None or str(actual).strip() != expected:
            wrong.append(f"{path.rsplit('.', 1)[-1]}={actual!r} (want {expected!r})")
    if wrong:
        return Verdict(False, "; ".join(wrong))
    return Verdict(True, "three datasheet values landed on the right elements")



# ---------------------------------------------------------------------------
# batch, prune, locate -- the three shapes the original eight did not cover
# ---------------------------------------------------------------------------

# 30 rows. The number is the point: PROGRESS.md open question 5 records that
# the original suite has no task writing thirty values, and that a composed
# pipeline only pays for itself when there is something to batch -- so the
# comparison measured the condition least favourable to its own hypothesis.
BATCH_ROWS = [(f"Param{i:02d}", f"{i * 7 + 3}.5") for i in range(1, 31)]

IMM_OPERATIONAL = "https://fraunhoferIPA/ids/sm/imm_sim/OperationalData"


def _setup_batch(context: Context) -> None:
    """An empty submodel and a CSV of 30 values beside it.

    The values are in a file rather than the instruction because thirty
    name/value pairs inlined would be a prompt-length experiment rather than a
    tool-surface one, and because reading a file is what an agent would really
    do with a supplier list.
    """
    with context.client() as client:
        client.create_submodel(
            make_submodel(f"{SCRATCH_PREFIX}sm:batch", "BatchTarget")
        )
    lines = ["name,value"] + [f"{name},{value}" for name, value in BATCH_ROWS]
    (context.workdir / "batch.csv").write_text("\n".join(lines) + "\n", encoding="utf-8")


def _check_batch(context: Context) -> Verdict:
    """Every one of the thirty, read back from the repository.

    All-or-nothing on purpose. A partial write is the failure mode worth
    catching: an agent that stops at twenty and reports success has produced a
    submodel that looks populated and is not.
    """
    submodel_id = f"{SCRATCH_PREFIX}sm:batch"
    if context.submodel(submodel_id) is None:
        return Verdict(False, "the submodel is gone")
    missing, wrong = [], []
    for name, expected in BATCH_ROWS:
        actual = context.value_at(submodel_id, name)
        if actual is None:
            missing.append(name)
        elif str(actual).strip() != expected:
            wrong.append(f"{name}={actual!r}")
    if missing:
        return Verdict(False, f"{len(missing)} of {len(BATCH_ROWS)} never written: "
                              f"{', '.join(missing[:5])}...")
    if wrong:
        return Verdict(False, f"{len(wrong)} wrong: {'; '.join(wrong[:5])}")
    return Verdict(True, f"all {len(BATCH_ROWS)} values written and read back")


def _setup_prune(context: Context) -> None:
    """A submodel with four filled slots and five empty ones."""
    with context.client() as client:
        client.create_submodel(make_submodel(f"{SCRATCH_PREFIX}sm:prune", "PruneTarget"))
        root = ElementPath.parse("")
        for name, value in (("Keep1", "a"), ("Keep2", "b"), ("Keep3", "c"), ("Keep4", "d")):
            client.add_element(f"{SCRATCH_PREFIX}sm:prune", root,
                               make_element("Property", name, value=value,
                                            value_type="xs:string"))
        for index in range(1, 6):
            client.add_element(f"{SCRATCH_PREFIX}sm:prune", root,
                               make_element("Property", f"Empty{index}", value="",
                                            value_type="xs:string"))


def _check_prune(context: Context) -> Verdict:
    """The four filled slots survive and the five empty ones are gone.

    The only task in the suite that deletes. Deletion is the one irreversible
    operation the CLI offers, and an over-eager agent failing this is a
    different and more serious result than one that under-deletes -- so both
    directions are reported separately.
    """
    submodel_id = f"{SCRATCH_PREFIX}sm:prune"
    submodel = context.submodel(submodel_id)
    if submodel is None:
        return Verdict(False, "the submodel was deleted; only its empty elements should be")
    present = {
        element.get("idShort")
        for element in (submodel.get("submodelElements") or [])
        if isinstance(element, dict)
    }
    lost = [n for n in ("Keep1", "Keep2", "Keep3", "Keep4") if n not in present]
    left = [f"Empty{i}" for i in range(1, 6) if f"Empty{i}" in present]
    if lost:
        return Verdict(False, f"deleted slots that held values: {', '.join(lost)}")
    if left:
        return Verdict(False, f"{len(left)} empty slots remain: {', '.join(left)}")
    return Verdict(True, "five empty slots removed, four populated ones untouched")


def _check_locate(context: Context) -> Verdict:
    """The report has to name the path, and the instruction never gave the id.

    Judged on the report rather than on the repository because the task is a
    read. Crude, and said so plainly -- but checkable, and it does not depend
    on a second model's opinion.
    """
    summary = context.summary
    if "BarrelZone1" not in summary:
        return Verdict(False, "the report does not name BarrelZone1")
    if IMM_OPERATIONAL.rsplit("/", 1)[-1] not in summary and "OperationalData" not in summary:
        return Verdict(False, "the report names the element but not where it lives")
    return Verdict(True, "found the element without being told which submodel holds it")


TASKS: List[Task] = [
    Task(
        name="read",
        instruction=(
            f"In the submodel {TECHNICAL_DATA_527}, report the exact idShortPath of the "
            "element holding the battery's rated capacity, and which collection it sits in."
        ),
        check=_check_read,
        max_steps=8,
    ),
    Task(
        name="explore",
        instruction=(
            f"List the submodel templates curated by the shell {SMT_COLLECTION} and report "
            "which one has the most mandatory top-level slots, and how many."
        ),
        check=_check_explore,
        max_steps=8,
    ),
    Task(
        name="author",
        instruction=(
            f"Create a shell {SCRATCH_PREFIX}aas:cell with idShort ExpCell, then instantiate "
            f"the template {NAMEPLATE_TEMPLATE} into a submodel {SCRATCH_PREFIX}sm:nameplate "
            "with idShort Nameplate, attached to that shell. Create only the mandatory "
            "slots. Report what you created."
        ),
        check=_check_author,
        teardown=purge_scratch,
        max_steps=12,
    ),
    Task(
        name="write",
        instruction=(
            f"In the submodel {SCRATCH_PREFIX}sm:write, set SerialNumber to SN-4711 and "
            "set ManufacturerName to 'Fraunhofer IPA' in English. Read both back and "
            "report what they now hold."
        ),
        check=_check_write,
        setup=_setup_write,
        teardown=purge_scratch,
        max_steps=10,
    ),
    Task(
        name="retrieve",
        instruction=(
            f"In the submodel {TECHNICAL_DATA_527}, find where the battery's nominal "
            "capacity and its lowest operating voltage belong. Report the exact "
            "idShortPath of each. Do not guess from names alone."
        ),
        check=_check_retrieve,
        max_steps=10,
    ),
    Task(
        name="complete",
        instruction=(
            f"The submodel {SCRATCH_PREFIX}sm:partial was instantiated from the template "
            f"{TECHNICAL_DATA_TEMPLATE}. Report which of its mandatory slots are still "
            "empty. List them by idShortPath."
        ),
        check=_check_complete,
        setup=_setup_complete,
        teardown=purge_scratch,
        max_steps=10,
    ),
    Task(
        name="validate",
        instruction=(
            f"Check the submodel {SCRATCH_PREFIX}sm:broken against the AAS metamodel "
            "constraints. Report every violation you find, naming the rule and the "
            "element it applies to."
        ),
        check=_check_validate,
        setup=_setup_validate,
        teardown=purge_scratch,
        max_steps=8,
    ),
    Task(
        name="chain",
        instruction=(
            "Build a Digital Product Passport from ./datasheet.pdf. Extract the datasheet, "
            f"instantiate the template {TECHNICAL_DATA_TEMPLATE} into the submodel "
            f"{SCRATCH_PREFIX}sm:chain with idShort TechnicalData including optional slots, "
            "match the extracted entities into it and apply the result, holding back any "
            "match that carries a warning. Report which mandatory slots remain empty."
        ),
        check=_check_chain,
        teardown=purge_scratch,
        fixtures=["datasheet.pdf"],
        max_steps=22,
    ),
    Task(
        name="batch",
        instruction=(
            "The file ./batch.csv holds 30 name/value pairs. Create one Property in "
            f"the submodel {SCRATCH_PREFIX}sm:batch for every row, with the row's name "
            "as its idShort and its value as the value, valueType xs:double. "
            "Report how many you wrote."
        ),
        check=_check_batch,
        setup=_setup_batch,
        teardown=purge_scratch,
        fixtures=[],
        max_steps=18,
    ),
    Task(
        name="prune",
        instruction=(
            f"The submodel {SCRATCH_PREFIX}sm:prune has some Properties holding values "
            "and some holding nothing. Delete every Property whose value is empty and "
            "leave the rest alone. Report which ones you removed."
        ),
        check=_check_prune,
        setup=_setup_prune,
        teardown=purge_scratch,
        max_steps=16,
    ),
    Task(
        name="locate",
        instruction=(
            "Somewhere in this repository there is a simulated injection moulding "
            "machine. Find the element holding the temperature of its first barrel "
            "heating zone and report its exact idShortPath and which submodel it is in."
        ),
        check=_check_locate,
        max_steps=10,
    ),
]
