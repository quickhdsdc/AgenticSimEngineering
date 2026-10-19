"""The AAS layer: path algebra, element construction, validation, compression."""

from __future__ import annotations

from typing import Any, Dict, List

import pytest

from a4s.aasctl.model import ModelError, delegation_qualifier, make_element, make_shell
from a4s.aasctl.paths import ElementPath, PathError, decode_id, encode_id
from a4s.kernel import context as ctx


# ---------------------------------------------------------------------------
# paths
# ---------------------------------------------------------------------------

@pytest.mark.parametrize(
    "raw",
    [
        "Nameplate",
        "Nameplate.ManufacturerName",
        "TechnicalData.ProductClassifications[0].ProductClassId",
        "A[3].B[0].C",
        "OperatingData.Measurements[12]",
    ],
)
def test_paths_round_trip(raw: str) -> None:
    assert ElementPath.parse(raw).render() == raw


def test_list_index_is_a_first_class_step() -> None:
    """SubmodelElementList children are addressed by position, not idShort.

    The predecessor system indexed only by idShort and so could not reach
    into a list at all; this is the case that regressed there.
    """
    path = ElementPath.parse("Measurements[2].Value")
    assert path.steps == ("Measurements", 2, "Value")
    assert ElementPath.parse("Measurements[2]").addresses_list_item
    # A list child has no idShort of its own, so the owning list names it.
    assert ElementPath.parse("Measurements[2]").leaf_id_short == "Measurements"


def test_parent_and_child_navigation() -> None:
    path = ElementPath.parse("A.B.C")
    assert path.parent.render() == "A.B"
    assert path.parent.child("D").render() == "A.B.D"
    assert ElementPath.parse("A").is_ancestor_of(path)


@pytest.mark.parametrize("raw", ["A..B", "[0].A", "A.", "A[x]"])
def test_malformed_paths_are_rejected(raw: str) -> None:
    with pytest.raises(PathError):
        ElementPath.parse(raw)


def test_identifier_encoding_round_trips() -> None:
    identifier = "https://example.com/aas/1?x=1&y=2"
    assert decode_id(encode_id(identifier)) == identifier


# ---------------------------------------------------------------------------
# element construction
# ---------------------------------------------------------------------------

def test_property_carries_value_type_and_semantics() -> None:
    element = make_element(
        "Property", "MaxTemperature", value=250, value_type="xs:double",
        semantic_id="0173-1#02-AAO677#002",
    )
    assert element["valueType"] == "xs:double"
    assert element["value"] == "250"
    assert element["semanticId"]["keys"][0]["value"] == "0173-1#02-AAO677#002"


def test_list_requires_its_element_type() -> None:
    """typeValueListElement is mandatory: a list without it is unreadable."""
    with pytest.raises(ModelError):
        make_element("SubmodelElementList", "Measurements")


def test_operation_binds_to_a_service_through_a_qualifier() -> None:
    element = make_element(
        "Operation", "Disassemble", qualifiers=[delegation_qualifier("http://rds:5010/go")]
    )
    assert element["qualifiers"][0]["type"] == "invocationDelegation"
    assert element["qualifiers"][0]["value"] == "http://rds:5010/go"


def test_shell_always_has_asset_information() -> None:
    """BaSyx-Java tolerates a null here; the strict Python SDK refuses to read it."""
    shell = make_shell("urn:x:aas:1", "P389")
    assert shell["assetInformation"]["globalAssetId"] == "urn:x:aas:1"


def test_unknown_model_type_is_refused() -> None:
    with pytest.raises(ModelError):
        make_element("NotAThing", "X")


# ---------------------------------------------------------------------------
# validation
# ---------------------------------------------------------------------------

def _validate(submodel: Dict[str, Any]) -> List[Dict[str, str]]:
    """Drive cmd_validate against a fixed submodel via a stub client."""
    import argparse

    from a4s.aasctl import __main__ as cli

    captured: Dict[str, Any] = {}

    class StubClient:
        def __enter__(self): return self
        def __exit__(self, *exc): return None
        def get_submodel(self, _id): return submodel

    cli._client = lambda _args: StubClient()  # type: ignore[assignment]

    class Capture(cli.Out):
        def lines(self, payload, lines): captured.update(payload)

    cli.cmd_validate(argparse.Namespace(submodel_id="urn:x:sm:1", endpoint=None), Capture(False))
    return captured["findings"]


def test_validate_flags_an_id_short_on_a_list_child() -> None:
    findings = _validate(
        {
            "modelType": "Submodel", "id": "urn:x:sm:1", "idShort": "T",
            "submodelElements": [
                {
                    "modelType": "SubmodelElementList", "idShort": "Items",
                    "typeValueListElement": "Property",
                    "value": [{"modelType": "Property", "idShort": "Illegal", "valueType": "xs:string"}],
                }
            ],
        }
    )
    assert any(f["rule"] == "AASd-120" for f in findings)


def test_validate_flags_a_list_child_of_the_wrong_type() -> None:
    findings = _validate(
        {
            "modelType": "Submodel", "id": "urn:x:sm:1", "idShort": "T",
            "submodelElements": [
                {
                    "modelType": "SubmodelElementList", "idShort": "Items",
                    "typeValueListElement": "Property",
                    "value": [{"modelType": "Range", "valueType": "xs:int"}],
                }
            ],
        }
    )
    assert any(f["rule"] == "AASd-108" for f in findings)


def test_validate_passes_a_clean_submodel() -> None:
    findings = _validate(
        {
            "modelType": "Submodel", "id": "urn:x:sm:1", "idShort": "T",
            "submodelElements": [
                {"modelType": "Property", "idShort": "Name", "valueType": "xs:string", "value": "x"}
            ],
        }
    )
    assert findings == []


def _prop(id_short: str, value: str = "x") -> Dict[str, Any]:
    return {
        "modelType": "Property", "idShort": id_short,
        "valueType": "xs:string", "value": value,
    }


def test_validate_flags_two_siblings_sharing_an_id_short() -> None:
    """AASd-022, and the reason it is checked here rather than trusted upstream.

    BaSyx does not enforce uniqueness among siblings. It appends the duplicate
    and the path then addresses neither -- so the damage is present in the
    submodel, absent from every path-addressed read, and reported by nothing
    unless this rule reports it.
    """
    findings = _validate(
        {
            "modelType": "Submodel", "id": "urn:x:sm:1", "idShort": "T",
            "submodelElements": [_prop("Twin", "first"), _prop("Twin", "second")],
        }
    )

    duplicates = [f for f in findings if f["rule"] == "AASd-022"]
    assert len(duplicates) == 1, findings
    assert duplicates[0]["path"] == "Twin"
    assert "2 siblings" in duplicates[0]["detail"]


def test_validate_flags_a_duplicate_nested_in_a_collection() -> None:
    """The nested case is the one BaSyx accepts with HTTP 200 and no error.

    At a submodel root the caller at least gets a 404 to react to; one level
    down the write reports success. A rule that only looked at the root would
    miss the case with nothing else to catch it.
    """
    findings = _validate(
        {
            "modelType": "Submodel", "id": "urn:x:sm:1", "idShort": "T",
            "submodelElements": [
                {
                    "modelType": "SubmodelElementCollection", "idShort": "Box",
                    "value": [_prop("Inner", "a"), _prop("Inner", "b")],
                }
            ],
        }
    )

    duplicates = [f for f in findings if f["rule"] == "AASd-022"]
    assert len(duplicates) == 1, findings
    assert duplicates[0]["path"] == "Box.Inner"


def test_validate_does_not_flag_list_children_that_share_nothing() -> None:
    """An SML child has no idShort, so it cannot collide and must not be flagged.

    A uniqueness rule that did not exempt lists would report every list in the
    repository as broken -- and `SubmodelElementList` is the type this CLI
    exists to reach.
    """
    findings = _validate(
        {
            "modelType": "Submodel", "id": "urn:x:sm:1", "idShort": "T",
            "submodelElements": [
                {
                    "modelType": "SubmodelElementList", "idShort": "Items",
                    "typeValueListElement": "Property",
                    "value": [
                        {"modelType": "Property", "valueType": "xs:string", "value": "a"},
                        {"modelType": "Property", "valueType": "xs:string", "value": "b"},
                    ],
                }
            ],
        }
    )

    assert findings == []


def test_validate_counts_three_copies_as_one_finding() -> None:
    """One finding per colliding name, not one per copy.

    An agent that reads three findings for one problem tries three fixes. The
    detail carries the count instead.
    """
    findings = _validate(
        {
            "modelType": "Submodel", "id": "urn:x:sm:1", "idShort": "T",
            "submodelElements": [_prop("Twin"), _prop("Twin"), _prop("Twin")],
        }
    )

    duplicates = [f for f in findings if f["rule"] == "AASd-022"]
    assert len(duplicates) == 1
    assert "3 siblings" in duplicates[0]["detail"]


# ---------------------------------------------------------------------------
# context compression
# ---------------------------------------------------------------------------

def test_elision_reclaims_old_successes_but_keeps_failures() -> None:
    """A stale error still earns its tokens: it stops the agent retrying."""
    messages = [
        {"role": "system", "content": "s"},
        {"role": "tool", "tool_call_id": "1", "content": "x" * 8000},
        {"role": "tool", "tool_call_id": "2", "content": "error: 404 on Nameplate.Foo" + "y" * 8000},
        *[{"role": "assistant", "content": "step"} for _ in range(8)],
    ]
    compressed, reclaimed = ctx.elide_old_tool_results(messages, keep_recent=4)

    assert reclaimed > 0
    assert compressed[1]["content"] == ctx.ELIDED
    assert compressed[2]["content"].startswith("error: 404")


def test_compression_is_a_no_op_below_the_trigger() -> None:
    messages = [{"role": "user", "content": "short"}]
    out, actions = ctx.compress(messages, ctx.Budget(context_window=128_000))
    assert out == messages and actions == []


def test_summary_keeps_the_original_instruction_verbatim() -> None:
    """A paraphrased goal is how a long run ends up solving a different problem."""
    messages = [
        {"role": "system", "content": "sys"},
        {"role": "user", "content": "Onboard printer 389 with its Capability submodel"},
        *[{"role": "assistant", "content": f"work {i}"} for i in range(30)],
    ]
    out = ctx.summarise_span(messages, lambda _t: "SUMMARY", keep_recent=5, keep_leading=2)

    assert out[1]["content"] == "Onboard printer 389 with its Capability submodel"
    assert "SUMMARY" in out[2]["content"]
    assert len(out) < len(messages)


def test_compression_refuses_to_silently_truncate() -> None:
    """Better to fail loudly than to hand the model a half-cut tool result."""
    messages = [{"role": "user", "content": "z" * 400_000}]
    with pytest.raises(RuntimeError, match="after compression"):
        ctx.compress(messages, ctx.Budget(context_window=8_000), lambda _t: "s")


# ---------------------------------------------------------------------------
# element URLs
# ---------------------------------------------------------------------------

def test_list_indices_are_percent_encoded_in_the_url() -> None:
    """`[` and `]` are RFC 3986 gen-delims. BaSyx answers a raw one with a bare
    400 and an HTML page; the stub these tests used to run against accepted
    either, which hid the break across the whole SubmodelElementList surface."""
    from a4s.aasctl.client import BasyxClient, Endpoint

    client = BasyxClient(Endpoint(base_url="http://x"))
    try:
        url = client._element_url("urn:sm", ElementPath.parse("Markings[1].MarkingName"))
    finally:
        client.close()
    assert url.endswith("/submodel-elements/Markings%5B1%5D.MarkingName")


def test_dots_stay_unescaped_as_the_path_separator() -> None:
    from a4s.aasctl.client import BasyxClient, Endpoint

    client = BasyxClient(Endpoint(base_url="http://x"))
    try:
        url = client._element_url("urn:sm", ElementPath.parse("A.B.C"))
    finally:
        client.close()
    assert url.endswith("/submodel-elements/A.B.C")


def test_an_empty_path_addresses_the_submodel_root() -> None:
    from a4s.aasctl.client import BasyxClient, Endpoint

    client = BasyxClient(Endpoint(base_url="http://x"))
    try:
        url = client._element_url("urn:sm", ElementPath.parse(""))
    finally:
        client.close()
    assert url.endswith("/submodel-elements")


def test_the_full_xsd_type_enumeration_is_accepted() -> None:
    """An incomplete list makes `aas validate` cry wolf on a valid IDTA
    template: the ZVEI TechnicalData template declares xs:gYearMonth."""
    from a4s.aasctl.model import VALUE_TYPES, make_element

    assert len(VALUE_TYPES) == 30
    for value_type in ("xs:gYearMonth", "xs:base64Binary", "xs:positiveInteger", "xs:hexBinary"):
        assert value_type in VALUE_TYPES
        element = make_element("Property", "X", value="1", value_type=value_type)
        assert element["valueType"] == value_type


def test_a_type_outside_the_enumeration_is_still_refused() -> None:
    from a4s.aasctl.model import ModelError, make_element

    with pytest.raises(ModelError, match="unknown valueType"):
        make_element("Property", "X", value="1", value_type="xs:money")


# ---------------------------------------------------------------------------
# delegation host remapping
# ---------------------------------------------------------------------------

def test_a_container_hostname_can_be_mapped_to_the_host(monkeypatch) -> None:
    """A delegation URL is written for the network the service lives on. From
    outside docker `printer-service` does not resolve, though the service is up
    on localhost:5060 -- neither the model nor the service is wrong."""
    from a4s.aasctl.__main__ import remap_host

    monkeypatch.setenv("A4S_DELEGATION_HOSTS", "printer-service=localhost")
    assert (
        remap_host("http://printer-service:5060/print/classify-defect")
        == "http://localhost:5060/print/classify-defect"
    )


def test_a_replacement_port_overrides_the_original(monkeypatch) -> None:
    from a4s.aasctl.__main__ import remap_host

    monkeypatch.setenv("A4S_DELEGATION_HOSTS", "mes-service=localhost:15030")
    assert remap_host("http://mes-service:5030/records") == "http://localhost:15030/records"


def test_unmapped_hosts_pass_through(monkeypatch) -> None:
    """The default has to be exactly what the model says."""
    from a4s.aasctl.__main__ import remap_host

    monkeypatch.setenv("A4S_DELEGATION_HOSTS", "printer-service=localhost")
    assert remap_host("http://other-service:5000/x") == "http://other-service:5000/x"


def test_no_mapping_means_no_rewriting(monkeypatch) -> None:
    from a4s.aasctl.__main__ import remap_host

    monkeypatch.delenv("A4S_DELEGATION_HOSTS", raising=False)
    assert remap_host("http://printer-service:5060/x") == "http://printer-service:5060/x"
    assert remap_host(None) is None


def test_several_mappings_are_accepted(monkeypatch) -> None:
    from a4s.aasctl.__main__ import remap_host

    monkeypatch.setenv("A4S_DELEGATION_HOSTS", "a-svc=localhost,b-svc=127.0.0.1:9000")
    assert remap_host("http://b-svc:80/p") == "http://127.0.0.1:9000/p"
    assert remap_host("http://a-svc:81/p") == "http://localhost:81/p"


def test_every_transport_failure_is_caught_not_just_three() -> None:
    """A BaSyx that accepted a connection then dropped it raised
    RemoteProtocolError, which fell through to a Python traceback."""
    import httpx

    from a4s.aasctl.__main__ import httpx_connect_errors

    caught = httpx_connect_errors()
    for kind in (
        httpx.ConnectError, httpx.ConnectTimeout, httpx.ReadTimeout,
        httpx.RemoteProtocolError, httpx.ReadError, httpx.WriteError,
    ):
        assert issubclass(kind, caught)


# ---------------------------------------------------------------------------
# how long to wait, and for what
# ---------------------------------------------------------------------------

def test_connecting_and_reading_have_different_deadlines() -> None:
    """A host that will not accept a connection is down; waiting longer does
    not change that. A repository that *has* answered may legitimately spend
    twenty seconds on a large submodel.

    Windows' own connect timeout is 21 seconds, which is what a mistyped
    endpoint in the web UI cost: twenty-one seconds of frozen page to learn
    that a host does not exist.
    """
    from a4s.aasctl.client import CONNECT_TIMEOUT, DEFAULT_TIMEOUT, BasyxClient, Endpoint

    client = BasyxClient(Endpoint(base_url="http://127.0.0.1:9"))
    try:
        timeout = client._http.timeout
        assert timeout.connect == CONNECT_TIMEOUT
        assert timeout.read == DEFAULT_TIMEOUT
        assert CONNECT_TIMEOUT < DEFAULT_TIMEOUT
    finally:
        client.close()


def test_a_short_overall_timeout_is_not_lengthened_by_the_connect_floor() -> None:
    """`timeout=1` means one second, not five. A caller that asked for a tight
    budget gets it."""
    from a4s.aasctl.client import BasyxClient, Endpoint

    client = BasyxClient(Endpoint(base_url="http://127.0.0.1:9"), timeout=1.0)
    try:
        assert client._http.timeout.connect == 1.0
    finally:
        client.close()


def test_a_slash_means_the_root_of_the_submodel() -> None:
    """`/` is how a person -- and an agent -- writes "the root". It is not
    idShortPath syntax, which has no leading separator, and sending it on
    produced a 400 whose body was a Tomcat error page. No idShort may contain
    a slash (AASd-002), so it can mean nothing else."""
    from a4s.aasctl.paths import ElementPath

    assert ElementPath.parse("/").steps == ()
    assert ElementPath.parse("/") == ElementPath.parse("")
    assert ElementPath.parse(".").steps == ()
    # And a real path is unaffected.
    assert len(ElementPath.parse("Press.Ready").steps) == 2
