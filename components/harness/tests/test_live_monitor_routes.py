import asyncio
import socket
from datetime import datetime, timezone
from types import SimpleNamespace
from unittest.mock import Mock

from a4s.aasctl import opc
from a4s.server import live_monitor_routes as routes


def test_ui_simulator_controls_serve_real_loopback_opc_nodes(monkeypatch):
    from a4s.machines.festo_cp_lab import opc_simulator

    with socket.socket() as reserved:
        reserved.bind(("127.0.0.1", 0))
        port = reserved.getsockname()[1]
    monkeypatch.setattr(opc_simulator, "ENDPOINT", f"opc.tcp://127.0.0.1:{port}/a4s-test/")
    controller = opc_simulator.SimulatorController()
    monkeypatch.setattr(routes, "simulator", controller)
    try:
        assert routes.simulator_start()["running"] is True
        node = "ns=2;s=Press.Fault"
        assert opc.read_evidence(opc_simulator.ENDPOINT, [node])[0]["value"] is False
        assert routes.simulator_state({"mode": "fault"})["mode"] == "fault"
        assert opc.read_evidence(opc_simulator.ENDPOINT, [node])[0]["value"] is True
        assert routes.simulator_state({"mode": "normal"})["mode"] == "normal"
        assert opc.read_evidence(opc_simulator.ENDPOINT, [node])[0]["value"] is False
    finally:
        assert routes.simulator_stop()["running"] is False


def test_simulation_uses_selected_instance_without_provisioning(monkeypatch):
    client = Mock()
    client.__enter__ = Mock(return_value=client)
    client.__exit__ = Mock(return_value=False)
    client.get_shell.return_value = {"idShort": "FestoWorkCenterStamping_362",
        "assetInformation": {"assetKind": "Instance"}, "submodels": []}
    monkeypatch.setattr(routes, "BasyxClient", lambda _: client)
    monitor = Mock()
    monkeypatch.setattr(routes, "service", monitor)
    selection = routes.target("urn:current:362", "http://aas.test", "simulator")
    routes.start(selection)
    assert monitor.start.call_args.args[0] == selection
    reader = Mock(return_value=[{"value": False, "status": "Good"}] * 8)
    monkeypatch.setattr(routes.opc, "read_evidence", reader)
    result = routes.read_target(selection)
    assert result["source"] == "simulator"
    assert len(result["values"]) == 8
    assert reader.call_args.args[0].startswith("opc.tcp://127.0.0.1:4849/")
    client.get_submodel.assert_not_called()
    client.create_shell.assert_not_called()
    client.create_submodel.assert_not_called()
    client.replace_submodel.assert_not_called()
    assert selection != routes.target("urn:current:362", "http://aas.test", "live")


def test_missing_selected_instance_is_not_recreated(monkeypatch):
    import pytest
    from fastapi import HTTPException
    client = Mock()
    client.__enter__ = Mock(return_value=client)
    client.__exit__ = Mock(return_value=False)
    client.get_shell.side_effect = routes.AasHttpError(404, "http://aas", "missing")
    monkeypatch.setattr(routes, "BasyxClient", lambda _: client)
    for source in ("live", "simulator"):
        with pytest.raises(HTTPException):
            routes.start({"asset_id": "missing", "endpoint": "http://aas", "source": source})
    client.create_shell.assert_not_called()
    client.create_submodel.assert_not_called()


def test_unknown_monitor_source_is_rejected():
    import pytest
    from fastapi import HTTPException
    with pytest.raises(HTTPException):
        routes.target("asset", "http://aas", "arbitrary")


def test_read_target_follows_actual_operational_reference_and_ignores_missing_context(monkeypatch):
    client = Mock()
    client.__enter__ = Mock(return_value=client)
    client.__exit__ = Mock(return_value=False)
    client.get_shell.return_value = {
        "id": "urn:dynamic",
        "idShort": "FestoWorkCenterStamping_967",
        "assetInformation": {"assetKind": "Instance"},
        "submodels": [
            {"keys": [{"type": "Submodel", "value": s}]}
            for s in ("urn:actual:opc", "urn:missing:context")
        ],
    }
    client.get_submodel.side_effect = [
        {"id": "urn:actual:opc", "idShort": "OperationalData", "kind": "Instance"},
        routes.AasHttpError(404, "http://aas.test", "missing"),
    ]
    monkeypatch.setattr(routes, "BasyxClient", lambda _: client)
    monkeypatch.setattr(
        routes,
        "_walk",
        lambda sm, prefix, found: found.extend(
            [{"path": "Press.Fault", "endpoint": "opc.tcp://test", "node": "ns=2;s=fault"}]
        ),
    )
    reader = Mock(return_value=[{"value": False, "status": "Good", "dataType": "Boolean"}])
    monkeypatch.setattr(routes.opc, "read_evidence", reader)
    result = routes.read_target({"asset_id": "urn:dynamic", "endpoint": "http://aas.test"})
    assert result["submodel"] == "urn:actual:opc"
    assert result["values"][0]["value"] is False
    reader.assert_called_once_with("opc.tcp://test", ["ns=2;s=fault"], timeout=8)
    client.get_shell.assert_called_once_with("urn:dynamic")


def test_opc_evidence_preserves_bad_quality_and_timestamps(monkeypatch):
    stamp = datetime(2026, 9, 23, tzinfo=timezone.utc)

    class Status:
        def is_good(self):
            return False

        def __str__(self):
            return "BadNoCommunication"

    class Node:
        async def read_data_value(self, raise_on_bad_status=True):
            assert raise_on_bad_status is False
            return SimpleNamespace(
                Value=SimpleNamespace(Value=True, VariantType=SimpleNamespace(name="Boolean")),
                StatusCode=Status(),
                SourceTimestamp=stamp,
                ServerTimestamp=stamp,
            )

    class Connection:
        async def __aenter__(self):
            return self

        async def __aexit__(self, *_):
            pass

        def get_node(self, _):
            return Node()

    monkeypatch.setattr(opc, "_connect", lambda *_: Connection())
    monkeypatch.setattr(opc, "_run", asyncio.run)
    result = opc.read_evidence("opc.tcp://test", ["ns=2;s=fault"])
    assert result[0]["status"] == "BadNoCommunication"
    assert result[0]["source_timestamp"] == stamp.isoformat()
    assert result[0]["dataType"] == "Boolean"
