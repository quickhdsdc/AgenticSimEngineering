import json

import httpx
import pytest

from a4s.mesctl.client import Config, MesClient, MesError
from a4s.mesctl.commands import build_parser, command_info
from a4s.mesctl.executor import execute


def run(words, handler):
    with MesClient(
        Config("http://mes.test", "secret"), transport=httpx.MockTransport(handler)
    ) as client:
        return execute(build_parser().parse_args(words.split()), client)


def test_create_readback_and_classification():
    body = {}

    def handler(req):
        if req.method == "POST":
            assert req.url.path == "/api/v1/maintenance-events"
            body.update(json.loads(req.content))
            return httpx.Response(201, json={"data": {"id": 9}})
        assert req.url.path == "/api/v1/maintenance-events/9"
        return httpx.Response(200, json={"data": {**body, "id": 9, "status": "PENDING"}})

    result = run("maintenance create --title Review --event-type corrective --line 7", handler)
    assert result["verified"]
    assert body["line_id"] == 7
    assert command_info("mes maintenance create --title Review --event-type corrective --line 7")[
        "mutates"
    ]
    assert not command_info(
        "mes maintenance create --title Review --event-type corrective --line 7 --dry-run"
    )["mutates"]


def test_dry_run_and_required_mes_target():
    def no_http(req):
        pytest.fail("Unexpected HTTP")

    result = run(
        "maintenance create --title Review --event-type corrective --line 7 --dry-run", no_http
    )
    assert result["dry_run"]
    with pytest.raises(MesError, match="Select a MES"):
        run("maintenance create --title Review --event-type corrective --dry-run", no_http)


@pytest.mark.parametrize(
    "verb,status", [("start", "IN_PROGRESS"), ("complete", "COMPLETED"), ("cancel", "CANCELLED")]
)
def test_transitions_are_read_back(verb, status):
    written = False

    def handler(req):
        nonlocal written
        if req.method == "POST":
            written = True
            return httpx.Response(200, json={"data": {"id": 9}})
        return httpx.Response(
            200,
            json={
                "data": {
                    "id": 9,
                    "status": status if written else "PENDING",
                    "resolution_notes": "Reviewed",
                }
            },
        )

    result = run(
        f"maintenance {verb} 9" + (" --resolution-notes Reviewed" if verb == "complete" else ""),
        handler,
    )
    assert result["verified"]


def test_write_verification_failure_is_not_replayed():
    writes = []

    def handler(req):
        if req.method == "POST":
            writes.append(req)
            return httpx.Response(201, json={"data": {"id": 9}})
        return httpx.Response(
            200, json={"data": {"id": 9, "title": "different", "status": "PENDING"}}
        )

    result = run("maintenance create --title Review --event-type corrective --line 7", handler)
    assert not result["verified"] and len(writes) == 1
