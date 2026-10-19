"""Fixtures for tests that run against a real BaSyx.

The layer these tests occupy did not exist. Above them are the unit tests,
which run against a stub; below them the agent tasks in `a4s/experiment`,
which need a model and cost tokens. Neither catches a protocol-level
disagreement between what this code believes BaSyx does and what it does.

The percent-encoded list index is the case that argues for the layer. `[` and
`]` are gen-delims in RFC 3986 and BaSyx rejects them raw, but the stub the
unit tests run against accepted either form -- so every index-addressed write
was broken against a real repository, invisibly, until someone tried one. A
test at this level would have failed the day the code was written.

**Everything created here is prefixed and purged by prefix.** Not by a
hand-kept list: a test that fails part-way through leaves elements behind, and
the next run has to start from a known state. `purge()` refuses to delete
anything outside the prefix, so pointing these tests at a populated repository
cannot damage it.

Run them with::

    pytest -m integration                     # against $A4S_AAS_ENDPOINT
    A4S_TEST_ENDPOINT=http://localhost:8091 pytest -m integration

They skip, rather than fail, when no repository answers. A laptop without
docker running should still be able to run `pytest`.
"""

from __future__ import annotations

import contextlib
import io
import os
import uuid
from dataclasses import dataclass
from typing import Any, Dict, Iterator, List, Optional, Sequence

import pytest

from a4s.aasctl.client import AasHttpError, BasyxClient, Endpoint
from a4s.aasctl.paths import ElementPath

# Every id these tests create starts with this. `purge` asserts on it before
# deleting anything, so a mistyped endpoint cannot take out real plant data.
TEST_PREFIX = "urn:a4s:it:"

PROBE_TIMEOUT = 4.0


def test_endpoint() -> Optional[str]:
    """The repository to test against, or None if none is configured.

    `A4S_TEST_ENDPOINT` is separate from `A4S_AAS_ENDPOINT` so a second,
    disposable BaSyx can be pointed at without changing the variable every
    other command reads. It falls back to the main one because most of the
    time there is only one.

    `.env` is loaded first, for the same reason every other entry point loads
    it: on this project the endpoint normally lives there rather than in the
    shell. Reading `os.environ` alone would skip the whole suite on a
    correctly configured machine, which is the worst kind of green.
    """
    from a4s.kernel.config import load_env

    load_env()
    return os.getenv("A4S_TEST_ENDPOINT") or os.getenv("A4S_AAS_ENDPOINT")


# ---------------------------------------------------------------------------
# running the CLI
# ---------------------------------------------------------------------------

@dataclass
class AasRun:
    """What one `aas` invocation produced."""

    code: int
    stdout: str
    stderr: str
    argv: Sequence[str]

    @property
    def ok(self) -> bool:
        return self.code == 0

    def json(self) -> Any:
        import json

        return json.loads(self.stdout)

    def __str__(self) -> str:  # pragma: no cover - diagnostics only
        return (
            f"$ aas {' '.join(self.argv)}\n"
            f"[exit {self.code}]\n{self.stdout}{self.stderr}"
        )


def run_aas(*argv: str, expect_ok: bool = True) -> AasRun:
    """Invoke the `aas` CLI in-process and capture what it printed.

    In-process rather than through `subprocess`: the parser, the element
    constructors, the path algebra and the HTTP client are all exercised
    either way, and a process launch per assertion would make a 55-case matrix
    slow enough that nobody runs it. What is *not* exercised is the PATH
    lookup and the shell quoting -- those belong to the harness, and
    `a4s/environments/local.py` owns them.

    `fail()` raises SystemExit, which is the CLI's normal way of reporting an
    error to an agent, so it is caught and turned into an exit code rather
    than allowed to end the test session.
    """
    from a4s.aasctl.__main__ import main

    out, err = io.StringIO(), io.StringIO()
    code = 0
    with contextlib.redirect_stdout(out), contextlib.redirect_stderr(err):
        try:
            code = main(list(argv)) or 0
        except SystemExit as exc:
            code = exc.code if isinstance(exc.code, int) else 1

    run = AasRun(code=code, stdout=out.getvalue(), stderr=err.getvalue(), argv=argv)
    if expect_ok and not run.ok:
        pytest.fail(f"expected success, got exit {run.code}\n{run}")
    return run


# ---------------------------------------------------------------------------
# the repository
# ---------------------------------------------------------------------------

def purge(client: BasyxClient) -> int:
    """Delete every submodel and shell under `TEST_PREFIX`. Returns the count.

    The assertion is not defensive programming for its own sake. These tests
    are meant to be runnable against the same populated repository the agent
    runs against, because that is where the interesting shapes live -- and a
    purge that could reach outside its prefix would make that unsafe.
    """
    removed = 0
    for submodel in client.list_submodels(limit=500):
        identifier = submodel.get("id") or ""
        if not identifier.startswith(TEST_PREFIX):
            continue
        assert identifier.startswith(TEST_PREFIX), "purge escaped its prefix"
        with contextlib.suppress(AasHttpError):
            client.delete_submodel(identifier)
            removed += 1
    for shell in client.list_shells(limit=500):
        identifier = shell.get("id") or ""
        if not identifier.startswith(TEST_PREFIX):
            continue
        assert identifier.startswith(TEST_PREFIX), "purge escaped its prefix"
        with contextlib.suppress(AasHttpError):
            client.delete_shell(identifier)
            removed += 1
    return removed


@pytest.fixture(scope="session")
def endpoint() -> str:
    """Skip the whole integration suite when no repository answers."""
    import httpx

    base = test_endpoint()
    if not base:
        pytest.skip("no A4S_TEST_ENDPOINT or A4S_AAS_ENDPOINT set")

    try:
        response = httpx.get(
            f"{base.rstrip('/')}/shells", params={"limit": 1}, timeout=PROBE_TIMEOUT
        )
        response.raise_for_status()
    except Exception as exc:  # noqa: BLE001 - any failure means the same thing
        pytest.skip(f"{base} unreachable: {type(exc).__name__}: {exc}")

    # Every command below reads the endpoint from the environment rather than
    # taking it as an argument, so it has to be in place before the first one
    # runs. Set unconditionally: A4S_TEST_ENDPOINT is meant to win here.
    os.environ["A4S_AAS_ENDPOINT"] = base
    return base


@pytest.fixture(scope="session")
def session_client(endpoint: str) -> Iterator[BasyxClient]:
    with BasyxClient(Endpoint.from_env(endpoint)) as client:
        purge(client)          # a previous run that crashed leaves debris
        yield client
        purge(client)


@pytest.fixture
def client(session_client: BasyxClient) -> BasyxClient:
    return session_client


@pytest.fixture
def scratch(client: BasyxClient) -> Iterator[str]:
    """An empty submodel with a unique id, deleted afterwards.

    Unique per test rather than shared: a matrix of 55 cases that all write
    into one submodel would make every failure look like it might have been
    caused by the case before it.
    """
    submodel_id = f"{TEST_PREFIX}sm:{uuid.uuid4().hex[:10]}"
    run_aas("sm", "create", submodel_id, "--id-short", "ScratchSubmodel")
    try:
        yield submodel_id
    finally:
        with contextlib.suppress(AasHttpError):
            client.delete_submodel(submodel_id)


@pytest.fixture
def scratch_shell(client: BasyxClient) -> Iterator[str]:
    shell_id = f"{TEST_PREFIX}aas:{uuid.uuid4().hex[:10]}"
    run_aas("shell", "create", shell_id, "--id-short", "ScratchShell")
    try:
        yield shell_id
    finally:
        with contextlib.suppress(AasHttpError):
            client.delete_shell(shell_id)


# ---------------------------------------------------------------------------
# assertions
# ---------------------------------------------------------------------------

def element_at(client: BasyxClient, submodel_id: str, path: str) -> Dict[str, Any]:
    return client.get_element(submodel_id, ElementPath.parse(path))


def value_at(client: BasyxClient, submodel_id: str, path: str) -> Any:
    return client.get_value(submodel_id, ElementPath.parse(path))


def assert_absent(client: BasyxClient, submodel_id: str, path: str) -> None:
    """A deleted element must be gone, not merely empty.

    An endpoint that answers 200 with a null value for a path that no longer
    exists is a different failure from one that answers 404, and an agent
    reading the two makes different decisions: the first reads as "the element
    is there and unset", the second as "create it".
    """
    try:
        element = element_at(client, submodel_id, path)
    except AasHttpError as exc:
        assert exc.status == 404, f"expected 404 for a deleted element, got {exc.status}"
        return
    pytest.fail(f"{path} still readable after deletion: {element}")


def children_of(client: BasyxClient, submodel_id: str, path: str = "") -> List[Dict[str, Any]]:
    node = (
        client.get_submodel(submodel_id)
        if not path
        else element_at(client, submodel_id, path)
    )
    for key in ("submodelElements", "value", "statements"):
        children = node.get(key)
        if isinstance(children, list):
            return [c for c in children if isinstance(c, dict)]
    return []
