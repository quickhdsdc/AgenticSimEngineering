"""The machine's own HTTP service -- what its AAS Operations delegate to.

This is the shape the architecture assumes for an edge node: the machine
offers *capabilities* over HTTP, and the AAS holds the semantic handle. An
Operation on the machine's Capability submodel carries an
``invocationDelegation`` qualifier pointing at one of these endpoints, so
``aas op ls`` discovers what the machine can do and ``aas op invoke`` does it.
No agent logic lives here, and nothing here knows what an LLM is -- the same
separation the MachineAgent this stands in for was built around.

Standard library only. The harness pins its dependencies exactly because a
transitive change under a shopfloor deployment is not acceptable, and a
simulator is a poor reason to add a web framework to that list.

Two endpoints are marked ``moves_machine`` in the schema: ``/start`` and
``/setpoint``. That flag is not decoration. It is what an approval gate reads
to decide whether an invocation needs a human, and it is published rather than
inferred because guessing which endpoints are dangerous from their names is
exactly the kind of thing that works until it does not.
"""

from __future__ import annotations

import json
import threading
from http.server import BaseHTTPRequestHandler, ThreadingHTTPServer
from pathlib import Path
from typing import Any, Callable, Dict, Optional, Tuple

from .machine import DEFAULT_DATASET, SETPOINT_LIMITS, SETTABLE, UNITS, Machine, MachineError

DEFAULT_PORT = 5080

# The service contract, published at /openapi.json. A delegated Operation's
# declared input variables can drift from the service that answers it -- it has
# happened in this repository's own data -- and a service that publishes its
# signature makes that difference checkable rather than a surprise at call time.
ENDPOINTS: Dict[str, Dict[str, Any]] = {
    "/status": {
        "method": "GET",
        "summary": "Live telemetry: every reading, its unit, and the active setpoints.",
        "inputs": {},
        "moves_machine": False,
    },
    "/model": {
        "method": "GET",
        "summary": "How the quality model was fitted and how well.",
        "inputs": {},
        "moves_machine": False,
    },
    "/read": {
        "method": "POST",
        "summary": "One named reading.",
        "inputs": {"name": "string, e.g. MaxInjectionPressure"},
        "moves_machine": False,
    },
    "/predict-quality": {
        "method": "POST",
        "summary": (
            "Predicted part weight, for supplied process parameters or for the "
            "machine's current state."
        ),
        "inputs": {"parameters": "object of process parameter -> number, optional"},
        "moves_machine": False,
    },
    "/start": {
        "method": "POST",
        "summary": "Start production.",
        "inputs": {},
        "moves_machine": True,
    },
    "/stop": {
        "method": "POST",
        "summary": "Stop production.",
        "inputs": {},
        "moves_machine": True,
    },
    "/setpoint": {
        "method": "POST",
        "summary": "Command a process setpoint. The process approaches it over several shots.",
        "inputs": {
            "name": f"string, one of: {', '.join(SETTABLE)}",
            "value": "number, within the endpoint's operating limits",
        },
        "moves_machine": True,
    },
    "/setpoint/clear": {
        "method": "POST",
        "summary": "Release a setpoint and let the process return to its measured value.",
        "inputs": {"name": "string"},
        "moves_machine": True,
    },
}


def routes(machine: Machine) -> Dict[Tuple[str, str], Callable[[Dict[str, Any]], Dict[str, Any]]]:
    """Map (method, path) to a handler over one machine."""

    def status(_: Dict[str, Any]) -> Dict[str, Any]:
        return machine.telemetry()

    def model(_: Dict[str, Any]) -> Dict[str, Any]:
        return machine.model.report()

    def read(body: Dict[str, Any]) -> Dict[str, Any]:
        name = str(body.get("name") or "")
        if not name:
            raise MachineError("read needs a 'name'")
        return {"name": name, "value": machine.reading(name), "unit": UNITS.get(name, "")}

    def predict(body: Dict[str, Any]) -> Dict[str, Any]:
        parameters = body.get("parameters")
        if parameters is not None and not isinstance(parameters, dict):
            raise MachineError("'parameters' must be an object of name -> number")
        return machine.predict_quality(parameters)

    def start(_: Dict[str, Any]) -> Dict[str, Any]:
        return machine.start()

    def stop(_: Dict[str, Any]) -> Dict[str, Any]:
        return machine.stop()

    def setpoint(body: Dict[str, Any]) -> Dict[str, Any]:
        name = str(body.get("name") or "")
        if "value" not in body:
            raise MachineError("setpoint needs a 'value'")
        try:
            value = float(body["value"])
        except (TypeError, ValueError):
            raise MachineError(f"'value' must be a number, got {body['value']!r}") from None
        return machine.set_point(name, value)

    def clear(body: Dict[str, Any]) -> Dict[str, Any]:
        return machine.clear_setpoint(str(body.get("name") or ""))

    def contract(_: Dict[str, Any]) -> Dict[str, Any]:
        return {
            "service": "injection-moulding-machine",
            "endpoints": ENDPOINTS,
            "settable": list(SETTABLE),
            "limits": {k: list(v) for k, v in SETPOINT_LIMITS.items()},
            "units": UNITS,
        }

    handlers = {
        "/status": status,
        "/model": model,
        "/openapi.json": contract,
        "/read": read,
        "/predict-quality": predict,
        "/start": start,
        "/stop": stop,
        "/setpoint": setpoint,
        "/setpoint/clear": clear,
    }
    # Every endpoint answers POST, and the read-only ones also answer GET.
    #
    # The first version served `/status` and `/model` on GET alone, because
    # they read nothing -- and `aas op invoke` could not call them, because an
    # Operation invocation is an RPC and it POSTs. The service was wrong, not
    # the CLI: an Operation is a procedure call, not a REST resource. GET is
    # kept on the safe ones so a person with a browser can still look.
    table: Dict[Tuple[str, str], Callable[[Dict[str, Any]], Dict[str, Any]]] = {
        ("POST", path): handler for path, handler in handlers.items()
    }
    for path in ("/status", "/model", "/openapi.json"):
        table[("GET", path)] = handlers[path]
    return table


def make_handler(machine: Machine) -> type:
    table = routes(machine)

    class Handler(BaseHTTPRequestHandler):
        protocol_version = "HTTP/1.1"
        server_version = "a4s-imm/1.0"

        def log_message(self, *_: Any) -> None:
            """Silence the per-request line; the ledger is the record that matters."""

        def _respond(self, status: int, payload: Dict[str, Any]) -> None:
            body = json.dumps(payload, ensure_ascii=False).encode("utf-8")
            self.send_response(status)
            self.send_header("Content-Type", "application/json; charset=utf-8")
            self.send_header("Content-Length", str(len(body)))
            self.end_headers()
            self.wfile.write(body)

        def _dispatch(self, method: str) -> None:
            handler = table.get((method, self.path.split("?", 1)[0]))
            if handler is None:
                self._respond(404, {
                    "error": f"no {method} {self.path}",
                    "available": sorted(f"{m} {p}" for m, p in table),
                })
                return
            body: Dict[str, Any] = {}
            length = int(self.headers.get("Content-Length") or 0)
            if length:
                try:
                    body = json.loads(self.rfile.read(length).decode("utf-8") or "{}")
                except json.JSONDecodeError as exc:
                    self._respond(400, {"error": f"body is not JSON: {exc}"})
                    return
            if not isinstance(body, dict):
                self._respond(400, {"error": "body must be a JSON object"})
                return
            try:
                self._respond(200, handler(body))
            except MachineError as exc:
                # The machine refusing is a 400, not a 500: the caller asked for
                # something the machine will not do, and the message says which.
                self._respond(400, {"error": str(exc)})
            except Exception as exc:  # noqa: BLE001 - never leak a traceback to a caller
                self._respond(500, {"error": f"{type(exc).__name__}: {exc}"})

        def do_GET(self) -> None:  # noqa: N802 - BaseHTTPRequestHandler's contract
            self._dispatch("GET")

        def do_POST(self) -> None:  # noqa: N802
            self._dispatch("POST")

    return Handler


def serve(
    port: int = DEFAULT_PORT,
    dataset: Path = DEFAULT_DATASET,
    seconds_per_shot: float = 2.0,
    autostart: bool = True,
) -> ThreadingHTTPServer:
    """Start the service and return the server, already listening."""
    machine = Machine.build(dataset=dataset, seconds_per_shot=seconds_per_shot)
    if autostart:
        machine.start()
    server = ThreadingHTTPServer(("0.0.0.0", port), make_handler(machine))
    server.machine = machine  # type: ignore[attr-defined]
    threading.Thread(target=server.serve_forever, daemon=True).start()
    return server


def main(argv: Optional[list] = None) -> int:
    import argparse

    parser = argparse.ArgumentParser(
        prog="a4s-imm",
        description="A simulated injection moulding machine, replaying a real production run.",
    )
    parser.add_argument("--port", type=int, default=DEFAULT_PORT)
    parser.add_argument("--dataset", type=Path, default=DEFAULT_DATASET)
    parser.add_argument("--seconds-per-shot", type=float, default=2.0)
    parser.add_argument("--stopped", action="store_true", help="do not start production")
    args = parser.parse_args(argv)

    server = serve(
        port=args.port, dataset=args.dataset,
        seconds_per_shot=args.seconds_per_shot, autostart=not args.stopped,
    )
    machine: Machine = server.machine  # type: ignore[attr-defined]
    print(f"injection moulding machine on http://localhost:{args.port}")
    print(f"  {len(machine.shots)} shots from {args.dataset.name}, "
          f"one every {args.seconds_per_shot}s, running={machine.running}")
    print(f"  quality model: R2={machine.model.r_squared} RMSE={machine.model.rmse} "
          f"{UNITS.get(machine.model.target, '')}")
    print("  GET /openapi.json for the contract")
    try:
        threading.Event().wait()
    except KeyboardInterrupt:
        server.shutdown()
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
