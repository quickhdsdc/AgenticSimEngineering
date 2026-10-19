"""Isolated loopback OPC test server. Never connects to a PLC or AAS."""

import asyncio
import threading

ENDPOINT = "opc.tcp://127.0.0.1:4849/a4s-test/"
PATHS = ("Press.Fault", "Turning.Fault", "Press.Ready", "Turning.Ready", "Press.Fault01",
         "Press.PressureBar", "Press.FlowLMin", "Safety.EmergencyStop")


def signal_values(command):
    leak = command in ("fault", "leak")
    estop = command == "estop"
    return {"Press.Fault": leak or estop, "Turning.Fault": estop,
            "Press.Ready": not (leak or estop), "Turning.Ready": not estop,
            "Press.Fault01": leak, "Press.PressureBar": 3.8 if leak else 6.0,
            "Press.FlowLMin": 48.0 if leak else 12.0, "Safety.EmergencyStop": estop}


def sources():
    return [{"path": path, "tag": "SIMULATED." + path, "node": "ns=2;s=" + path,
             "endpoint": ENDPOINT} for path in PATHS]


class SimulatorController:
    """Own one loopback server in the backend process; never touch a PLC/AAS."""

    def __init__(self):
        self._lock = threading.RLock()
        self._thread = None
        self._loop = None
        self._stop = None
        self._nodes = {}
        self._mode = "normal"
        self._error = None
        self._ready = threading.Event()

    def status(self):
        with self._lock:
            running = bool(self._thread and self._thread.is_alive() and self._nodes)
            return {"running": running, "mode": self._mode, "endpoint": ENDPOINT,
                    "error": self._error, "nodes": [
                        {**item, "value": signal_values(self._mode)[item["path"]]}
                        for item in sources()]}

    def start(self):
        with self._lock:
            if self._thread and self._thread.is_alive():
                return self.status()
            self._ready.clear()
            self._error = None
            self._mode = "normal"
            self._thread = threading.Thread(target=self._thread_main, name="a4s-opc-simulator", daemon=True)
            self._thread.start()
        if not self._ready.wait(10):
            raise RuntimeError("Simulated OPC server did not start within 10 seconds")
        state = self.status()
        if not state["running"]:
            raise RuntimeError(state["error"] or "Simulated OPC server did not start")
        return state

    def _thread_main(self):
        try:
            asyncio.run(self._serve_managed())
        except Exception as exc:
            with self._lock:
                self._error = f"{type(exc).__name__}: {exc}"
            self._ready.set()
        finally:
            with self._lock:
                self._loop = None
                self._stop = None
                self._nodes = {}

    async def _serve_managed(self):
        from asyncua import Server, ua

        server = Server()
        await server.init()
        server.set_endpoint(ENDPOINT)
        server.set_security_policy([ua.SecurityPolicyType.NoSecurity])
        index = await server.register_namespace("urn:a4s:isolated-opc-test")
        if index != 2:
            raise RuntimeError("Unexpected OPC namespace index")
        obj = await server.nodes.objects.add_object(index, "SimulatedFesto")
        nodes = {path: await obj.add_variable(ua.NodeId(path, index), path, value)
                 for path, value in signal_values("normal").items()}
        async with server:
            with self._lock:
                self._loop = asyncio.get_running_loop()
                self._stop = asyncio.Event()
                self._nodes = nodes
            self._ready.set()
            await self._stop.wait()

    def set_mode(self, mode):
        if mode not in ("normal", "fault", "leak", "estop"):
            raise ValueError("Unknown simulated OPC mode")
        with self._lock:
            if not self._loop or not self._nodes:
                raise RuntimeError("Start the simulated OPC server first")
            future = asyncio.run_coroutine_threadsafe(self._write_mode(mode), self._loop)
        future.result(timeout=10)
        return self.status()

    async def _write_mode(self, mode):
        for path, value in signal_values(mode).items():
            await self._nodes[path].write_value(value)
        with self._lock:
            self._mode = mode

    def stop(self):
        with self._lock:
            loop, event, thread = self._loop, self._stop, self._thread
            if loop and event:
                loop.call_soon_threadsafe(event.set)
        if thread and thread.is_alive():
            thread.join(timeout=10)
        return self.status()


controller = SimulatorController()


async def serve():
    from asyncua import Server, ua

    server = Server()
    await server.init()
    server.set_endpoint(ENDPOINT)
    server.set_security_policy([ua.SecurityPolicyType.NoSecurity])
    index = await server.register_namespace("urn:a4s:isolated-opc-test")
    assert index == 2
    obj = await server.nodes.objects.add_object(index, "SimulatedFesto")
    nodes = {}
    for path, value in signal_values("normal").items():
        nodes[path] = await obj.add_variable(ua.NodeId(path, index), path, value)
    async with server:
        print("SIMULATED OPC only:", ENDPOINT, flush=True)
        print("Start UI monitoring on Test OPC. Enter fault/leak, estop, normal, or quit here.", flush=True)
        while True:
            command = (await asyncio.to_thread(input, "> ")).strip().lower()
            if command in ("quit", "exit"):
                break
            if command not in ("fault", "leak", "estop", "normal"):
                print("Commands: fault/leak | estop | normal | quit", flush=True)
                continue
            for path, value in signal_values(command).items():
                await nodes[path].write_value(value)
            print("SIMULATED", command.upper(), flush=True)


if __name__ == "__main__":
    try:
        asyncio.run(serve())
    except (KeyboardInterrupt, EOFError):
        pass
