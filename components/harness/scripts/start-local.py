"""Start the installed local workspace without activating a venv or exporting variables.

Uses existing .env configuration; never rewrites secrets, initializes databases,
creates demo records or stops existing processes. Docker dependencies must have
been provisioned already. --agent runs a CLI task instead of starting the web UI.
"""

from __future__ import annotations

import argparse
import os
from pathlib import Path
import shutil
import socket
import subprocess
import sys
import time
from urllib.parse import urlsplit
import uuid
import webbrowser

import httpx

from a4s.kernel.config import load_env

ROOT = Path(__file__).resolve().parents[1]
BACKEND = "http://localhost:8000"
FRONTEND = "http://localhost:5173"
HIDDEN = subprocess.CREATE_NO_WINDOW if os.name == "nt" else 0


def configure():
    # Existing process variables still take precedence over the selected file.
    # Load before chdir so an explicit relative A4S_ENV_FILE works as expected.
    explicit = os.getenv("A4S_ENV_FILE")
    if explicit and not Path(explicit).is_file():
        raise RuntimeError("A4S_ENV_FILE does not point to a readable file")
    if explicit:
        explicit = str(Path(explicit).resolve())
        os.environ["A4S_ENV_FILE"] = explicit
    load_env(explicit or str(ROOT / ".env"))
    os.chdir(ROOT)
    os.environ.setdefault("A4S_AAS_ENDPOINT", "http://localhost:8081")
    paths = [str(Path(sys.executable).parent)]
    if os.name == "nt":
        candidates = [Path(os.getenv("LOCALAPPDATA", "")) / "Programs/Git/usr/bin",
                      Path(os.getenv("ProgramFiles", "C:/Program Files")) / "Git/usr/bin"]
        for candidate in candidates:
            if (candidate / "bash.exe").is_file():
                paths.append(str(candidate))
                break
    os.environ["PATH"] = os.pathsep.join(paths + [os.getenv("PATH", "")])
    os.environ.setdefault("PYTHONUTF8", "1")


def fetch(url):
    try:
        return httpx.get(url, timeout=3, trust_env=False)
    except httpx.HTTPError:
        return None


def aas_ready(endpoint):
    response = fetch(endpoint.rstrip("/") + "/shells?limit=1")
    if response is None or response.status_code != 200:
        return False
    try:
        body = response.json()
        return isinstance(body, dict) and isinstance(body.get("result"), list)
    except ValueError:
        return False


def mes_ready(endpoint):
    response = fetch(endpoint.rstrip("/") + "/api/health")
    return response is not None and response.status_code == 200


def backend_ready(endpoint):
    response = fetch(BACKEND + "/health")
    if response is None or response.status_code != 200:
        return False
    try:
        data = response.json()
        return data.get("ok") is True and data.get("repository", "").rstrip("/") == endpoint.rstrip("/")
    except (ValueError, AttributeError):
        return False


def frontend_ready():
    response = fetch(FRONTEND + "/src/components/RecoveryWorkspace.vue")
    return response is not None and response.status_code == 200 and "Phone housing production" in response.text


def occupied(port):
    for host in ("127.0.0.1", "::1"):
        try:
            with socket.create_connection((host, port), timeout=.5):
                return True
        except OSError:
            pass
    return False


def local_endpoint(endpoint, port):
    try:
        parsed = urlsplit(endpoint)
        return (parsed.scheme == "http" and parsed.hostname in ("localhost", "127.0.0.1")
                and parsed.port == port and parsed.path in ("", "/")
                and not parsed.username and not parsed.password and not parsed.query and not parsed.fragment)
    except ValueError:
        return False


def wait_ready(name, probe, process=None, seconds=90):
    deadline = time.monotonic() + seconds
    last_notice = 0
    while time.monotonic() < deadline:
        if probe():
            print(f"[ready] {name}", flush=True)
            return
        if process is not None and process.poll() is not None:
            raise RuntimeError(f"{name} exited. Inspect this startup's log directory")
        if time.monotonic() - last_notice > 10:
            print(f"[wait] {name} ...", flush=True)
            last_notice = time.monotonic()
        time.sleep(1)
    raise RuntimeError(f"{name} did not become ready; inspect logs. No process was killed or replayed")


def start_compose(name, arguments, log_dir):
    docker = shutil.which("docker")
    if not docker:
        raise RuntimeError(f"{name} is unavailable. Start Docker Desktop first")
    base = [docker, "compose", *arguments]
    result = subprocess.run([*base, "ps", "-a", "--format", "json"], cwd=ROOT,
                            capture_output=True, timeout=30, creationflags=HIDDEN)
    if result.returncode or not result.stdout.strip() or result.stdout.strip() == b"[]":
        raise RuntimeError(f"{name}: Docker unavailable or deployment not initialized. See README; no provisioning attempted")
    with (log_dir / f"{name}.log").open("ab") as log:
        subprocess.run([*base, "start"], cwd=ROOT, stdout=log, stderr=log,
                       check=True, timeout=60, creationflags=HIDDEN)


def launch(name, command, cwd, log_dir, env=None):
    with (log_dir / f"{name}.log").open("ab") as log:
        return subprocess.Popen(command, cwd=cwd, env=env, stdin=subprocess.DEVNULL,
                                stdout=log, stderr=log, creationflags=HIDDEN,
                                start_new_session=os.name != "nt")


def main(argv=None):
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--check", action="store_true", help="Read-only service health check")
    parser.add_argument("--no-browser", action="store_true", help="Do not open a browser")
    parser.add_argument("--skip-mes", action="store_true", help="Start AAS/Agent UI without requiring MES")
    parser.add_argument("--agent", metavar="TASK", help="Run a CLI Agent task in this terminal instead")
    args = parser.parse_args(argv)
    if args.agent and args.check:
        parser.error("--agent and --check cannot be combined")
    configure()
    if args.agent:
        from a4s.cli import main as agent_main
        return agent_main(["run", args.agent])

    endpoint = os.environ["A4S_AAS_ENDPOINT"].rstrip("/")
    mes = os.getenv("A4S_MES_ENDPOINT", "http://localhost:8092").rstrip("/")
    probes = {"BaSyx": lambda: aas_ready(endpoint), "A4S API": lambda: backend_ready(endpoint),
              "Frontend": frontend_ready}
    if not args.skip_mes:
        probes["OpenMES"] = lambda: mes_ready(mes)
    if args.check:
        statuses = {name: probe() for name, probe in probes.items()}
        for name, ready in statuses.items():
            print(f"[{'ready' if ready else 'unavailable'}] {name}")
        return 0 if all(statuses.values()) else 1

    # Do not start any child if basic application prerequisites are missing.
    from a4s.environments.local import check_shell
    shell = check_shell()
    if not shell["works"] or not shell["posix"]:
        raise RuntimeError("No working POSIX shell for Agent commands. Install Git for Windows or fix A4S_SHELL")
    vite = ROOT / "frontend/node_modules/vite/bin/vite.js"
    node = shutil.which("node")
    if not node or not vite.is_file():
        raise RuntimeError("Frontend dependencies missing. Install Node.js, then run npm install in frontend")
    log_dir = Path(os.getenv("A4S_WORK_DIRECTORY", str(ROOT / "temp"))) / "startup" / (time.strftime("%Y%m%d-%H%M%S") + "-" + uuid.uuid4().hex[:6])
    log_dir.mkdir(parents=True)
    print(f"Configuration loaded (values hidden). Logs: {log_dir}", flush=True)
    if not aas_ready(endpoint):
        if not local_endpoint(endpoint, 8081):
            raise RuntimeError("Configured remote/custom AAS is unavailable; no local replacement started")
        start_compose("BaSyx", ["-f", str(ROOT / "BaSyxMinimal/docker-compose.yml")], log_dir)
        wait_ready("BaSyx", lambda: aas_ready(endpoint))
    if not args.skip_mes and not mes_ready(mes):
        if not local_endpoint(mes, 8092):
            raise RuntimeError("Configured remote/custom MES is unavailable; no local replacement started")
        if not (ROOT / ".env.openmes").is_file():
            raise RuntimeError("OpenMES is not initialized. Follow the OpenMES setup in the tutorial")
        start_compose("OpenMES", ["--env-file", str(ROOT / ".env.openmes"), "-p", "a4s-openmes",
                                  "-f", str(ROOT / "docker-compose.openmes.yml")], log_dir)
        wait_ready("OpenMES", lambda: mes_ready(mes))
    if not backend_ready(endpoint):
        if occupied(8000):
            raise RuntimeError("Port 8000 is occupied by an unready or differently configured service; refusing to replace it")
        process = launch("backend", [sys.executable, "-c", "from a4s.server import main; raise SystemExit(main())",
                                     "--endpoint", endpoint, "--host", "127.0.0.1", "--port", "8000"], ROOT, log_dir)
        wait_ready("A4S API", lambda: backend_ready(endpoint), process)
    if not frontend_ready():
        if occupied(5173):
            raise RuntimeError("Port 5173 is occupied by a different service; refusing to replace it")
        env = dict(os.environ, VITE_APP_BACKEND_URL=BACKEND, VITE_APP_AAS_SERVER_URL=endpoint)
        process = launch("frontend", [node, str(vite), "--host", "127.0.0.1", "--port", "5173", "--strictPort"],
                         ROOT / "frontend", log_dir, env)
        wait_ready("Frontend", frontend_ready, process)
    print(f"Ready: {FRONTEND}  (Agent chat is on the right)", flush=True)
    print("Existing services reused; no database initialization, demo provisioning or PLC operation.")
    print("Existing processes keep their original environment; restart them explicitly after changing .env.")
    if not args.skip_mes and not os.getenv("A4S_MES_TOKEN"):
        print("Warning: MES is reachable, but A4S_MES_TOKEN is missing; authenticated MES actions need setup.")
    if not args.no_browser:
        webbrowser.open(FRONTEND)
    return 0


if __name__ == "__main__":
    try:
        raise SystemExit(main())
    except (RuntimeError, OSError, subprocess.SubprocessError) as exc:
        # Do not dump subprocess output, request bodies or environment values.
        message = str(exc) if isinstance(exc, RuntimeError) else type(exc).__name__
        print(f"Startup failed: {message}", file=sys.stderr)
        raise SystemExit(1) from None
