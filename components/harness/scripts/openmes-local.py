"""Operator-only bootstrap for the isolated local OpenMES demo.

Generates local secrets without printing them, starts Compose, and provisions a
Supervisor token. Never targets a remote deployment or replaces existing MES
configuration. Run from any directory with the harness virtualenv's Python.
"""
from __future__ import annotations

import argparse
import base64
import json
import os
from pathlib import Path
import secrets
import subprocess
import time

import httpx

ROOT = Path(__file__).resolve().parents[1]
SECRETS = ROOT / ".env.openmes"
ENDPOINT = "http://localhost:8092"


def compose(*args, input=None, private=False):
    command = ["docker", "compose", "--env-file", str(SECRETS), "-p", "a4s-openmes",
               "-f", str(ROOT / "docker-compose.openmes.yml"), *args]
    result = subprocess.run(command, cwd=ROOT, text=True, input=input,
                            capture_output=private, check=False)
    if result.returncode:
        # Provisioning output could contain a token: never echo it on failure.
        raise RuntimeError(f"Compose operation {args[0]} failed (exit {result.returncode})")
    return result.stdout


def provision():
    php = (ROOT / "scripts/openmes-provision.php").read_text(encoding="utf-8")
    result = compose("exec", "-T", "backend", "php", input=php, private=True)
    marker = "A4S_PROVISION="
    for line in result.splitlines():
        if line.startswith(marker):
            return json.loads(line[len(marker):])
    raise RuntimeError("Provisioning did not return its result; no credentials were printed")


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--build", action="store_true", help="Build the upstream image before starting")
    args = parser.parse_args()
    env_path = ROOT / ".env"
    existing = env_path.read_text(encoding="utf-8") if env_path.exists() else ""
    configured = {}
    for line in existing.splitlines():
        key, sep, value = line.strip().removeprefix("export ").partition("=")
        if sep:
            configured[key.strip()] = value.strip().strip("\"'")
    if configured.get("A4S_MES_ENDPOINT", ENDPOINT) != ENDPOINT:
        raise RuntimeError(".env already selects another MES; local setup will not replace it")
    if os.getenv("A4S_MES_ENDPOINT", ENDPOINT) != ENDPOINT:
        raise RuntimeError("The process environment selects another MES; use a clean operator shell")
    if not SECRETS.exists():
        with SECRETS.open("x", encoding="utf-8") as stream:
            stream.write("# Local OpenMES secrets. Never commit or paste into agent chat.\n")
            stream.write(f"OPENMES_DB_PASSWORD={secrets.token_hex(24)}\n")
            stream.write(f"OPENMES_ADMIN_PASSWORD={secrets.token_hex(24)}\n")
            stream.write("OPENMES_APP_KEY=base64:" + base64.b64encode(secrets.token_bytes(32)).decode() + "\n")
        if os.name != "nt":
            SECRETS.chmod(0o600)
    compose("up", "-d", "--build" if args.build else "--no-build")
    with httpx.Client(timeout=3, trust_env=False) as client:
        for attempt in range(90):
            try:
                response = client.get(ENDPOINT + "/api/health")
                if response.status_code == 200:
                    break
            except httpx.HTTPError:
                pass
            if attempt % 10 == 0:
                print("Waiting for OpenMES migrations and HTTP health...", flush=True)
            time.sleep(2)
        else:
            raise RuntimeError("OpenMES did not become healthy; inspect backend logs")
    if configured.get("A4S_MES_TOKEN"):
        if "A4S_MES_TIMEZONE" not in configured:
            with env_path.open("a", encoding="utf-8") as stream:
                stream.write("\nA4S_MES_TIMEZONE=Europe/Berlin\n")
        print("OpenMES is healthy. Existing MES token preserved; provisioning skipped.")
        return
    result = provision()
    # Append only missing settings; preserve every existing key and credential.
    with env_path.open("a", encoding="utf-8") as stream:
        stream.write("\n# Local OpenMES service identity (operator-provisioned)\n")
        if "A4S_MES_ENDPOINT" not in configured:
            stream.write(f"A4S_MES_ENDPOINT={ENDPOINT}\n")
        stream.write("A4S_MES_TOKEN=" + result.pop("token") + "\n")
        if "A4S_MES_TIMEZONE" not in configured:
            stream.write("A4S_MES_TIMEZONE=Europe/Berlin\n")
    if os.name != "nt":
        env_path.chmod(0o600)
    print(json.dumps({"endpoint": ENDPOINT, "identity": "a4s-agent", "role": "Supervisor",
                      "credentials": "Saved locally in .env and .env.openmes; not printed", **result}))


if __name__ == "__main__":
    try:
        main()
    except (RuntimeError, OSError) as exc:
        raise SystemExit(str(exc)) from None
