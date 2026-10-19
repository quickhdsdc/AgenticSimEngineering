"""Write the Festo work center into an AAS repository, or to a file.

Two destinations because they answer different needs. `--out` writes an AAS
environment JSON, which is what `BaSyxMinimal/aas` loads at boot and therefore
what survives the database being thrown away; `--endpoint` posts into a running
repository, which is what you want when the model changed and restarting the
stack would cost more than it is worth.
"""

from __future__ import annotations

import json
import sys
from pathlib import Path
from typing import Optional

from ...aasctl.client import AasHttpError, BasyxClient, Endpoint
from .model import build


def write_file(path: Path) -> int:
    environment = build()
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(environment, ensure_ascii=False, indent=2),
                    encoding="utf-8")
    print(f"{path}: 1 shell, {len(environment['submodels'])} submodels")
    for submodel in environment["submodels"]:
        print(f"  {submodel['idShort']:22} {submodel['id']}")
    return 0


def push(endpoint: Optional[str], replace: bool) -> int:
    environment = build()
    with BasyxClient(Endpoint.from_env(endpoint)) as client:
        for submodel in environment["submodels"]:
            if replace:
                try:
                    client.delete_submodel(submodel["id"])
                except AasHttpError:
                    pass
            try:
                client.create_submodel(submodel)
                print(f"  created {submodel['idShort']}")
            except AasHttpError as exc:
                print(f"  {submodel['idShort']}: {exc}", file=sys.stderr)
        for shell in environment["assetAdministrationShells"]:
            if replace:
                try:
                    client.delete_shell(shell["id"])
                except AasHttpError:
                    pass
            try:
                client.create_shell(shell)
                print(f"  created {shell['idShort']}")
            except AasHttpError as exc:
                print(f"  {shell['idShort']}: {exc}", file=sys.stderr)
    return 0


def main(argv: Optional[list] = None) -> int:
    import argparse

    parser = argparse.ArgumentParser(
        prog="a4s-festo-onboard",
        description="Model the Festo CP Lab work center in an AAS repository.",
    )
    parser.add_argument("--out", type=Path,
                        help="write an AAS environment JSON here instead of posting")
    parser.add_argument("--endpoint", help="AAS repository (else $A4S_AAS_ENDPOINT)")
    parser.add_argument("--replace", action="store_true",
                        help="delete and recreate what is already there")
    args = parser.parse_args(argv)

    if args.out:
        return write_file(args.out)
    return push(args.endpoint, args.replace)


if __name__ == "__main__":
    raise SystemExit(main())
