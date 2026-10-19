"""Read an asset's actual referenced submodels; no inference or mutation."""
import argparse
import json
import os
from datetime import datetime, timezone

from ...aasctl.client import BasyxClient, Endpoint
from ...kernel.config import load_env


def read_context(asset, endpoint=None):
    with BasyxClient(Endpoint.from_env(endpoint)) as client:
        shell = client.get_shell(asset)
        models = []
        for ref in shell.get("submodels", []):
            for key in ref.get("keys", []):
                if key.get("type") == "Submodel":
                    models.append(client.get_submodel(key["value"]))
        return {"repository": client.endpoint.base_url, "shell": shell, "submodels": models,
                "current_time": datetime.now(timezone.utc).isoformat(),
                "trusted_mes_endpoint": os.getenv("A4S_MES_ENDPOINT", ""),
                "mes_timezone": os.getenv("A4S_MES_TIMEZONE", "")}


def main():
    load_env()
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--asset", required=True)
    parser.add_argument("--endpoint")
    args = parser.parse_args()
    print(json.dumps(read_context(args.asset, args.endpoint), ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()
