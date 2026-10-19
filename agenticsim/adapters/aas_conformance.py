"""Isolated official AAS 3.0 checker; no device clients or LLM credentials."""
import importlib.metadata
import json
from pathlib import Path
import sys


def main():
    from aas_test_engines.file import check_json_data
    request = json.loads(sys.stdin.read())
    version = importlib.metadata.version("aas-test-engines")
    if version != "1.0.3":
        raise ValueError("pinned aas-test-engines 1.0.3 required")
    output = []
    directory = Path(request["directory"])
    for case in request["cases"]:
        result = check_json_data(case["aas"])
        raw = result.to_dict()
        (directory / (case["case_id"] + ".json")).write_text(json.dumps(raw, indent=2), encoding="utf-8")
        output.append({"case_id": case["case_id"], "schema_ok": result.ok()})
    print(json.dumps({"tool": "aas-test-engines", "version": version,
                      "scope": "AAS 3.0 Environment JSON checks; custom submodel is not an IDTA template", "results": output}))


if __name__ == "__main__":
    main()
