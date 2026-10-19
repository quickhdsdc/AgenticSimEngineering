"""Download and inventory the pinned public MERL dataset without loading models.

The dataset is CC-BY-SA-4.0, not covered by this project's software license.
No archive payload is imported or deserialized. Outputs stay under work/.
"""
import argparse
from datetime import datetime, timezone
import hashlib
import json
from pathlib import Path
import pickletools
import re
import time
import urllib.request
import zipfile

ROOT = Path(__file__).resolve().parents[2]
DIRECTORY = ROOT / "work/cache/datasets/merl-lip4rid-v1"
URL = "https://zenodo.org/records/12516500/files/PUB-5510-LIP4RID.zip?download=1"
EXPECTED_MD5 = "d3e29fbd280fc4cf08d008eb50559338"
MAX_BYTES = 600_000_000
REPORT = ROOT / "work/runs/datasets/merl-lip4rid-v1/intake.json"


def fingerprints(path):
    md5, sha = hashlib.md5(), hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            md5.update(chunk)
            sha.update(chunk)
    return md5.hexdigest(), sha.hexdigest()


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--check", action="store_true", help="Verify existing download; never access network")
    args = parser.parse_args()
    archive = DIRECTORY / "PUB-5510-LIP4RID.zip"
    if not archive.exists():
        if args.check:
            parser.error("dataset not downloaded")
        DIRECTORY.mkdir(parents=True, exist_ok=True)
        partial = archive.with_suffix(".partial")
        request = urllib.request.Request(URL, headers={"User-Agent": "AgenticSimEngineering-data-intake"})
        total, last = 0, time.monotonic()
        with urllib.request.urlopen(request, timeout=60) as response, partial.open("wb") as output:
            length = response.headers.get("Content-Length")
            if length is not None and int(length) > MAX_BYTES:
                raise ValueError("unexpected dataset size")
            while chunk := response.read(1024 * 1024):
                total += len(chunk)
                if total > MAX_BYTES:
                    raise ValueError("download exceeded pinned dataset size bound")
                output.write(chunk)
                if time.monotonic() - last > 10:
                    print(json.dumps({"downloaded_mb": round(total / 1_000_000, 1)}), flush=True)
                    last = time.monotonic()
        md5, _ = fingerprints(partial)
        if md5 != EXPECTED_MD5:
            raise ValueError("download checksum does not match the published Zenodo record")
        partial.replace(archive)
    md5, sha = fingerprints(archive)
    if md5 != EXPECTED_MD5:
        raise ValueError("cached archive differs from the published record")
    with zipfile.ZipFile(archive) as bundle:
        inventory = [{"path": item.filename, "bytes": item.file_size, "crc32": f"{item.CRC:08x}"}
                     for item in bundle.infolist() if not item.is_dir()]
        for item in inventory:
            relative = Path(item["path"])
            if relative.is_absolute() or ".." in relative.parts or ":" in item["path"]:
                raise ValueError("invalid archive member path")
        # Only plain text notices are copied out, bounded individually.
        notices = []
        for item in inventory:
            name = Path(item["path"]).name.lower()
            if name.startswith(("readme", "license", "citation")) and item["bytes"] <= 100_000:
                data = bundle.read(item["path"])
                text = data.decode("utf-8")
                notices.append({"path": item["path"], "text": text})
        experiments = []
        for item in inventory:
            if "/Robots/" not in item["path"] or "/Experiments/" not in item["path"]:
                continue
            if item["bytes"] > 10_000_000:
                raise ValueError("unexpected experiment payload size")
            data = bundle.read(item["path"])
            # Inspect string opcodes only, without invoking pickle globals.
            strings = {arg for op, arg, _ in pickletools.genops(data)
                       if op.name in {"SHORT_BINUNICODE", "BINUNICODE", "UNICODE"}}
            labels = sorted(s for s in strings if re.fullmatch(r"(?:q|dq|ddq|tau)(?:_filt|_interp|_smd)?_[1-7]|t", s))
            robot = item["path"].split("/Robots/", 1)[1].split("/", 1)[0]
            derived = "dynamics_components" in item["path"]
            experiments.append({"path": item["path"], "robot": robot,
                                "source": "derived_dynamics_components" if derived else "published_robot_experiment",
                                "sha256": hashlib.sha256(data).hexdigest(),
                                "candidate_labels_from_pickle_strings": labels,
                                "numeric_values_validated": False})
    report = {
        "schema_version": "1.0", "checked_at": datetime.now(timezone.utc).isoformat(),
        "record_url": "https://zenodo.org/records/12516500", "download_url": URL,
        "license": "CC-BY-SA-4.0", "attribution": "Mitsubishi Electric Research Laboratories (MERL), 2024",
        "archive": str(archive.relative_to(ROOT)), "bytes": archive.stat().st_size,
        "published_md5": EXPECTED_MD5, "md5": md5, "sha256": sha,
        "status": "downloaded_and_inventoried", "physical_validation": "not_established",
        "payloads_deserialized": False, "models_executed": False,
        "scope": "Public real-robot and simulated subsets; inventory is not model calibration or physical acceptance",
        "files": inventory, "notices": notices, "experiment_files": experiments,
        "open_mapping_questions": [
            "Archive README describes q as torque and Panda as six joints; actual Panda labels include joint 7",
            "Archive lists 18 MELFA experiments but contains 22; choose splits explicitly",
            "Confirm SI units, joint signs/order, coordinate frames and robot/tool configuration from acquisition source",
            "Confirm tau versus tau_interp and filtering/derivative provenance; model-derived components are not measurements",
            "Validate numeric samples and time axis before parameter fitting; MELFA string labels contain no t",
            "Keep raw/filtered/dynamics variants of one excitation in the same split; avoid trajectory leakage",
            "The registered KUKA fit is incompatible with Panda/MELFA; corresponding model adapter remains unimplemented",
        ],
    }
    REPORT.parent.mkdir(parents=True, exist_ok=True)
    REPORT.write_text(json.dumps(report, indent=2, ensure_ascii=False) + "\n", encoding="utf-8")
    summary = {key: report[key] for key in (
        "schema_version", "checked_at", "record_url", "license", "attribution", "archive", "sha256",
        "status", "physical_validation", "payloads_deserialized", "models_executed", "scope", "open_mapping_questions")}
    summary["counts"] = {
        "archive_files": len(inventory), "robot_experiment_files": len(experiments),
        "panda_original_experiment_files": sum(e["robot"] == "PANDA" and "as_unfiltered.pkl" in e["path"] for e in experiments),
        "panda_filtered_experiment_files": sum(e["robot"] == "PANDA" and "as_filtered_fcut4.0.pkl" in e["path"] for e in experiments),
        "panda_derived_component_files": sum(e["robot"] == "PANDA" and e["source"] == "derived_dynamics_components" for e in experiments),
        "melfa_experiment_files": sum(e["robot"] == "MELFA_RV4FRL" for e in experiments),
    }
    summary["representative_files"] = [next(e for e in experiments if e["robot"] == robot)
                                       for robot in ("PANDA", "MELFA_RV4FRL")]
    (REPORT.parent / "summary.json").write_text(json.dumps(summary, indent=2) + "\n", encoding="utf-8")
    print(json.dumps({"status": report["status"], "files": len(inventory),
                      "bytes": report["bytes"], "sha256": sha, "report": str(REPORT)}))


if __name__ == "__main__":
    main()
