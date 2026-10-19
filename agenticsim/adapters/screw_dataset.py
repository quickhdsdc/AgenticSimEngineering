"""Read only the pinned real s01 archive; never use station outcomes as features."""
import csv
import hashlib
import io
import json
import math
from pathlib import Path
import statistics
import zipfile

from agenticsim.domain import require

ARCHIVE_SHA256 = "b39d42eb3d65bf9974db7e7597675377fc0ca6c6f0cb5ac42decdd3447487bc7"
FEATURES = ["duration_s", "peak_torque_nm", "end_torque_nm", "angle_span_deg",
            "mean_torque_nm", "torque_std_nm"]


def features(document):
    require(document.get("torque unit") == "Nm", "unsupported torque unit")
    torques, angles, durations = [], [], []
    duplicate_times = 0
    steps = document.get("tightening steps")
    require(isinstance(steps, list) and steps, "missing measured steps")
    for step in steps:
        graph = step.get("graph", {})
        clock, torque, angle = [graph.get(key) for key in ("time values", "torque values", "angle values")]
        require(all(isinstance(v, list) and v for v in (clock, torque, angle)), "empty measurement graph")
        require(len(clock) == len(torque) == len(angle), "measurement graph shape mismatch")
        require(all(type(x) in (int, float) and math.isfinite(x) for v in (clock, torque, angle) for x in v),
                "non-finite or non-numeric measurement")
        require(all(b >= a for a, b in zip(clock, clock[1:])), "decreasing measured clock")
        duplicate_times += sum(b == a for a, b in zip(clock, clock[1:]))
        durations.append(clock[-1] - clock[0])
        torques.extend(torque)
        angles.extend(angle)
    values = [sum(durations), max(torques), torques[-1], max(angles) - min(angles),
              statistics.fmean(torques), statistics.pstdev(torques)]
    return dict(zip(FEATURES, values)), {"observed_steps": len(steps), "duplicate_times": duplicate_times}


def read_archive(path):
    path = Path(path)
    require(hashlib.sha256(path.read_bytes()).hexdigest() == ARCHIVE_SHA256, "pinned archive hash mismatch")
    records, labels, quality = [], {}, {"duplicate_times": 0, "incomplete_step_operations": 0}
    with zipfile.ZipFile(path) as archive:
        members = archive.infolist()
        require(len(members) == 5002 and sum(m.file_size for m in members) < 150_000_000,
                "unexpected archive inventory")
        by_name = {Path(m.filename).name: m.filename for m in members}
        require(len(by_name) == len(members), "ambiguous member basenames")
        require(all(not Path(m.filename).is_absolute() and ".." not in Path(m.filename).parts
                    and ":" not in m.filename and "\\" not in m.filename for m in members), "unsafe archive path")
        raw = archive.read("labels.csv").decode("utf-8-sig")
        for row in csv.DictReader(io.StringIO(raw, newline=None)):
            require(row["result_value"] in ("OK", "NOK"), "unsupported station label")
            name = row["file_name"]
            require(Path(name).name == name and name in by_name, "missing or unsafe record name")
            require(name not in labels, "duplicate operation")
            document = json.loads(archive.read(by_name[name]))
            require(str(document["cycle"]) == row["run_id"] and str(document["id code"]) == row["workpiece_id"],
                    "operation identity mismatch")
            vector, flags = features(document)
            quality["duplicate_times"] += flags["duplicate_times"]
            quality["incomplete_step_operations"] += flags["observed_steps"] < 4
            # Explicit whitelist: no result, quality code, thresholds, batch/rework
            # outcomes, filename class path, usage count or embedded network metadata.
            records.append({"operation_id": name, "workpiece_id": row["workpiece_id"],
                            "cycle": int(row["run_id"]), "timestamp_local": document["date"],
                            "location": row["workpiece_location"], "features": vector})
            labels[name] = int(row["result_value"] == "NOK")
        require(len(records) == 5000 and len({r["workpiece_id"] for r in records}) == 100,
                "unexpected operation or workpiece count")
        notice = archive.read("README.md").decode("utf-8-sig")
    return records, labels, quality, notice
