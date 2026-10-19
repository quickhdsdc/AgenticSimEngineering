"""Fetch the pinned screw-driving archive; preserve its embedded MIT notice."""
import argparse
import hashlib
import json
from pathlib import Path
import urllib.request
import zipfile

ROOT = Path(__file__).resolve().parents[2]
DIRECTORY = ROOT / "work/cache/datasets/screw-s01-v1.1.3"
URL = "https://zenodo.org/records/14887942/files/s01_thread-degradation.zip?download=1"
MD5 = "5e1e69ffaaa9f8b40570674962dcd7a3"


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--check", action="store_true")
    args = parser.parse_args()
    DIRECTORY.mkdir(parents=True, exist_ok=True)
    path = DIRECTORY / "s01_thread-degradation.zip"
    if not path.exists():
        if args.check:
            parser.error("download the dataset first")
        partial = path.with_suffix(".partial")
        request = urllib.request.Request(URL, headers={"User-Agent": "AgenticSimEngineering-research"})
        with urllib.request.urlopen(request, timeout=45) as response, partial.open("wb") as out:
            total = 0
            while chunk := response.read(1024 * 1024):
                total += len(chunk)
                if total > 40_000_000:
                    raise ValueError("unexpected archive size")
                out.write(chunk)
        if hashlib.md5(partial.read_bytes()).hexdigest() != MD5:
            raise ValueError("publisher checksum mismatch")
        partial.replace(path)
    data = path.read_bytes()
    if hashlib.md5(data).hexdigest() != MD5:
        raise ValueError("cached archive checksum mismatch")
    with zipfile.ZipFile(path) as bundle:
        members = bundle.infolist()
        if sum(m.file_size for m in members) > 150_000_000:
            raise ValueError("unexpected expanded size")
        notices = {}
        for member in members:
            if Path(member.filename).name.lower().startswith(("readme", "license")):
                if member.file_size > 100_000:
                    raise ValueError("unexpected notice size")
                notices[member.filename] = bundle.read(member).decode("utf-8-sig")
    report = {"url": URL, "source": "measurement", "md5": MD5,
              "sha256": hashlib.sha256(data).hexdigest(), "bytes": len(data),
              "members": len(members), "notices": notices,
              "license": "MIT (embedded README of this pinned archive)",
              "license_note": "Later online documentation declares CC BY 4.0; retained archive terms govern this version",
              "attribution": "Nikolai West and Jochen Deuse; West & Deuse, HICSS 2024, https://hdl.handle.net/10125/106504",
              "scope": "offline intake; not an operational plant connection"}
    (DIRECTORY / "intake.json").write_text(json.dumps(report, indent=2), encoding="utf-8")
    print(json.dumps({k: v for k, v in report.items() if k != "notices"}))


if __name__ == "__main__":
    main()
