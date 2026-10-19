"""Fetch pinned Panda XML and notices for geometry-free inverse dynamics."""
from concurrent.futures import ThreadPoolExecutor
import hashlib
import json
from pathlib import Path
import urllib.request

ROOT = Path(__file__).resolve().parents[1]
DEST = ROOT / "assets/robots/panda-identification"
COMMIT = "8161bba264d7fa7c99ca301e91e7fb44737676ad"
REPO = "google-deepmind/mujoco_menagerie"
FILES = ("panda.xml", "LICENSE", "README.md")


def fetch(url):
    with urllib.request.urlopen(urllib.request.Request(url, headers={"User-Agent": "AgenticSimEngineering"}), timeout=60) as response:
        return response.read()


def main():
    tree = json.loads(fetch(f"https://api.github.com/repos/{REPO}/git/trees/{COMMIT}?recursive=1"))
    if tree.get("truncated"):
        raise ValueError("incomplete reference tree")
    entries = {e["path"]: e for e in tree["tree"] if e["type"] == "blob"}
    DEST.mkdir(parents=True, exist_ok=True)

    def copy(name):
        source = f"franka_emika_panda/{name}"
        path = DEST / name
        data = path.read_bytes() if path.exists() else fetch(f"https://raw.githubusercontent.com/{REPO}/{COMMIT}/{source}")
        git_hash = hashlib.sha1(b"blob " + str(len(data)).encode() + b"\0" + data).hexdigest()
        if git_hash != entries[source]["sha"]:
            raise ValueError("reference differs from pinned upstream")
        if not path.exists():
            path.write_bytes(data)
        return {"path": name, "upstream_path": source, "sha256": hashlib.sha256(data).hexdigest()}

    with ThreadPoolExecutor(max_workers=3) as pool:
        records = list(pool.map(copy, FILES))
    manifest = {"repository": f"https://github.com/{REPO}", "commit": COMMIT, "files": records,
                "scope": "Original XML and notices only; mesh-free calibration model generation, not visualization assets"}
    (DEST / "manifest.json").write_text(json.dumps(manifest, indent=2) + "\n", encoding="utf-8")
    print(json.dumps({"status": "verified", "files": len(records), "commit": COMMIT}))


if __name__ == "__main__":
    main()
