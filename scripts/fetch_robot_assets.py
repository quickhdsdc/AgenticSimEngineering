"""Vendor two robot descriptions, meshes and licenses at one immutable commit.

Explicit setup command only. Simulation never downloads assets.
"""
import argparse
from concurrent.futures import ThreadPoolExecutor
import hashlib
import json
from pathlib import Path
import urllib.request

ROOT = Path(__file__).resolve().parents[1]
DEST = ROOT / "assets/robots/menagerie"
REPO = "google-deepmind/mujoco_menagerie"
MODELS = ("kuka_iiwa_14", "unitree_g1")


def fetch(url):
    request = urllib.request.Request(url, headers={"User-Agent": "AgenticSimEngineering-asset-setup"})
    with urllib.request.urlopen(request, timeout=120) as response:
        return response.read()


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--revision", help="Commit SHA; defaults to existing manifest, or resolves main on first setup")
    args = parser.parse_args()
    manifest_path = DEST / "manifest.json"
    existing = json.loads(manifest_path.read_text()) if manifest_path.exists() else None
    revision = args.revision or (existing["commit"] if existing else "main")
    commit = json.loads(fetch(f"https://api.github.com/repos/{REPO}/commits/{revision}"))["sha"]
    tree = json.loads(fetch(f"https://api.github.com/repos/{REPO}/git/trees/{commit}?recursive=1"))
    if tree.get("truncated"):
        raise RuntimeError("incomplete upstream tree")
    entries = [e for e in tree["tree"] if e["type"] == "blob" and e["path"].split("/")[0] in MODELS]
    DEST.mkdir(parents=True, exist_ok=True)

    def download(entry):
        relative = entry["path"]
        path = (DEST / relative).resolve()
        if not path.is_relative_to(DEST.resolve()):
            raise ValueError("asset path escapes destination")
        if path.exists():
            data = path.read_bytes()
        else:
            data = fetch(f"https://raw.githubusercontent.com/{REPO}/{commit}/{relative}")
        git_hash = hashlib.sha1(b"blob " + str(len(data)).encode() + b"\0" + data).hexdigest()
        if git_hash != entry["sha"]:
            raise ValueError(f"asset differs from pinned upstream: {relative}")
        path.parent.mkdir(parents=True, exist_ok=True)
        if not path.exists():
            path.write_bytes(data)
        return {"path": relative, "sha256": hashlib.sha256(data).hexdigest(), "bytes": len(data)}

    with ThreadPoolExecutor(max_workers=6) as pool:
        records = list(pool.map(download, entries))
    for model in MODELS:
        if not (DEST / model / "LICENSE").is_file():
            raise RuntimeError(f"missing license: {model}")
    manifest = {"repository": f"https://github.com/{REPO}", "commit": commit,
                "models": list(MODELS), "files": records,
                "license": "Per-model LICENSE files retain their original terms"}
    manifest_path.write_text(json.dumps(manifest, indent=2) + "\n", encoding="utf-8")
    print(json.dumps({"commit": commit, "files": len(records), "bytes": sum(r["bytes"] for r in records)}))


if __name__ == "__main__":
    main()
