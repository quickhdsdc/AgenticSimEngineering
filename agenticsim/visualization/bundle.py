"""Build a portable local viewer and verify every served artifact."""
import hashlib
import json
from pathlib import Path
import shutil
from uuid import uuid4

from agenticsim.domain import require
from .robot import playback

ROOT = Path(__file__).resolve().parents[2]
WEB = Path(__file__).parent / "web"
THREE_VERSION = "0.186.1"


def sha256(path):
    return hashlib.sha256(path.read_bytes()).hexdigest()


def export(study_path, output):
    study_path = Path(study_path).resolve()
    replay, extra = playback(study_path)
    three = ROOT / "components/visualization/node_modules/three"
    require((three / "package.json").is_file(), "Install local viewer resources: npm ci --prefix components/visualization")
    require(json.loads((three / "package.json").read_text(encoding="utf-8"))["version"] == THREE_VERSION,
            "Three.js version differs from pinned viewer dependency")
    copies = {p.name: p for p in WEB.iterdir() if p.is_file()}
    copies.update(extra)
    copies.update({"vendor/three.module.js": three / "build/three.module.js",
                   "vendor/three.core.js": three / "build/three.core.js",
                   "vendor/OrbitControls.js": three / "examples/jsm/controls/OrbitControls.js",
                   "vendor/RoomEnvironment.js": three / "examples/jsm/environments/RoomEnvironment.js",
                   "vendor/THREE-LICENSE.txt": three / "LICENSE", "study.json": study_path})
    require(all(p.is_file() for p in copies.values()), "viewer dependency files missing")
    destination = Path(output).resolve() / f"replay-{uuid4().hex}"
    destination.mkdir(parents=True, exist_ok=False)
    for name, source in copies.items():
        target = destination / name
        target.parent.mkdir(parents=True, exist_ok=True)
        shutil.copyfile(source, target)
    (destination / "replay.json").write_text(json.dumps(replay, ensure_ascii=False, allow_nan=False, indent=2), encoding="utf-8")
    manifest = {"schema_version": "1.0", "study_id": replay["study_id"],
                "study_payload_sha256": replay["study_sha256"], "renderer": f"three.js/{THREE_VERSION}",
                "files": {str(p.relative_to(destination)).replace("\\", "/"): sha256(p)
                          for p in destination.rglob("*") if p.is_file()}}
    (destination / "bundle.json").write_text(json.dumps(manifest, indent=2), encoding="utf-8")
    verify(destination)
    return {"status": "exported", "study_id": replay["study_id"], "bundle": str(destination),
            "entrypoint": str(destination / "index.html"), "manifest": str(destination / "bundle.json")}


def verify(directory):
    directory = Path(directory).resolve()
    manifest = json.loads((directory / "bundle.json").read_text(encoding="utf-8"))
    require(manifest.get("schema_version") == "1.0" and isinstance(manifest.get("files"), dict), "invalid bundle manifest")
    require({"index.html", "viewer.js", "scene.js", "state.js", "style.css", "replay.json", "study.json",
             "vendor/three.module.js", "vendor/three.core.js", "vendor/OrbitControls.js", "vendor/THREE-LICENSE.txt"}
            <= set(manifest["files"]), "incomplete playback bundle")
    for name, expected in manifest["files"].items():
        path = (directory / name).resolve()
        require(path.is_relative_to(directory) and path.is_file(), "invalid bundle file path")
        require(sha256(path) == expected, f"bundle checksum mismatch: {name}")
    replay, _ = playback(directory / "study.json", directory)
    require(replay == json.loads((directory / "replay.json").read_text(encoding="utf-8")), "replay does not match source Study")
    require(manifest.get("study_payload_sha256") == replay["study_sha256"], "manifest Study mismatch")
    require(manifest.get("study_id") == replay["study_id"], "manifest Study ID mismatch")
    return manifest
