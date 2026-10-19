"""Download/check immutable Tensile2D resources without decoding sample pickles."""
import argparse
import hashlib
import json
from pathlib import Path
import urllib.request

ROOT = Path(__file__).resolve().parents[1]
CACHE = ROOT / "work/cache/datasets/plaid-tensile2d"
SOURCE = ROOT / "examples/plaid-tensile2d-source.json"


def check_file(path, entry):
    raw = path.read_bytes()
    if len(raw) != entry['bytes']:
        raise ValueError('publisher size mismatch: ' + entry['path'])
    digest = hashlib.sha256(raw).hexdigest()
    if 'sha256' in entry and digest != entry['sha256']:
        raise ValueError('publisher SHA-256 mismatch: ' + entry['path'])
    if 'git_blob' in entry:
        blob = hashlib.sha1(b'blob ' + str(len(raw)).encode() + b'\0' + raw).hexdigest()
        if blob != entry['git_blob']:
            raise ValueError('publisher Git blob mismatch: ' + entry['path'])
    return digest


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--check', action='store_true')
    args = parser.parse_args()
    source = json.loads(SOURCE.read_text(encoding='utf-8'))
    hashes = {}
    for entry in source['files']:
        path = CACHE / entry['path']
        path.parent.mkdir(parents=True, exist_ok=True)
        if not path.exists():
            if args.check:
                parser.error('missing dataset file: ' + str(path))
            url = f"https://huggingface.co/datasets/{source['dataset']}/resolve/{source['revision']}/{entry['path']}"
            temporary = path.with_suffix('.partial')
            request = urllib.request.Request(url, headers={'User-Agent': 'AgenticSimEngineering-research'})
            with urllib.request.urlopen(request, timeout=90) as response, temporary.open('wb') as out:
                total = 0
                while chunk := response.read(1024 * 1024):
                    total += len(chunk)
                    if total > entry['bytes']:
                        raise ValueError('download exceeds declared size')
                    out.write(chunk)
            check_file(temporary, entry)
            temporary.replace(path)
        hashes[entry['path']] = check_file(path, entry)
    (CACHE / 'intake.json').write_text(json.dumps(dict(source, hashes=hashes), indent=2), encoding='utf-8')
    print(json.dumps({'status': 'verified', 'cache': str(CACHE), 'revision': source['revision'], 'hashes': hashes}))


if __name__ == '__main__':
    main()
