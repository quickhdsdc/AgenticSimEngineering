"""Snapshot the public official competition protocol; never log in or submit."""
import hashlib
import json
from pathlib import Path
import urllib.request
from datetime import datetime, timezone

ROOT = Path(__file__).resolve().parents[1]
BASE = 'https://plaidcompetitions-tensile2dbenchmark.hf.space'


def main():
    output = ROOT/'work/cache/datasets/plaid-tensile2d/protocol'
    output.mkdir(parents=True, exist_ok=True)
    hashes = {}
    for name in ('dataset_info', 'submission_info', 'rules', 'competition_info'):
        raw = urllib.request.urlopen(BASE+'/'+name, timeout=45).read()
        json.loads(raw)
        hashes[name+'.json'] = hashlib.sha256(raw).hexdigest()
        (output/(name+'.json')).write_bytes(raw)
    (output/'manifest.json').write_text(json.dumps({'base_url':BASE, 'retrieved_utc':datetime.now(timezone.utc).isoformat(),
        'sha256':hashes, 'status':'public_read_only_snapshot', 'official_submission':'requires competition login'},indent=2),encoding='utf-8')
    print(json.dumps({'output':str(output), 'sha256':hashes}))


if __name__ == '__main__':
    main()
