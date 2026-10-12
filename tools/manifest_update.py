"""manifest_update.py — insert/update a manifest.json entry from an
export fragment. Sets entry.url to the release-asset URL and bumps the
manifest version to today's date.

    python tools/manifest_update.py --fragment dist/chronos-2/fragment.json \
        --manifest manifest.json \
        --repo powerdecksystems/powerdeck-models --tag models-2026-10-12
"""
import argparse
import datetime
import json
from pathlib import Path


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument('--fragment', required=True, type=Path)
    ap.add_argument('--manifest', required=True, type=Path)
    ap.add_argument('--repo', required=True)
    ap.add_argument('--tag', required=True)
    args = ap.parse_args()

    frag = json.loads(args.fragment.read_text())
    frag['url'] = (f"https://github.com/{args.repo}/releases/download/"
                   f"{args.tag}/{frag['name']}.onnx")

    manifest = json.loads(args.manifest.read_text())
    models = [m for m in manifest.get('models', [])
              if m.get('name') != frag['name']]
    models.append(frag)
    manifest['models'] = sorted(models, key=lambda m: m['name'])
    manifest['version'] = datetime.date.today().isoformat()
    args.manifest.write_text(json.dumps(manifest, indent=2) + '\n')
    print(f"manifest: +{frag['name']} → {frag['url']} "
          f"({len(models)} entries, version {manifest['version']})")


if __name__ == '__main__':
    main()
