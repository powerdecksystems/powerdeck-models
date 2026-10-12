# powerdeck-models

PowerDeck AI model catalog — manifest + release-hosted artifacts.

## Layout

- `manifest.json` — the model catalog. Each `models[]` entry:
  `name` (stable id), `title`, `description`, `use_case`, `tier`,
  `kind`, `format` (`onnx`|`gguf`), `artifact` (cache-relative path),
  `url` (full download URL — release asset), `sha256`, `bytes`.
- Releases — model artifacts as release assets (max 2GB/asset).
  A release tag = a version; bump = new tag + manifest repoint.
- `tools/export.py` — ONNX export + parity check.
- `tools/manifest_update.py` — insert/update a manifest entry.
- `.github/workflows/export.yml` — `workflow_dispatch`: export an HF
  model to ONNX, upload to a release, update the manifest.

## Contract

Appliances fetch the manifest from the license-gated distribution
surface (cf-stub in dev, Worker in prod), then pull artifacts straight
from each entry's `url` — any host works; integrity is the sha256 pin.
No credentials are sent to artifact hosts.
