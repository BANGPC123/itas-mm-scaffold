# Repository hygiene / GitHub publication (2026-10-14)

This repository intentionally commits **source code, configuration, unit/integration tests, documentation, manifests, and small reproducibility references**. The original media, trained weights, downloaded datasets, machine-local runtimes, large research outputs and cached logs remain on the development computer, not in Git history.

## Files deliberately excluded from commits

- `models/`, `*.pt`, `*.pth`, `*.npz`, `*.bin`: large pretrained/fine-tuned weights and reference banks. Code needs locally provisioned model files to run inference.
- `data/raw/`, `data/processed/`, `data/audit/`, review image packs: external training corpora, derivatives, and locally generated evidence.
- `output/`, `docs/evidence/`, `tmp/`: runtime data, rendered video/audio, images of recorded streets, logs, experiment caches. The full annotated video remains in `output/video_recheck_full_20261014/` locally.
- `data/regulations/raw/`, `data/regulations/current_raw/`, `data/regulations/official_*/`: downloaded copies of official source documents (use documented source acquisition locally).
- `node_modules/`, `frontend/dist/`, Python environments, OCR runtime, `__pycache__/`, `.pytest_cache/`: rebuildable dependencies or caches.
- `.env`, key/certificate patterns: credentials must never enter Git.

## Deploy / test

Install dependencies from `requirements.txt` and `frontend/package-lock.json`; provision models and legal source data as described in `data/LOCAL_SETUP.md` / `data/README.md`, and configure `.env.example`. A fresh Git clone contains **no trained binary models or video**. After provisioning, run `python -m pytest -q tests` and `npm --prefix frontend run build` and `start.ps1` for the local app.

## Branch protection

This publication uses new topic branches only. It does not reset, amend, merge, force push, or delete `main`, `backend/system-refactor` (Onsra), or other existing branches. Releases are research prototypes, not validated road-safety decision systems.

## Preservation rule

Local model weights, downloaded datasets, output videos, and legal documents are excluded, **not deleted**. Only rebuildable Python test/cache directories may be removed locally. Deleting or rewriting generated scientific evidence is not part of this cleanup.
