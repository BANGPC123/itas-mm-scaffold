# Local dataset setup

Source: https://drive.google.com/drive/folders/1lusdJKxeES-gXswoCeCXozIY3Xw4guI0

Extract the Drive download directly into `D:\itas-mm\data`. The resulting layout must be:

```text
data/
  raw/
  processed/canonical/itas_mm_v1/
    localization/
      data.yaml
      samples.csv
      boxes.csv
      images/{train,val,test}/
      labels/{train,val,test}/
    semantic/
  audit/semantic/final_v1_1/
    recognition_manifest_v1.csv
    recognition_classes_v1.csv
    FINAL_MANIFEST.json
  tools/
  README.md
  sources.json
  download-manifest.csv
```

Do not put another `ITAS-MM` or `data` directory between `data/` and these folders. Preserve `data/regulations/`, which belongs to the project's separate RAG corpus.

From `D:\itas-mm`, run:

```powershell
python scripts/verify_dataset.py
```

The verifier checks release artifact hashes, canonical metadata hashes, all image content hashes, label tree hash and split counts. It writes `data/local-import-report.json` and exits with a failure until all required training files are present and correct.

Expected canonical images: TRAIN 10,952; VAL 2,325; TEST 2,326. Recognition: TRAIN 5,673; VAL 1,272; TEST 3,058; 30 classes. Keep these splits unchanged.

If labels are missing, restore them exactly from the frozen `boxes.csv`:

```powershell
python scripts/verify_dataset.py --restore-labels
```

Recognition uses images from `processed/canonical/itas_mm_v1/localization/` and the final recognition manifest from `audit/semantic/final_v1_1/`. The older `processed/canonical/itas_mm_v1/semantic/` files are evidence, not the final recognition training labels.

The eight Drive ZIP parts downloaded on 2026-10-08 have been imported into this layout. All five original source archives match the byte counts and SHA-256 hashes in `download-manifest.csv`; every source archive member is present in its extracted directory. See `data/zip-import-report.json`, `data/raw-archive-verification.json`, and `data/local-import-report.json` for local verification results.

For YOLO training on this checkout, use `configs/sign_dataset.yaml`, whose dataset root points to `D:/itas-mm/data/processed/canonical/itas_mm_v1/localization`. This leaves the original release's `data.yaml` and provenance files unchanged. Historical absolute paths in `sources.json` and audit reports refer to the source machine.
