# ITAS-MM dataset_v2 (WP1)

Version `dataset_v2.0`, built 2026-10-26 for WP1 of [`docs/HANDOFF_KIET_DATASET_V2_20261022.md`](HANDOFF_KIET_DATASET_V2_20261022.md) and issue #2.

## What dataset_v2 is

The handoff assumed the original dataset was lost. It was not: the owner's machine holds the canonical V1 localization set, and all 15,603 images and 15,603 label files were re-hashed on 2026-10-23 and 2026-10-26 and match the 2026-10-07 upload manifest byte for byte. dataset_v2 therefore reuses those verified pixels instead of reacquiring them, and adds the owner-reviewed field pilot as new real-video evidence.

| Part | Samples | Boxes | Split | Labels |
|---|---:|---:|---|---|
| Canonical V1 localization (5 public sources) | 15,603 | 30,558 | TRAIN 10,952 / VAL 2,325 / TEST 2,326 | Boxes: source annotations, canonicalised. Sign codes: AI-assisted semantic V1.1. No human review. |
| Field pilot, one user-supplied driving video | 120 frames | 66 | TEST only | Boxes: owner bulk approval of detector proposals, exhaustive per owner attestation. Codes and lanes not reviewed. |

Git holds only the reference manifest in `data/manifests/dataset_v2/`; image and video bytes stay outside Git.

| File | Content |
|---|---|
| `sources.csv` | One row per source: URI, version, reported license, archive SHA256 and size, counts |
| `samples.csv` | One row per sample: model-independent key, image SHA256, split, split group, source IDs, original split, augmentation flag, image path, size, box count, label and review status, frame timestamp for video |
| `MANIFEST.json` | Input SHA256, per-split counts and frozen fingerprints, label status summary, pilot review method, leakage result, output SHA256 |

## Source inventory

| Need (handoff WP1) | Available source | Status |
|---|---|---|
| Sign detection | 5 public Vietnamese sign datasets (4 Kaggle, 1 Roboflow), archive SHA256 recorded | Available, verified |
| Sign code recognition | Same images; recognition manifest V1.1 (30 codes) and semantic V1.2 (71 IDs) | Available; codes AI-assisted, not human GT |
| Vietnam lane geometry | Field pilot frames only; no lane annotations | Missing: no reviewed lane polylines or masks |
| Supplementary signs, vehicle and time context | Field pilot frames; parent links supported by the review schema | Missing: no reviewed context labels yet |

Reported licenses are recorded as published (two "Unknown", one CC BY-NC-SA 4.0). License review was skipped by owner decision because the project runs locally; media must still not be redistributed through the public repository.

## Split and leakage policy

- Canonical splits are fixed by the V1 split-constraint groups (`split_group_id`) from the lineage-split audit, which keeps related images (near-duplicates, shared lineage) in one split. They are not re-split.
- V1 lineage components are exact-content groups (one per image SHA256); `split_group_id` is the stricter grouping.
- All field pilot frames share one group (`video:<source video SHA256>`) and are TEST only, so contiguous frames cannot leak across splits.
- Checks, all PASS on 2026-10-26: no `split_group_id`, image SHA256 or sample key occurs in more than one split, and no image SHA256 appears twice.
- Frozen split fingerprints (SHA256 over the sorted `sample_key|image_sha256` lines):

| Split | Samples | Groups | Frozen SHA256 |
|---|---:|---:|---|
| TRAIN | 10,952 | 6,861 | `c5de7105dd962e983344862b126cb2bd7d8115998e30280e637fd83ac4f9c560` |
| VAL | 2,325 | 1,354 | `30362e09b64cc1138d31663adf3c52c4616a445b79e120e63939611cca52b144` |
| TEST | 2,446 | 1,404 | `2167cb380ecae1185fc645767993364fd8ec143ac64b6a741598c340fb02979d` |

TEST is frozen: do not tune models, thresholds or data on it. The canonical TEST split was used in earlier project phases, so it is a frozen evaluation split, not a pristine holdout.

## Storage, restore and verification

Raw and processed media live outside Git in the dataset root (layout: `processed/canonical/itas_mm_v1/localization/...`, `sources.json`, `download-manifest.csv`). The current copy is the owner's shared Google Drive folder `ITAS-MM` (created 2026-10-09, same layout as the local root); ask the owner for the link. An earlier copy was uploaded on 2026-10-08 (76,306 files, size and MD5 verified at upload time).

Download drill from the shared Drive copy, 2026-10-26 (files fetched into an empty temporary directory, hashed, then deleted):

| Check | Result |
|---|---|
| `sources.json`, `download-manifest.csv`, `samples.csv`, `boxes.csv` | 4/4 SHA256 match `MANIFEST.json` inputs |
| Image folder listing | VAL 2,325 and TEST 2,326 entries, equal to the manifest; TRAIN listing stopped at 5,500 entries (listing-method limit, full count not confirmed) |
| Random 30 canonical images (seed 20261026) | 18 downloaded, 18/18 SHA256 match; 12 were TRAIN images beyond the listed entries, 0 mismatches |

```powershell
# point to the restored dataset root
$env:ITASMM_DATASET_ROOT = "<path to ITAS-MM dataset root>"
# check manifest integrity, leakage, frozen splits and that every image exists
python scripts/build_dataset_v2_manifest.py --verify
# additionally recompute SHA256 of all 15,603 canonical images
python scripts/build_dataset_v2_manifest.py --verify --rehash
# rebuild (requires the dataset root and, optionally, the private pilot pack)
python scripts/build_dataset_v2_manifest.py --pilot-pack <pilot pack dir>
```

Rebuilding from the same inputs reproduces identical frozen fingerprints. Verification on 2026-10-26 (`--rehash`, with pilot pack): PASS, 15,603 canonical files checked.

## Open items

1. Restore drill on a second machine from the Drive copy has not been done; the current state of the Drive folder is unverified. This belongs to the issue #2 backup gate.
2. Lane, sign-code and context human review (WP2) are not done; the field pilot comes from a single development video and is not an independent field holdout.
3. A source-disjoint field TEST set needs additional videos from different routes.
