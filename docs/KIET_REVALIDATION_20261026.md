# ITAS-MM WP3/WP4 revalidation - 2026-10-26

Status: Local revalidation report for 2026-10-26. A copy is prepared at docs/KIET_REVALIDATION_20261026.md for a local documentation commit. The release gate remains a draft and is not approved.

## Repository and environment

- Repository root: .
- Branch: work/kiet-dataset-v2
- Evaluation base HEAD before the requested commits: a5cf04166e30a0ae64ad24b410bad3424a58314b
- Worktree at evaluation time: existing local changes remained; no staged changes. The requested work is limited to two local commits; no push, merge, reset, stash, or branch switch.
- Runtime: Windows 11, conda environment itas-mm, Python 3.12.15, torch 2.11.0+cu128, CUDA 12.8, NVIDIA GeForce RTX 3050 Laptop GPU, 4 GB.
- PYTHONNOUSERSITE=1. No .venv was created and torch was not reinstalled.
- Dataset files were not edited; scripts/verify_dataset.py was not run. Benchmark reads used the existing canonical junction and final_v1_1 labels.

## WP3 detector

The selected candidate is output/sign_detector_revalidation_20261026/best_selected.pt, SHA256 c7fc0b607d603c88d33e02fa9986a787d527f8050b74d204fce40b365149541f. Baseline models/sign_detector/retrained_20261009.pt SHA256 is 4fe92aafe504f083636bb6e17303a557d36686766c42cb520dddaba41ee22073.

Primary metrics below use the Phase A evaluator: confidence 0.35, NMS IoU 0.45, match IoU 0.50, 640 px, batch 1, 101-point AP; mAP50-95 averages IoU 0.50 to 0.95 in 0.05 steps.

| Model | Split | Images | GT boxes | Precision | Recall | mAP50 | mAP50-95 |
|---|---|---:|---:|---:|---:|---:|---:|
| Baseline | VAL | 2325 | 4747 | 0.9211 | 0.8850 | 0.9390 | 0.7347 |
| best_selected | VAL | 2325 | 4747 | 0.9409 | 0.9016 | 0.9477 | 0.7553 |
| Baseline | TEST | 2326 | 4888 | 0.9232 | 0.8509 | 0.9194 | 0.7168 |
| best_selected | TEST | 2326 | 4888 | 0.9361 | 0.8754 | 0.9357 | 0.7360 |

Paired image-level TEST bootstrap, 1,000 resamples, seed 20261026: best_selected minus baseline recall 95% CI [0.01702, 0.03175]; mAP50 95% CI [0.01220, 0.01982]. Both intervals exclude zero. No CI was computed for mAP50-95. TEST was used only for reporting and the registered bootstrap; no model or threshold was adjusted from it.

Ground-truth provenance correction: detector evaluation includes every canonical box in each split, not only AI_VERIFIED_HIGH. Read-only ID comparison found exact equality across 30,558 unique canonical_box_id values in boxes.csv and semantic_labels_v1_1.csv; VAL has 4,747 rows and TEST has 4,888. Evidence tiers are mixed: 19,313 AI_VERIFIED_HIGH, 595 deterministic, 10 safe propagated, and 10,640 blank. These labels are not independent human ground truth.

Size-stratified recall and bootstrap detail are in output/kiet_wp_20261026/D/detector_summary.md. The separate Ultralytics VAL table is retained there with its conf 0.001 / NMS 0.70 settings; its P/R summary is at the smoothed maximum-F1 point, not the fixed Phase A 0.35 point.

## WP3 recognizer by crop size

Recognizer checkpoint: models/sign_recognizer/retrained_20261009.pt, SHA256 c1dad57daf61c104d4f0b6cb2387bc2c53ff397eaf948934e4a53a63babb859b. Crops come from GT bbox coordinates in final_v1_1/recognition_manifest_v1.csv. Prediction is raw argmax; no threshold was changed. Macro-F1 is the unweighted mean over classes with GT support in that group.

| Split / size group | Crops | Supported classes | Accuracy | Macro-F1 |
|---|---:|---:|---:|---:|
| VAL >=32 px | 1272 | 30 | 0.9890 | 0.9870 |
| TEST <16 px | 1077 | 22 | 0.5135 | 0.5119 |
| TEST 16-32 px | 802 | 25 | 0.9302 | 0.9265 |
| TEST 32-64 px | 734 | 30 | 0.9223 | 0.9165 |
| TEST >=64 px | 445 | 26 | 0.9685 | 0.9460 |
| TEST >=32 px | 1179 | 30 | 0.9398 | 0.9178 |

TRAIN and VAL contain no crops below 32 px. TRAIN size composition is in output/kiet_wp_20261026/D/recognizer_size_groups.json. The confusion matrix for TEST >=32 is in recognizer_test_ge32_confusion_matrix.csv and includes all 31 checkpoint output axes; unknown has no true-class support. This GT-crop benchmark is not detector end-to-end recognition accuracy.

## WP3 latency

The previous Phase A latency run used 50 VAL images on CUDA, 10 standalone model warmups and 5 pipeline warmups. Audio synthesis was disabled. The offline pipeline includes lane/context/offline legal retrieval and is not video E2E.

| Component | Model / input | Latency | Throughput |
|---|---|---:|---:|
| Detector | baseline | 11.726 ms/image | 85.28 FPS |
| Detector | best_selected | 10.787 ms/image | 92.71 FPS |
| Recognizer | 50 GT crops, batch 50 | 0.118 ms/crop | 8457 crops/s |
| Offline pipeline | baseline | 46.029 ms/image | 21.73 FPS |
| Offline pipeline | best_selected | 46.181 ms/image | 21.65 FPS |

## WP4 current checks and gate draft

Latest local full-suite run used basetemp in %TEMP%:

- Command: conda run --no-capture-output -n itas-mm python -m pytest tests -q -rf --tb=line -p no:cacheprovider --basetemp=%TEMP%\itas-mm-wp-20261026-D-pytest
- Result: 455 passed, 41 failed in 18.19 seconds. Full log: output/kiet_wp_20261026/D/pytest_full_current.log.
- Classification: 35 legacy-artifact, 5 missing-real-input, 0 env-dependency, 1 config/test contract mismatch. See output/kiet_wp_20261026/D/test_failure_classification.md for every failure and one-line evidence.
- The single config/test contract mismatch is test_model_configuration_still_points_to_known_baselines: the local perception config points to the owner-proposed fine-tuned model while the test still asserts the 2026-10-09 baseline. This does not establish an inference defect.
- The owner reports 7/7 Node tests, npm ci plus frontend build, check_rebuild_readiness --strict-core, and evaluate_offline_retrieval.py all passed. Local verification found UFLD HEAD c903880678454dfd9b55a63022368db05c00bc6d, the expected loss.py, four official PDFs matching sources.json SHA256/byte sizes, 703 processed legal chunks, and transformers 5.20.2. The owner-run commands were not all repeated in this turn.
- Core CI gate is PASS for the owner-reported local workflow-equivalent checks and the current full-suite evidence. The GitHub-hosted Ubuntu/Python 3.11 status remains UNVERIFIED.
- Draft gate: output/kiet_wp_20261026/D/release_gate_v2_draft.json. It is not approved. Overall release decision remains HOLD.

## WP2 owner-reviewed pilot and D4

The owner export is present at output/kiet_wp_20261026/C/pilot_test/human_review.json. Its SHA256 is 7abe3862e4c300058254c0bc70e89065ddacdc36c4d524170323afc195759ee4. The import report records 120/120 human-confirmed exhaustive frames, 66 confirmed signs, and 0 confirmed negatives. The score report's frames_jsonl SHA256 matches the current imported file.

### Exact review_method from human_review.json

```json
{
  "type": "owner_bulk_approval_of_ai_boxes",
  "reviewer": "Kiet (owner)",
  "written_by": "Claude on owner instruction",
  "codes_reviewed": false,
  "lanes_reviewed": false,
  "exhaustive_checked": true,
  "frames_without_boxes_reviewed": true,
  "attestation": "Owner stated all 120 frames reviewed and no visible sign missed (chat, 2026-10-26 12:36 +07)",
  "circularity_warning": "GT boxes come from old detector proposals; scores of that detector on this set are circular."
}
```

The reviewed labels are an owner bulk approval of detector-proposed boxes. The owner attested in chat at 2026-10-26 12:36 +07 that all 120 frames were checked and no visible sign was missed. Sign codes and lane were not reviewed. Since the GT boxes originate from detector proposals, the pilot can be biased in favor of that detector; its score is circular and is not independent validation.

| Model | Frames | GT signs | TP | FP | FN | Precision | Recall |
|---|---:|---:|---:|---:|---:|---:|---:|
| Baseline | 120 | 66 | 50 | 28 | 16 | 0.6410 | 0.7576 |
| best_selected | 120 | 66 | 51 | 19 | 15 | 0.7286 | 0.7727 |

Scoring used confidence 0.35, NMS IoU 0.45, match IoU 0.50 and took 4.74 seconds on the RTX 3050. The output identifies the source as test_video_dev_20261015; this is an owner-reviewed pilot, not an independent TEST holdout. The importer explicitly reports separate field-test readiness false. Therefore the human ground-truth release gate remains BLOCKED and E2E video remains BLOCKED.

The C after_owner_review.ps1 script was not rerun in this turn because its guard found import_entries.json, human_corpus, and human_gt_detector_scores.json already present and refuses to overwrite them. I verified the current review SHA, imported frames SHA, import counts, checkpoint hashes, and score metrics from those existing artifacts.

## Commit scope and review boundary

The two local commits in this task are:

1. `feat(review): enforce sign-only parent links` for exactly the six parent_id implementation and test files listed in output/kiet_wp_20261026/D/commit_scope_and_pr.md.
2. `docs(revalidation): summarize 2026-10-26 benchmark results` for this report at docs/KIET_REVALIDATION_20261026.md.

The parent_id commit body records rejection of non-sign parent/supplementary links, clearing stale child parent_id when a parent changes away from sign, and regression coverage for not_sign/uncertain parents and a not_sign child. The report commit body records detector/recognizer provenance, release-gate evidence and failures, owner-reviewed pilot limits, and that the report contains no images or dataset files.

Status update 2026-10-26: branch work/kiet-dataset-v2 was pushed by the owner and PR #3 (base integration/itas-mm-release-candidate-20261021) is open and not merged. Do not include configs/sign_dataset.yaml, configs/perception.yaml, models/, output/, or the untracked architecture documents in these commits. The local detector proposal remains pending group review: configs/perception.yaml points to ignored models/sign_detector/finetuned_20261026.pt (SHA256 c7fc0b607d603c88d33e02fa9986a787d527f8050b74d204fce40b365149541f), while the legacy test asserts the 2026-10-09 baseline. The report and parent_id changes do not resolve that separate decision.

The PR #3 description was derived from output/kiet_wp_20261026/D/commit_scope_and_pr.md (local, ignored by Git).

## Remaining work and deviations from the research proposal

Status as of 2026-10-26. Items below are open; none is claimed as complete.

Update 2026-10-28: dataset_v2.1 includes AI-reviewed records for 394 canonical TEST frames across 11 clip IDs and 120 copied pilot frames. Their reviewer is `ai-review (Codex)`, `human_confirmed=false`, and `review_method.type=ai_reviewed`. TEST membership is unchanged; label provenance changed. The prior owner bulk approval applies to pilot boxes only, while codes and lanes are AI proposals. Human ground truth remains BLOCKED. See `docs/DATASET_V2.md` for provenance and the label artifact hash.

### Open work

- **End-to-end video demo: NOT YET VERIFIED.** The historical V0 video has been located and its SHA256 matches the pilot source. The owner accepted skipping license review. Run the offline pipeline and verify overlays, TTS audio, legal citations, seek and stale-frame handling. Proposal Tasks 5 and 7 depend on this.
- **Lane quality: NOT MEASURED.** No human-reviewed Vietnamese lane ground truth exists, so lane precision/recall/F1 promised in proposal Task 6 cannot be reported. The owner-reviewed pilot did not review lanes.
- **Recognizer on small crops.** TEST accuracy is 0.5135 for crops with short side <16 px, versus 0.9398 for >=32 px. TRAIN and VAL contain no crops below 32 px, so the gap reflects an untrained size range. A future retraining should add small crops; TEST must not be used for that tuning.
- **Sign codes in the pilot: AI-REVIEWED ONLY.** Some AI codes remain `unknown`; no human-verified recognizer accuracy exists.
- **Independent holdout: MISSING.** All human-reviewed frames come from one development video source group; a source-disjoint field TEST set is still required.

### Deviations from the research proposal

- Contrastive-learning feature separation is not implemented; the repository itself records that SAHI/contrastive improvements lack validated ablations.
- SAHI tiling exists in code but is disabled by default (`sahi.enabled: false`) and has no evaluation.
- Road-zone context uses simulated GPS only, as scoped in the proposal; no real GPS integration.
- The proposal cites QCVN 41:2019/BGTVT; the repository corpus uses QCVN 41:2024/BGTVT.
- Decree 168/2024/ND-CP is a scanned PDF; 0 of 111 pages yield searchable text, so its content is not retrievable offline.

### Legal retrieval benchmark (development regression only)

`scripts/evaluate_offline_retrieval.py`, run 2026-10-26 on the restored corpus (corpus SHA256 1086c043d97db61dd0c0f253144e2803a060d8746b05486e206ae5945b88c065):

| Cases | Recall@1 | Recall@3 | Recall@4 | MRR |
|---:|---:|---:|---:|---:|
| 24 | 0.5417 | 0.8333 | 0.8750 | 0.6840 |

Method: BM25 with positional bigram/trigram evidence, no generation. The script reports `independent_field_validation: false` and `legal_currency_verified: false`; this is an existing development suite, not unseen legal QA.

### Detector replacement (pending group decision)

If the group approves the fine-tuned detector, open a separate PR that:

1. points `configs/perception.yaml` to the new checkpoint;
2. updates `tests/unit/test_release_gate_20261011.py::test_model_configuration_still_points_to_known_baselines` to the new expected contract;
3. adds `models/sign_detector/finetuned_20261026.pt` (SHA256 c7fc0b607d603c88d33e02fa9986a787d527f8050b74d204fce40b365149541f) through Git LFS and updates `docs/MODEL_WEIGHTS_LFS_MANIFEST_20261018.json` or a successor manifest.

Until then, the shared default remains `models/sign_detector/retrained_20261009.pt`.

## Local JSON artifact fingerprints

All artifacts under output/ are local files ignored by Git and are not present on GitHub. SHA256 values below identify the JSON snapshots used for this report; they can be recomputed locally for comparison.

| Artifact | SHA256 |
|---|---|
| output/kiet_wp_20261026/A/summary.json | 1eb2e1adfccd7fa6b08266e0b75fffa620e5b30fa5569496dc25c379836c9c17 |
| output/kiet_wp_20261026/A/detector_baseline_val.json | d7f81a16fa2e9ddca29cbdf5fc5c7fbad789e9285aba3bb42bdb385ff3eda583 |
| output/kiet_wp_20261026/A/detector_baseline_test.json | 91e442b32844a8ff30955ac3dfa85c7450a8277b547754b72716c3a9ed79acfc |
| output/kiet_wp_20261026/A/detector_best_selected_val.json | c8f576802c684955eb27b07a8a086e4f5c5435eac4a3470d55b103d17d3925bd |
| output/kiet_wp_20261026/A/detector_best_selected_test.json | 33be045743d9e0a77c60c65f074f67fb127e6ed518b5adfe956455437f855df4 |
| output/kiet_wp_20261026/A/detector_map50_95.json | 59dbaf9453402b2fa215f48b56e30b0db1409499ac04f136add86bdc39283969 |
| output/kiet_wp_20261026/A/detector_test_bootstrap.json | f8fab31f6c75ab069716b7522e9e1506931ac19dc01f6bae4c345ac53d6c4d78 |
| output/kiet_wp_20261026/A/recognizer_val.json | e363a664095e626ae87e130510d078266c38e98753b21ef6c2068fece15ec034 |
| output/kiet_wp_20261026/A/recognizer_test.json | e86f09f616500e2b3ff3e518cb7c50cce197f3d8faef6f8a883798c2dde8af00 |
| output/kiet_wp_20261026/A/latency_val_50.json | 8378c475e288314811ed574616b9cf17657de25826d4828616107b95b077f30b |
| output/kiet_wp_20261026/C/human_gt_detector_scores.json | 780a3a8176aa1521458f7f0cb544c45a794bc4b66b9a99672a5f0b90b48d0255 |
| output/kiet_wp_20261026/C/human_corpus/report.json | 91a6ea4819668afc0f1f824faefa8ae029f8ec749e82bc65a65b4d9caa3327a9 |
| output/kiet_wp_20261026/C/pilot_test/human_review.json | 7abe3862e4c300058254c0bc70e89065ddacdc36c4d524170323afc195759ee4 |
| output/kiet_wp_20261026/D/recognizer_size_groups.json | bc1a201b71df46c5c7253d03baa442aa180fbec800aca7f38f379ed356bfe7ac |
| output/kiet_wp_20261026/D/release_gate_v2_draft.json | bd52d03360ac737455bff15d422db7111fdd6c0e092b7b394108d282593df358 |

## Commands, elapsed time, and errors

- Recognizer size report: conda run --no-capture-output -n itas-mm python output/kiet_wp_20261026/D/run_recognizer_size_groups.py; 20.36 seconds total (19.88 seconds inference), CUDA, scoring 4,330 VAL/TEST crops; TRAIN is used only for size counts. The first attempt failed before inference because the script import path did not include the repo root; I corrected the runner and the final run completed.
- Current full pytest: the command above, 18.19 seconds, exit code 1 because 41 tests failed as classified.
- Phase A detector/recognizer and latency runs were completed earlier in the same work package; their commands and provenance remain in the A JSON artifacts.
- The pytest basetemp was created under %TEMP%. Local command policy rejected the separate recursive cleanup command even after its resolved path was checked. It remains outside the repository at %TEMP%\itas-mm-wp-20261026-D-pytest.
- No dataset verification script, dataset mutation, TEST-driven tuning, push, merge, reset, stash, or branch switch was performed. The requested local commits are limited to the two scopes above.
