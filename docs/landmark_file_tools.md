# UL/UR landmark file tools

These tools validate, organize, normalize and summarize **existing schema 1.1 UL/UR outputs**. They do not run extraction, download models or reconstruct 3D points. They use the existing [schema reader](../preprocessing/ul_ur_landmarks/schema.py) and coordinate helpers included with the visualiser.

For historical LL/LR files, use the separate [lower-view importer](visualiser.md#normalize-and-import-historical-lower-views). The formats and commands are different.

## Requirements and scope

Use Python 3.10–3.12 and the environment described in [the visualiser guide](visualiser.md). NumPy and h5py are required; no GPU or inference packages are needed. Normalization additionally uses `fcntl` locks and directory `fsync`, so use Linux for this workflow. It has not been tested on Windows or macOS.

Run commands from the repository root. Keep recordings, manifests, landmarks, model configuration and generated reports local and outside the repository. Work on one frozen collection at a time, with extraction and other writers stopped during organization or normalization.

This workflow requires collection records produced alongside the existing extraction. A portable viewer dataset containing only `videos/` and `landmarks/` is sufficient for viewing, but **not** for these archive-management commands. Do not recreate missing provenance by guessing hashes, identities or model configuration.

## Required inputs

```text
collection/
├── run_scope.json
├── run_ledger.jsonl
├── <model-key>_configuration.json
└── results/<model-key>/<pair-id>/
    ├── CAM_UL.h5
    └── CAM_UR.h5

inventory.json
videos/<task>/<original-video-stem>.mp4
```

- **Inventory:** a JSON array of unique pairs. Each row supplies `pair_id`, `pid` and `views`, with exactly `CAM_UL` and `CAM_UR` for organization/normalization. Each view records `relpath`, `width`, `height` and `declared_frames`; `subtask_dir` is optional. The relative video filename must retain its camera token. `pair_id` is a relative path within the archive, such as the synthetic example `T/P01-T-0_1`.
- **`run_scope.json`:** supplies `manifest_sha256`, the SHA-256 of the exact inventory bytes, and `input_root`, the source-video directory used for extraction. This record is preserved.
- **Model configuration:** `<model-key>_configuration.json` must match the model metadata stored in the HDF5 files. It describes provenance; it does not load a checkpoint.
- **Ledger:** newline-delimited JSON recording at least `pair_id`, `view`, `model_key`, `status`, and `frames` for successful records. Optional `elapsed_s` supports progress estimates. Successful statuses are `written` and `skipped_valid`; failures use `failed`. The latest record for each clip determines progress and coverage inclusion. A partial final line is ignored when inspecting a live ledger.
- **Validation manifests:** newline-delimited JSON produced by validation, organization or normalization. Consumers require one `validated` record per expected clip, with `pair_id`, `view`, `output`, `output_sha256` and `frames`. Organization adds `original_output`, which normalization needs to locate the preserved archive.

Keep the inventory and record files unchanged after validation. Hashes identify exact file contents, including whitespace in JSON.

## 1. Inspect and validate the archived outputs

```bash
.venv-viewer/bin/python -m preprocessing.ul_ur_landmarks.collection_status \
  --inventory /path/to/inventory.json \
  --output-root /path/to/collection \
  --model-key rtmw_collection_v1_1 \
  --validate-all
```

Replace the example model key with the key used by your collection. Without `--validate-all`, this command summarizes the ledger and writes `collection_status.json`; that is **not** a fresh audit of saved outputs.

With `--validate-all`, it checks every expected HDF5 file against the current source-video hash, stored source identity, schema, model configuration and declared frame count. It reports unexpected UL/UR outputs and exits with an error if validation is incomplete. It hashes source files but does not decode videos or reassess model accuracy.

The collection directory receives `collection_status.json`, a timestamped `validation_*.json` report and a `validated_outputs_*.jsonl` manifest. Read `validated_output_manifest` and `validated_output_manifest_sha256` from the status report for the next step.

If source videos have moved since extraction, organization accepts their current location explicitly in step 2 and checks their hashes before copying. Use a retained, successful archive-validation manifest for this case. After organization, subsequent validation uses the new location descriptor.

## 2. Organize independent copies by source filename

```bash
.venv-viewer/bin/python -m preprocessing.ul_ur_landmarks.organize \
  --collection-root /path/to/collection \
  --inventory /path/to/inventory.json \
  --validated-manifest /path/to/collection/validated_outputs_TIMESTAMP.jsonl \
  --validated-manifest-sha256 FULL_SHA256_FROM_VALIDATION_REPORT \
  --video-root /path/to/dataset/videos \
  --landmarks-root /path/to/dataset/landmarks \
  --audit-root /path/to/organization-audit \
  --model-key rtmw_collection_v1_1
```

The default is a dry run: it checks the scope, manifest digest, paths and destination collisions and prints the plan without creating output directories. It does not perform all the content checks of an applied run. Add `--apply` to check source identity/video hashes and copy the files.

The result is `landmarks/<task>/<exact-video-stem>.h5`, alongside existing AV/LL/LR files. Copies are independent of their archived originals. Existing targets must be byte-identical; conflicting files are never overwritten. Camera names, participant case and decimal timestamps are retained. Existing files in the destination are hashed before and after the operation.

The audit directory must be new for each applied attempt. It receives `existing_files_before.json`, `validated_outputs.jsonl` and `summary.json`. The summary records the new manifest path and digest. An interrupted run can be resumed with a new audit directory; completed identical copies are reused. The operation is atomic per file, not a transaction over the whole collection.

After success, the collection receives `data_locations.json`, separate from frozen extraction provenance. It records `layout: "video_filename_v1"`, the inventory hash, current video and landmark roots, and the model key. Status and coverage commands then resolve canonical files through this descriptor. An existing descriptor must match the requested configuration.

## 3. Add normalized coordinates to the delivered copies

Use the **organization manifest** from step 2, which records both the delivered pixel files and their archived originals:

```bash
.venv-viewer/bin/python -m preprocessing.ul_ur_landmarks.normalize \
  --collection-root /path/to/collection \
  --inventory /path/to/inventory.json \
  --validated-manifest /path/to/organization-audit/validated_outputs.jsonl \
  --validated-manifest-sha256 FULL_SHA256_FROM_ORGANIZATION_SUMMARY \
  --audit-root /path/to/normalization-audit \
  --model-key rtmw_collection_v1_1 \
  --workers 4
```

This is also a dry run by default. Add `--limit 2 --apply` for a bounded check, then omit `--limit` to complete the collection. Matching reruns use the same pinned organization manifest and normalization audit directory. Its recorded configuration must match, including code hashes; use a new audit directory after intentional code changes.

For every selected clip, normalization verifies the archived hash and identity, creates a temporary independent copy, adds the arrays below, and compares **every original dataset and attribute** against the archive before atomically replacing the delivered copy. Existing normalized files are validated and reused; changed pixel files or conflicting normalized data are rejected. Archived files remain intact. AV files are hashed before and after normalization.

| Added dataset | Coordinates |
| --- | --- |
| `participant/pose/xy_norm` | COCO WholeBody 133 points |
| `participant/face/xy_norm` | MediaPipe 478 dense face points |
| `hands/xy_norm` | Hand observations with their existing topology and actor |
| `participant/bbox_xyxy_norm` | Participant boxes |
| `hands/bbox_xyxy_norm` | Hand boxes |
| `objects/bbox_xyxy_norm` | Object boxes |

Point coordinates are **x / source width, y / source height**. Boxes retain `(x1, y1, x2, y2)`, with x divided by width and y by height. Values are not clipped; predictions outside the image can remain outside `[0, 1]`. Original pixel arrays, NaNs, validity masks, confidence, timestamps, labels, tracks, model provenance and crop transforms are preserved. `(0, 0)` can be a valid point: consult the stored validity mask.

The additive extension keeps schema version 1.1 and records `normalization_version="1.0"` and `normalization_json`. The audit directory contains configuration and status, an AV hash snapshot, a run lock and timestamped validated-output manifests. Check `collection_complete`; a run using `--limit` can finish successfully without completing the collection.

**Do not run organization over normalized delivered files.** Their bytes now differ from the original pixel archive, so organization correctly refuses to overwrite them. Keep the original organization manifest for normalization reruns, and use the new normalization manifest (or fresh validation) when auditing current files.

## 4. Validate current files and inspect coverage

Repeat step 1 after normalization to obtain a fresh validation manifest for the current files. Then:

```bash
.venv-viewer/bin/python -m preprocessing.ul_ur_landmarks.collection_quality \
  --inventory /path/to/inventory.json \
  --output-root /path/to/collection \
  --validated-manifest /path/to/collection/validated_outputs_NEW_TIMESTAMP.jsonl \
  --report-root /path/to/coverage-report
```

Use a new report directory outside the video, landmark, archive-results and repository trees. The coverage tool reads successful ledger entries, checks their output hashes against the supplied manifest, and writes `summary.json`, `clip_quality.jsonl` and `audit_candidates.json`. With a manifest supplied, incomplete coverage or read errors cause a nonzero exit. Without it, the report may describe a partial collection and is not a final validation.

Coverage measures returned predictions, not correctness or physical visibility. “Body” coverage uses the first 11 COCO points for the upper-body check. Diagnostic flags include missing body/face points, large frame-to-frame changes and unusually many hands; these require visual review. Track counts do not measure identity switches. Reports contain participant/clip identifiers and should remain local.

To show the report in the viewer, optionally set `QUB_PHEO_QUALITY_ROOT` to the report directory in `.env` and restart the viewer. File organization and normalization do not change the viewer's frame timing or recompute predictions.

## Synthetic checks

Install `pytest` alongside the viewer requirements and run:

```bash
python -m pytest -q tests/test_normalization.py tests/test_landmark_locations.py \
  tests/test_collection_status.py tests/test_collection_quality.py
```

These fixtures cover provenance, path checks, independent copies, dry runs, normalization and reruns, missing-point handling, source changes, ledger handling, and validation/coverage before and after normalization. They do not process a research dataset.
