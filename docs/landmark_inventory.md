# UL/UR inventory and source snapshots

These utilities prepare local input records for landmark work and retain the source code used by a run. They do not extract landmarks, load checkpoints or change recordings. Generated inventories and snapshots stay outside the repository.

## Requirements

- Python 3.10–3.12; tested on Python 3.12/Linux.
- `ffprobe` on `PATH` for video metadata. `ffmpeg` is also needed for the synthetic video test.
- These utilities use Python's standard library; they do not require GPU or model packages.

Run commands from the repository root. The [visualiser environment](visualiser.md) can also run them.

## Inventory all local UL/UR clips

```bash
python -m preprocessing.ul_ur_landmarks.inventory \
  --input-root /path/to/dataset/videos \
  --all-participants \
  --output-root /path/to/inventory-run \
  --inventory-only
```

Scope must be explicit. `--all-participants` includes every parsed local participant ID in UL/UR. Alternatively, the historical `--consented /path/to/ids.txt` flag filters by an exact newline-delimited ID list. IDs are compared case-insensitively but retain their digits: `P01` does not match `P010`. The list is never modified. The flag records an inclusion rule; the inventory is not a publication or transfer operation.

Only `.mp4` filenames matching `P<digits>-CAM_UL-...` or `P<digits>-CAM_UR-...` are considered. Filename matching is case-insensitive for the ID/view tokens. Pair identity includes the relative parent directory, normalized participant ID and unchanged remainder of the stem. Original relative video paths, including case and decimal timestamps, are preserved in each view record. AV, LL and LR files are ignored by this UL/UR inventory command.

The command creates:

| File | Contents |
| --- | --- |
| `inventory.json` | Readable complete pairs with relative paths, participant/task identity, codec, size, dimensions, declared frame count, nominal rate, duration and start time |
| `inventory_issues.json` | Duplicate or missing views, unreadable clips, paths resolving outside the input root and metadata mismatches |
| `inventory_scope.json` | Explicit inclusion mode, input root and optional participant-list path |

Duplicate-view pairs are excluded from the inventory rather than choosing one source arbitrarily. Files resolving outside the input root are reported and not probed. Missing or unreadable pairs are also excluded. Readable complete pairs with metadata differences remain in the inventory with `metadata_match=false` for review.

Metadata comparison checks dimensions, declared frame counts, frame rates (tolerance 0.001 fps), and durations (tolerance one frame at the higher reported rate). Start times are recorded but not compared. This is a metadata inventory, not a full decode, source-hash validation or physical synchronization measurement. Issue reports do not by themselves produce a nonzero exit; inspect them before freezing the inventory for downstream use.

The output directory must be outside the repository and source-video tree. Use a new output directory for each inventory intended as a research record: rerunning in the same scope replaces the three inventory files. A changed scope is rejected. `--inventory-only` leaves any existing pilot selection untouched. Frozen downstream records identify the inventory by its hash; later edits will invalidate that association.

The inventory provides the input-list structure used by the [landmark file tools](landmark_file_tools.md), but does not create extraction ledgers, model configuration or a frozen `run_scope.json`. Those must accompany the actual extraction run.

## Optional provisional pilot selection

Omit `--inventory-only` and optionally set `--pilot-pairs N` (default 20, maximum 20). Selection requires enough metadata-matched pairs between 3 and 8 seconds long. It deterministically favors new subtasks, then new participant IDs, shorter clips and filename order. A new selection is saved as `pilot_pairs.json`; a valid existing selection is preserved.

This is a duration/metadata heuristic. It does not inspect gloves, hand presence, occlusion or visual difficulty. Review those properties separately. Inventory-only mode is appropriate when collecting the complete input list or when there are insufficient eligible pilot clips.

## Retain a source snapshot

`source_snapshot.py` supplies a Python helper for callers that need to retain the exact package files used in a run:

```python
from pathlib import Path
from preprocessing.ul_ur_landmarks.source_snapshot import snapshot_code

record = snapshot_code(Path('/path/to/run/source_snapshots'))
print(record)  # sha256, absolute snapshot path, and number of hashed files
```

The destination should be outside the repository and source-video tree. The helper copies the immediate `preprocessing/ul_ur_landmarks/` files ending in `.py`, `.html`, `.js`, `.txt` or `.md`, plus `preprocessing/__init__.py`. It includes matching uncommitted files currently on disk. It does not copy nested directories, videos, weights, `.env`, the repository-root launcher or installed dependencies, and is not a full environment backup.

Each file's SHA-256 is recorded in `manifest.json`; the manifest hash names the snapshot directory. Matching snapshots are reused, changed source contents produce a new directory, and conflicting existing contents are rejected. The record should be stored with the run's provenance. Snapshots preserve whatever local code/documentation the helper finds, so keep their contents local unless separately reviewed for release.

## Synthetic checks

Install `pytest` and run:

```bash
python -m pytest -q tests/test_inventory.py tests/test_source_snapshot.py
```

Fixtures cover exact IDs, explicit scope, pairing and duplicates, path containment, unreadable/mismatched metadata, provisional selection, preservation of recordings/selections, actual probing of generated clips, and snapshot reuse/versioning/corruption. No research recordings are processed by these tests.
