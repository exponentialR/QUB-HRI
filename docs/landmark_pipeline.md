# UL/UR extraction and evaluation

This guide covers the local research tools in `preprocessing/ul_ur_landmarks/`. They read segmented `CAM_UL` and `CAM_UR` videos and produce per-view 2D landmarks and object detections. They preserve model/source identity, frame indices, relative presentation times and explicit missingness. They do not synchronize raw recordings or reconstruct 3D points.

For viewing existing results, use the lighter [visualiser setup](visualiser.md). For inventory and final file layout, see [inventory](landmark_inventory.md) and [landmark file tools](landmark_file_tools.md).

## Environments and models

The extraction setup was exercised with Python 3.12 on Linux. Use a separate environment from the viewer. FFmpeg and `ffprobe` must be on `PATH`. The GPU adapters require a working NVIDIA driver, CUDA-capable PyTorch, and the ONNX Runtime CUDA provider; they reject an implicit CPU fallback. Collection refinement uses Linux `fcntl`; following a live producer additionally requires user systemd.

### GPU collection environment

Run these commands from the repository root, choosing an environment location outside the checkout:

```bash
python3 -m venv /path/to/environments/qub-landmarks
source /path/to/environments/qub-landmarks/bin/activate
python -m pip install torch==2.5.1 torchvision==0.20.1 \
  --index-url https://download.pytorch.org/whl/cu124
python -m pip install -r preprocessing/ul_ur_landmarks/requirements-gpu.txt
python -m pip install --no-deps rtmlib==0.0.16 mediapipe==0.10.35
```

The `--no-deps` step deliberately retains `onnxruntime-gpu`, NumPy 1.26 and one OpenCV wheel. RTMLib's dependency metadata requests CPU ONNX Runtime and both OpenCV distributions; MediaPipe requests the contrib distribution. This setup uses the tested `opencv-python` functions instead. Consequently, `pip check` can report those distribution-name mismatches. Installing the missing names blindly would install conflicting runtime/OpenCV packages. This is a tested research environment recipe, not a complete transitive dependency lock or a promise of compatibility with arbitrary package upgrades. Linux OpenCV may also require the distribution's OpenGL/GLib runtime libraries; sounddevice may need PortAudio.

The separate `requirements.txt` in this package records the CPU MediaPipe baseline environment. `requirements-yolo.txt` records the CPU detector experiment environment and requires separately installed PyTorch/torchvision CPU wheels. Do not combine these three files in one environment. The repository-root `requirements.txt` is for historical scripts.

For inference commands, this launcher exposes the isolated environment's packaged CUDA libraries before Python imports ONNX Runtime:

```bash
python -m preprocessing.ul_ur_landmarks.gpu_run \
  preprocessing.ul_ur_landmarks.readiness --output /path/to/work/readiness.json
```

Readiness checks runtime availability; actual checkpoint loading/inference and reference quality remain separate checks. The helper only changes its child process environment; it does not install drivers or modify the system.

### Checkpoints

Store models outside the repository and source-video tree:

```bash
python -m preprocessing.ul_ur_landmarks.download_models \
  --output-dir /path/to/work/models
python -m preprocessing.ul_ur_landmarks.download_comparison_models \
  --output-dir /path/to/work/models
```

These commands download pinned official MediaPipe task bundles and OpenMMLab exports and verify SHA-256 hashes. The comparison downloader also retains the pinned Sapiens decoder source and its license in the sibling `model_sources/` directory. Model downloads contact their upstream hosts; recordings and annotations are not inputs to these commands.

| Component | Required local artifact |
| --- | --- |
| Participant detector | `yolox_m_humanart.onnx` |
| RTMW-l body/face/native hands | `rtmw_l_384x288.onnx` |
| Hand5 hand pose | `rtmpose_m_hand5_256.onnx` |
| Optional generic hand detector experiment | `rtmdet_nano_hand_320.onnx` |
| MediaPipe baseline/crop tasks | `holistic_landmarker.task`, `face_landmarker.task`, `hand_landmarker.task` |
| Collection hand detector | Separately supplied YOLO checkpoint with `participant_hand` and `surrogate_hand` classes |
| LEGO detector | Separately supplied YOLO checkpoint with the exact class vocabulary below |

The hand and LEGO YOLO weights are **not distributed or downloaded by this repository**. Full collection requires both. The LEGO vocabulary is `two_two_block`, `four_two_block`, `assembly_base`, `biah_hole`, `lefthand`, `mini_stairway`, `righthand`, `stacked_bridge`, `stacked_stairs`, `stacked_tower`, `surrogate_hand`, `tower_head`. A generic YOLO checkpoint is not interchangeable: the adapter checks these class names. The LEGO hand class names are historical detector labels; they are not anatomical handedness or persistent identities.

Sapiens is optional. To use the 133-point adapter, obtain the exact `sapiens_1b_coco_wholebody_best_coco_wholebody_AP_727_torchscript.pt2` checkpoint through the publisher's access process and pass its local path as `--sapiens-file` to `download_comparison_models`. An absent checkpoint means that comparison is unavailable. Review the terms accompanying each model/code artifact before use or redistribution; repository citations do not grant model or dataset access. This guide does not claim a Sapiens comparison result.

## 1. Inventory and a bounded comparison

Use [the inventory CLI](landmark_inventory.md) to record an explicit scope. `--all-participants --inventory-only` inventories all parsed local UL/UR IDs; `--consented /path/to/ids.txt` filters exact IDs. Outputs and source paths are frozen into subsequent runs, so use a new work directory when changing inputs, model settings or code. Keep generated artifacts local and outside the repository and source-video tree.

For a provisional pilot selection, omit `--inventory-only` and use up to 20 pairs. Duration/task diversity is only a starting point: review view coverage, occlusion, gloves and hand-absent frames separately. The original MediaPipe pilot and sampler use an explicit participant list:

```bash
python -m preprocessing.ul_ur_landmarks.runner \
  --input-root /path/to/dataset/videos --consented /path/to/ids.txt \
  --selection /path/to/pilot/pilot_pairs.json \
  --output-root /path/to/pilot --dry-run
```

Remove `--dry-run` and add `--models-dir /path/to/work/models` to run the MediaPipe baseline. It writes schema 1.0 files under `results/mediapipe/<pair-id>/<view>.h5`, with pose 33, dense face 478 and independent hand 21-point observations. `--face-crop focused` or `pose_guided` selects separately named baseline variants. Baseline output does not include LEGO detection.

Export local reference images and open the annotation page:

```bash
python -m preprocessing.ul_ur_landmarks.sample \
  --input-root /path/to/dataset/videos --consented /path/to/ids.txt \
  --selection /path/to/pilot/pilot_pairs.json --output-root /path/to/pilot
python -m preprocessing.ul_ur_landmarks.annotate \
  --reference /path/to/pilot/reference
```

The sampler exports five frames per clip (200 for 20 pairs), including a short consecutive sequence when possible, and refuses to overwrite an existing reference. The annotation server binds to `127.0.0.1:8765`; saving changes `annotations.json` and retains the preceding version as `annotations.backup.json`. Use one editing session per reference set. Hand boxes use **pixel `[x, y, width, height]`**; actor, anatomical side, selected joints and reference track identity are separate fields. Body and face anchors use pixel `[x, y]`. Explicitly check hand-absent frames; unlabelled frames are not negatives. Full review requires hands, body and face checks, including face visibility.

## 2. Resumable collection

Create the all-participant inventory first. Inspect its issues before using it for extraction. This example records the configuration used for the available-model collection; it does not claim that the choices are optimal for another camera arrangement:

```bash
python -m preprocessing.ul_ur_landmarks.gpu_run \
  preprocessing.ul_ur_landmarks.runner \
  --input-root /path/to/dataset/videos --all-participants \
  --inventory /path/to/inventory-run/inventory.json \
  --output-root /path/to/collection --backend rtmw \
  --models-dir /path/to/work/models \
  --lego-weights /path/to/checkpoints/lego.pt \
  --hand-weights /path/to/checkpoints/hands.pt \
  --hand-mode native_hand5 --hand-joint-threshold 0.2 \
  --hand-box-expansion 1.4 --object-mode full_lower \
  --face-mode crop_fallback --other-hand-nms-iou 0.3 \
  --max-hands-per-actor 2 --batch-size 8 --cpu-workers 2 \
  --overlap-face-tasks --dry-run
```

The dry run validates manifest selection and scope without loading models or writing extraction outputs. Remove `--dry-run` and initially add `--pair-id '<pair-id-from-inventory>'` for a bounded real inference check. Inspect its decoded-frame alignment and overlays before the full run. Removing `--pair-id` processes the full inventory. Sapiens uses `--backend sapiens` with its separate checkpoint; compare on the same labelled frames and participant boxes where applicable.

`native_hand5` uses detector-supported native participant hands and Hand5 fallbacks; it disables MediaPipe hand passes. `hybrid` adds independent/full-frame and cropped MediaPipe hands. Dense face points continue to use MediaPipe. The person selection rule favors the upper seated participant; the hand-capacity option assumes at most two hands per actor. These camera-layout and dyadic priors need checking on new data.

Outputs are:

- `results/rtmw_collection_v1_1/<pair-id>/CAM_UL.h5` and `CAM_UR.h5` (a separate model key for Sapiens).
- `run_scope.json`, the model configuration JSON, and package source snapshots.
- `run_ledger.jsonl`, with completion/failure records and timings.

Rerun the same command, environment, code and frozen inputs to validate and skip completed files and retry missing/failed clips. Different provenance is rejected; changing source files, model settings or code requires a new output root. Source snapshots include matching local package files, even uncommitted ones. A fresh clone will therefore not necessarily reproduce the identity of a historical worktree. Only run **one extraction process per output root**; the runner does not implement a collection-wide process lock. It runs in the foreground unless you explicitly manage it as a detached job/service.

Monitor and validate using `collection_status`; see the [file tools guide](landmark_file_tools.md). A ledger completion count is not a quality metric. Files are written atomically after schema checks, and failures exit nonzero after the runner has attempted the selected clips. Interruptions may leave temporary files or an incomplete final ledger line; preserve the records and validate completed outputs before reuse.

## 3. Hand refinement, delivery and normalization

`refine_hands` adds unsupported, strong native RTMW participant-hand observations to separate files, clears ambiguous side assignments by default and recomputes tracks. It preserves existing landmark coordinates and non-hand arrays. Added boxes are landmark extents, not independent detections, and physical visibility is not inferred. It requires the RTMW-specific response threshold; other pose models are rejected.

```bash
python -m preprocessing.ul_ur_landmarks.refine_collection \
  --inventory /path/to/inventory-run/inventory.json \
  --input-root /path/to/collection --output-root /path/to/refined \
  --model-key rtmw_collection_v1_1 --dry-run
```

Remove `--dry-run` to write refined copies. This command checks parent/source/configuration identity, locks its output directory and audits preservation. It can follow an explicitly named user systemd producer with `--follow-service`. A later rerun validates existing outputs. Keep the parent collection intact.

Use [organization, validation and normalization](landmark_file_tools.md) to deliver copies under `landmarks/<task>/<video-stem>.h5`, then [launch the viewer](visualiser.md). Normalization adds `x/width` and `y/height` arrays alongside original pixels; it does not replace coordinates, resample frames or clip values. The extraction runner itself writes pixels.

## 4. Measure quality separately from coverage

```bash
python -m preprocessing.ul_ur_landmarks.compare_reference \
  --annotations /path/to/pilot/reference/annotations.json \
  --candidate mediapipe=/path/to/pilot/results/mediapipe \
  --candidate rtmw=/path/to/collection/results/rtmw_collection_v1_1 \
  --output /path/to/work/anchor-comparison.json
python -m preprocessing.ul_ur_landmarks.hand_quality \
  --annotations /path/to/pilot/reference/annotations.json \
  --results /path/to/collection/results/rtmw_collection_v1_1 \
  --output /path/to/work/hand-quality.json
```

Create output parents first and choose fresh report paths. Body/face comparison scores the same selected visible anchors and includes missing predictions in availability denominators. Hand quality separates box matches, actor assignment, selected joints and short reference tracks. Legacy `evaluate` supports the original full baseline gate or `--scope hands`; landmark-in-box matching and detector box-IoU matching are different measurements and should not be compared as identical recall definitions.

Initial research targets were 90% visible second-actor hand recall **in each view**, 95% correct actor assignment and no systematic participant failure on labelled visible points. These are targets, not guaranteed properties of an output file. Model confidence/validity means an available prediction, not verified physical visibility. Hidden joints can be inferred, face detections can be absent, gloves can merge, tracks can fragment and LEGO detections can include clothing patterns or miss objects. Dense finger accuracy and long-sequence tracking require their own reference labels. An inconclusive optional Sapiens comparison is not evidence to replace RTMW.

### Additional research commands

Every CLI is invoked as `python -m preprocessing.ul_ur_landmarks.<module> --help`; use `gpu_run` for GPU inference. These bounded experiments require the reference formats described below and separately supplied checkpoints.

| Modules | Purpose / inputs |
| --- | --- |
| `pose_reference` | RTMW/Sapiens predictions and overlays on reference images; `--person-boxes` reuses saved person boxes |
| `face_reference`, `probe_face`, `face_variants` | Dense face crop/alignment experiments from reference images, saved poses or results |
| `detector_pilot`, `detector_variants` | Local hand/object checkpoint diagnostics on reference boxes |
| `hand_reference`, `hand_crop_reference`, `hand_pose_variants` | Generic detector, reference-box, crop and rotation experiments; reference-box results are oracle diagnostics |
| `object_quality`, `object_variants` | LEGO loose-brick/assembly box evaluation with explicitly labelled ignore regions |
| `propagate_reference` | Optical-flow drafts from independent seed labels; always require visual review |
| `make_hand_holdout` | Freeze fresh participant clips excluded by the supplied development selection, then export reference images |
| `train_hands`, `resume_hand_training` | Separate participant training/validation sets, local YOLO fine-tuning and verified checkpoint resume |
| `batch_benchmark`, `performance_benchmark` | Bounded batch/scheduling timing with output comparisons; excludes whole-run startup/I/O costs |

`train_hands` prepares copied images, YOLO labels and an experiment manifest by default; `--train` starts GPU training. Accepted hand labels need `hands_checked`, known actor labels and matching `hand_annotation_provenance.image_sha256`; ambiguity-flagged frames are excluded. Freeze the participant split and evaluate fresh participants before claiming generalization. Prior checkpoint training overlap may be unknown. `resume_hand_training` expects the saved `experiment.json` and `fit/weights/last.pt`; it checks copied image/label identity before continuing. Its recorded determinism settings may differ from an interrupted earlier run.

The `sample` annotation format has `schema_version: 1` and a `frames` list. The challenge exporter also supplies video/image hashes. Seed propagation uses a JSON `frames` object keyed by image ID; entries contain `pose`, `face` and `provenance.image_sha256`. LEGO references use a `frames` object keyed by ID, with each frame's `objects` list (`group`, `bbox_xyxy_px`, `ignore_for_box_score`) and `ignore_regions_xyxy_px`; `group` is `loose_brick` or `assembly`. These specialized references are created locally and are not included in this repository. Inspect the relevant reader before adapting another annotation format.

`performance_benchmark` uses pair indices 0 and 7 and the first 32 frames of both views: supply at least eight eligible pairs with sufficient frames. It runs repeated warmed inference; its generated timestamps are nominal benchmark timestamps, not a substitute for decoded source PTS in extraction. Use a fresh output directory for every experimental variant. Some diagnostic commands overwrite their explicitly named reports; they are not all resumable collection runners.

## Verification and reproducibility

In the GPU environment, run:

```bash
python -m pytest -q tests
```

Tests use synthetic videos/HDF5 data and mocked inference to cover scope, timing, transforms, missingness, schema, resume validation, actor/tracking logic, refinement preservation and comparison math. The RTMLib preprocessing regression requires RTMLib installed, but no downloaded checkpoint or working GPU. Passing tests does not replace actual inference checks for new drivers/models or validation against independently reviewed labels.

Keep generated reports, reference images, annotation drafts, run manifests, checkpoints and source snapshots outside Git. This public guide describes reusable commands; machine recovery notes and historical run-specific report generation remain local. The paper describes the published dataset methodology; these later extraction experiments should retain their own versioned provenance and measured limitations.
