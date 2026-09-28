# QUB-PHEO visualiser

View saved aerial (AV), upper-left/right (UL/UR) and lower-left/right (LL/LR) landmarks over their source videos, with synchronized playback, gaze and object layers. The viewer runs locally in your browser. Dataset files are read only; nothing is uploaded or inferred.

## Requirements

- **Python 3.10–3.12**; the isolated environment is tested on Python 3.12/Linux.
- **FFmpeg**, including `ffprobe` on your `PATH`. Check with `ffprobe -version`.
- A local dataset copy containing the videos and saved HDF5 landmarks. Recordings, landmarks and model weights are not bundled with this launcher.

On Ubuntu/Debian, install the system prerequisites if necessary:

```bash
sudo apt install python3-venv ffmpeg
```

The viewer's Python requirements are separate from the historical extraction requirements. It needs NumPy, h5py and headless OpenCV; no GPU, PyTorch, MediaPipe or model checkpoints are required.

## Quick start

```bash
git clone https://github.com/exponentialR/QUB-HRI.git
cd QUB-HRI
python3 -m venv .venv-viewer
.venv-viewer/bin/python -m pip install -r requirements-viewer.txt
cp .env.example .env
```

Edit `.env` and set the parent directory containing your dataset:

```dotenv
QUB_PHEO_DATASET_ROOT="/path/to/qub-pheo-dataset"
QUB_PHEO_VIEWER_PORT=8767
```

Then start the viewer:

```bash
.venv-viewer/bin/python visualise.py --check
.venv-viewer/bin/python visualise.py
```

Open **http://127.0.0.1:8767/**. Stop the server with **Ctrl+C**. The `--check` command checks configuration, dependencies and matching filenames; it does not decode or certify every clip. Selected clips are checked when opened.

On Windows, create the environment with `py -3.12 -m venv .venv-viewer`, use `.venv-viewer\Scripts\python.exe` in place of `.venv-viewer/bin/python`, and copy `.env.example` to `.env` with your editor or `Copy-Item`. Install FFmpeg separately and put its `bin` directory on `PATH`. Use forward slashes in `.env` paths, for example `"D:/datasets/QUB-PHEO"`. Windows and macOS have not been exercised in this checkout.

## Dataset layout

```text
qub-pheo-dataset/
├── videos/
│   └── <task>/
│       ├── pXX-CAM_AV-<task-and-time-identifiers>.mp4
│       ├── pXX-CAM_UL-<task-and-time-identifiers>.mp4
│       ├── pXX-CAM_UR-<task-and-time-identifiers>.mp4
│       ├── pXX-CAM_LL-<task-and-time-identifiers>.mp4
│       └── pXX-CAM_LR-<task-and-time-identifiers>.mp4
└── landmarks/
    └── <task>/
        ├── pXX-CAM_AV-<task-and-time-identifiers>.h5
        ├── pXX-CAM_UL-<task-and-time-identifiers>.h5
        ├── pXX-CAM_UR-<task-and-time-identifiers>.h5
        ├── pXX-CAM_LL-<task-and-time-identifiers>.h5
        └── pXX-CAM_LR-<task-and-time-identifiers>.h5
```

Keep the original relative directories, filenames and camera tokens. Views are grouped by the filename with its camera token removed. Any subset of the five views is supported, including lower-only clips. Files missing their video or landmark counterpart are counted and excluded; duplicate identities fail with an error.

You do **not** need `ul_ur_landmarks_full/`, a consent-list file, an extraction inventory, run logs, training artifacts or the original machine's absolute paths. Source hashes and model provenance are read from the saved UL/UR HDF5 files. Archived paths stored inside their normalization metadata remain provenance only and are not opened by the viewer.

### Supported landmark formats

- **UL/UR:** this repository's schema 1.1, with COCO WholeBody 133 pose points, MediaPipe 478 face points, hand observations and object boxes. Both pixel-only files and files with the additive normalized-coordinate extension work. The viewer draws the preserved pixel arrays.
- **AV:** the historical `left_landmarks`, `right_landmarks`, `norm_gaze`, `rec_bboxes`, `surrogate_hands` and `timestamps` layout. Normalized coordinates are converted to pixels for display.
- **LL/LR:** the historical `reconstruction/sideview_keyextraction.py` layout: `pose_landmarks` (33), `face_landmarks` (478), `left_hand_landmarks` (21) and `right_hand_landmarks` (21), each shaped `(frames, points, 2)` with integer-valued float32 pixel coordinates. Original files and copies with the normalized extension below work. This topology is drawn separately from the UL/UR COCO topology.

Other historical HDF5 formats are not automatically converted. A view that fails validation is reported separately; other valid views of the clip remain usable.

For existing UL/UR extraction archives, see [the landmark file tools](landmark_file_tools.md) to validate, organize and normalize saved outputs before viewing them. Those archive-management commands require the original collection records; the viewer itself does not.

### Normalize and import historical lower views

The old LL/LR writer multiplied MediaPipe coordinates by the image dimensions and truncated to integer pixels. These files are **not normalized**. Use this command with the viewer environment to import independent copies from `CAM_L/CAM_LL/<task>/*.h5` and `CAM_L/CAM_LR/<task>/*.h5`:

```bash
.venv-viewer/bin/python -m preprocessing.ul_ur_landmarks.normalize_lower \
  --input-root /path/to/CAM_L \
  --video-root /path/to/dataset/videos \
  --output-root /path/to/dataset/landmarks \
  --audit-root /path/to/lower-normalization-audit
```

This is a filename/path dry run. Add `--apply` to perform conversion, optionally with `--limit 2` for a small check or `--workers 4`. Inputs, outputs and audit directories must be separate and outside the repository. Originals are left intact. Conflicting existing outputs are rejected; matching normalized outputs are validated and reused.

Each copy retains all four original datasets and attributes. It adds `<original-name>_norm` (float32) and `<original-name>_valid` (boolean). For example, read `pose_landmarks_norm` and `pose_landmarks_valid`. Normalization is **x / video width, y / video height**, without clipping: predicted points outside the image can fall outside `[0, 1]`. The original subpixel precision cannot be recovered from integer pixels.

A whole zero-filled landmark group is absent/unknown in this historical format. Its normalized points are NaN and its mask is false. A point at `(0, 0)` inside an otherwise nonzero group remains valid. Nonfinite points are invalid. These masks describe stored predictions, not visibility or accuracy; the old writer cannot distinguish missing detections from unprocessed trailing frames.

`lower_normalization_json` records source dimensions, matching video hash, original HDF5 hash/path, converter hash, formula and limitations. Every conversion checks the video header frame count and first decoded frame, normalized arrays and preservation of original values. The audit manifest records each source/output hash. The viewer additionally checks full decoded video PTS/frame count when a selected clip is opened; original LL/LR files contain no independent landmark timestamps. Handedness is preserved as the historical detector label; actor identity remains unknown. No models are rerun, and no lower-view object detections are added.

## Configuration

| Setting | Purpose | Default |
| --- | --- | --- |
| `QUB_PHEO_DATASET_ROOT` | Parent directory of the dataset | Required unless both directory overrides are set |
| `QUB_PHEO_VIDEO_ROOT` | Explicit source-video directory | `<dataset-root>/videos` |
| `QUB_PHEO_LANDMARKS_ROOT` | Explicit HDF5 directory | `<dataset-root>/landmarks` |
| `QUB_PHEO_VIEWER_PORT` | Local HTTP port | `8767` |
| `QUB_PHEO_QUALITY_ROOT` | Optional directory containing an existing `summary.json` coverage report | None |

For each setting, precedence is **command-line argument → environment variable → `.env` → default**. Directory overrides take precedence over directories derived from the dataset root. `.env` is automatically found beside `visualise.py`, even if the command is run from another working directory.

Configuration is read at startup. Restart the viewer after changing `.env`.

Values in `.env` are literal: shell commands and `${VARIABLE}` expressions are not executed or expanded. Quotes, comments, optional `export` prefixes and `~` are supported. Quote paths containing spaces. Relative paths in an environment file are relative to that file; paths passed as CLI arguments or environment variables are relative to the current working directory. Duplicate keys and malformed lines are rejected.

Alternative examples:

```bash
# Explicit dataset, no .env required:
.venv-viewer/bin/python visualise.py --dataset-root /data/qub-pheo --port 8770

# Another configuration file:
.venv-viewer/bin/python visualise.py --env-file /path/to/research.env

# Separate video and landmark directories:
.venv-viewer/bin/python visualise.py --video-root /data/videos --landmarks-root /results/landmarks
```

`.env`, other `.env.*` files and the virtual environments are ignored by Git; `.env.example` is the shareable template. Keep dataset files outside the repository.

## Controls and synchronization

- Filter by participant, task or filename.
- Play/pause with **Space**; step with **← / →**; scrub with the timeline.
- Select **All five**, **Aerial only**, **UL + UR**, or **LL + LR** under **Views**. On wide screens, **All five** places UL above LL on the left, UR above LR on the right, and AV in the center. Narrow screens stack the panels with AV first.
- Toggle source video, hands, AV gaze, body/face and object layers. Hiding source video shows landmarks only.
- Click a point to inspect its pixel coordinates and available score. **Save visible views** downloads an image locally; five-view exports retain the camera arrangement, while paired-view exports sit side by side.

UL is the reference timeline when available; otherwise AV, UR, LL, then LR. Each other view uses its nearest relative clip timestamp, with no extrapolation beyond its coverage. AV landmark timestamps must match decoded video timing within 2 ms. LL/LR landmark row indices are mapped to their source videos' relative presentation timestamps. This uses the clips' existing synchronization; it does not estimate a new camera offset.

Validity means a prediction exists, not that it is accurate or physically visible. AV has no saved confidence scores, object classes or persistent track IDs. Its generic boxes are labelled “AV detection”; all-zero gaze has ambiguous validity and is omitted. UL/UR field validation is implemented in [schema.py](../preprocessing/ul_ur_landmarks/schema.py); coverage alone does not establish model accuracy.

LL/LR have no saved confidence scores, actor identities or tracks. Their hands are orange and labelled with unknown actor identity, rather than assigning the historical left/right hand slots to the participant or surrogate. These files cannot establish whether a point is occluded.

## Troubleshooting

- **“ensurepip is not available”:** install the virtual-environment package for the selected Python version (for example `python3.12-venv`), then recreate the environment.
- **Missing Python module:** install `requirements-viewer.txt` with the same interpreter used to launch the viewer.
- **Missing ffprobe:** install FFmpeg and verify `ffprobe -version` in the same terminal.
- **No matching clips:** check the dataset root, folder names and matching `.mp4`/`.h5` stems with `--check`.
- **Source hash mismatch:** the video differs from the one used for UL/UR extraction. Supply the corresponding original clip; verification is not bypassed.
- **Port already in use:** if the viewer is already running, open its existing browser URL. Otherwise stop the previous instance or use `python visualise.py --port 8770`.

The server binds to `127.0.0.1` and serves explicit viewer/API routes, not arbitrary files. Hosting it for remote users is outside this launcher.
