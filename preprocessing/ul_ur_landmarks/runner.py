"""Local UL/UR runner with explicit pilot or full-inventory scope."""

from __future__ import annotations

import argparse
from contextlib import ExitStack
from datetime import datetime, timezone
import json
import os
from pathlib import Path
import subprocess
import tempfile
import time

from .inventory import VIEWS, consented_ids, parse_clip
from .schema import sha256, validate_hdf5, write_hdf5


def frame_pts_ms(path: Path) -> list[int]:
    command = ["ffprobe", "-v", "error", "-select_streams", "v:0", "-show_frames",
               "-show_entries", "frame=best_effort_timestamp_time", "-of", "json", str(path)]
    result = subprocess.run(command, check=True, capture_output=True, text=True)
    frames = json.loads(result.stdout)["frames"]
    values = [float(frame["best_effort_timestamp_time"]) for frame in frames]
    if not values:
        raise ValueError(f"No decoded video frames: {path}")
    relative = [round((value - values[0]) * 1000) for value in values]
    if relative[0] < 0 or any(b <= a for a, b in zip(relative, relative[1:])):
        raise ValueError(f"Video timestamps are not strictly increasing in milliseconds: {path}")
    return relative


def selected_rows(path: Path, allowed: set[str] | None, input_root: Path, *,
                  max_pairs: int | None = 20) -> list[dict]:
    rows = json.loads(path.read_text())
    if not isinstance(rows, list) or not rows or (max_pairs is not None and len(rows) > max_pairs):
        raise ValueError(f"Selection must be a nonempty list of at most {max_pairs} pairs"
                         if max_pairs is not None else "Inventory must be a nonempty list")
    seen = set()
    for row in rows:
        pair_id = row["pair_id"]
        if pair_id in seen or Path(pair_id).is_absolute() or ".." in Path(pair_id).parts:
            raise ValueError(f"Invalid or duplicate pair ID: {pair_id}")
        seen.add(pair_id)
        if (allowed is not None and row["pid"] not in allowed) or set(row["views"]) != set(VIEWS):
            raise ValueError(f"Unlisted participant or missing view: {pair_id}")
        for view in VIEWS:
            source = (input_root / row["views"][view]["relpath"]).resolve()
            if not source.is_relative_to(input_root) or not source.is_file():
                raise ValueError(f"Invalid source path: {source}")
            if parse_clip(source, input_root, allowed) != (pair_id, view):
                raise ValueError(f"Pair identity mismatch: {source}")
            if parse_clip(source, input_root, {row["pid"]}) != (pair_id, view):
                raise ValueError(f"Participant identity mismatch: {source}")
    return rows


def freeze_run_scope(path: Path, identity: dict) -> None:
    """Publish the run identity atomically and never replace a different scope."""
    if path.exists():
        if json.loads(path.read_text()) != identity:
            raise ValueError("Output root belongs to a different frozen inventory")
        return
    fd, name = tempfile.mkstemp(prefix=".run-scope.", suffix=".partial", dir=path.parent)
    temporary = Path(name)
    try:
        with os.fdopen(fd, "w") as stream:
            json.dump(identity, stream, indent=2)
            stream.write("\n")
            stream.flush()
            os.fsync(stream.fileno())
        try:
            os.link(temporary, path)
        except FileExistsError:
            if json.loads(path.read_text()) != identity:
                raise ValueError("Output root belongs to a different frozen inventory")
    finally:
        temporary.unlink(missing_ok=True)


def process_clip(path: Path, output: Path, *, input_root: Path, row: dict, view: str,
                 models_dir: Path, face_crop: str = "full", backend=None) -> dict:
    import cv2
    from .mediapipe_backend import MediaPipeBackend

    source_hash = sha256(path)
    width, height = (int(row["views"][view][key]) for key in ("width", "height"))
    start = time.perf_counter()
    with ExitStack() as stack:
        if backend is None:
            backend = stack.enter_context(MediaPipeBackend(models_dir, width, height, face_crop=face_crop))
        else:
            backend.reset(width,height)
        if output.exists():
            result = validate_hdf5(output, source_sha256=source_hash, model=backend.provenance)
            return {"status": "skipped_valid", "output": str(output), **result}
        pts = frame_pts_ms(path)
        capture = cv2.VideoCapture(str(path))
        if not capture.isOpened():
            raise ValueError(f"Unable to decode {path}")
        frames = []
        batch_images, batch_times = [], []
        batch_size = getattr(backend, "batch_size", 1)
        try:
            for index, timestamp in enumerate(pts):
                okay, bgr = capture.read()
                if not okay:
                    raise ValueError(f"Decoder stopped at frame {index}, expected {len(pts)}")
                if bgr.shape[:2] != (height, width):
                    raise ValueError(f"Unexpected decoded dimensions at frame {index}")
                image = bgr if getattr(backend,"expects_bgr",False) else cv2.cvtColor(bgr, cv2.COLOR_BGR2RGB)
                if hasattr(backend, "process_batch"):
                    batch_images.append(image)
                    batch_times.append(timestamp)
                    if len(batch_images) == batch_size:
                        results = backend.process_batch(batch_images, batch_times)
                        if len(results) != len(batch_times) or any(f.timestamp_ms != t for f,t in zip(results,batch_times)):
                            raise ValueError("Backend changed batch frame count or timestamps")
                        frames.extend(results)
                        batch_images, batch_times = [], []
                else:
                    frames.append(backend.process(image, timestamp))
            if batch_images:
                results = backend.process_batch(batch_images, batch_times)
                if len(results) != len(batch_times) or any(f.timestamp_ms != t for f,t in zip(results,batch_times)):
                    raise ValueError("Backend changed batch frame count or timestamps")
                frames.extend(results)
            okay, _ = capture.read()
            if okay:
                raise ValueError("Decoder returned more frames than ffprobe")
        finally:
            capture.release()
        declared = row["views"][view]["declared_frames"]
        if declared is not None and len(frames) != declared:
            raise ValueError(f"Declared {declared} frames; decoded {len(frames)}")
        source = {"relpath": str(path.relative_to(input_root)), "sha256": source_hash,
                  "pair_id": row["pair_id"], "pid": row["pid"], "view": view,
                  "width": width, "height": height, "declared_frames": declared,
                  "timestamp_origin": "first decoded presentation timestamp"}
        write_hdf5(output, frames, source=source, model=backend.provenance,
                   pose_topology=backend.pose_topology, pose_count=backend.pose_count,
                   schema_version=getattr(backend,"schema_version","1.0"))
        result = validate_hdf5(output, source_sha256=source_hash, model=backend.provenance)
    return {"status": "written", "output": str(output), "elapsed_s": round(time.perf_counter() - start, 3),
            "inference_stage_s": getattr(backend,"timings",{}), **result}


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--input-root", required=True, type=Path)
    scope = parser.add_mutually_exclusive_group(required=True)
    scope.add_argument("--consented", type=Path, help="Original pilot: exact participant list")
    scope.add_argument("--all-participants", action="store_true",
                       help="Use the explicitly recorded all-participant inventory scope")
    inputs = parser.add_mutually_exclusive_group(required=True)
    inputs.add_argument("--selection", type=Path, help="Pilot selection, at most 20 pairs")
    inputs.add_argument("--inventory", type=Path, help="Full metadata inventory, without the pilot limit")
    parser.add_argument("--output-root", required=True, type=Path)
    parser.add_argument("--models-dir", type=Path)
    parser.add_argument("--pair-id", help="Run a single selected pair")
    parser.add_argument("--face-crop", choices=("full", "focused", "pose_guided"), default="full")
    parser.add_argument("--dry-run", action="store_true")
    parser.add_argument("--backend", choices=("mediapipe","rtmw","sapiens"), default="mediapipe")
    parser.add_argument("--lego-weights", type=Path)
    parser.add_argument("--hand-weights", type=Path)
    parser.add_argument("--batch-size", type=int, default=8, help="Collection backend frame batch, 1–32")
    parser.add_argument("--cpu-workers", type=int, default=2, help="Independent MediaPipe task workers, 1–4")
    parser.add_argument("--overlap-face-tasks", action="store_true",
                        help="Overlap independent CPU face tasks with GPU hand/object detection")
    parser.add_argument("--hand-mode", choices=("hybrid","native_hand5"), default="hybrid")
    parser.add_argument("--hand-joint-threshold", type=float, default=.3, help="Hand5 response threshold only")
    parser.add_argument("--hand-box-expansion", type=float, default=1., help="Hand5 input box scale before intrinsic padding")
    parser.add_argument("--object-mode", choices=("full","full_lower"), default="full",
                        help="Original LEGO full frame, or full frame plus fixed lower crop with duplicate suppression")
    parser.add_argument("--face-mode", choices=("single","crop_fallback"), default="single",
                        help="Try two additional crop scales only if dense face detection fails")
    parser.add_argument("--other-hand-nms-iou", type=float, default=.7,
                        help="Additional NMS IoU for the second actor's hand proposals")
    parser.add_argument("--max-hands-per-actor", type=int, choices=(0,1,2), default=0,
                        help="Explicit dyadic hand capacity; 0 preserves unlimited detector proposals")
    args = parser.parse_args()
    input_root = args.input_root.resolve()
    output_root = args.output_root.resolve()
    repository_root = Path(__file__).resolve().parents[2]
    if output_root.is_relative_to(input_root) or output_root.is_relative_to(repository_root):
        parser.error("Output must be outside the input tree and repository")
    if args.all_participants and not args.inventory:
        parser.error("--all-participants requires --inventory; preserve the original pilot selection")
    manifest = args.inventory or args.selection
    scope_record = None
    if args.inventory:
        scope_path = manifest.with_name("inventory_scope.json")
        if not scope_path.is_file():
            parser.error("Full inventory requires its adjacent inventory_scope.json")
        scope_record = json.loads(scope_path.read_text())
        expected_scope = "all_local_ul_ur" if args.all_participants else "participant_list"
        if (scope_record.get("scope") != expected_scope or
                scope_record.get("input_root") != str(input_root) or
                (args.consented and scope_record.get("participant_list") != str(args.consented.resolve()))):
            parser.error("Inventory scope does not match the requested inputs")
    rows = selected_rows(manifest, None if args.all_participants else consented_ids(args.consented),
                          input_root, max_pairs=None if args.inventory else 20)
    if args.pair_id:
        rows = [row for row in rows if row["pair_id"] == args.pair_id]
        if not rows:
            parser.error("--pair-id is not in the selection")
    if args.dry_run:
        print(json.dumps({"pairs": len(rows), "clips": len(rows) * 2,
                          "output_root": str(output_root),
                          "scope": scope_record or "original_participant_list_pilot",
                          "manifest_sha256": sha256(manifest),
                          "pair_ids": [row["pair_id"] for row in rows] if len(rows) <= 20 else None,
                          "backend": args.backend}, indent=2))
        return
    if not args.models_dir:
        parser.error("--models-dir is required except for --dry-run")
    if args.backend != "mediapipe" and (not args.lego_weights or not args.hand_weights):
        parser.error("Collection backends require --lego-weights and --hand-weights")
    if args.backend != "mediapipe" and args.face_crop != "full":
        parser.error("Collection backends define face crops in their recorded configuration")
    output_root.mkdir(parents=True, exist_ok=True)
    run_identity = {"manifest_sha256": sha256(manifest), "input_root": str(input_root),
                    "scope": scope_record or "original_participant_list_pilot"}
    if args.inventory:
        freeze_run_scope(output_root / "run_scope.json", run_identity)
    failures = 0
    model_key = ({"full": "mediapipe", "focused": "mediapipe_focused",
                  "pose_guided": "mediapipe_pose_guided"}[args.face_crop] if args.backend == "mediapipe"
                 else args.backend + "_collection_v1_1")
    with ExitStack() as stack:
        backend = None
        if args.backend != "mediapipe":
            from .collection_backend import CollectionBackend
            from .source_snapshot import snapshot_code
            snapshot=snapshot_code(output_root/"source_snapshots")
            backend = stack.enter_context(CollectionBackend(args.models_dir,args.lego_weights,args.hand_weights,
                                                             output_root/"yolo_config",pose_model=args.backend,
                                                             batch_size=args.batch_size,cpu_workers=args.cpu_workers,
                                                             hand_mode=args.hand_mode,
                                                             hand_joint_threshold=args.hand_joint_threshold,
                                                             hand_box_expansion=args.hand_box_expansion,
                                                             object_mode=args.object_mode,face_mode=args.face_mode,
                                                             other_hand_nms_iou=args.other_hand_nms_iou,
                                                             max_hands_per_actor=args.max_hands_per_actor,
                                                             overlap_face_tasks=args.overlap_face_tasks))
            backend.provenance['source_snapshot_sha256']=snapshot['sha256']
            freeze_run_scope(output_root / f"{model_key}_configuration.json", backend.provenance)
        for row in rows:
            for view in VIEWS:
                source = input_root / row["views"][view]["relpath"]
                output = output_root / "results" / model_key / row["pair_id"] / f"{view}.h5"
                try:
                    result = process_clip(source, output, input_root=input_root, row=row,
                                          view=view, models_dir=args.models_dir, face_crop=args.face_crop,backend=backend)
                    record = {"pair_id": row["pair_id"], "view": view, **result}
                except Exception as exc:
                    failures += 1
                    record = {"pair_id": row["pair_id"], "view": view,
                              "status": "failed", "error": str(exc)}
                serialized = json.dumps({"model_key": model_key,"finished_utc":datetime.now(timezone.utc).isoformat(), **record})
                with (output_root / "run_ledger.jsonl").open("a") as stream:
                    stream.write(serialized + "\n")
                    stream.flush()
                    os.fsync(stream.fileno())
                print(serialized, flush=True)
    if failures:
        raise SystemExit(f"{failures} clips failed; no failed output was published")


if __name__ == "__main__":
    main()
