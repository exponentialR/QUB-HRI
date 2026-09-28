"""HDF5 v1 contract for pilot landmark results."""

from __future__ import annotations

from dataclasses import dataclass, field
import hashlib
import json
import os
from pathlib import Path
import tempfile

import numpy as np

from . import SCHEMA_VERSION


ACTORS = {"unknown": 0, "participant": 1, "other_actor": 2}
HANDEDNESS = {"unknown": 0, "left": 1, "right": 2}


@dataclass
class Hand:
    xy: np.ndarray
    confidence: np.ndarray
    actor: str = "unknown"
    handedness: str = "unknown"
    track_id: int = -1
    topology: str = "mediapipe_21"
    model_id: str = "mediapipe_hand_landmarker"
    bbox_xyxy: np.ndarray | None = None
    detection_confidence: float | None = None


@dataclass
class ObjectDetection:
    bbox_xyxy: np.ndarray
    confidence: float
    class_id: int
    class_name: str
    model_id: str


@dataclass
class Frame:
    timestamp_ms: float
    pose_xy: np.ndarray | None = None
    pose_confidence: np.ndarray | None = None
    face_xy: np.ndarray | None = None
    face_confidence: np.ndarray | None = None
    face_crop_rect: tuple[int, int, int, int] | None = None
    hands: list[Hand] = field(default_factory=list)
    objects: list[ObjectDetection] = field(default_factory=list)
    person_bbox_xyxy: np.ndarray | None = None
    face_crop_to_source: np.ndarray | None = None


def sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as stream:
        for chunk in iter(lambda: stream.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def _points(xy: np.ndarray | None, confidence: np.ndarray | None, count: int) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    if xy is None:
        return (np.full((count, 2), np.nan, dtype=np.float32),
                np.zeros(count, dtype=np.float32), np.zeros(count, dtype=bool))
    xy = np.asarray(xy, dtype=np.float32)
    if xy.shape != (count, 2):
        raise ValueError(f"Expected ({count}, 2) points, got {xy.shape}")
    score = np.ones(count, dtype=np.float32) if confidence is None else np.asarray(confidence, dtype=np.float32)
    if score.shape != (count,):
        raise ValueError(f"Expected ({count},) confidence, got {score.shape}")
    valid = np.isfinite(xy).all(axis=1) & np.isfinite(score) & (score > 0)
    xy = xy.copy()
    xy[~valid] = np.nan
    score = score.copy()
    score[~valid] = 0
    return xy, score, valid


def write_hdf5(path: Path, frames: list[Frame], *, source: dict, model: dict,
               pose_topology: str, pose_count: int, face_topology: str = "mediapipe_478",
               schema_version: str = SCHEMA_VERSION) -> None:
    """Write once, atomically. Existing files are never replaced."""
    import h5py

    if not frames or pose_count < 1:
        raise ValueError("No frames or invalid pose topology")
    if schema_version not in {"1.0", "1.1"}:
        raise ValueError("Unsupported schema version")
    if schema_version == "1.0" and any(
            f.objects or f.person_bbox_xyxy is not None or f.face_crop_to_source is not None or any(
                h.topology != "mediapipe_21" or h.model_id != "mediapipe_hand_landmarker" or
                h.bbox_xyxy is not None or h.detection_confidence is not None for h in f.hands) for f in frames):
        raise ValueError("Extended observations require schema 1.1")
    if any(not np.isfinite(frame.timestamp_ms) for frame in frames):
        raise ValueError("Frame timestamps must be finite")
    if any(b.timestamp_ms <= a.timestamp_ms for a, b in zip(frames, frames[1:])):
        raise ValueError("Frame timestamps must increase strictly")
    if path.exists():
        raise FileExistsError(f"Existing output is not overwritten: {path}")
    path.parent.mkdir(parents=True, exist_ok=True)
    temp_fd, temp_name = tempfile.mkstemp(prefix=f".{path.name}.", suffix=".partial", dir=path.parent)
    os.close(temp_fd)
    temp = Path(temp_name)
    try:
        with h5py.File(temp, "w") as out:
            out.attrs["schema_version"] = schema_version
            out.attrs["source_json"] = json.dumps(source, sort_keys=True)
            out.attrs["model_json"] = json.dumps(model, sort_keys=True)
            out.attrs["pose_topology"] = pose_topology
            out.attrs["face_topology"] = face_topology
            out.attrs["hand_topology"] = "mediapipe_21" if schema_version == "1.0" else "per_observation"
            out.attrs["crop_to_frame_3x3"] = np.eye(3, dtype=np.float32)
            out.attrs["actor_enum_json"] = json.dumps(ACTORS, sort_keys=True)
            out.attrs["handedness_enum_json"] = json.dumps(HANDEDNESS, sort_keys=True)
            out.require_group("participant/pose").attrs["confidence_kind"] = model.get(
                "pose_confidence_kind", "visibility_or_presence_proxy")
            out.require_group("participant/face").attrs["confidence_kind"] = "presence_proxy_not_calibrated"
            out.require_group("hands").attrs["confidence_kind"] = (
                "presence_proxy_not_calibrated" if schema_version == "1.0" else "see_model_provenance_for_each_model_id")
            out.create_dataset("frames/index", data=np.arange(len(frames), dtype=np.int32))
            out.create_dataset("frames/timestamp_ms", data=np.asarray([f.timestamp_ms for f in frames], dtype=np.float64))
            if schema_version == "1.1":
                rects = [f.face_crop_rect if f.face_crop_rect is not None else [-1]*4 for f in frames]
                out.create_dataset("frames/face_crop_rect_xyxy_px", data=np.asarray(rects,dtype=np.int32))
                out.create_dataset("frames/face_crop_valid", data=np.array([f.face_crop_rect is not None for f in frames],dtype=bool))
                matrices = []
                for frame in frames:
                    if frame.face_crop_rect is None:
                        matrices.append(np.full((3,3),np.nan))
                    elif frame.face_crop_to_source is not None:
                        matrices.append(frame.face_crop_to_source)
                    else:
                        matrix = np.eye(3)
                        matrix[:2,2] = frame.face_crop_rect[:2]
                        matrices.append(matrix)
                out.create_dataset("frames/face_crop_to_source_3x3",data=np.asarray(matrices,dtype=np.float64))
            elif all(frame.face_crop_rect is not None for frame in frames):
                out.create_dataset("frames/face_crop_rect_xyxy_px",
                                   data=np.asarray([f.face_crop_rect for f in frames], dtype=np.int32))
            for group, count, xy_name, conf_name in (
                ("participant/pose", pose_count, "pose_xy", "pose_confidence"),
                ("participant/face", 478, "face_xy", "face_confidence"),
            ):
                processed = [_points(getattr(f, xy_name), getattr(f, conf_name), count) for f in frames]
                for name, idx in (("xy_px", 0), ("confidence", 1), ("valid", 2)):
                    out.create_dataset(f"{group}/{name}", data=np.stack([p[idx] for p in processed]),
                                       compression="gzip", compression_opts=1)
            observations = [(index, hand) for index, frame in enumerate(frames) for hand in frame.hands]
            out.create_dataset("hands/frame_index", data=np.asarray([i for i, _ in observations], dtype=np.int32))
            out.create_dataset("hands/track_id", data=np.asarray([h.track_id for _, h in observations], dtype=np.int32))
            out.create_dataset("hands/actor", data=np.asarray([ACTORS[h.actor] for _, h in observations], dtype=np.uint8))
            out.create_dataset("hands/handedness", data=np.asarray([HANDEDNESS[h.handedness] for _, h in observations], dtype=np.uint8))
            processed_hands = [_points(h.xy, h.confidence, 21) for _, h in observations]
            for name, idx, shape, dtype in (("xy_px", 0, (0, 21, 2), np.float32),
                                           ("confidence", 1, (0, 21), np.float32),
                                           ("valid", 2, (0, 21), bool)):
                values = np.stack([p[idx] for p in processed_hands]) if processed_hands else np.empty(shape, dtype=dtype)
                out.create_dataset(f"hands/{name}", data=values, compression="gzip", compression_opts=1)
            if schema_version == "1.1":
                out.attrs["validity_semantics"] = "prediction_available; physical visibility is not measured"
                string_type = h5py.string_dtype("utf-8")
                for name in ("topology", "model_id"):
                    out.create_dataset(f"hands/{name}", data=[getattr(h, name) for _, h in observations], dtype=string_type)
                boxes = [h.bbox_xyxy if h.bbox_xyxy is not None else [np.nan] * 4 for _, h in observations]
                out.create_dataset("hands/bbox_xyxy_px", data=np.asarray(boxes, dtype=np.float32).reshape(-1, 4))
                out.create_dataset("hands/bbox_valid", data=np.array([h.bbox_xyxy is not None for _,h in observations], dtype=bool))
                out.create_dataset("hands/detection_confidence", data=np.asarray([
                    h.detection_confidence if h.detection_confidence is not None else np.nan for _,h in observations], dtype=np.float32))
                out["hands/detection_confidence"].attrs["missing"] = "NaN means no separate detector score"
                person_boxes = [f.person_bbox_xyxy if f.person_bbox_xyxy is not None else [np.nan] * 4 for f in frames]
                out.create_dataset("participant/bbox_xyxy_px", data=np.asarray(person_boxes, dtype=np.float32))
                out.create_dataset("participant/bbox_valid", data=np.array([f.person_bbox_xyxy is not None for f in frames], dtype=bool))
                objects = [(index, detection) for index, frame in enumerate(frames) for detection in frame.objects]
                out.create_dataset("objects/frame_index", data=np.array([i for i,_ in objects], dtype=np.int32))
                out.create_dataset("objects/bbox_xyxy_px", data=np.asarray([d.bbox_xyxy for _,d in objects], dtype=np.float32).reshape(-1,4))
                out.create_dataset("objects/confidence", data=np.array([d.confidence for _,d in objects], dtype=np.float32))
                out.create_dataset("objects/class_id", data=np.array([d.class_id for _,d in objects], dtype=np.int32))
                for name in ("class_name", "model_id"):
                    out.create_dataset(f"objects/{name}", data=[getattr(d, name) for _,d in objects], dtype=string_type)
        validate_hdf5(temp, source_sha256=source["sha256"], model=model)
        # A second process might have completed the same output while this one ran.
        if path.exists():
            raise FileExistsError(f"Existing output is not overwritten: {path}")
        os.link(temp, path)
    finally:
        temp.unlink(missing_ok=True)


def validate_hdf5(path: Path, *, source_sha256: str, model: dict) -> dict:
    import h5py

    with h5py.File(path, "r") as data:
        if data.attrs["schema_version"] not in {"1.0", "1.1"}:
            raise ValueError("Schema version mismatch")
        source = json.loads(data.attrs["source_json"])
        if source["sha256"] != source_sha256:
            raise ValueError("Source hash mismatch")
        if json.loads(data.attrs["model_json"]) != model:
            raise ValueError("Model provenance mismatch")
        from .normalization import validate_normalized
        validate_normalized(data)
        count = len(data["frames/index"])
        if count == 0:
            raise ValueError("Output contains no frames")
        if count != len(data["frames/timestamp_ms"]) or any(
                len(data[key]) != count for key in
                ("participant/pose/xy_px", "participant/pose/confidence", "participant/pose/valid",
                 "participant/face/xy_px", "participant/face/confidence", "participant/face/valid")):
            raise ValueError("Frame dataset length mismatch")
        if not np.array_equal(data["frames/index"][:], np.arange(count)):
            raise ValueError("Frame indices are not contiguous")
        timestamps = data["frames/timestamp_ms"][:]
        if not np.isfinite(timestamps).all() or np.any(timestamps < 0):
            raise ValueError("Frame timestamps must be finite and nonnegative")
        if np.any(np.diff(timestamps) <= 0):
            raise ValueError("Frame timestamps are not increasing")
        if source.get("declared_frames") is not None and count != source["declared_frames"]:
            raise ValueError("Output frame count differs from the recorded source")
        if "frames/face_crop_rect_xyxy_px" in data:
            rects = data["frames/face_crop_rect_xyxy_px"][:]
            crop_valid = data["frames/face_crop_valid"][:].astype(bool) if data.attrs["schema_version"] == "1.1" else np.ones(count,dtype=bool)
            if rects.shape != (count,4) or crop_valid.shape != (count,):
                raise ValueError("Invalid face crop shape")
            active_rects = rects[crop_valid]
            if (np.any(active_rects < 0) or
                    np.any(active_rects[:, 2:] <= active_rects[:, :2]) or
                    ("width" in source and np.any(active_rects[:, 2] > source["width"])) or
                    ("height" in source and np.any(active_rects[:, 3] > source["height"])) or
                    np.any(rects[~crop_valid] != -1)):
                raise ValueError("Invalid face crop rectangles")
            if data.attrs["schema_version"] == "1.1":
                matrices = data["frames/face_crop_to_source_3x3"][:]
                if (matrices.shape != (count,3,3) or not np.isfinite(matrices[crop_valid]).all() or
                        not np.isnan(matrices[~crop_valid]).all() or
                        not np.allclose(matrices[crop_valid,2,:], [0,0,1]) or
                        np.any(np.abs(np.linalg.det(matrices[crop_valid])) < 1e-8)):
                    raise ValueError("Invalid face crop transforms")
        hand_frames = data["hands/frame_index"][:]
        hand_count = len(hand_frames)
        if any(len(data[f"hands/{key}"]) != hand_count for key in
               ("track_id", "actor", "handedness", "xy_px", "confidence", "valid")):
            raise ValueError("Hand observation dataset length mismatch")
        if np.any((hand_frames < 0) | (hand_frames >= count)):
            raise ValueError("Hand observation references a missing frame")
        if np.any(np.diff(hand_frames) < 0):
            raise ValueError("Hand observations are not in frame order")
        if not np.isin(data["hands/actor"][:], list(ACTORS.values())).all():
            raise ValueError("Invalid hand actor enum")
        if not np.isin(data["hands/handedness"][:], list(HANDEDNESS.values())).all():
            raise ValueError("Invalid handedness enum")
        for group, rows, points in (("participant/pose", count, None),
                                     ("participant/face", count, 478), ("hands", hand_count, 21)):
            shape = data[f"{group}/xy_px"].shape
            if (len(shape) != 3 or shape[0] != rows or shape[-1] != 2 or
                    shape[1] < 1 or (points is not None and shape[1] != points) or
                    data[f"{group}/confidence"].shape != shape[:2] or
                    data[f"{group}/valid"].shape != shape[:2]):
                raise ValueError(f"Invalid landmark shapes in {group}")
            # Bound temporary validation arrays independently of clip length.
            for start in range(0, rows, 256):
                xy = data[f"{group}/xy_px"][start:start + 256]
                score = data[f"{group}/confidence"][start:start + 256]
                valid = data[f"{group}/valid"][start:start + 256].astype(bool)
                if (not np.isfinite(xy[valid]).all() or not np.isfinite(score).all() or
                        np.any(score[valid] <= 0) or not np.isnan(xy[~valid]).all() or
                        np.any(score[~valid] != 0)):
                    raise ValueError(f"Inconsistent validity, coordinates, or confidence in {group}")
        object_count = 0
        if data.attrs["schema_version"] == "1.1":
            def check_boxes(group, rows, optional):
                boxes = data[f"{group}/bbox_xyxy_px"][:]
                valid = data[f"{group}/bbox_valid"][:].astype(bool) if optional else np.ones(rows, dtype=bool)
                if boxes.shape != (rows,4) or valid.shape != (rows,):
                    raise ValueError(f"Invalid box shape in {group}")
                good = boxes[valid]
                if (not np.isfinite(good).all() or np.any(good < 0) or np.any(good[:,2:] <= good[:,:2]) or
                        not np.isnan(boxes[~valid]).all() or
                        ("width" in source and np.any(good[:,2] > source["width"])) or
                        ("height" in source and np.any(good[:,3] > source["height"]))):
                    raise ValueError(f"Invalid box coordinates in {group}")
            check_boxes("participant", count, True)
            check_boxes("hands", hand_count, True)
            for name in ("topology", "model_id", "detection_confidence"):
                if len(data[f"hands/{name}"]) != hand_count:
                    raise ValueError("Hand metadata length mismatch")
            for name in ("topology", "model_id"):
                if any(not x for x in data[f"hands/{name}"].asstr()[:]):
                    raise ValueError("Missing hand model or topology")
            scores = data["hands/detection_confidence"][:]
            if np.isinf(scores).any() or np.any(scores[np.isfinite(scores)] < 0):
                raise ValueError("Invalid hand detector scores")
            indices = data["objects/frame_index"][:]
            object_count = len(indices)
            if np.any((indices < 0) | (indices >= count)) or np.any(np.diff(indices) < 0):
                raise ValueError("Invalid object frame references")
            for name in ("class_id", "class_name", "model_id", "confidence", "bbox_xyxy_px"):
                if len(data[f"objects/{name}"]) != object_count:
                    raise ValueError("Object dataset length mismatch")
            check_boxes("objects", object_count, False)
            scores = data["objects/confidence"][:]
            if not np.isfinite(scores).all() or np.any((scores <= 0) | (scores > 1)):
                raise ValueError("Invalid object confidence")
            if np.any(data["objects/class_id"][:] < 0):
                raise ValueError("Invalid object class ID")
            for name in ("class_name", "model_id"):
                if any(not x for x in data[f"objects/{name}"].asstr()[:]):
                    raise ValueError("Missing object class or model")
        return {"frames": count, "hand_observations": hand_count, "object_observations": object_count,
                "pose_topology": data.attrs["pose_topology"]}
