"""Compare GPU pose models on identical saved reference frames and person boxes."""

from __future__ import annotations

import argparse
import json
from pathlib import Path
import time

import numpy as np

from .detector_pilot import write_json
from .pose_models import PersonDetector, RTMW, Sapiens, select_participant_box, visible_points
from .schema import sha256


def reference_frames(reference: Path, limit: int) -> tuple[list[dict], dict]:
    frames = json.loads((reference / "annotations.json").read_text())["frames"][:limit]
    if not frames or len({f["id"] for f in frames}) != len(frames):
        raise ValueError("Reference frames must be nonempty and unique")
    identities = []
    for frame in frames:
        path = (reference / frame["image"]).resolve()
        if not path.is_relative_to(reference / "images") or not path.is_file():
            raise ValueError(f"Invalid reference path: {frame['id']}")
        identities.append({key: frame[key] for key in
                           ("id", "pair_id", "view", "frame_index", "timestamp_ms", "width", "height")}
                          | {"image_sha256": sha256(path)})
    return frames, {"frames": identities}


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--reference", type=Path, required=True)
    parser.add_argument("--models-dir", type=Path, required=True)
    parser.add_argument("--output-root", type=Path, required=True)
    parser.add_argument("--model", choices=("rtmw", "sapiens"), required=True)
    parser.add_argument("--person-boxes", type=Path,
                        help="Reuse the saved detector results for identical comparison crops")
    parser.add_argument("--max-frames", type=int, default=200)
    parser.add_argument("--threshold", type=float, default=.3)
    parser.add_argument("--person-threshold", type=float, default=.7)
    args = parser.parse_args()
    if not 1 <= args.max_frames <= 200 or not 0 < args.threshold < 1 or not 0 < args.person_threshold < 1:
        parser.error("Use 1–200 frames and a threshold in (0, 1)")
    reference, output = args.reference.resolve(), args.output_root.resolve()
    if any(output.is_relative_to(p) for p in (reference, Path(__file__).resolve().parents[2])):
        parser.error("Output must be outside the reference directory and repository")
    output.mkdir(parents=True, exist_ok=True)
    frames, identity = reference_frames(reference, args.max_frames)
    import cv2

    cv2.setNumThreads(2)
    if args.person_boxes:
        boxes_data = json.loads(args.person_boxes.read_text())
        if boxes_data["reference"] != identity:
            raise ValueError("Person-box reference frames differ from this comparison")
    else:
        box_path = output / "person_boxes.json"
        detector = PersonDetector(args.models_dir / "yolox_m_humanart.onnx", args.person_threshold)
        if box_path.exists():
            boxes_data = json.loads(box_path.read_text())
            if boxes_data["reference"] != identity or boxes_data["model"] != detector.provenance:
                raise ValueError("Existing person boxes have different provenance")
        else:
            detected = []
            for frame in frames:
                image = cv2.imread(str(reference / frame["image"]))
                if image is None or image.shape[:2] != (frame["height"], frame["width"]):
                    raise ValueError(f"Invalid reference image: {frame['id']}")
                started = time.perf_counter()
                boxes = detector(image)
                # Preserve all boxes and the choice for visual participant audit.
                selected = select_participant_box(boxes, frame["height"])
                detected.append({"id": frame["id"], "boxes_xyxy_px": boxes.tolist(),
                                 "box_confidence": detector.last_scores.tolist(),
                                 "selected_index": selected,
                                 "selection_rule": "uppermost_center_among_person_boxes_starting_in_upper_half",
                                 "inference_s": time.perf_counter() - started})
            boxes_data = {"schema": "reference_person_boxes_v1", "reference": identity,
                          "model": detector.provenance, "frames": detected}
            write_json(box_path, boxes_data)
        del detector
    if [row["id"] for row in boxes_data["frames"]] != [f["id"] for f in frames]:
        raise ValueError("Person-box frame order differs")
    model = (RTMW(args.models_dir / "rtmw_l_384x288.onnx") if args.model == "rtmw" else
             Sapiens(args.models_dir / "sapiens_1b_coco_wholebody_best_coco_wholebody_AP_727_torchscript.pt2",
                     args.models_dir.parent / "model_sources" / "sapiens_pose_utils.py"))
    provenance = {"reference": identity, "model": model.provenance,
                  "detector": boxes_data, "threshold": args.threshold,
                  "validity_kind": "finite, in-frame, above response threshold, inside crop; physical visibility unmeasured",
                  "runner_sha256": sha256(Path(__file__)),
                  "adapter_sha256": sha256(Path(__file__).with_name("pose_models.py"))}
    destination = output / "predictions.json"
    if destination.exists():
        saved = json.loads(destination.read_text())
        if saved.get("provenance") != provenance:
            raise ValueError("Existing pose predictions have different provenance; use a new directory")
        print(json.dumps({"status": "skipped_valid", "frames": len(saved["frames"])}))
        return
    predictions = []
    overlays = output / "overlays"
    overlays.mkdir(exist_ok=True)
    wall_start = time.perf_counter()
    body_edges = ((5, 6), (5, 7), (7, 9), (6, 8), (8, 10), (5, 11), (6, 12))
    for frame, detection in zip(frames, boxes_data["frames"]):
        image = cv2.imread(str(reference / frame["image"]))
        if image is None or image.shape[:2] != (frame["height"], frame["width"]):
            raise ValueError(f"Invalid reference image: {frame['id']}")
        index = detection["selected_index"]
        box = detection["boxes_xyxy_px"][index] if index is not None else None
        start = time.perf_counter()
        if box is None:
            points = np.full((model.point_count, 2), np.nan, dtype=np.float32)
            scores = np.zeros(model.point_count, dtype=np.float32)
        else:
            points, scores = model(image, np.array(box))
        elapsed = time.perf_counter() - start
        xy, raw, valid = visible_points(points, scores, frame["width"], frame["height"], args.threshold)
        crop_valid = model.last_crop_valid if box is not None else np.zeros(model.point_count, dtype=bool)
        valid &= crop_valid
        xy[~valid] = np.nan
        for a, b in body_edges:
            if valid[a] and valid[b]:
                cv2.line(image, tuple(xy[a].astype(int)), tuple(xy[b].astype(int)), (30, 200, 250), 2)
        for i in np.flatnonzero(valid):
            color = (20, 240, 40) if i < 23 else ((255, 160, 0) if i < 91 else (0, 160, 255))
            cv2.circle(image, tuple(xy[i].astype(int)), 3 if i < 23 else 2, color, -1)
        if box is not None:
            x0, y0, x1, y1 = map(round, box)
            cv2.rectangle(image, (x0, y0), (x1, y1), (255, 100, 100), 2)
        if not cv2.imwrite(str(overlays / Path(frame["image"]).name), image):
            raise OSError("Failed to save pose overlay")
        predictions.append({"id": frame["id"], "frame_index": frame["frame_index"],
                            "timestamp_ms": frame["timestamp_ms"], "bbox_xyxy_px": box,
                            "xy_px": [[float(x), float(y)] if okay else [None, None]
                                      for (x, y), okay in zip(xy, valid)],
                            "confidence": [float(s) if np.isfinite(s) else None for s in raw],
                            "raw_xy_px": [[float(x), float(y)] if np.isfinite([x, y]).all() else [None, None]
                                          for x, y in points],
                            "crop_interior": crop_valid.tolist(),
                            "valid": valid.tolist(), "inference_s": elapsed})
        if len(predictions) % 20 == 0:
            print(json.dumps({"model": args.model, "completed_frames": len(predictions)}), flush=True)
    saved = {"schema": "participant_pose_reference_v1", "topology": model.topology,
             "provenance": provenance, "frames": predictions,
             "wall_s_including_overlays": time.perf_counter() - wall_start}
    write_json(destination, saved)
    print(json.dumps({"model": args.model, "frames": len(predictions),
                      "detector_misses": sum(f["bbox_xyxy_px"] is None for f in predictions),
                      "pose_inference_s": sum(f["inference_s"] for f in predictions),
                      "wall_s_including_overlays": saved["wall_s_including_overlays"]}))


if __name__ == "__main__":
    main()
