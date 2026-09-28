"""Test an existing local object/hand detector on at most 200 pilot reference images."""

from __future__ import annotations

import argparse
from collections import Counter, defaultdict
from importlib.metadata import version
import json
import math
import os
from pathlib import Path
import tempfile
import time

from .inventory import consented_ids
from .runner import selected_rows
from .schema import sha256


# Model class semantics; separate from anatomical handedness or a track identity.
HAND_ACTORS = {"lefthand": "participant", "righthand": "participant",
               "participant_hand": "participant",
               "surrogate_hand": "other_actor"}


def box_iou(first: list[float], second: list[float]) -> float:
    """Both boxes use source-pixel x_min, y_min, x_max, y_max."""
    intersection = max(0, min(first[2], second[2]) - max(first[0], second[0])) * max(
        0, min(first[3], second[3]) - max(first[1], second[1]))
    area = lambda box: max(0, box[2] - box[0]) * max(0, box[3] - box[1])
    union = area(first) + area(second) - intersection
    return intersection / union if union else 0.0


def match_boxes(labels: list[dict], predictions: list[dict], threshold: float) -> list[tuple[int, int]]:
    """Greedy highest-IoU matches, independent of actor; never reuse a box."""
    candidates = []
    for i, hand in enumerate(labels):
        x, y, width, height = hand["bbox"]
        for j, prediction in enumerate(predictions):
            overlap = box_iou([x, y, x + width, y + height], prediction["bbox_xyxy_px"])
            if overlap >= threshold:
                candidates.append((-overlap, i, j))
    used_labels, used_predictions, matches = set(), set(), []
    for _, i, j in sorted(candidates):
        if i not in used_labels and j not in used_predictions:
            matches.append((i, j))
            used_labels.add(i)
            used_predictions.add(j)
    return matches


def score_hands(frames: list[dict], predictions: list[dict], threshold: float = .5) -> dict:
    if not 0 < threshold <= 1:
        raise ValueError("IoU threshold must be in (0, 1]")
    predicted = {frame["id"]: frame for frame in predictions}
    by_view, details, excluded = defaultdict(Counter), [], []
    for frame in frames:
        if (not frame.get("hands_checked") or
                frame.get("hand_annotation_provenance", {}).get("needs_review")):
            excluded.append(frame["id"])
            continue
        if frame["id"] not in predicted:
            raise ValueError(f"Missing prediction frame: {frame['id']}")
        hands = [item for item in predicted[frame["id"]]["detections"] if item["class_name"] in HAND_ACTORS]
        labels = frame["hands"]
        matches = match_boxes(labels, hands, threshold)
        counts = by_view[frame["view"]]
        counts["frames"] += 1
        counts["predicted_hands"] += len(hands)
        counts["matched_hands"] += len(matches)
        counts["unmatched_hand_predictions"] += len(hands) - len(matches)
        for hand in labels:
            counts[f"labelled_{hand['actor']}_hands"] += 1
        for i, j in matches:
            actor = labels[i]["actor"]
            counts[f"detected_{actor}_hands"] += 1
            if actor != "unknown":
                counts["actor_scored"] += 1
                correct = actor == HAND_ACTORS[hands[j]["class_name"]]
                counts["correct_actor"] += correct
                counts[f"correct_{actor}_assignment"] += correct
        other_absent = not any(hand["actor"] == "other_actor" for hand in labels)
        counts["other_actor_absent_frames"] += other_absent
        counts["false_other_actor_frames"] += other_absent and any(
            HAND_ACTORS[hand["class_name"]] == "other_actor" for hand in hands)
        matched_ids = {i for i, _ in matches}
        details.append({"id": frame["id"], "missed_other_hand_indices": [
            i for i, hand in enumerate(labels) if hand["actor"] == "other_actor" and i not in matched_ids]})
    views = {}
    ratio = lambda n, d: n / d if d else None
    for view, counts in sorted(by_view.items()):
        views[view] = {**counts,
            "other_actor_recall": ratio(counts["detected_other_actor_hands"], counts["labelled_other_actor_hands"]),
            "participant_recall": ratio(counts["detected_participant_hands"], counts["labelled_participant_hands"]),
            "actor_accuracy_matched": ratio(counts["correct_actor"], counts["actor_scored"]),
            "other_actor_assignment_accuracy_matched": ratio(counts["correct_other_actor_assignment"], counts["detected_other_actor_hands"]),
            "other_actor_end_to_end_recall": ratio(counts["correct_other_actor_assignment"], counts["labelled_other_actor_hands"]),
            "hand_detection_precision": ratio(counts["matched_hands"], counts["predicted_hands"]),
            "false_other_actor_frame_rate": ratio(counts["false_other_actor_frames"], counts["other_actor_absent_frames"])}
    return {"iou_threshold": threshold, "views": views, "excluded_frame_ids": excluded,
            "frame_diagnostics": details, "decision_ready": False,
            "limitations": ["Provisional assistant hand boxes; ambiguous frames excluded.",
                            "Checkpoint training-data overlap with these participants is unknown.",
                            "Box IoU scoring differs from MediaPipe landmark-in-box matching; scores are not directly interchangeable.",
                            "No hand joints, tracking, or LEGO ground truth are evaluated."]}


def write_json(path: Path, value: dict) -> None:
    fd, temporary = tempfile.mkstemp(prefix=".detector.", suffix=".json", dir=path.parent)
    try:
        with os.fdopen(fd, "w") as stream:
            json.dump(value, stream, indent=2, allow_nan=False)
            stream.write("\n")
            stream.flush()
            os.fsync(stream.fileno())
        os.replace(temporary, path)
    finally:
        Path(temporary).unlink(missing_ok=True)


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    for name in ("input-root", "selection", "reference", "weights", "output-root"):
        parser.add_argument(f"--{name}", required=True, type=Path)
    scope = parser.add_mutually_exclusive_group(required=True)
    scope.add_argument("--consented", type=Path)
    scope.add_argument("--all-participants", action="store_true",
                       help="Allow the selected local IDs outside the historical consent list")
    parser.add_argument("--device", default="cpu")
    parser.add_argument("--imgsz", type=int, default=640)
    parser.add_argument("--confidence", type=float, default=.25)
    parser.add_argument("--max-frames", type=int, default=200)
    parser.add_argument("--dry-run", action="store_true")
    parser.add_argument("--training-experiment", type=Path,
                        help="Report participant-held-out validation separately for a locally trained hand model")
    args = parser.parse_args()
    if not 1 <= args.max_frames <= 200 or args.imgsz < 32 or not 0 < args.confidence < 1:
        parser.error("Use 1–200 reference frames, imgsz >= 32, and confidence in (0, 1)")
    root, output, reference = args.input_root.resolve(), args.output_root.resolve(), args.reference.resolve()
    repo = Path(__file__).resolve().parents[2]
    if any(output.is_relative_to(path) for path in (root, repo, reference)):
        parser.error("Output must be outside source clips, repository, and reference annotations")
    if not args.weights.is_file():
        parser.error("--weights must be an existing local checkpoint")
    rows = selected_rows(args.selection, None if args.all_participants else consented_ids(args.consented), root)
    selection = {row["pair_id"]: row for row in rows}
    annotation_path = reference / "annotations.json"
    frames = json.loads(annotation_path.read_text())["frames"][:args.max_frames]
    if not frames or len({f["id"] for f in frames}) != len(frames):
        parser.error("Empty or duplicate reference frames")
    images = []
    for frame in frames:
        if frame["pair_id"] not in selection or frame["view"] not in selection[frame["pair_id"]]["views"]:
            parser.error("Reference frame is not in the selected pairs")
        image = (reference / frame["image"]).resolve()
        if not image.is_relative_to(reference / "images") or not image.is_file():
            parser.error("Invalid reference image path")
        images.append(image)
    provenance = {"weights_sha256": sha256(args.weights), "weights_path": str(args.weights.resolve()),
                  "annotations_sha256": sha256(annotation_path), "selection_sha256": sha256(args.selection),
                  "image_sha256": {f["id"]: sha256(p) for f, p in zip(frames, images)},
                  "code_sha256": sha256(Path(__file__)), "device": args.device,
                  "imgsz": args.imgsz, "confidence": args.confidence, "nms_iou": .7,
                  "max_detections": 100, "hand_actor_class_mapping": HAND_ACTORS,
                  "participant_scope": "all_local_selected" if args.all_participants else "consent_list",
                  "consent_list_sha256": sha256(args.consented) if args.consented else None}
    experiment = None
    if args.training_experiment:
        experiment = json.loads((args.training_experiment / "experiment.json").read_text())
        completion = json.loads((args.training_experiment / "training_completion.json").read_text())
        if completion["sha256"] != provenance["weights_sha256"]:
            parser.error("Weights do not match the completed training experiment")
        if completion["experiment_sha256"] != sha256(args.training_experiment / "experiment.json"):
            parser.error("Training experiment identity changed")
        provenance["training_experiment_sha256"] = completion["experiment_sha256"]
    if args.dry_run:
        print(json.dumps({"frames": len(frames), "output_root": str(output), **provenance}, indent=2))
        return
    output.mkdir(parents=True, exist_ok=True)
    os.environ["YOLO_CONFIG_DIR"] = str(output / "yolo_config")
    os.environ["YOLO_AUTOINSTALL"] = "false"
    os.environ["YOLO_OFFLINE"] = "true"
    import cv2
    import torch
    from ultralytics import YOLO, settings

    settings.update({"sync": False})
    torch.set_num_threads(4)
    provenance["packages"] = {name: version(name) for name in ("torch", "torchvision", "ultralytics", "numpy", "opencv-python")}
    path = output / "predictions.json"
    if path.exists():
        saved = json.loads(path.read_text())
        if saved.get("schema_version") != "object_detections_v1" or saved.get("provenance") != provenance:
            raise ValueError("Existing detections have different provenance; choose a new output directory")
        print(json.dumps({"status": "skipped_valid", "frames": len(saved["frames"])}))
        return
    model = YOLO(str(args.weights.resolve()), task="detect")
    predictions = []
    output.joinpath("overlays").mkdir(exist_ok=True)
    started = time.perf_counter()
    for frame, image_path in zip(frames, images):
        image = cv2.imread(str(image_path))
        if image is None or image.shape[:2] != (frame["height"], frame["width"]):
            raise ValueError(f"Invalid reference dimensions: {frame['id']}")
        start = time.perf_counter()
        result = model.predict(image, device=args.device, imgsz=args.imgsz, conf=args.confidence,
                               iou=.7, max_det=100, verbose=False, save=False)[0]
        elapsed = time.perf_counter() - start
        detections = []
        for box in result.boxes:
            class_id, confidence = int(box.cls.item()), float(box.conf.item())
            xyxy = box.xyxy[0].tolist()
            if not all(math.isfinite(value) for value in [*xyxy, confidence]):
                raise ValueError("Detector returned nonfinite values")
            label = model.names[class_id]
            detections.append({"class_id": class_id, "class_name": label,
                               "confidence": confidence, "bbox_xyxy_px": xyxy})
            color = (255, 200, 0) if label == "surrogate_hand" else (60, 220, 60)
            x0, y0, x1, y1 = map(round, xyxy)
            cv2.rectangle(image, (x0, y0), (x1, y1), color, 2)
            cv2.putText(image, f"{label} {confidence:.2f}", (x0, max(18, y0-5)),
                        cv2.FONT_HERSHEY_SIMPLEX, .55, color, 1)
        if not cv2.imwrite(str(output / "overlays" / image_path.name), image):
            raise OSError("Failed to save detector overlay")
        predictions.append({key: frame[key] for key in ("id", "pair_id", "view", "frame_index", "timestamp_ms", "width", "height")}
                           | {"detections": detections, "inference_s": elapsed})
        if len(predictions) % 25 == 0:
            print(json.dumps({"completed_frames": len(predictions)}), flush=True)
    saved = {"schema_version": "object_detections_v1", "provenance": provenance,
             "class_names": model.names, "frames": predictions,
             "elapsed_s_including_overlays": time.perf_counter() - started}
    write_json(path, saved)
    metrics = {"model_sha256": provenance["weights_sha256"],
               "predictions_sha256": sha256(path),
               "annotations_sha256": provenance["annotations_sha256"],
               "hand_iou_050": score_hands(frames, predictions, .5),
               "hand_iou_030": score_hands(frames, predictions, .3),
               "object_accuracy": None,
               "object_accuracy_reason": "No independently labelled LEGO reference set yet"}
    if experiment is not None:
        pair_participants = {row["pair_id"]: row["pid"] for row in rows}
        metrics["split_metrics"] = {}
        for split in ("train", "val", "fresh"):
            selected = [frame for frame in frames if experiment["participant_splits"].get(pair_participants[frame["pair_id"]], "fresh") == split]
            metrics["split_metrics"][split] = score_hands(selected, predictions, .5)
        metrics["split_note"] = "Train and val are the frozen fine-tuning/development splits. Fresh IDs were absent from both splits. Prior pretraining overlap is unknown."
    write_json(output / "metrics.json", metrics)
    print(json.dumps({"frames": len(predictions), "class_counts": dict(Counter(
        item["class_name"] for frame in predictions for item in frame["detections"])),
        "hand_iou_050": metrics["hand_iou_050"]["views"],
        "validation_only": metrics.get("split_metrics", {}).get("val")}, indent=2))


if __name__ == "__main__":
    main()
