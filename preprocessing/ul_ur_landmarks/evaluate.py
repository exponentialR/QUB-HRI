"""Score reviewed pilot frames without treating unlabelled frames as negatives."""

from __future__ import annotations

import argparse
from collections import defaultdict
import hashlib
import json
import math
from pathlib import Path

import numpy as np

from .schema import ACTORS


POSE_INDEX = {"mediapipe_pose_33": {"left_shoulder": 11, "left_elbow": 13, "left_wrist": 15,
                                      "right_shoulder": 12, "right_elbow": 14, "right_wrist": 16},
              "coco_wholebody_133": {"left_shoulder": 5, "left_elbow": 7, "left_wrist": 9,
                                     "right_shoulder": 6, "right_elbow": 8, "right_wrist": 10}}
HAND_INDEX = {"wrist": 0, "thumb_tip": 4, "index_tip": 8, "middle_tip": 12,
              "ring_tip": 16, "pinky_tip": 20}
# Anatomical sides, as named by MediaPipe FaceLandmarksConnections.
FACE_INDEX = {"nose": (1,), "left_eye": (263, 362),
              "right_eye": (33, 133), "mouth": (13, 14)}


def matching_hand(gt: dict, predicted: list[dict], used: set[int]) -> int | None:
    bbox = gt.get("bbox")
    if not bbox or bbox[2] <= 0 or bbox[3] <= 0:
        return None
    x, y, width, height = bbox
    candidates = []
    for index, hand in enumerate(predicted):
        if index in used:
            continue
        points = hand["xy"][hand["valid"]]
        if len(points) == 0:
            continue
        inside = ((points[:, 0] >= x - .1 * width) & (points[:, 0] <= x + 1.1 * width) &
                  (points[:, 1] >= y - .1 * height) & (points[:, 1] <= y + 1.1 * height))
        if inside.sum() >= 3:
            center = np.asarray([x + width / 2, y + height / 2])
            distance = float(np.linalg.norm(np.median(points[inside], axis=0) - center))
            candidates.append((-int(inside.sum()), distance, index))
    return min(candidates)[2] if candidates else None


def score_frame(frame: dict, output: Path) -> dict:
    import h5py

    index = frame["frame_index"]
    with h5py.File(output, "r") as data:
        if index >= len(data["frames/index"]):
            raise ValueError(f"Reference frame {index} is missing in {output}")
        topology = str(data.attrs["pose_topology"])
        pose = data["participant/pose/xy_px"][index]
        pose_valid = data["participant/pose/valid"][index]
        face = data["participant/face/xy_px"][index]
        face_valid = data["participant/face/valid"][index]
        face_detected = bool(np.any(face_valid))
        hand_rows = np.flatnonzero(data["hands/frame_index"][:] == index)
        predicted = [{"xy": data["hands/xy_px"][row], "valid": data["hands/valid"][row],
                      "actor": int(data["hands/actor"][row]),
                      "track_id": int(data["hands/track_id"][row])} for row in hand_rows]
    if topology not in POSE_INDEX:
        raise ValueError(f"No evaluation map for {topology}")
    result = {"gt_other_hands": 0, "detected_other_hands": 0, "correct_actor": 0,
              "gt_participant_hands": 0, "detected_participant_hands": 0,
              "correct_other_actor": 0, "correct_participant_actor": 0,
              "actor_scored_hands": 0, "false_other_actor_frame": 0,
              "hand_errors": [], "pose_errors": [], "face_visible": 0,
              "face_detected_when_visible": 0, "face_false_positive": 0,
              "face_errors": [], "track_matches": {}, "hand_motion_points": {},
              "pose_motion_points": {}, "hand_matches": []}
    used = set()
    for gt_hand in frame["hands"]:
        if gt_hand["actor"] == "other_actor":
            result["gt_other_hands"] += 1
        elif gt_hand["actor"] == "participant":
            result["gt_participant_hands"] += 1
        matched = matching_hand(gt_hand, predicted, used)
        result["hand_matches"].append({"actor": gt_hand["actor"], "matched": matched is not None,
                                       "predicted_actor": predicted[matched]["actor"] if matched is not None else None})
        if matched is None:
            continue
        used.add(matched)
        pred = predicted[matched]
        if gt_hand["actor"] != "unknown":
            result["actor_scored_hands"] += 1
            if pred["actor"] == ACTORS[gt_hand["actor"]]:
                result["correct_actor"] += 1
                result["correct_other_actor" if gt_hand["actor"] == "other_actor"
                       else "correct_participant_actor"] += 1
        if gt_hand["actor"] == "other_actor":
            result["detected_other_hands"] += 1
        elif gt_hand["actor"] == "participant":
            result["detected_participant_hands"] += 1
        if gt_hand["actor"] != "unknown" and gt_hand.get("handedness") in {"left", "right"}:
            label_key = f"{gt_hand['actor']}:{gt_hand['handedness']}"
            if label_key in result["track_matches"]:
                result["track_matches"][label_key] = None
            else:
                result["track_matches"][label_key] = pred["track_id"]
        diagonal = math.hypot(gt_hand["bbox"][2], gt_hand["bbox"][3])
        if gt_hand["actor"] != "unknown" and gt_hand.get("handedness") in {"left", "right"}:
            key = f"{gt_hand['actor']}:{gt_hand['handedness']}"
            if "wrist" in gt_hand.get("points", {}) and pred["valid"][0] and diagonal > 0:
                if key in result["hand_motion_points"]:
                    result["hand_motion_points"][key] = None
                else:
                    result["hand_motion_points"][key] = {
                        "gt": gt_hand["points"]["wrist"], "pred": pred["xy"][0].tolist(),
                        "scale": diagonal}
        for name, point in gt_hand.get("points", {}).items():
            landmark_index = HAND_INDEX.get(name)
            if landmark_index is not None and pred["valid"][landmark_index] and diagonal > 0:
                result["hand_errors"].append(float(np.linalg.norm(pred["xy"][landmark_index] - point) / diagonal))
    result["unmatched_predictions"] = len(predicted) - len(used)
    result["predicted_hands"] = len(predicted)
    if result["gt_other_hands"] == 0 and any(hand["actor"] == ACTORS["other_actor"] for hand in predicted):
        result["false_other_actor_frame"] = 1
    shoulders = frame.get("pose", {})
    if "left_shoulder" in shoulders and "right_shoulder" in shoulders:
        scale = float(np.linalg.norm(np.asarray(shoulders["left_shoulder"]) - shoulders["right_shoulder"]))
        if scale > 0:
            for name, point in shoulders.items():
                landmark_index = POSE_INDEX[topology].get(name)
                if landmark_index is not None and pose_valid[landmark_index]:
                    result["pose_errors"].append(float(np.linalg.norm(pose[landmark_index] - point) / scale))
                    result["pose_motion_points"][name] = {"gt": point,
                                                          "pred": pose[landmark_index].tolist(),
                                                          "scale": scale}
    face_visible = frame.get("face", {}).get("visible")
    if face_visible is True:
        result["face_visible"] = 1
        result["face_detected_when_visible"] = int(face_detected)
        anchors = frame["face"].get("anchors", {})
        if "left_eye" in anchors and "right_eye" in anchors:
            scale = float(np.linalg.norm(np.asarray(anchors["left_eye"]) - anchors["right_eye"]))
            if scale > 0:
                for name, point in anchors.items():
                    indices = FACE_INDEX.get(name)
                    if indices and all(face_valid[i] for i in indices):
                        predicted_point = np.mean(face[list(indices)], axis=0)
                        result["face_errors"].append(float(np.linalg.norm(predicted_point - point) / scale))
    elif face_visible is False:
        result["face_false_positive"] = int(face_detected)
    return result


def evaluate(annotations: Path, output_root: Path, model: str = "mediapipe") -> dict:
    frames = json.loads(annotations.read_text())["frames"]
    by_view = defaultdict(list)
    sequences = defaultdict(list)
    for frame in frames:
        if not frame.get("reviewed"):
            continue
        for area in ("hands", "body", "face"):
            if not frame.get(f"{area}_checked"):
                raise ValueError(f"Reviewed frame lacks explicit {area} check: {frame['id']}")
        if frame.get("face", {}).get("visible") is None:
            raise ValueError(f"Reviewed frame lacks face visibility: {frame['id']}")
        output = output_root / "results" / model / frame["pair_id"] / f"{frame['view']}.h5"
        if not output.is_file():
            raise FileNotFoundError(f"Missing pilot output for reviewed frame: {output}")
        scored = score_frame(frame, output)
        by_view[frame["view"]].append(scored)
        sequences[(frame["pair_id"], frame["view"])].append((frame["frame_index"], scored))
    summary = {"model": model, "total_frames": len(frames),
               "reviewed_frames": sum(len(values) for values in by_view.values()), "views": {}}
    for view, values in sorted(by_view.items()):
        totals = {key: sum(item[key] for item in values) for key in
                  ("gt_other_hands", "detected_other_hands", "correct_actor", "actor_scored_hands",
                   "false_other_actor_frame", "face_visible", "face_detected_when_visible", "face_false_positive")}
        hand_errors = [error for item in values for error in item["hand_errors"]]
        pose_errors = [error for item in values for error in item["pose_errors"]]
        face_errors = [error for item in values for error in item["face_errors"]]
        negative_frames = sum(item["gt_other_hands"] == 0 for item in values)
        track_links = track_switches = 0
        pose_temporal = []
        hand_temporal = []
        for (pair_id, sequence_view), sequence in sequences.items():
            if sequence_view != view:
                continue
            ordered = sorted(sequence)
            for (previous_index, previous), (next_index, following) in zip(ordered, ordered[1:]):
                if next_index != previous_index + 1:
                    continue
                previous_tracks, following_tracks = previous["track_matches"], following["track_matches"]
                for key in previous_tracks.keys() & following_tracks.keys():
                    if previous_tracks[key] is not None and following_tracks[key] is not None:
                        track_links += 1
                        track_switches += previous_tracks[key] != following_tracks[key]
            for (a, first), (b, middle), (c, last) in zip(ordered, ordered[1:], ordered[2:]):
                if (b, c) != (a + 1, a + 2):
                    continue
                for name, destination in (("pose_motion_points", pose_temporal),
                                          ("hand_motion_points", hand_temporal)):
                    one, two, three = first[name], middle[name], last[name]
                    for key in one.keys() & two.keys() & three.keys():
                        points = (one[key], two[key], three[key])
                        if any(point is None for point in points):
                            continue
                        scale = float(np.mean([point["scale"] for point in points]))
                        if scale > 0:
                            gt = [np.asarray(point["gt"]) for point in points]
                            pred = [np.asarray(point["pred"]) for point in points]
                            residual = (pred[2] - 2 * pred[1] + pred[0]) - (gt[2] - 2 * gt[1] + gt[0])
                            destination.append(float(np.linalg.norm(residual) / scale))
        summary["views"][view] = {"reviewed_frames": len(values), **totals,
            "other_actor_recall": totals["detected_other_hands"] / totals["gt_other_hands"] if totals["gt_other_hands"] else None,
            "actor_accuracy_matched": totals["correct_actor"] / totals["actor_scored_hands"] if totals["actor_scored_hands"] else None,
            "false_other_actor_frame_rate": totals["false_other_actor_frame"] / negative_frames if negative_frames else None,
            "median_hand_error_by_box_diagonal": float(np.median(hand_errors)) if hand_errors else None,
            "median_pose_error_by_shoulder_width": float(np.median(pose_errors)) if pose_errors else None,
            "face_recall_when_visible": totals["face_detected_when_visible"] / totals["face_visible"] if totals["face_visible"] else None,
            "median_face_anchor_error_by_eye_distance": float(np.median(face_errors)) if face_errors else None,
            "track_links": track_links, "track_switches": track_switches,
            "median_pose_temporal_residual_by_shoulder_width": float(np.median(pose_temporal)) if pose_temporal else None,
            "median_hand_temporal_residual_by_box_diagonal": float(np.median(hand_temporal)) if hand_temporal else None}
    summary["labels_complete"] = summary["reviewed_frames"] == len(frames) and set(summary["views"]) == {"CAM_UL", "CAM_UR"}
    pairs = {frame["pair_id"] for frame in frames}
    summary["comparison_outputs_complete"] = all(
        (output_root / "results" / candidate / pair_id / f"{view}.h5").is_file()
        for candidate in ("rtmw", "sapiens") for pair_id in pairs for view in ("CAM_UL", "CAM_UR"))
    summary["decision_ready"] = summary["labels_complete"] and summary["comparison_outputs_complete"]
    return summary


def evaluate_hands(annotations: Path, output_root: Path, model: str = "mediapipe",
                   *, include_uncertain: bool = False) -> dict:
    """Diagnostic box scoring; hand drafts can never complete the model decision gate."""
    content = annotations.read_bytes()
    frames = json.loads(content)["frames"]
    by_view = defaultdict(list)
    details, excluded = [], []
    counts = ("gt_other_hands", "detected_other_hands", "gt_participant_hands",
              "detected_participant_hands", "correct_actor", "actor_scored_hands",
              "correct_other_actor", "correct_participant_actor", "false_other_actor_frame",
              "unmatched_predictions", "predicted_hands")
    for frame in frames:
        provenance = frame.get("hand_annotation_provenance", {})
        if not frame.get("hands_checked"):
            excluded.append({"id": frame["id"], "reason": "hands_not_checked"})
            continue
        if provenance.get("needs_review") and not include_uncertain:
            excluded.append({"id": frame["id"], "reason": "ambiguous_visual_label"})
            continue
        output = output_root / "results" / model / frame["pair_id"] / f"{frame['view']}.h5"
        if not output.is_file():
            raise FileNotFoundError(f"Missing pilot output for hand-checked frame: {output}")
        for hand in frame["hands"]:
            box = hand.get("bbox", [])
            if (len(box) != 4 or not all(math.isfinite(v) for v in box)
                    or box[0] < 0 or box[1] < 0 or box[2] <= 0 or box[3] <= 0
                    or box[0] + box[2] > frame["width"] or box[1] + box[3] > frame["height"]):
                raise ValueError(f"Invalid hand box: {frame['id']}")
        scored = score_frame({**frame, "pose": {}, "face": {"visible": None}}, output)
        by_view[frame["view"]].append(scored)
        details.append({"id": frame["id"], "view": frame["view"],
                        "annotation_status": provenance.get("status", "unspecified"),
                        **{key: scored[key] for key in counts}, "hand_matches": scored["hand_matches"]})
    summary = {"scope": "hands_only", "model": model,
               "annotation_sha256": hashlib.sha256(content).hexdigest(),
               "total_frames": len(frames), "scored_frames": len(details),
               "provisional_frames": sum(item["annotation_status"] == "provisional_assistant_visual"
                                          for item in details),
               "include_uncertain": include_uncertain, "excluded_frames": excluded,
               "labels_complete": False, "decision_ready": False, "views": {},
               "matching_rule": "One-to-one greedy match: at least 3 valid predicted points inside a box expanded by 10%; actor not used in matching.",
               "limitations": ["Assistant visual boxes are provisional, not independently validated reference labels.",
                               "Box matches measure spatial detection coverage, not joint accuracy.",
                               "Assignment accuracy on matched hands excludes missed detections; consult per-actor recall and end-to-end recall.",
                               "Adjacent frames are correlated; counts are diagnostic, not independent statistical samples.",
                               "Hands-only scoring does not evaluate upper body, face, handedness, or tracking."],
               "frames": details}
    for view, values in sorted(by_view.items()):
        totals = {key: sum(item[key] for item in values) for key in counts}
        ratio = lambda n, d: n / d if d else None
        negative = sum(item["gt_other_hands"] == 0 for item in values)
        summary["views"][view] = {
            "scored_frames": len(values), **totals,
            "other_actor_absent_frames": negative,
            "all_hands_absent_frames": sum(item["gt_other_hands"] + item["gt_participant_hands"] == 0 for item in values),
            "other_actor_recall": ratio(totals["detected_other_hands"], totals["gt_other_hands"]),
            "participant_recall": ratio(totals["detected_participant_hands"], totals["gt_participant_hands"]),
            "actor_accuracy_matched": ratio(totals["correct_actor"], totals["actor_scored_hands"]),
            "other_actor_accuracy_matched": ratio(totals["correct_other_actor"], totals["detected_other_hands"]),
            "participant_actor_accuracy_matched": ratio(totals["correct_participant_actor"], totals["detected_participant_hands"]),
            "other_actor_end_to_end_recall": ratio(totals["correct_other_actor"], totals["gt_other_hands"]),
            "false_other_actor_frame_rate": ratio(totals["false_other_actor_frame"], negative),
        }
    return summary


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--annotations", required=True, type=Path)
    parser.add_argument("--output-root", required=True, type=Path)
    parser.add_argument("--model", default="mediapipe")
    parser.add_argument("--scope", choices=("full", "hands"), default="full")
    parser.add_argument("--include-uncertain", action="store_true",
                        help="Hands-only sensitivity check including flagged draft boxes")
    args = parser.parse_args()
    if args.scope == "hands":
        result = evaluate_hands(args.annotations, args.output_root, args.model,
                                include_uncertain=args.include_uncertain)
    else:
        if args.include_uncertain:
            parser.error("--include-uncertain requires --scope hands")
        result = evaluate(args.annotations, args.output_root, args.model)
    print(json.dumps(result, indent=2))


if __name__ == "__main__":
    main()
