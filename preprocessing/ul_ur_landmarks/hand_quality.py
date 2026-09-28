"""Score hand boxes, selected joints and adjacent-frame tracks separately."""

import argparse
from collections import defaultdict
import json
import math
from pathlib import Path

import numpy as np

from .compare_reference import point_score, read_hdf5, summarize
from .detector_pilot import box_iou
from .evaluate import HAND_INDEX
from .schema import ACTORS, sha256


def match_boxes(truth, predictions, threshold=.5):
    candidates = []
    for i, gt in enumerate(truth):
        x, y, w, h = gt["bbox"]
        for j, pred in enumerate(predictions):
            if pred["box"] is None:
                continue
            overlap = box_iou([x, y, x + w, y + h], pred["box"])
            if overlap >= threshold:
                candidates.append((-overlap, i, j))
    matches, used = {}, set()
    for _, i, j in sorted(candidates):
        if i not in matches and j not in used:
            matches[i] = j
            used.add(j)
    return matches


def evaluate(annotations, root, include_uncertain=False):
    import h5py

    frames = json.loads(annotations.read_text())["frames"]
    groups, sequences, hashes, details = defaultdict(list), defaultdict(list), {}, []
    for f in frames:
        if not f.get("hands_checked") or (f.get("hand_annotation_provenance", {}).get("needs_review") and not include_uncertain):
            continue
        *_, path = read_hdf5(f, root)
        if str(path) not in hashes:
            hashes[str(path)] = sha256(path)
        with h5py.File(path) as h:
            rows = np.flatnonzero(h["hands/frame_index"][:] == f["frame_index"])
            preds = []
            for row in rows:
                xy, valid = h["hands/xy_px"][row], h["hands/valid"][row]
                points = xy[valid]
                box = None
                if "hands/bbox_xyxy_px" in h:
                    candidate = h["hands/bbox_xyxy_px"][row]
                    if np.isfinite(candidate).all() and np.all(candidate[2:] > candidate[:2]):
                        box = candidate
                if box is None and len(points) >= 3:
                    box = np.r_[points.min(axis=0), points.max(axis=0)]
                model = h["hands/model_id"][row] if "hands/model_id" in h else "mediapipe_hand_landmarker"
                if isinstance(model, bytes):
                    model = model.decode()
                preds.append({"box": box, "xy": xy, "valid": valid, "model": str(model),
                              "actor": int(h["hands/actor"][row]), "track": int(h["hands/track_id"][row])})
        matches = match_boxes(f["hands"], preds)
        tracks = {}
        for i, hand in enumerate(f["hands"]):
            pred = preds[matches[i]] if i in matches else None
            scored = {"id": f["id"], "hand_index": i, "actor": hand["actor"],
                      "box_matched": pred is not None,
                      "correct_actor": pred is not None and pred["actor"] == ACTORS[hand["actor"]],
                      "valid_joint_count": int(pred["valid"].sum()) if pred else 0,
                      "model": pred["model"] if pred else None, "points": {}}
            diagonal = math.hypot(f["width"], f["height"])
            for name, point in hand.get("points", {}).items():
                index = HAND_INDEX[name]
                s = point_score(point, pred["xy"][index] if pred else [np.nan, np.nan],
                                bool(pred and pred["valid"][index]), diagonal)
                scale = math.hypot(hand["bbox"][2], hand["bbox"][3])
                s["error_by_hand_box_diagonal"] = s["error_px"] / scale if s["available"] else None
                scored["points"][name] = s
            key = hand.get("reference_track_id")
            if key:
                if key in tracks:
                    raise ValueError(f"Duplicate reference track {key} in {f['id']}")
                wrist = scored["points"].get("wrist")
                tracks[key] = {"predicted_track": pred["track"] if pred else None,
                               "gt_wrist": hand.get("points", {}).get("wrist"),
                               "pred_wrist": pred["xy"][0].tolist() if wrist and wrist["available"] else None}
            groups[(f["view"], hand["actor"])].append(scored)
            details.append(scored)
        sequences[(f["pair_id"], f["view"])].append((f["frame_index"], tracks))
    views = {}
    for view in ("CAM_UL", "CAM_UR"):
        views[view] = {}
        for actor in ("participant", "other_actor"):
            rows = groups[(view, actor)]
            points = [p for r in rows for p in r["points"].values()]
            normalized = [p["error_by_hand_box_diagonal"] for p in points if p["available"]]
            matched = sum(r["box_matched"] for r in rows)
            views[view][actor] = {"labelled_hands": len(rows), "box_matches": matched,
                "correct_actor_matches": sum(r["correct_actor"] for r in rows),
                "matches_with_at_least_3_valid_joints": sum(r["valid_joint_count"] >= 3 for r in rows),
                "selected_joint_scores": summarize(points),
                "median_joint_error_by_hand_box_diagonal_when_available": float(np.median(normalized)) if normalized else None}
        links = matched_links = switches = 0
        temporal = []
        for (_, v), sequence in sequences.items():
            if v != view:
                continue
            sequence.sort()
            for (a, first), (b, second) in zip(sequence, sequence[1:]):
                if b != a + 1:
                    continue
                for key in first.keys() & second.keys():
                    links += 1
                    one, two = first[key]["predicted_track"], second[key]["predicted_track"]
                    if one is not None and two is not None:
                        matched_links += 1
                        switches += one != two
            for (a, first), (b, second), (c, third) in zip(sequence, sequence[1:], sequence[2:]):
                if (b, c) != (a + 1, a + 2):
                    continue
                for key in first.keys() & second.keys() & third.keys():
                    points = [first[key], second[key], third[key]]
                    if any(p["gt_wrist"] is None or p["pred_wrist"] is None for p in points):
                        continue
                    residuals = [np.asarray(p["pred_wrist"]) - p["gt_wrist"] for p in points]
                    temporal.append(float(np.linalg.norm(residuals[2] - 2 * residuals[1] + residuals[0])))
        views[view]["tracking"] = {"reference_adjacent_links": links, "both_boxes_matched_links": matched_links,
            "links_with_a_missing_box": links - matched_links, "track_switches_on_matched_links": switches,
            "wrist_temporal_triples": len(temporal),
            "median_wrist_temporal_residual_px": float(np.median(temporal)) if temporal else None}
    return {"schema": "hand_quality_v1", "annotations_sha256": sha256(annotations),
            "include_uncertain": include_uncertain, "output_sha256": hashes, "views": views, "details": details,
            "limitations": "Provisional assistant labels; approximate glove wrist centers, no independent human validation. Box matching uses IoU 0.5 and ignores actor class. Point availability is not physical visibility. Track scores cover adjacent sampled frames only."}


def main():
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--annotations", type=Path, required=True)
    p.add_argument("--results", type=Path, required=True)
    p.add_argument("--output", type=Path, required=True)
    p.add_argument("--include-uncertain", action="store_true")
    a = p.parse_args()
    result = evaluate(a.annotations, a.results, a.include_uncertain)
    content = json.dumps(result, indent=2, allow_nan=False) + "\n"
    if a.output.exists() and a.output.read_text() != content:
        raise FileExistsError("Use a separate output for a different evaluation")
    a.output.write_text(content)
    print(json.dumps(result["views"], indent=2))


if __name__ == "__main__":
    main()
