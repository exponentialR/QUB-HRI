"""Audit visible loose bricks and assembly extents without changing detector class names."""

import argparse
from collections import Counter, defaultdict
import json
from pathlib import Path

import numpy as np

from .compare_reference import read_hdf5
from .detector_pilot import box_iou
from .hand_quality import match_boxes
from .schema import sha256


GROUPS = {"two_two_block": "loose_brick", "four_two_block": "loose_brick",
          **{name: "assembly" for name in ("assembly_base", "biah_hole", "mini_stairway",
                                           "stacked_bridge", "stacked_stairs", "stacked_tower", "tower_head")}}


def inside_center(box, region):
    center = (np.asarray(box[:2]) + box[2:]) / 2
    return bool(np.all(center >= region[:2]) and np.all(center < region[2:]))


def score_frame(frame, predictions, iou):
    labels = [o for o in frame["objects"] if not o["ignore_for_box_score"]]
    eligible = [p for p in predictions if p["class_name"] in GROUPS and
                inside_center(p["box"], frame["roi_xyxy_px"]) and
                not any(inside_center(p["box"], r) for r in frame["ignore_regions_xyxy_px"])]
    results, details = {}, []
    assemblies = [o["bbox_xyxy_px"] for o in frame["objects"] if o["group"] == "assembly"]
    for group in ("loose_brick", "assembly"):
        truth = [o for o in labels if o["group"] == group]
        pred = [p for p in eligible if GROUPS[p["class_name"]] == group]
        gt_boxes = [{"bbox": [o["bbox_xyxy_px"][0], o["bbox_xyxy_px"][1],
                                o["bbox_xyxy_px"][2] - o["bbox_xyxy_px"][0],
                                o["bbox_xyxy_px"][3] - o["bbox_xyxy_px"][1]]} for o in truth]
        matches = match_boxes(gt_boxes, pred, iou)
        used = set(matches.values())
        # Attached bricks and subassemblies were not exhaustively annotated. They are unassessed.
        unassessed = [j for j, p in enumerate(pred) if j not in used and
                      any(inside_center(p["box"], a) for a in assemblies)]
        remaining = len(pred) - len(unassessed)
        results[group] = {"labels": len(truth), "matched": len(matches),
                          "scored_predictions": remaining, "unmatched_predictions": remaining - len(matches),
                          "unassessed_component_predictions": len(unassessed)}
        for i, obj in enumerate(truth):
            details.append({"group": group, "truth": obj, "prediction": pred[matches[i]] if i in matches else None})
    return results, details


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    for name in ("annotations", "reference", "results", "output-root"):
        parser.add_argument("--" + name, type=Path, required=True)
    a = parser.parse_args()
    import cv2
    import h5py

    reference = json.loads(a.annotations.read_text())
    if len(reference["frames"]) > 200:
        raise ValueError("Object audit must remain bounded to reference frames")
    a.output_root.mkdir(parents=True, exist_ok=True)
    if (a.output_root / "metrics.json").exists():
        raise FileExistsError("Keep previous object metrics; choose a new output directory")
    (a.output_root / "overlays").mkdir(exist_ok=True)
    totals, details, hashes = defaultdict(Counter), [], {}
    for f in reference["frames"].values():
        image_path = a.reference / f["image"]
        if sha256(image_path) != f["image_sha256"]:
            raise ValueError("Reference image changed")
        *_, path = read_hdf5(f, a.results)
        hashes[str(path)] = sha256(path)
        with h5py.File(path) as h:
            indices = np.flatnonzero(h["objects/frame_index"][:] == f["frame_index"])
            predictions = [{"box": h["objects/bbox_xyxy_px"][i].tolist(),
                            "confidence": float(h["objects/confidence"][i]),
                            "class_name": h["objects/class_name"].asstr()[i]} for i in indices]
        for threshold in (.3, .5):
            scores, matches = score_frame(f, predictions, threshold)
            for group, row in scores.items():
                totals[(threshold, f["view"], group)].update(row)
            details.append({"id": f["id"], "iou": threshold, "scores": scores, "matches": matches})
        image = cv2.imread(str(image_path))
        for obj in f["objects"]:
            x, y, xx, yy = obj["bbox_xyxy_px"]
            cv2.rectangle(image, (x, y), (xx, yy), (40, 220, 40), 2)
        for prediction in predictions:
            if prediction["class_name"] not in GROUPS:
                continue
            x, y, xx, yy = map(round, prediction["box"])
            cv2.rectangle(image, (x, y), (xx, yy), (230, 140, 20), 1)
            cv2.putText(image, prediction["class_name"], (x, max(12, y - 3)), cv2.FONT_HERSHEY_SIMPLEX, .35, (230, 140, 20), 1)
        if not cv2.imwrite(str(a.output_root / "overlays" / (f["id"] + ".jpg")), image):
            raise OSError("Cannot save object overlay")
    summary = {}
    for (threshold, view, group), row in totals.items():
        summary.setdefault(str(threshold), {}).setdefault(view, {})[group] = {
            **row, "recall": row["matched"] / row["labels"] if row["labels"] else None,
            "precision": row["matched"] / row["scored_predictions"] if row["scored_predictions"] else None}
    result = {"schema": "visible_object_audit_v1", "annotations_sha256": sha256(a.annotations),
              "reference_provenance": reference["provenance"], "coarse_evaluation_mapping": GROUPS,
              "output_sha256": hashes, "summary": summary, "details": details,
              "limitations": "Loose visible bricks and assembly outer bounds within annotated ROI only. Attached components and predictions outside the ROI/border scope are unassessed. Exact model class semantics and prior training overlap are unverified; assembly extent matches are a coarse spatial diagnostic, not exact class accuracy."}
    (a.output_root / "metrics.json").write_text(json.dumps(result, indent=2, allow_nan=False) + "\n")
    print(json.dumps(summary, indent=2))


if __name__ == "__main__":
    main()
