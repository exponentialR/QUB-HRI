"""Compare selected visible body/face anchors, counting unavailable predictions explicitly."""

from __future__ import annotations

import argparse
from collections import defaultdict
from itertools import combinations
import json
import math
from pathlib import Path

import numpy as np

from .evaluate import FACE_INDEX, POSE_INDEX
from .schema import sha256


def point_score(truth, prediction, valid, diagonal):
    """An unavailable point remains in the denominator and never becomes zero error."""
    available = bool(valid) and bool(np.isfinite(prediction).all())
    error = float(np.linalg.norm(np.asarray(prediction) - truth)) if available else None
    return {"available": available, "error_px": error,
            "error_by_image_diagonal": error / diagonal if available else None}


def summarize(points):
    errors = [p["error_px"] for p in points if p["available"]]
    normalized = [p["error_by_image_diagonal"] for p in points if p["available"]]
    return {"labelled_points": len(points), "available_points": len(errors),
            "availability": len(errors) / len(points) if points else None,
            "median_error_px_when_available": float(np.median(errors)) if errors else None,
            "p90_error_px_when_available": float(np.percentile(errors, 90)) if errors else None,
            "fraction_within_30px_including_missing": sum(e <= 30 for e in errors) / len(points) if points else None,
            "fraction_within_0_03_image_diagonal_including_missing": sum(e <= .03 for e in normalized) / len(points) if points else None}


def read_hdf5(frame, root):
    import h5py

    path = root / frame["pair_id"] / (frame["view"] + ".h5")
    with h5py.File(path, "r") as h:
        source = json.loads(h.attrs["source_json"])
        for key in ("pair_id", "view", "width", "height"):
            if source[key] != frame[key]:
                raise ValueError(f"Source mismatch for {frame['id']}: {key}")
        index = frame["frame_index"]
        if h["frames/index"][index] != index or abs(float(h["frames/timestamp_ms"][index]) - frame["timestamp_ms"]) > 1:
            raise ValueError(f"Frame/time mismatch for {frame['id']}")
        return (str(h.attrs["pose_topology"]), h["participant/pose/xy_px"][index],
                h["participant/pose/valid"][index], h["participant/face/xy_px"][index],
                h["participant/face/valid"][index], path)


def compare(annotations: Path, candidates: dict[str, Path]):
    reference = json.loads(annotations.read_text())
    frames = reference["frames"]
    if len({f["id"] for f in frames}) != len(frames):
        raise ValueError("Duplicate reference frame IDs")
    output = {"schema": "selected_anchor_comparison_v1", "annotations_sha256": sha256(annotations),
              "reference_provenance": reference.get("supplement_provenance"),
              "limitations": "Assistant visual reference, not independently validated. Selected visible anchors only; clothed joint centers are approximate. Missing labels are not negatives. Face availability is not face-anchor correctness.",
              "candidates": {}, "paired_comparisons": {}}
    scores = {}
    for name, root in candidates.items():
        groups, all_points, identities, faces = defaultdict(list), {}, {}, defaultdict(list)
        for frame in frames:
            if not frame.get("body_checked") or not frame.get("face_checked"):
                continue
            topology, pose, pose_valid, face, face_valid, path = read_hdf5(frame, root)
            if topology not in POSE_INDEX:
                raise ValueError(f"Unmapped pose topology {topology}")
            if str(path) not in identities:
                identities[str(path)] = sha256(path)
            diagonal = math.hypot(frame["width"], frame["height"])
            visible = frame["face"].get("visible")
            faces[frame["view"]].append({"visible": visible, "returned": bool(face_valid.any())})
            for area, labelled in (("pose", frame.get("pose", {})),
                                   ("face", frame["face"].get("anchors", {}) if visible else {})):
                for joint, truth in labelled.items():
                    if area == "pose":
                        i = POSE_INDEX[topology][joint]
                        predicted, valid = pose[i], pose_valid[i]
                    else:
                        indices = list(FACE_INDEX[joint])
                        predicted, valid = face[indices].mean(axis=0), face_valid[indices].all()
                    scored = point_score(truth, predicted, valid, diagonal)
                    scored.update(id=frame["id"], view=frame["view"], area=area, joint=joint)
                    key = f"{frame['id']}/{area}/{joint}"
                    all_points[key] = scored
                    groups[(frame["view"], area)].append(scored)
        views = {}
        for view in sorted(faces):
            visible = [f for f in faces[view] if f["visible"] is True]
            absent = [f for f in faces[view] if f["visible"] is False]
            views[view] = {area: summarize(groups[(view, area)]) for area in ("pose", "face")}
            views[view]["pose_by_joint"] = {
                joint: summarize([p for p in groups[(view, "pose")] if p["joint"] == joint])
                for joint in sorted(POSE_INDEX["coco_wholebody_133"])}
            views[view]["face_visible_frames"] = len(visible)
            views[view]["face_returned_when_visible"] = sum(f["returned"] for f in visible)
            views[view]["false_face_rate"] = sum(f["returned"] for f in absent) / len(absent) if absent else None
        output["candidates"][name] = {"views": views, "output_sha256": identities,
                                      "point_scores": all_points}
        scores[name] = all_points
    for a, b in combinations(scores, 2):
        paired = {}
        if scores[a].keys() != scores[b].keys():
            raise ValueError("Candidates were not scored on identical labelled points")
        for view in ("CAM_UL", "CAM_UR"):
            paired[view] = {}
            for area in ("pose", "face"):
                keys = [k for k, p in scores[a].items() if p["view"] == view and p["area"] == area]
                common = [k for k in keys if scores[a][k]["available"] and scores[b][k]["available"]]
                paired[view][area] = {"same_labelled_points": len(keys), "both_available_points": len(common),
                                     a: summarize([scores[a][k] for k in common]),
                                     b: summarize([scores[b][k] for k in common])}
        output["paired_comparisons"][f"{a} vs {b}"] = paired
    return output


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--annotations", type=Path, required=True)
    parser.add_argument("--candidate", action="append", required=True, help="NAME=/path/to/results/model")
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()
    candidates = {}
    for argument in args.candidate:
        name, path = argument.split("=", 1)
        if not name or name in candidates:
            parser.error("Candidate names must be unique and nonempty")
        candidates[name] = Path(path).resolve()
    result = compare(args.annotations.resolve(), candidates)
    content = json.dumps(result, indent=2, allow_nan=False) + "\n"
    if args.output.exists() and args.output.read_text() != content:
        raise FileExistsError("Use a new output path for a different comparison")
    args.output.write_text(content)
    print(json.dumps({name: data["views"] for name, data in result["candidates"].items()}, indent=2))


if __name__ == "__main__":
    main()
