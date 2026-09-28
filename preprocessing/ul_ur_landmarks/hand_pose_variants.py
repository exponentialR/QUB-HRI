"""Bounded development diagnostic for Hand5 crop/orientation and response thresholds."""

import argparse
import json
from pathlib import Path
import time

import numpy as np

from .hand_models import HandPose
from .hand_quality import match_boxes
from .schema import sha256
from .task_crops import square_rect


def rotated_crop(image, box, quarter_turns, expansion):
    """Return an exact 90-degree crop transform and the expanded inference box."""
    x0, y0, x1, y1 = square_rect(box, image.shape[1], image.shape[0], padding=2.5)
    crop = image[y0:y1, x0:x1]
    h, w = crop.shape[:2]
    matrices = (np.eye(3), np.array([[0, 1, 0], [-1, 0, w - 1], [0, 0, 1]]),
                np.array([[-1, 0, w - 1], [0, -1, h - 1], [0, 0, 1]]),
                np.array([[0, -1, h - 1], [1, 0, 0], [0, 0, 1]]))
    if quarter_turns not in range(4) or expansion <= 0:
        raise ValueError("Invalid rotation or box expansion")
    source_to_crop = matrices[quarter_turns] @ np.array([[1, 0, -x0], [0, 1, -y0], [0, 0, 1]])
    box = np.asarray(box, dtype=float)
    center, size = (box[:2] + box[2:]) / 2, (box[2:] - box[:2]) * expansion
    lo, hi = center - size / 2, center + size / 2
    corners = np.array([[lo[0], lo[1], 1], [lo[0], hi[1], 1], [hi[0], lo[1], 1], [hi[0], hi[1], 1]])
    transformed = (corners @ source_to_crop.T)[:, :2]
    transformed_box = np.r_[transformed.min(axis=0), transformed.max(axis=0)]
    return np.ascontiguousarray(np.rot90(crop, quarter_turns)), transformed_box, np.linalg.inv(source_to_crop)


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    for name in ("reference", "results", "models-dir", "output"):
        parser.add_argument("--" + name, type=Path, required=True)
    args = parser.parse_args()
    import cv2
    import h5py

    cv2.setNumThreads(2)
    frames = json.loads((args.reference / "annotations_quality_v1.json").read_text())["frames"]
    if len(frames) > 200:
        raise ValueError("This diagnostic is limited to 200 reference frames")
    if args.output.exists():
        raise FileExistsError("Preserve the previous diagnostic; choose a new output")
    model = HandPose(args.models_dir / "rtmpose_m_hand5_256.onnx")
    rows, hashes, elapsed = [], {}, 0.
    for f in frames:
        image = cv2.imread(str(args.reference / f["image"]))
        if image is None or image.shape[:2] != (f["height"], f["width"]):
            raise ValueError("Reference image dimensions differ")
        path = args.results / f["pair_id"] / (f["view"] + ".h5")
        if str(path) not in hashes:
            hashes[str(path)] = sha256(path)
        with h5py.File(path) as h:
            indices = np.flatnonzero(h["hands/frame_index"][:] == f["frame_index"])
            proposals = [{"box": h["hands/bbox_xyxy_px"][i], "actor": int(h["hands/actor"][i])} for i in indices]
        matches = {j: i for i, j in match_boxes(f["hands"], proposals).items()}
        for j, proposal in enumerate(proposals):
            if not np.isfinite(proposal["box"]).all():
                continue
            variants, images, boxes, inverses = [], [], [], []
            for expansion in (1., 1.4):
                for rotation in range(4):
                    crop, box, inverse = rotated_crop(image, proposal["box"], rotation, expansion)
                    images.append(crop); boxes.append(box); inverses.append(inverse)
                    variants.append({"box_expansion": expansion, "quarter_turns": rotation})
            start = time.perf_counter()
            coordinates, scores, interior = model.batch(images, boxes)
            elapsed += time.perf_counter() - start
            for variant, xy, confidence, inside, inverse in zip(variants, coordinates, scores, interior, inverses):
                xy = (np.c_[xy, np.ones(21)] @ inverse.T)[:, :2]
                margin = .1 * (proposal["box"][2:] - proposal["box"][:2])
                support = ((xy >= proposal["box"][:2] - margin) & (xy <= proposal["box"][2:] + margin)).all(axis=1)
                in_frame = (xy[:, 0] >= 0) & (xy[:, 0] < f["width"]) & (xy[:, 1] >= 0) & (xy[:, 1] < f["height"])
                variant.update(xy_px=xy.tolist(), confidence=confidence.tolist(), crop_interior=inside.tolist(),
                               detector_support=support.tolist(), in_frame=in_frame.tolist())
            rows.append({"id": f["id"], "view": f["view"], "hand_row": j,
                         "reference_hand_index": matches.get(j), "actor": proposal["actor"],
                         "bbox_xyxy_px": proposal["box"].tolist(), "image_sha256": sha256(args.reference / f["image"]),
                         "variants": variants})
    result = {"schema": "hand_pose_variants_v1", "model": model.provenance, "rows": rows,
              "annotations_sha256": sha256(args.reference / "annotations_quality_v1.json"),
              "code_sha256": sha256(Path(__file__)), "source_output_sha256": hashes,
              "inference_s": elapsed, "scope": "development diagnostic on original reference; not fresh validation",
              "note": "Uses detector proposals saved from decoded clips, then runs all variants on the same JPEG reference image. Labels are used only for subsequent scoring, never to choose crops."}
    args.output.write_text(json.dumps(result, indent=2, allow_nan=False) + "\n")
    print(json.dumps({"observations": len(rows), "inference_s": elapsed}))


if __name__ == "__main__":
    main()
