"""Reproduce the full-frame versus focused-crop face spot check."""

from __future__ import annotations

import argparse
import json
from pathlib import Path


def probe(reference: Path, model: Path) -> dict:
    import cv2
    import mediapipe as mp
    import numpy as np

    frames = json.loads((reference / "annotations.json").read_text())["frames"]
    chosen = {}
    for frame in frames:
        chosen.setdefault((frame["pair_id"], frame["view"]), []).append(frame)
    middle = [sorted(group, key=lambda item: item["frame_index"])[len(group) // 2]
              for group in chosen.values()]
    results = {"full": 0, "focused": 0}
    with mp.tasks.vision.FaceLandmarker.create_from_model_path(str(model)) as landmarker:
        for frame in middle:
            bgr = cv2.imread(str(reference / frame["image"]))
            if bgr is None:
                raise ValueError(f"Missing reference image: {frame['image']}")
            rgb = cv2.cvtColor(bgr, cv2.COLOR_BGR2RGB)
            height, width = rgb.shape[:2]
            crops = {"full": rgb,
                     "focused": rgb[:int(.6 * height), int(.22 * width):int(.78 * width)]}
            for name, crop in crops.items():
                image = mp.Image(image_format=mp.ImageFormat.SRGB, data=np.ascontiguousarray(crop))
                results[name] += bool(landmarker.detect(image).face_landmarks)
    return {"inspected_frames": len(middle), "detections": results,
            "focused_crop": "x=[0.22w,0.78w), y=[0,0.60h)"}


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--reference", type=Path, required=True)
    parser.add_argument("--model", type=Path, required=True)
    parser.add_argument("--output", type=Path)
    args = parser.parse_args()
    report = probe(args.reference, args.model)
    result = json.dumps(report, indent=2) + "\n"
    print(result, end="")
    if args.output:
        args.output.write_text(result)


if __name__ == "__main__":
    main()
