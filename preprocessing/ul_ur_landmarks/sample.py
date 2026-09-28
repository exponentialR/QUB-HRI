"""Export a 200-frame local reference set and contact sheets from 20 pilot pairs."""

from __future__ import annotations

import argparse
import json
from pathlib import Path

from .inventory import VIEWS, consented_ids
from .runner import frame_pts_ms, selected_rows


def sample_positions(frame_count: int, count: int = 5) -> list[int]:
    if frame_count < count:
        raise ValueError(f"Need at least {count} frames, got {frame_count}")
    if count != 5:
        raise ValueError("The pilot reference contract uses exactly five frames per clip")
    middle = frame_count // 2
    chosen = {round(.2 * (frame_count - 1)), middle - 1, middle, middle + 1,
              round(.8 * (frame_count - 1))}
    if len(chosen) != 5:
        chosen = {round((frame_count - 1) * (index + .5) / count) for index in range(count)}
    if len(chosen) != 5:
        raise ValueError(f"Could not choose five distinct frames from {frame_count}")
    return sorted(chosen)


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--input-root", required=True, type=Path)
    parser.add_argument("--consented", required=True, type=Path)
    parser.add_argument("--selection", required=True, type=Path)
    parser.add_argument("--output-root", required=True, type=Path)
    args = parser.parse_args()
    import cv2
    import numpy as np

    input_root = args.input_root.resolve()
    output_root = args.output_root.resolve()
    if output_root.is_relative_to(input_root) or output_root.is_relative_to(Path(__file__).resolve().parents[2]):
        parser.error("Output must be outside the input tree and repository")
    rows = selected_rows(args.selection, consented_ids(args.consented), input_root)
    frames_dir = output_root / "reference" / "images"
    frames_dir.mkdir(parents=True, exist_ok=True)
    annotations_file = output_root / "reference" / "annotations.json"
    if annotations_file.exists():
        raise FileExistsError(f"Preserving existing annotations: {annotations_file}")
    entries = []
    contact = {view: [] for view in VIEWS}
    for pair_index, row in enumerate(rows):
        for view in VIEWS:
            path = input_root / row["views"][view]["relpath"]
            pts = frame_pts_ms(path)
            chosen = sample_positions(len(pts))
            capture = cv2.VideoCapture(str(path))
            if not capture.isOpened():
                raise ValueError(f"Cannot decode {path}")
            wanted = set(chosen)
            seen = set()
            index = 0
            try:
                while wanted - seen:
                    okay, frame = capture.read()
                    if not okay:
                        raise ValueError(f"Decoder stopped before sampled frames in {path}")
                    if index in wanted:
                        filename = f"{pair_index:02d}_{view}_{index:05d}.jpg"
                        image_path = frames_dir / filename
                        if image_path.exists():
                            raise FileExistsError(f"Preserving existing reference image: {image_path}")
                        if not cv2.imwrite(str(image_path), frame, [cv2.IMWRITE_JPEG_QUALITY, 90]):
                            raise OSError(f"Could not write {image_path}")
                        entries.append({"id": filename.removesuffix(".jpg"),
                                        "image": f"images/{filename}",
                                        "pair_id": row["pair_id"], "view": view,
                                        "frame_index": index, "timestamp_ms": pts[index],
                                        "width": frame.shape[1], "height": frame.shape[0],
                                        "reviewed": False, "hands_checked": False,
                                        "body_checked": False, "face_checked": False,
                                        "hands": [], "pose": {},
                                        "face": {"visible": None, "anchors": {}}})
                        if index == chosen[len(chosen) // 2]:
                            thumb = cv2.resize(frame, (320, 180))
                            cv2.putText(thumb, f"{pair_index:02d} {row['pid']} {row['subtask_dir']}",
                                        (8, 22), cv2.FONT_HERSHEY_SIMPLEX, 0.55, (255, 255, 255), 2)
                            contact[view].append(thumb)
                        seen.add(index)
                    index += 1
            finally:
                capture.release()
    for view, thumbnails in contact.items():
        if len(thumbnails) != len(rows):
            raise ValueError(f"Missing contact thumbnails for {view}")
        canvas = np.zeros((900, 1280, 3), dtype=np.uint8)
        for index, thumbnail in enumerate(thumbnails):
            y, x = divmod(index, 4)
            canvas[y * 180:(y + 1) * 180, x * 320:(x + 1) * 320] = thumbnail
        cv2.imwrite(str(output_root / "reference" / f"contact_{view}.jpg"), canvas)
    annotations_file.write_text(json.dumps({"schema_version": 1, "frames": entries}, indent=2) + "\n")
    print(json.dumps({"pairs": len(rows), "reference_frames": len(entries),
                      "annotations": str(annotations_file)}))


if __name__ == "__main__":
    main()
