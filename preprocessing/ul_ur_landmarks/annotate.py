"""Serve the reference images and save manual labels on localhost only."""

from __future__ import annotations

import argparse
import hashlib
from http.server import BaseHTTPRequestHandler, HTTPServer
import json
import os
from pathlib import Path
import shutil
import tempfile
from urllib.parse import unquote


def prepare_annotations(old: dict, new: dict, reference: Path) -> None:
    """Validate an edit and bind manually checked hand labels to their image."""
    reference = reference.resolve()
    if not (old.get("schema_version") == 1 or
            old.get("schema") == "local_hand_challenge_annotations_v1"):
        raise ValueError("Unsupported reference schema")
    if any(old.get(key) != new.get(key) for key in ("schema", "schema_version")):
        raise ValueError("Reference schema changed")
    if [frame["id"] for frame in old["frames"]] != [frame["id"] for frame in new["frames"]]:
        raise ValueError("Frame IDs changed")
    if old.get("assistant_annotation_provenance") != new.get("assistant_annotation_provenance"):
        raise ValueError("Annotation source changed; reload the page to fetch current drafts")
    for before, after in zip(old["frames"], new["frames"]):
        for key in ("image", "pair_id", "view", "frame_index", "timestamp_ms", "width", "height",
                    "image_sha256", "source_video_sha256", "decoded_bgr_sha256"):
            if before.get(key) != after.get(key):
                raise ValueError(f"Immutable reference metadata changed: {key}")
        if after.get("reviewed"):
            for area in ("hands", "body", "face"):
                if not after.get(f"{area}_checked"):
                    raise ValueError(f"{area.title()} has not been checked: {after['id']}")
            if after.get("face", {}).get("visible") not in (True, False):
                raise ValueError(f"Face visibility is unknown: {after['id']}")
        if after.get("hands_checked"):
            for hand in after.get("hands", []):
                bbox = hand.get("bbox")
                if not bbox or len(bbox) != 4 or bbox[2] <= 0 or bbox[3] <= 0:
                    raise ValueError(f"Hand needs a nonempty box: {after['id']}")
            changed = any(before.get(key) != after.get(key)
                          for key in ("hands", "hands_checked", "reviewed"))
            if changed or not before.get("hand_annotation_provenance"):
                image = (reference / after["image"]).resolve()
                if not image.is_relative_to(reference / "images") or not image.is_file():
                    raise ValueError("Reference image must be inside the local images directory")
                digest = hashlib.sha256(image.read_bytes()).hexdigest()
                recorded = before.get("image_sha256") or before.get("hand_annotation_provenance", {}).get("image_sha256")
                if recorded and recorded != digest:
                    raise ValueError("Reference image differs from annotation provenance")
                after["hand_annotation_provenance"] = {
                    "status": "manual_visual", "annotator": "local_annotation_ui",
                    "image_sha256": digest, "needs_review": False,
                    "note": "Manually checked through the local annotation page; not independently validated."}
            elif before.get("hand_annotation_provenance") != after.get("hand_annotation_provenance"):
                raise ValueError("Hand provenance changed without a label edit; reload the page")


def serve(reference: Path, port: int) -> None:
    reference = reference.resolve()
    annotations = reference / "annotations.json"
    html = Path(__file__).with_name("annotator.html")
    if not annotations.is_file() or not html.is_file():
        raise FileNotFoundError("Reference set or annotator page is missing")

    class Handler(BaseHTTPRequestHandler):
        def do_GET(self) -> None:
            if self.path == "/":
                data, mime = html.read_bytes(), "text/html; charset=utf-8"
            elif self.path == "/annotations.json":
                data, mime = annotations.read_bytes(), "application/json"
            elif self.path.startswith("/images/"):
                name = unquote(self.path.removeprefix("/images/"))
                if Path(name).name != name or not name.endswith(".jpg"):
                    self.send_error(400)
                    return
                image = reference / "images" / name
                if not image.is_file():
                    self.send_error(404)
                    return
                data, mime = image.read_bytes(), "image/jpeg"
            else:
                self.send_error(404)
                return
            self.send_response(200)
            self.send_header("Content-Type", mime)
            self.send_header("Content-Length", str(len(data)))
            self.send_header("Cache-Control", "no-store")
            self.end_headers()
            self.wfile.write(data)

        def do_POST(self) -> None:
            if self.path != "/save":
                self.send_error(404)
                return
            length = int(self.headers.get("Content-Length", "0"))
            if not 0 < length < 10_000_000:
                self.send_error(413)
                return
            try:
                new = json.loads(self.rfile.read(length))
                old = json.loads(annotations.read_text())
                prepare_annotations(old, new, reference)
                temp_fd, temp_name = tempfile.mkstemp(prefix=".annotations.", suffix=".json", dir=reference)
                with os.fdopen(temp_fd, "w") as stream:
                    json.dump(new, stream, indent=2)
                    stream.write("\n")
                    stream.flush()
                    os.fsync(stream.fileno())
                shutil.copy2(annotations, reference / "annotations.backup.json")
                os.replace(temp_name, annotations)
            except (ValueError, KeyError, TypeError, json.JSONDecodeError) as exc:
                self.send_error(400, str(exc))
                return
            data = json.dumps(new).encode()
            self.send_response(200)
            self.send_header("Content-Type", "application/json")
            self.send_header("Content-Length", str(len(data)))
            self.end_headers()
            self.wfile.write(data)

    server = HTTPServer(("127.0.0.1", port), Handler)
    print(f"Local annotation page: http://127.0.0.1:{port}/", flush=True)
    server.serve_forever()


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--reference", required=True, type=Path)
    parser.add_argument("--port", type=int, default=8765)
    args = parser.parse_args()
    serve(args.reference, args.port)


if __name__ == "__main__":
    main()
