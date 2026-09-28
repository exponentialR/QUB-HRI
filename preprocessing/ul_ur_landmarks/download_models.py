"""Fetch pinned official MediaPipe task bundles to a local pilot directory."""

from __future__ import annotations

import argparse
import hashlib
import os
from pathlib import Path
import tempfile
from urllib.request import urlopen


MODELS = {
    "hand_landmarker": "fbc2a30080c3c557093b5ddfc334698132eb341044ccee322ccf8bcf3607cde1",
    "face_landmarker": "64184e229b263107bc2b804c6625db1341ff2bb731874b0bcc2fe6544e0bc9ff",
    "holistic_landmarker": "e2dab61191e2dcd0a15f943d8e3ed1dce13c82dfa597b9dd39f562975a50c3f8",
}


def download(output_dir: Path) -> None:
    output_dir.mkdir(parents=True, exist_ok=True)
    for name, expected in MODELS.items():
        path = output_dir / f"{name}.task"
        if path.exists():
            actual = hashlib.sha256(path.read_bytes()).hexdigest()
            if actual != expected:
                raise ValueError(f"Existing model hash mismatch: {path}")
            print(f"verified {path}")
            continue
        url = f"https://storage.googleapis.com/mediapipe-models/{name}/{name}/float16/1/{name}.task"
        fd, temp_name = tempfile.mkstemp(prefix=f".{name}.", dir=output_dir)
        temp = Path(temp_name)
        digest = hashlib.sha256()
        try:
            with os.fdopen(fd, "wb") as output, urlopen(url, timeout=60) as response:
                while chunk := response.read(1024 * 1024):
                    output.write(chunk)
                    digest.update(chunk)
            if digest.hexdigest() != expected:
                raise ValueError(f"Downloaded model hash mismatch: {name}")
            os.link(temp, path)
            print(f"downloaded {path}")
        finally:
            temp.unlink(missing_ok=True)


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--output-dir", type=Path, required=True)
    args = parser.parse_args()
    download(args.output_dir)


if __name__ == "__main__":
    main()
