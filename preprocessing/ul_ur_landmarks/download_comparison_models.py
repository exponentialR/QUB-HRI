"""Fetch pinned OpenMMLab exports and optionally import a local gated Sapiens file."""

from __future__ import annotations

import argparse
import hashlib
import json
import os
from pathlib import Path
import shutil
import tempfile
import urllib.request
import zipfile


BASE = "https://download.openmmlab.com/mmpose/v1/projects/"
ARTIFACTS = {
    "rtmw_l_384x288": (
        "rtmw/onnx_sdk/rtmw-dw-x-l_simcc-cocktail14_270e-384x288_20231122.zip",
        "a87e1af41a0a067776dba7d46e1c21c8f6e9f18e247e0e606718dd1f31e96ffd",
        "bd033156e5104c4f5d2edfe0453e02661e30a2f3da453ec93c8764d561b83054"),
    "yolox_m_humanart": (
        "rtmposev1/onnx_sdk/yolox_m_8xb8-300e_humanart-c2c7a14a.zip",
        "a000224fd8ba283202bc62d4a5fcdfe353adb9f468777dbac1ea2ada2093adde",
        "3dea6513388889f0fff4b77bf7a26013600321b9eb9ceb0e9a400a82572f5f23"),
    "rtmpose_m_hand5_256": (
        "rtmposev1/onnx_sdk/rtmpose-m_simcc-hand5_pt-aic-coco_210e-256x256-74fb594_20230320.zip",
        "45c9e1aa21b8fe33859973c49bce4647645d312e7a226e398a42f9137d44a56d",
        "39e858936bca0f94c09847d4e70b68a51d6c0adac61f36b457fcadb54621cd29"),
    "rtmdet_nano_hand_320": (
        "rtmposev1/onnx_sdk/rtmdet_nano_8xb32-300e_hand-267f9c8f.zip",
        "9c0370a43c02b2fe42b4382aba7383d97cfa3ed35623b655cac4f0c25cfde402",
        "568d3ea97a5b142488366b67e036b6a5cb0a1fef9087a710cb8e66b6979fbac2"),
}
SAPIENS_NAME = "sapiens_1b_coco_wholebody_best_coco_wholebody_AP_727_torchscript.pt2"
SAPIENS_SHA256 = "911c4a26dbede2d21fa1a3bffb8d1b5ecac6bcb4983951408ba80092bc07b054"
SAPIENS_COMMIT = "2cb07227a740cf09896309ea3a3b8fa44429865c"


def digest(path: Path) -> str:
    value = hashlib.sha256()
    with path.open("rb") as stream:
        for chunk in iter(lambda: stream.read(8 * 1024 * 1024), b""):
            value.update(chunk)
    return value.hexdigest()


def publish(path: Path, expected: str, source) -> None:
    """Validate before publication; an existing mismatched file is preserved."""
    if path.exists():
        if digest(path) != expected:
            raise ValueError(f"Existing checkpoint has a different checksum: {path}")
        return
    fd, name = tempfile.mkstemp(prefix=".model.", suffix=".partial", dir=path.parent)
    temporary = Path(name)
    try:
        with os.fdopen(fd, "wb") as out:
            shutil.copyfileobj(source, out, 8 * 1024 * 1024)
        if digest(temporary) != expected:
            raise ValueError(f"Downloaded/imported checkpoint checksum mismatch: {path.name}")
        try:
            os.link(temporary, path)
        except FileExistsError:
            if digest(path) != expected:
                raise ValueError(f"Concurrent checkpoint has a different checksum: {path}")
    finally:
        temporary.unlink(missing_ok=True)


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--output-dir", type=Path, required=True)
    parser.add_argument("--sapiens-file", type=Path,
                        help="Already downloaded 133-point checkpoint after publisher access acceptance")
    args = parser.parse_args()
    root = args.output_dir.resolve()
    if root.is_relative_to(Path(__file__).resolve().parents[2]):
        parser.error("Keep model downloads outside the repository")
    root.mkdir(parents=True, exist_ok=True)
    for name, (suffix, archive_hash, onnx_hash) in ARTIFACTS.items():
        path = root / f"{name}.zip"
        if not path.exists():
            with urllib.request.urlopen(BASE + suffix, timeout=60) as response:
                publish(path, archive_hash, response)
        elif digest(path) != archive_hash:
            raise ValueError(f"Existing archive checksum mismatch: {path}")
        with zipfile.ZipFile(path) as archive:
            members = [entry for entry in archive.namelist() if entry.endswith(".onnx")]
            if len(members) != 1:
                raise ValueError(f"Expected one ONNX model: {path}")
            # Read the member directly; never extract archive-controlled paths.
            with archive.open(members[0]) as source:
                publish(root / f"{name}.onnx", onnx_hash, source)
        print(json.dumps({"name": name, "source": BASE + suffix, "sha256": onnx_hash}), flush=True)
    sources = root.parent / "model_sources"
    sources.mkdir(exist_ok=True)
    decoder = sources / "sapiens_pose_utils.py"
    decoder_hash = "78aa6c9d4631c00fe05362ea1e9066821e8a1e4c7b1f58d1a34cf2715e3342f7"
    if not decoder.exists():
        url = f"https://raw.githubusercontent.com/facebookresearch/sapiens/{SAPIENS_COMMIT}/lite/demo/pose_utils.py"
        with urllib.request.urlopen(url, timeout=60) as response:
            publish(decoder, decoder_hash, response)
    elif digest(decoder) != decoder_hash:
        raise ValueError("Existing Sapiens decoder source has a different checksum")
    license_path = sources / "sapiens_LICENSE.txt"
    if not license_path.exists():
        url = f"https://raw.githubusercontent.com/facebookresearch/sapiens/{SAPIENS_COMMIT}/LICENSE"
        with urllib.request.urlopen(url, timeout=60) as response:
            license_path.write_bytes(response.read())
    if args.sapiens_file:
        with args.sapiens_file.open("rb") as source:
            publish(root / SAPIENS_NAME, SAPIENS_SHA256, source)
        print(json.dumps({"name": SAPIENS_NAME, "sha256": SAPIENS_SHA256, "license": "CC-BY-NC-4.0"}))


if __name__ == "__main__":
    main()
