"""Launch an inference module with CUDA libraries from its isolated environment.

Example: .gpu-venv/bin/python -m preprocessing.ul_ur_landmarks.gpu_run \
    preprocessing.ul_ur_landmarks.pose_reference --help
"""

from __future__ import annotations

import argparse
import os
from pathlib import Path
import site
import sys


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("module", help="Python module to run")
    parser.add_argument("arguments", nargs=argparse.REMAINDER)
    args = parser.parse_args()
    if sys.prefix == sys.base_prefix:
        parser.error("Use the isolated GPU environment's Python interpreter")
    libraries = []
    for folder in site.getsitepackages():
        libraries.extend(str(path) for path in sorted((Path(folder) / "nvidia").glob("*/lib"))
                         if path.is_dir())
    if not libraries:
        parser.error("No packaged NVIDIA runtime libraries found in this environment")
    env = dict(os.environ)
    env.pop("PYTHONPATH", None)
    env.pop("PYTHONHOME", None)
    inherited = env.get("LD_LIBRARY_PATH")
    env["LD_LIBRARY_PATH"] = os.pathsep.join(libraries + ([inherited] if inherited else []))
    # The loader reads LD_LIBRARY_PATH at process startup. Changing it after
    # importing torch/onnxruntime cannot reliably expose cuDNN's split libraries.
    os.execve(sys.executable, [sys.executable, "-m", args.module, *args.arguments], env)


if __name__ == "__main__":
    main()
