"""Record local GPU and comparison-model readiness without changing the machine."""

from __future__ import annotations

import argparse
from importlib.util import find_spec
import json
from pathlib import Path
import platform
import shutil
import subprocess


def command_result(arguments: list[str]) -> dict:
    try:
        result = subprocess.run(arguments, capture_output=True, text=True, check=False, timeout=20)
        return {"returncode": result.returncode, "stdout": result.stdout.strip(),
                "stderr": result.stderr.strip()}
    except (OSError, subprocess.TimeoutExpired) as exc:
        return {"returncode": None, "error": str(exc)}


def inspect() -> dict:
    devices = sorted(str(path) for path in Path("/dev").glob("nvidia*"))
    command = shutil.which("nvidia-smi")
    report = {"nvidia_device_nodes": devices, "nvidia_smi": command,
              "torch_installed": find_spec("torch") is not None,
              "mmpose_installed": find_spec("mmpose") is not None,
              "rtmlib_installed": find_spec("rtmlib") is not None,
              "onnxruntime_installed": find_spec("onnxruntime") is not None,
              "cuda_operation_verified": False,
              "cuda_available": False, "kernel": platform.release(),
              "reboot_required": Path("/var/run/reboot-required").is_file()}
    lspci = command_result(["lspci", "-nnk", "-d", "10de:"])
    report["nvidia_pci"] = lspci.get("stdout", "").splitlines()
    report["secure_boot"] = command_result(["mokutil", "--sb-state"])
    packages = command_result(["dpkg-query", "-W", "-f=${binary:Package}\t${Version}\t${db:Status-Status}\n",
                               "nvidia-driver-*", "linux-modules-nvidia-*"])
    report["installed_driver_packages"] = [line for line in packages.get("stdout", "").splitlines()
                                           if line.endswith("\tinstalled")]
    report["nvidia_smi_check"] = command_result(
        [command, "--query-gpu=name,driver_version,memory.total", "--format=csv,noheader"]
    ) if command else {"returncode": None, "error": "nvidia-smi is not installed"}
    report["driver_accessible"] = bool(devices) and report["nvidia_smi_check"]["returncode"] == 0
    if report["torch_installed"]:
        try:
            import torch

            report["torch_version"] = torch.__version__
            report["cuda_available"] = bool(torch.cuda.is_available())
            if report["cuda_available"]:
                report["cuda_devices"] = [torch.cuda.get_device_name(index)
                                          for index in range(torch.cuda.device_count())]
                matrix = torch.ones((32, 32), device="cuda")
                product = matrix @ matrix
                torch.cuda.synchronize()
                report["cuda_operation_verified"] = bool(torch.all(product == 32).item())
        except Exception as exc:
            report["torch_error"] = str(exc)
    report["torch_cuda_build"] = None
    if report["torch_installed"] and "torch_error" not in report:
        report["torch_cuda_build"] = torch.version.cuda
    report["onnx_providers"] = []
    if report["onnxruntime_installed"]:
        try:
            import onnxruntime

            report["onnxruntime_version"] = onnxruntime.__version__
            report["onnx_providers"] = onnxruntime.get_available_providers()
        except Exception as exc:
            report["onnxruntime_error"] = str(exc)
    report["rtmw_runtime_available"] = (report["mmpose_installed"] or
        (report["rtmlib_installed"] and "CUDAExecutionProvider" in report["onnx_providers"]))
    report["comparison_ready"] = (report["driver_accessible"] and report["cuda_operation_verified"]
                                  and report["rtmw_runtime_available"])
    report["readiness_scope"] = "runtime_only; checkpoint, actual-model inference, and reference-label gates remain separate"
    return report


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--output", type=Path, help="Optional local JSON report path")
    args = parser.parse_args()
    report = inspect()
    serialized = json.dumps(report, indent=2) + "\n"
    print(serialized, end="")
    if args.output:
        args.output.parent.mkdir(parents=True, exist_ok=True)
        args.output.write_text(serialized)


if __name__ == "__main__":
    main()
