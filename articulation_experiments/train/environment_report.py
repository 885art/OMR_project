"""Collect a reproducible environment report for articulation experiments.

This script is intentionally diagnostic only. It does not install packages,
download model weights, convert data, or start training.
"""

from __future__ import annotations

import argparse
import importlib
import importlib.metadata
import json
import os
import platform
import random
import shutil
import subprocess
import sys
from datetime import datetime, timezone
from pathlib import Path
from typing import Any


DEFAULT_SEED = 20260717


def run_command(command: list[str], timeout: int = 15) -> dict[str, Any]:
    """Run a command without raising and return serializable diagnostics."""
    try:
        completed = subprocess.run(
            command,
            capture_output=True,
            check=False,
            text=True,
            timeout=timeout,
        )
    except (OSError, subprocess.TimeoutExpired) as exc:
        return {
            "ok": False,
            "returncode": None,
            "stdout": "",
            "stderr": str(exc),
        }

    return {
        "ok": completed.returncode == 0,
        "returncode": completed.returncode,
        "stdout": completed.stdout.strip(),
        "stderr": completed.stderr.strip(),
    }


def package_metadata(package_name: str) -> dict[str, Any]:
    """Report package installation metadata without requiring an import."""
    try:
        metadata = importlib.metadata.metadata(package_name)
        return {
            "installed": True,
            "metadata_version": importlib.metadata.version(package_name),
            "license": metadata.get("License"),
            "requires_python": metadata.get("Requires-Python"),
        }
    except importlib.metadata.PackageNotFoundError:
        return {
            "installed": False,
            "metadata_version": None,
            "license": None,
            "requires_python": None,
        }


def import_package(package_name: str) -> dict[str, Any]:
    """Import a package and retain any import error in the report."""
    result = package_metadata(package_name)
    try:
        module = importlib.import_module(package_name)
        result.update(
            {
                "importable": True,
                "runtime_version": getattr(module, "__version__", None),
                "module_path": getattr(module, "__file__", None),
                "import_error": None,
            }
        )
    except Exception as exc:  # Diagnostic script must preserve the full failure.
        result.update(
            {
                "importable": False,
                "runtime_version": None,
                "module_path": None,
                "import_error": f"{type(exc).__name__}: {exc}",
            }
        )
    return result


def collect_torch() -> tuple[dict[str, Any], Any | None]:
    result = import_package("torch")
    if not result["importable"]:
        result.update(
            {
                "cuda_build": None,
                "cuda_available": False,
                "cuda_device_count": 0,
                "cudnn_version": None,
                "devices": [],
            }
        )
        return result, None

    torch = importlib.import_module("torch")
    cuda_available = bool(torch.cuda.is_available())
    device_count = int(torch.cuda.device_count())
    devices = []
    for index in range(device_count):
        properties = torch.cuda.get_device_properties(index)
        devices.append(
            {
                "index": index,
                "name": torch.cuda.get_device_name(index),
                "compute_capability": list(torch.cuda.get_device_capability(index)),
                "total_memory_bytes": int(properties.total_memory),
            }
        )

    result.update(
        {
            "cuda_build": torch.version.cuda,
            "cuda_available": cuda_available,
            "cuda_device_count": device_count,
            "cudnn_version": torch.backends.cudnn.version(),
            "devices": devices,
        }
    )
    return result, torch


def collect_nvidia_gpus() -> dict[str, Any]:
    executable = shutil.which("nvidia-smi")
    if executable is None:
        return {"nvidia_smi_found": False, "gpus": [], "error": "nvidia-smi not found"}

    query = run_command(
        [
            executable,
            "--query-gpu=index,name,driver_version,memory.total,compute_cap",
            "--format=csv,noheader,nounits",
        ]
    )
    gpus = []
    if query["ok"]:
        for line in query["stdout"].splitlines():
            fields = [field.strip() for field in line.split(",")]
            if len(fields) == 5:
                gpus.append(
                    {
                        "index": int(fields[0]),
                        "name": fields[1],
                        "driver_version": fields[2],
                        "memory_mib": int(fields[3]),
                        "compute_capability": fields[4],
                    }
                )

    return {
        "nvidia_smi_found": True,
        "nvidia_smi_path": executable,
        "gpus": gpus,
        "error": None if query["ok"] else query["stderr"] or query["stdout"],
    }


def read_git_head_without_git(repo_root: Path) -> str | None:
    """Resolve a standard .git/HEAD as a fallback when Git is unavailable."""
    git_dir = repo_root / ".git"
    head_path = git_dir / "HEAD"
    if not head_path.is_file():
        return None
    head = head_path.read_text(encoding="utf-8").strip()
    if not head.startswith("ref: "):
        return head or None
    ref_path = git_dir / head.removeprefix("ref: ")
    if ref_path.is_file():
        return ref_path.read_text(encoding="utf-8").strip() or None
    return None


def collect_git(repo_root: Path) -> dict[str, Any]:
    git = shutil.which("git")
    if git is None:
        return {
            "commit_hash": read_git_head_without_git(repo_root),
            "dirty": None,
            "status_short": None,
            "error": "git executable not found; commit read from .git/HEAD when possible",
        }

    common = [git, "-c", f"safe.directory={repo_root}", "-C", str(repo_root)]
    commit = run_command(common + ["rev-parse", "HEAD"])
    status = run_command(common + ["status", "--short"])
    return {
        "commit_hash": commit["stdout"] if commit["ok"] else read_git_head_without_git(repo_root),
        "dirty": bool(status["stdout"]) if status["ok"] else None,
        "status_short": status["stdout"].splitlines() if status["ok"] else None,
        "error": None
        if commit["ok"] and status["ok"]
        else "; ".join(part for part in (commit["stderr"], status["stderr"]) if part),
    }


def set_random_seed(seed: int, torch: Any | None) -> dict[str, Any]:
    random.seed(seed)
    os.environ["PYTHONHASHSEED"] = str(seed)
    torch_seeded = False
    if torch is not None:
        torch.manual_seed(seed)
        if torch.cuda.is_available():
            torch.cuda.manual_seed_all(seed)
        torch_seeded = True
    return {
        "value": seed,
        "python_random_seeded": True,
        "python_hash_seed_for_child_processes": str(seed),
        "torch_seeded": torch_seeded,
    }


def build_report(repo_root: Path, dataset_path: Path, seed: int) -> dict[str, Any]:
    ultralytics_config_dir = Path(
        os.environ.get(
            "YOLO_CONFIG_DIR",
            repo_root
            / "articulation_experiments"
            / "outputs"
            / "ultralytics_config",
        )
    ).expanduser().resolve()
    ultralytics_config_dir.mkdir(parents=True, exist_ok=True)
    os.environ["YOLO_CONFIG_DIR"] = str(ultralytics_config_dir)

    torch_report, torch_module = collect_torch()
    torchvision_report = import_package("torchvision")
    ultralytics_report = import_package("ultralytics")
    ultralytics_report["config_dir"] = str(ultralytics_config_dir)
    gpu_report = collect_nvidia_gpus()

    checks = {
        "python_3_11": sys.version_info[:2] == (3, 11),
        "torch_importable": bool(torch_report["importable"]),
        "torchvision_importable": bool(torchvision_report["importable"]),
        "nvidia_gpu_detected": bool(gpu_report["gpus"]),
        "pytorch_cuda_available": bool(torch_report["cuda_available"]),
        "ultralytics_importable": bool(ultralytics_report["importable"]),
        "dataset_path_exists": dataset_path.exists(),
    }
    blocking_issues = [name for name, passed in checks.items() if not passed]

    return {
        "generated_at_utc": datetime.now(timezone.utc).isoformat(),
        "python": {
            "version": platform.python_version(),
            "version_full": sys.version,
            "executable": sys.executable,
            "implementation": platform.python_implementation(),
        },
        "os": {
            "platform": platform.platform(),
            "system": platform.system(),
            "release": platform.release(),
            "version": platform.version(),
            "machine": platform.machine(),
        },
        "gpu": gpu_report,
        "cuda": {
            "pytorch_cuda_build": torch_report["cuda_build"],
            "available_to_pytorch": torch_report["cuda_available"],
            "cudnn_version": torch_report["cudnn_version"],
        },
        "torch": torch_report,
        "torchvision": torchvision_report,
        "ultralytics": ultralytics_report,
        "random_seed": set_random_seed(seed, torch_module),
        "git": collect_git(repo_root),
        "dataset": {
            "path": str(dataset_path),
            "exists": dataset_path.exists(),
            "is_directory": dataset_path.is_dir(),
        },
        "readiness": {
            "checks": checks,
            "ready_for_yolo_gpu_training": all(checks.values()),
            "blocking_issues": blocking_issues,
        },
    }


def parse_args() -> argparse.Namespace:
    script_path = Path(__file__).resolve()
    default_repo_root = script_path.parents[2]
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--repo-root", type=Path, default=default_repo_root)
    parser.add_argument(
        "--dataset-path",
        type=Path,
        default=default_repo_root.parent / "ds2_dense",
    )
    parser.add_argument("--seed", type=int, default=DEFAULT_SEED)
    parser.add_argument(
        "--output",
        type=Path,
        default=default_repo_root
        / "articulation_experiments"
        / "outputs"
        / "environment_report.json",
    )
    return parser.parse_args()


def main() -> int:
    args = parse_args()
    repo_root = args.repo_root.expanduser().resolve()
    dataset_path = args.dataset_path.expanduser().resolve()
    output_path = args.output.expanduser().resolve()

    report = build_report(repo_root, dataset_path, args.seed)
    output_path.parent.mkdir(parents=True, exist_ok=True)
    output_path.write_text(
        json.dumps(report, ensure_ascii=False, indent=2) + "\n",
        encoding="utf-8",
    )

    print(json.dumps(report, ensure_ascii=False, indent=2))
    print(f"\nEnvironment report written to: {output_path}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
