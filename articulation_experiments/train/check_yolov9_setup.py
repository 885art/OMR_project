"""Validate local files and environment before official YOLOv9 training."""

from __future__ import annotations

import argparse
import importlib
import json
from pathlib import Path

import torch
import yaml


def class_count(dataset_yaml: Path) -> int:
    document = yaml.safe_load(dataset_yaml.read_text(encoding="utf-8"))
    names = document["names"]
    return len(names)


def require_passed_validation(dataset_yaml: Path) -> None:
    report_path = dataset_yaml.parent / "validation_report.json"
    if not report_path.is_file():
        raise FileNotFoundError(f"Missing dataset validation report: {report_path}")
    report = json.loads(report_path.read_text(encoding="utf-8"))
    if report.get("passed") is not True:
        raise RuntimeError(f"Dataset validation did not pass: {report_path}")
    reported_root = Path(report.get("dataset_root", "")).resolve()
    if reported_root != dataset_yaml.parent.resolve():
        raise RuntimeError(
            f"Validation report belongs to another dataset: {reported_root}"
        )


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--yolov9-root", type=Path, required=True)
    parser.add_argument("--weights", type=Path, required=True)
    parser.add_argument("--symbol-data", type=Path, required=True)
    parser.add_argument("--curve-data", type=Path, required=True)
    args = parser.parse_args()
    required = {
        "official train_dual.py": args.yolov9_root / "train_dual.py",
        "official yolov9-s.yaml": args.yolov9_root / "models" / "detect" / "yolov9-s.yaml",
        "pretrained weights": args.weights,
        "40-class dataset": args.symbol_data,
        "2-class curve dataset": args.curve_data,
    }
    missing = [f"{label}: {path}" for label, path in required.items() if not path.is_file()]
    if missing:
        raise FileNotFoundError("Missing YOLOv9 inputs:\n" + "\n".join(missing))
    dependencies = (
        "pandas", "seaborn", "thop", "pycocotools", "git", "tensorboard"
    )
    for package in dependencies:
        importlib.import_module(package)
    if class_count(args.symbol_data) != 40:
        raise RuntimeError("Symbol dataset must contain exactly 40 classes")
    if class_count(args.curve_data) != 2:
        raise RuntimeError("Curve dataset must contain exactly 2 classes")
    require_passed_validation(args.symbol_data)
    require_passed_validation(args.curve_data)
    if not torch.cuda.is_available():
        raise RuntimeError("CUDA is unavailable; YOLOv9 training would run on CPU")
    print(
        "YOLOV9 PREFLIGHT PASSED: "
        f"CUDA={torch.cuda.get_device_name(0)}, "
        "symbol_classes=40, curve_classes=2, datasets=validated"
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
