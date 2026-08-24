"""Run official YOLOv9 validation and save machine-readable per-class AP."""

from __future__ import annotations

import argparse
import json
import os
import sys
from pathlib import Path
from typing import Any, Sequence

import torch
import yaml

try:
    from .yolov9_compat_launcher import patch_pillow_font_getsize
except ImportError:
    from yolov9_compat_launcher import patch_pillow_font_getsize


def class_names(data_path: Path) -> list[str]:
    with data_path.open("r", encoding="utf-8") as handle:
        document = yaml.safe_load(handle)
    raw = document.get("names")
    if isinstance(raw, dict):
        return [str(raw[index]) for index in sorted(raw, key=int)]
    if isinstance(raw, list):
        return [str(value) for value in raw]
    raise ValueError(f"Missing class names in {data_path}")


def report_document(
    names: list[str],
    aggregate: Sequence[float],
    maps: Sequence[float],
    timing_ms_per_image: Sequence[float],
    arguments: dict[str, Any],
) -> dict[str, Any]:
    if len(maps) != len(names):
        raise ValueError(f"Expected {len(names)} class AP values, received {len(maps)}")
    labels = (
        "precision",
        "recall",
        "map50",
        "map50_95",
        "val_box_loss",
        "val_obj_loss",
        "val_cls_loss",
    )
    return {
        "schema_version": 1,
        "kind": "yolov9_complete_all136_validation",
        "arguments": arguments,
        "aggregate": {
            key: float(value) for key, value in zip(labels, aggregate)
        },
        "timing_ms_per_image": {
            key: float(value)
            for key, value in zip(("preprocess", "inference", "nms"), timing_ms_per_image)
        },
        "per_class": [
            {
                "class_id": class_id,
                "name": name,
                "map50_95": float(maps[class_id]),
            }
            for class_id, name in enumerate(names)
        ],
    }


def parse_args(argv: Sequence[str] | None = None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--yolov9-root", type=Path, required=True)
    parser.add_argument("--data", type=Path, required=True)
    parser.add_argument("--weights", type=Path, required=True)
    parser.add_argument("--output-dir", type=Path, required=True)
    parser.add_argument("--imgsz", type=int, default=1024)
    parser.add_argument("--batch-size", type=int, default=24)
    parser.add_argument("--device", default="0")
    parser.add_argument("--workers", type=int, default=16)
    args = parser.parse_args(argv)
    if args.imgsz < 1 or args.batch_size < 1 or args.workers < 0:
        parser.error("imgsz/batch-size must be positive and workers non-negative")
    return args


def main(argv: Sequence[str] | None = None) -> int:
    args = parse_args(argv)
    root = args.yolov9_root.expanduser().resolve()
    data = args.data.expanduser().resolve()
    weights = args.weights.expanduser().resolve()
    output_dir = args.output_dir.expanduser().resolve()
    for path in (root / "val_dual.py", data, weights):
        if not path.is_file():
            raise FileNotFoundError(path)
    if output_dir.exists():
        raise FileExistsError(output_dir)

    config_dir = output_dir.parent / "_yolov9_config"
    config_dir.mkdir(parents=True, exist_ok=True)
    os.environ["YOLOV5_CONFIG_DIR"] = str(config_dir)
    sys.path.insert(0, str(root))
    original_load = torch.load

    def compatible_load(*load_args, **load_kwargs):
        load_kwargs.setdefault("weights_only", False)
        return original_load(*load_args, **load_kwargs)

    torch.load = compatible_load
    patch_pillow_font_getsize()
    from val_dual import run

    aggregate, maps, timing = run(
        data=str(data),
        weights=str(weights),
        batch_size=args.batch_size,
        imgsz=args.imgsz,
        task="val",
        device=args.device,
        workers=args.workers,
        verbose=True,
        project=output_dir.parent,
        name=output_dir.name,
        exist_ok=True,
        half=True,
        plots=False,
    )
    document = report_document(
        class_names(data),
        aggregate,
        maps,
        timing,
        {
            "yolov9_root": str(root),
            "data": str(data),
            "weights": str(weights),
            "output_dir": str(output_dir),
            "imgsz": args.imgsz,
            "batch_size": args.batch_size,
            "device": args.device,
            "workers": args.workers,
        },
    )
    destination = output_dir / "per_class_metrics.json"
    destination.write_text(
        json.dumps(document, ensure_ascii=False, indent=2) + "\n",
        encoding="utf-8",
    )
    print(f"WROTE {destination}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
