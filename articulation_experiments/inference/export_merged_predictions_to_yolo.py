"""Export merged full-page predictions as a portable YOLO annotation dataset."""

from __future__ import annotations

import argparse
import json
import shutil
from pathlib import Path

import yaml


def load_names(data_yaml: Path) -> list[str]:
    document = yaml.safe_load(data_yaml.read_text(encoding="utf-8"))
    raw_names = document.get("names")
    if isinstance(raw_names, list):
        return [str(name) for name in raw_names]
    if isinstance(raw_names, dict):
        indexed = {int(key): str(value) for key, value in raw_names.items()}
        expected = list(range(len(indexed)))
        if sorted(indexed) != expected:
            raise ValueError("dataset.yaml names must use contiguous class IDs starting at 0")
        return [indexed[index] for index in expected]
    raise ValueError("dataset.yaml does not contain a supported names list or mapping")


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--input-dir", type=Path, required=True)
    parser.add_argument("--prediction-dir", type=Path, required=True)
    parser.add_argument("--data-yaml", type=Path, required=True)
    parser.add_argument("--output-dir", type=Path, required=True)
    parser.add_argument("--glob", default="*.jpeg")
    args = parser.parse_args()

    input_dir = args.input_dir.expanduser().resolve()
    prediction_dir = args.prediction_dir.expanduser().resolve()
    output_dir = args.output_dir.expanduser().resolve()
    if output_dir.exists() and any(output_dir.iterdir()):
        raise FileExistsError(f"Output directory is not empty: {output_dir}")

    names = load_names(args.data_yaml.expanduser().resolve())
    image_dir = output_dir / "images"
    label_dir = output_dir / "labels"
    image_dir.mkdir(parents=True, exist_ok=True)
    label_dir.mkdir(parents=True, exist_ok=True)

    sources = sorted(path for path in input_dir.glob(args.glob) if path.is_file())
    if not sources:
        raise FileNotFoundError(f"No images matching {args.glob!r} in {input_dir}")

    total_boxes = 0
    clipped_boxes = 0
    page_records = []
    for index, source in enumerate(sources, 1):
        merged_path = prediction_dir / f"{source.stem}.merged.json"
        if not merged_path.is_file():
            raise FileNotFoundError(f"Missing merged prediction: {merged_path}")
        document = json.loads(merged_path.read_text(encoding="utf-8"))
        width = float(document["source_width"])
        height = float(document["source_height"])
        if width <= 0 or height <= 0:
            raise ValueError(f"Invalid source dimensions in {merged_path}")

        lines: list[str] = []
        for prediction in document.get("predictions", []):
            class_id = int(prediction["class_id"])
            if not 0 <= class_id < len(names):
                raise ValueError(f"Invalid class ID {class_id} in {merged_path}")
            raw_box = [float(value) for value in prediction["bbox_xyxy"]]
            x1 = min(max(raw_box[0], 0.0), width)
            y1 = min(max(raw_box[1], 0.0), height)
            x2 = min(max(raw_box[2], 0.0), width)
            y2 = min(max(raw_box[3], 0.0), height)
            if [x1, y1, x2, y2] != raw_box:
                clipped_boxes += 1
            if x2 <= x1 or y2 <= y1:
                raise ValueError(f"Empty bbox after clipping in {merged_path}: {raw_box}")
            center_x = ((x1 + x2) / 2.0) / width
            center_y = ((y1 + y2) / 2.0) / height
            box_width = (x2 - x1) / width
            box_height = (y2 - y1) / height
            lines.append(
                f"{class_id} {center_x:.8f} {center_y:.8f} {box_width:.8f} {box_height:.8f}"
            )

        shutil.copy2(source, image_dir / source.name)
        (label_dir / f"{source.stem}.txt").write_text(
            "\n".join(lines) + ("\n" if lines else ""), encoding="utf-8"
        )
        total_boxes += len(lines)
        page_records.append({"image": source.name, "label": f"{source.stem}.txt", "box_count": len(lines)})
        print(f"YOLO {index}/{len(sources)} {source.name}: {len(lines)} boxes", flush=True)

    yaml_lines = ["path: .", "train: images", "val: images", "names:"]
    yaml_lines.extend(f"  {index}: {name}" for index, name in enumerate(names))
    (output_dir / "dataset.yaml").write_text("\n".join(yaml_lines) + "\n", encoding="utf-8")
    (output_dir / "classes.txt").write_text("\n".join(names) + "\n", encoding="utf-8")
    manifest = {
        "schema_version": 1,
        "format": "YOLO detection labels: class_id x_center y_center width height",
        "coordinates": "normalized to 0..1 relative to the original full-page image",
        "source": "model-generated merged predictions, not human-verified ground truth",
        "page_count": len(sources),
        "class_count": len(names),
        "box_count": total_boxes,
        "clipped_box_count": clipped_boxes,
        "pages": page_records,
    }
    (output_dir / "export_manifest.json").write_text(
        json.dumps(manifest, ensure_ascii=False, indent=2) + "\n", encoding="utf-8"
    )
    (output_dir / "README.md").write_text(
        "# BPSD Complete all136 YOLO pre-annotations\n\n"
        "- `images/`: original full-page score images\n"
        "- `labels/`: one YOLO `.txt` label file per image\n"
        "- Label line: `class_id x_center y_center width height`\n"
        "- Coordinates are normalized to the original image size\n"
        "- `classes.txt` and `dataset.yaml`: the 136-class order\n\n"
        "These are model-generated pre-annotations and must not be described as "
        "human-verified ground truth. Standard YOLO labels do not preserve confidence; "
        "the original `*.merged.json` files remain the source for confidence values.\n",
        encoding="utf-8",
    )
    print(
        f"EXPORTED pages={len(sources)} classes={len(names)} boxes={total_boxes} "
        f"clipped={clipped_boxes} output={output_dir}",
        flush=True,
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
