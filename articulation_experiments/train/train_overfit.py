"""Prepare and train a deterministic YOLO11n articulation overfit sanity check."""

from __future__ import annotations

import argparse
import csv
import json
import os
import random
import shutil
import subprocess
from collections import Counter, defaultdict
from contextlib import contextmanager
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Iterator

import yaml
from PIL import Image, ImageDraw, ImageFont


CLASS_COLORS = {
    0: "#e41a1c",
    1: "#ff7f00",
    2: "#377eb8",
    3: "#4daf4a",
    4: "#984ea3",
    5: "#a65628",
}


def load_json(path: Path) -> dict[str, Any]:
    with path.open("r", encoding="utf-8") as handle:
        return json.load(handle)


def write_json(path: Path, value: Any) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_suffix(path.suffix + ".tmp")
    temporary.write_text(
        json.dumps(value, ensure_ascii=False, indent=2) + "\n",
        encoding="utf-8",
    )
    temporary.replace(path)


@contextmanager
def working_directory(path: Path) -> Iterator[None]:
    previous = Path.cwd()
    path.mkdir(parents=True, exist_ok=True)
    os.chdir(path)
    try:
        yield
    finally:
        os.chdir(previous)


def parse_label_lines(path: Path) -> list[tuple[int, float, float, float, float]]:
    labels = []
    for line_number, line in enumerate(path.read_text(encoding="utf-8").splitlines(), start=1):
        if not line.strip():
            continue
        fields = line.split()
        if len(fields) != 5:
            raise ValueError(f"{path}:{line_number} has {len(fields)} fields")
        class_id = int(fields[0])
        values = tuple(float(value) for value in fields[1:])
        if not 0 <= class_id <= 5 or not all(0.0 <= value <= 1.0 for value in values):
            raise ValueError(f"Invalid label at {path}:{line_number}: {line}")
        labels.append((class_id, *values))
    return labels


def apply_diagnostic_minimum_bbox_height(
    path: Path,
    class_ids: set[int],
    minimum_height_pixels: float,
    image_height: int = 1024,
) -> list[dict[str, float | int]]:
    """Expand selected tiny boxes in a disposable subset for diagnosis only."""
    labels = parse_label_lines(path)
    minimum_normalized = minimum_height_pixels / image_height
    adjusted: list[dict[str, float | int]] = []
    output_lines: list[str] = []
    for class_id, cx, cy, width, height in labels:
        original_height = height
        if class_id in class_ids and height < minimum_normalized:
            height = minimum_normalized
            cy = min(max(cy, height / 2.0), 1.0 - height / 2.0)
            adjusted.append(
                {
                    "class_id": class_id,
                    "original_height_pixels": original_height * image_height,
                    "adjusted_height_pixels": height * image_height,
                }
            )
        output_lines.append(
            f"{class_id} {cx:.10f} {cy:.10f} {width:.10f} {height:.10f}"
        )
    path.write_text(
        "\n".join(output_lines) + ("\n" if output_lines else ""), encoding="utf-8"
    )
    return adjusted


def select_subset_records(
    manifest_tiles: list[dict[str, Any]],
    positive_count: int,
    minimum_tiles_per_class: int,
    negative_count: int,
    class_count: int,
    seed: int,
) -> tuple[list[dict[str, Any]], dict[str, list[int]]]:
    positives = [record for record in manifest_tiles if not record["is_negative"]]
    negatives = [record for record in manifest_tiles if record["is_negative"]]
    if not 16 <= positive_count <= 32:
        raise ValueError("Overfit positive tile count must be between 16 and 32")
    if positive_count < class_count * minimum_tiles_per_class:
        raise ValueError("positive_tiles is too small for minimum_tiles_per_class")

    rng = random.Random(seed)
    random_rank = {record["tile_filename"]: rng.random() for record in positives}
    selected: list[dict[str, Any]] = []
    selected_filenames: set[str] = set()
    selected_annotation_ids: set[str] = set()
    reasons: dict[str, list[int]] = defaultdict(list)

    for class_id in range(class_count):
        candidates = [
            record
            for record in positives
            if any(
                int(annotation["yolo_class_id"]) == class_id
                for annotation in record["annotations"]
            )
        ]
        candidates.sort(
            key=lambda record: (
                len(set(record["annotation_ids"]) & selected_annotation_ids),
                bool(record["clipped_annotation_ids"]),
                bool(record["padding"]["right"] or record["padding"]["bottom"]),
                abs(
                    sum(
                        int(annotation["yolo_class_id"] == class_id)
                        for annotation in record["annotations"]
                    )
                    - 3
                ),
                len(record["annotations"]),
                random_rank[record["tile_filename"]],
            )
        )
        added = 0
        for record in candidates:
            if record["tile_filename"] in selected_filenames:
                reasons[record["tile_filename"]].append(class_id)
                continue
            selected.append(record)
            selected_filenames.add(record["tile_filename"])
            selected_annotation_ids.update(record["annotation_ids"])
            reasons[record["tile_filename"]].append(class_id)
            added += 1
            if added == minimum_tiles_per_class:
                break
        if added < minimum_tiles_per_class:
            raise RuntimeError(
                f"Could only select {added} distinct tiles for class {class_id}"
            )

    if len(selected) > positive_count:
        raise RuntimeError(
            f"Required per-class coverage selected {len(selected)} tiles, above target {positive_count}"
        )
    remaining = [
        record for record in positives if record["tile_filename"] not in selected_filenames
    ]
    remaining.sort(
        key=lambda record: (
            len(set(record["annotation_ids"]) & selected_annotation_ids),
            bool(record["clipped_annotation_ids"]),
            bool(record["padding"]["right"] or record["padding"]["bottom"]),
            len(record["annotations"]),
            random_rank[record["tile_filename"]],
        )
    )
    for record in remaining[: positive_count - len(selected)]:
        selected.append(record)
        selected_filenames.add(record["tile_filename"])
        selected_annotation_ids.update(record["annotation_ids"])
        reasons[record["tile_filename"]].append(-1)

    negative_rng = random.Random(seed + 1)
    ordered_negatives = sorted(
        negatives,
        key=lambda record: (
            bool(record["padding"]["right"] or record["padding"]["bottom"]),
            record["source_image"],
            record["tile_offset"]["y"],
            record["tile_offset"]["x"],
        ),
    )
    unpadded = [
        record
        for record in ordered_negatives
        if not record["padding"]["right"] and not record["padding"]["bottom"]
    ]
    negative_pool = unpadded if len(unpadded) >= negative_count else ordered_negatives
    selected_negatives = negative_rng.sample(negative_pool, negative_count)
    for record in selected_negatives:
        reasons[record["tile_filename"]].append(-2)

    selected.sort(key=lambda record: record["tile_filename"])
    selected_negatives.sort(key=lambda record: record["tile_filename"])
    return selected + selected_negatives, dict(reasons)


def prepare_subset(
    full_dataset_root: Path,
    subset_dir: Path,
    config: dict[str, Any],
    overwrite: bool,
) -> dict[str, Any]:
    manifest_path = subset_dir / "manifest.json"
    if manifest_path.exists() and not overwrite:
        manifest = load_json(manifest_path)
        validate_subset(subset_dir, manifest)
        print(f"REUSING VALID SUBSET {subset_dir}", flush=True)
        return manifest

    owned_paths = [
        subset_dir / "images" / "train",
        subset_dir / "labels" / "train",
        subset_dir / "labels" / "train.cache",
        subset_dir / "dataset.yaml",
        subset_dir / "manifest.json",
        subset_dir / "statistics.json",
    ]
    existing = [path for path in owned_paths if path.exists()]
    if existing and not overwrite:
        raise FileExistsError(f"Subset outputs already exist: {existing}")
    if overwrite:
        for path in existing:
            if path.is_dir():
                if path.parent.name not in {"images", "labels"}:
                    raise RuntimeError(f"Refusing to remove unexpected directory: {path}")
                shutil.rmtree(path)
            else:
                path.unlink()

    image_dir = subset_dir / "images" / "train"
    label_dir = subset_dir / "labels" / "train"
    image_dir.mkdir(parents=True, exist_ok=True)
    label_dir.mkdir(parents=True, exist_ok=True)

    full_statistics = load_json(full_dataset_root / "statistics.json")
    class_stats = full_statistics["totals"]["class_statistics"]
    class_names = [class_stats[str(index)]["name"] for index in range(len(class_stats))]
    source_manifest = load_json(full_dataset_root / "manifests" / "train_tiles.json")
    subset_config = config["subset"]
    selected, reasons = select_subset_records(
        source_manifest["tiles"],
        positive_count=int(subset_config["positive_tiles"]),
        minimum_tiles_per_class=int(subset_config["minimum_tiles_per_class"]),
        negative_count=int(subset_config["negative_tiles"]),
        class_count=len(class_names),
        seed=int(config["seed"]),
    )

    items: list[dict[str, Any]] = []
    class_instances: Counter[int] = Counter()
    class_tiles: Counter[int] = Counter()
    diagnostic_transform = subset_config.get("diagnostic_bbox_transform")
    transformed_box_count = 0
    for record in selected:
        source_image = full_dataset_root / record["image_path"]
        source_label = full_dataset_root / record["label_path"]
        destination_image = image_dir / record["tile_filename"]
        destination_label = label_dir / Path(record["label_path"]).name
        shutil.copy2(source_image, destination_image)
        shutil.copy2(source_label, destination_label)
        adjustments: list[dict[str, float | int]] = []
        if diagnostic_transform and diagnostic_transform.get("enabled", False):
            adjustments = apply_diagnostic_minimum_bbox_height(
                destination_label,
                class_ids={int(value) for value in diagnostic_transform["class_ids"]},
                minimum_height_pixels=float(
                    diagnostic_transform["minimum_height_pixels"]
                ),
            )
            transformed_box_count += len(adjustments)
        labels = parse_label_lines(destination_label)
        classes_in_tile = {label[0] for label in labels}
        class_instances.update(label[0] for label in labels)
        class_tiles.update(classes_in_tile)
        items.append(
            {
                "tile_filename": record["tile_filename"],
                "is_negative": record["is_negative"],
                "source_image": record["source_image"],
                "source_image_id": record["source_image_id"],
                "tile_offset": record["tile_offset"],
                "annotation_ids": record["annotation_ids"],
                "classes_present": sorted(classes_in_tile),
                "selected_for_classes": reasons[record["tile_filename"]],
                "full_dataset_image_path": record["image_path"],
                "full_dataset_label_path": record["label_path"],
                "subset_image_path": destination_image.relative_to(subset_dir).as_posix(),
                "subset_label_path": destination_label.relative_to(subset_dir).as_posix(),
                "diagnostic_bbox_adjustments": adjustments,
            }
        )

    yaml_lines = [
        f"path: {subset_dir.resolve().as_posix()}",
        "train: images/train",
        "val: images/train",
        "names:",
    ]
    yaml_lines.extend(f"  {index}: {name}" for index, name in enumerate(class_names))
    (subset_dir / "dataset.yaml").write_text(
        "\n".join(yaml_lines) + "\n", encoding="utf-8"
    )
    manifest = {
        "schema_version": 1,
        "generated_at_utc": datetime.now(timezone.utc).isoformat(),
        "purpose": "YOLO11n overfit sanity check; train and val intentionally use the same tiles",
        "seed": int(config["seed"]),
        "full_dataset_root": str(full_dataset_root),
        "subset_root": str(subset_dir),
        "selection_configuration": subset_config,
        "diagnostic_bbox_transform": diagnostic_transform,
        "transformed_box_count": transformed_box_count,
        "positive_tile_count": sum(not item["is_negative"] for item in items),
        "negative_tile_count": sum(item["is_negative"] for item in items),
        "items": items,
    }
    statistics = {
        "tile_count": len(items),
        "positive_tile_count": manifest["positive_tile_count"],
        "negative_tile_count": manifest["negative_tile_count"],
        "instance_count": sum(class_instances.values()),
        "transformed_box_count": transformed_box_count,
        "class_statistics": {
            str(class_id): {
                "name": class_names[class_id],
                "instance_count": class_instances[class_id],
                "tile_count": class_tiles[class_id],
            }
            for class_id in range(len(class_names))
        },
    }
    write_json(manifest_path, manifest)
    write_json(subset_dir / "statistics.json", statistics)
    validate_subset(subset_dir, manifest)
    print(
        f"WROTE OVERFIT SUBSET: {manifest['positive_tile_count']} positive + "
        f"{manifest['negative_tile_count']} negative tiles, {statistics['instance_count']} instances",
        flush=True,
    )
    return manifest


def validate_subset(subset_dir: Path, manifest: dict[str, Any]) -> None:
    image_dir = subset_dir / "images" / "train"
    label_dir = subset_dir / "labels" / "train"
    image_stems = {path.stem for path in image_dir.glob("*.png")}
    label_stems = {path.stem for path in label_dir.glob("*.txt")}
    if image_stems != label_stems:
        raise RuntimeError("Overfit subset image/label mismatch")
    if len(image_stems) != len(manifest["items"]):
        raise RuntimeError("Overfit subset manifest count mismatch")
    class_tiles: Counter[int] = Counter()
    positive = 0
    negative = 0
    for item in manifest["items"]:
        image_path = subset_dir / item["subset_image_path"]
        label_path = subset_dir / item["subset_label_path"]
        if not image_path.is_file() or not label_path.is_file():
            raise RuntimeError(f"Missing subset file for {item['tile_filename']}")
        with Image.open(image_path) as image:
            if image.size != (1024, 1024) or image.mode != "RGB":
                raise RuntimeError(f"Invalid subset image: {image_path}")
        labels = parse_label_lines(label_path)
        if item["is_negative"] != (len(labels) == 0):
            raise RuntimeError(f"Negative/empty-label mismatch: {label_path}")
        if item["is_negative"]:
            negative += 1
        else:
            positive += 1
        class_tiles.update({label[0] for label in labels})
    if positive != int(manifest["selection_configuration"]["positive_tiles"]):
        raise RuntimeError("Positive subset count mismatch")
    if negative != int(manifest["selection_configuration"]["negative_tiles"]):
        raise RuntimeError("Negative subset count mismatch")
    minimum = int(manifest["selection_configuration"]["minimum_tiles_per_class"])
    missing = [class_id for class_id in range(6) if class_tiles[class_id] < minimum]
    if missing:
        raise RuntimeError(f"Classes below minimum tile coverage: {missing}")


def git_snapshot(repo_root: Path) -> dict[str, Any]:
    common = [
        "git",
        "-c",
        f"safe.directory={repo_root}",
        "-C",
        str(repo_root),
    ]
    commit = subprocess.run(
        common + ["rev-parse", "HEAD"], capture_output=True, text=True, check=False
    )
    status = subprocess.run(
        common + ["status", "--short"], capture_output=True, text=True, check=False
    )
    return {
        "commit_hash": commit.stdout.strip() if commit.returncode == 0 else None,
        "status_short": status.stdout.splitlines() if status.returncode == 0 else None,
    }


def load_font(size: int) -> ImageFont.ImageFont:
    for name in ("arial.ttf", "DejaVuSans.ttf"):
        try:
            return ImageFont.truetype(name, size=size)
        except OSError:
            continue
    return ImageFont.load_default()


def draw_box(
    draw: ImageDraw.ImageDraw,
    box: list[float],
    color: str,
    label: str,
    offset_x: int,
    offset_y: int,
    font: ImageFont.ImageFont,
) -> None:
    x1, y1, x2, y2 = box
    shifted = (x1 + offset_x, y1 + offset_y, x2 + offset_x, y2 + offset_y)
    draw.rectangle(shifted, outline=color, width=3)
    text_x = int(x1 + offset_x)
    text_y = max(offset_y, int(y1 + offset_y) - 15)
    text_box = draw.textbbox((text_x, text_y), label, font=font)
    draw.rectangle(
        (text_box[0] - 1, text_box[1] - 1, text_box[2] + 1, text_box[3] + 1),
        fill="white",
    )
    draw.text((text_x, text_y), label, fill=color, font=font)


def yolo_to_xyxy(label: tuple[int, float, float, float, float], size: int) -> list[float]:
    _, cx, cy, width, height = label
    return [
        (cx - width / 2.0) * size,
        (cy - height / 2.0) * size,
        (cx + width / 2.0) * size,
        (cy + height / 2.0) * size,
    ]


def create_comparisons(
    model: Any,
    subset_dir: Path,
    run_dir: Path,
    class_names: list[str],
    count: int,
    imgsz: int,
    device: int | str,
    confidence: float,
) -> list[dict[str, Any]]:
    manifest = load_json(subset_dir / "manifest.json")
    positives = [item for item in manifest["items"] if not item["is_negative"]]
    selected = positives[:count]
    sources = [str(subset_dir / item["subset_image_path"]) for item in selected]
    predictions = model.predict(
        source=sources,
        imgsz=imgsz,
        conf=confidence,
        iou=0.5,
        device=device,
        verbose=False,
    )
    output_dir = run_dir / "overfit_comparisons"
    output_dir.mkdir(parents=True, exist_ok=True)
    font = load_font(12)
    items = []
    for index, (item, prediction) in enumerate(zip(selected, predictions), start=1):
        image_path = subset_dir / item["subset_image_path"]
        label_path = subset_dir / item["subset_label_path"]
        with Image.open(image_path) as opened:
            source = opened.convert("RGB")
        banner = 30
        canvas = Image.new("RGB", (source.width * 2, source.height + banner), "white")
        canvas.paste(source, (0, banner))
        canvas.paste(source, (source.width, banner))
        draw = ImageDraw.Draw(canvas)
        draw.rectangle((0, 0, source.width, banner), fill="#1b5e20")
        draw.rectangle((source.width, 0, source.width * 2, banner), fill="#b71c1c")
        draw.text((8, 8), "GROUND TRUTH", fill="white", font=font)
        draw.text((source.width + 8, 8), "PREDICTIONS", fill="white", font=font)
        gt_labels = parse_label_lines(label_path)
        for label in gt_labels:
            class_id = label[0]
            draw_box(
                draw,
                yolo_to_xyxy(label, source.width),
                CLASS_COLORS[class_id],
                f"{class_id}:{class_names[class_id]}",
                0,
                banner,
                font,
            )
        prediction_count = 0
        if prediction.boxes is not None:
            xyxy = prediction.boxes.xyxy.detach().cpu().tolist()
            classes = prediction.boxes.cls.detach().cpu().tolist()
            confidences = prediction.boxes.conf.detach().cpu().tolist()
            prediction_count = len(xyxy)
            for box, class_value, conf_value in zip(xyxy, classes, confidences):
                class_id = int(class_value)
                draw_box(
                    draw,
                    box,
                    CLASS_COLORS.get(class_id, "#000000"),
                    f"{class_id}:{class_names[class_id]} {conf_value:.2f}",
                    source.width,
                    banner,
                    font,
                )
        destination = output_dir / f"{index:02d}_{Path(item['tile_filename']).stem}.png"
        canvas.save(destination, format="PNG", compress_level=6)
        items.append(
            {
                "tile_filename": item["tile_filename"],
                "ground_truth_count": len(gt_labels),
                "prediction_count_at_threshold": prediction_count,
                "confidence_threshold": confidence,
                "output": str(destination),
            }
        )
    write_json(output_dir / "manifest.json", {"items": items})
    return items


def summarize_csv(path: Path) -> dict[str, Any]:
    if not path.is_file():
        return {"row_count": 0, "first": None, "last": None, "minimum_losses": {}}
    with path.open("r", encoding="utf-8-sig", newline="") as handle:
        rows = [
            {key.strip(): value for key, value in row.items()}
            for row in csv.DictReader(handle)
        ]
    loss_keys = [
        key for key in rows[0] if key.startswith("train/") and key.endswith("_loss")
    ] if rows else []
    return {
        "row_count": len(rows),
        "first": rows[0] if rows else None,
        "last": rows[-1] if rows else None,
        "minimum_losses": {
            key: min(float(row[key]) for row in rows if row.get(key)) for key in loss_keys
        },
    }


def train_overfit(
    repo_root: Path,
    subset_dir: Path,
    runs_dir: Path,
    config_path: Path,
    config: dict[str, Any],
    run_name: str,
    overrides: dict[str, Any],
) -> dict[str, Any]:
    os.environ["YOLO_CONFIG_DIR"] = str(
        repo_root / "articulation_experiments" / "outputs" / "ultralytics_config"
    )
    from ultralytics import YOLO

    training = dict(config["training"])
    training.update({key: value for key, value in overrides.items() if value is not None})
    model_name = str(training.pop("model"))
    training.pop("pretrained", None)
    run_dir = runs_dir / run_name
    if run_dir.exists():
        raise FileExistsError(
            f"Run directory already exists: {run_dir}. Choose a different --run-name."
        )

    pretrained_dir = repo_root / "articulation_experiments" / "outputs" / "pretrained"
    pretrained_path = pretrained_dir / model_name
    if pretrained_path.is_file():
        model = YOLO(str(pretrained_path))
    else:
        with working_directory(pretrained_dir):
            model = YOLO(model_name)
        if not pretrained_path.is_file():
            raise FileNotFoundError(f"Expected downloaded model at {pretrained_path}")

    training.update(config["augmentation"])
    model.train(
        data=str(subset_dir / "dataset.yaml"),
        project=str(runs_dir),
        name=run_name,
        exist_ok=False,
        seed=int(config["seed"]),
        **training,
    )
    best_path = run_dir / "weights" / "best.pt"
    last_path = run_dir / "weights" / "last.pt"
    if not best_path.is_file() or not last_path.is_file():
        raise FileNotFoundError("Training did not produce best.pt and last.pt")

    best_model = YOLO(str(best_path))
    metrics = best_model.val(
        data=str(subset_dir / "dataset.yaml"),
        imgsz=int(training["imgsz"]),
        batch=int(training["batch"]),
        device=training["device"],
        workers=int(training["workers"]),
        plots=True,
        project=str(run_dir),
        name="final_validation",
        exist_ok=True,
        verbose=True,
    )
    metrics_dict = {
        key: float(value) if hasattr(value, "__float__") else value
        for key, value in metrics.results_dict.items()
    }
    class_names = [
        value["name"]
        for _, value in sorted(
            load_json(subset_dir / "statistics.json")["class_statistics"].items(),
            key=lambda item: int(item[0]),
        )
    ]
    comparisons = create_comparisons(
        best_model,
        subset_dir,
        run_dir,
        class_names,
        count=int(config["output"]["comparison_images"]),
        imgsz=int(training["imgsz"]),
        device=training["device"],
        confidence=float(config["output"]["prediction_confidence"]),
    )

    criteria = config["overfit_criteria"]
    precision = metrics_dict.get("metrics/precision(B)", 0.0)
    recall = metrics_dict.get("metrics/recall(B)", 0.0)
    map50 = metrics_dict.get("metrics/mAP50(B)", 0.0)
    criteria_results = {
        "precision": {
            "value": precision,
            "threshold": float(criteria["precision"]),
            "passed": precision >= float(criteria["precision"]),
        },
        "recall": {
            "value": recall,
            "threshold": float(criteria["recall"]),
            "passed": recall >= float(criteria["recall"]),
        },
        "map50": {
            "value": map50,
            "threshold": float(criteria["map50"]),
            "passed": map50 >= float(criteria["map50"]),
        },
    }
    summary = {
        "schema_version": 1,
        "generated_at_utc": datetime.now(timezone.utc).isoformat(),
        "overfit_passed": all(value["passed"] for value in criteria_results.values()),
        "criteria": criteria_results,
        "metrics": metrics_dict,
        "training_csv": summarize_csv(run_dir / "results.csv"),
        "configuration": {"training": training, "seed": int(config["seed"])},
        "paths": {
            "run_dir": str(run_dir),
            "best_weights": str(best_path),
            "last_weights": str(last_path),
            "subset": str(subset_dir),
            "pretrained_weights": str(pretrained_path),
        },
        "git": git_snapshot(repo_root),
        "comparison_count": len(comparisons),
    }
    write_json(run_dir / "overfit_summary.json", summary)
    shutil.copy2(config_path, run_dir / "overfit_config.yaml")
    shutil.copy2(subset_dir / "manifest.json", run_dir / "subset_manifest.json")
    shutil.copy2(subset_dir / "statistics.json", run_dir / "subset_statistics.json")
    environment_report = (
        repo_root / "articulation_experiments" / "outputs" / "environment_report.json"
    )
    if environment_report.is_file():
        shutil.copy2(environment_report, run_dir / "environment_report.json")
    print(f"OVERFIT PASSED={summary['overfit_passed']}")
    print(json.dumps(criteria_results, indent=2))
    print(f"SUMMARY {run_dir / 'overfit_summary.json'}")
    return summary


def parse_args() -> argparse.Namespace:
    script_path = Path(__file__).resolve()
    repo_root = script_path.parents[2]
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--config",
        type=Path,
        default=repo_root / "articulation_experiments" / "configs" / "overfit.yaml",
    )
    parser.add_argument(
        "--full-dataset-root",
        type=Path,
        default=repo_root / "articulation_experiments" / "outputs" / "yolo_dataset",
    )
    parser.add_argument(
        "--subset-dir",
        type=Path,
        default=repo_root / "articulation_experiments" / "outputs" / "overfit_subset",
    )
    parser.add_argument(
        "--runs-dir",
        type=Path,
        default=repo_root / "articulation_experiments" / "outputs" / "runs",
    )
    parser.add_argument("--run-name")
    parser.add_argument("--prepare-only", action="store_true")
    parser.add_argument("--overwrite-subset", action="store_true")
    parser.add_argument("--epochs", type=int)
    parser.add_argument("--batch", type=int)
    parser.add_argument("--device")
    return parser.parse_args()


def main() -> int:
    args = parse_args()
    script_path = Path(__file__).resolve()
    repo_root = script_path.parents[2]
    config_path = args.config.expanduser().resolve()
    config = yaml.safe_load(config_path.read_text(encoding="utf-8"))
    full_dataset_root = args.full_dataset_root.expanduser().resolve()
    subset_dir = args.subset_dir.expanduser().resolve()
    runs_dir = args.runs_dir.expanduser().resolve()
    manifest = prepare_subset(
        full_dataset_root,
        subset_dir,
        config,
        args.overwrite_subset,
    )
    print(
        f"SUBSET VALID: {manifest['positive_tile_count']} positive, "
        f"{manifest['negative_tile_count']} negative",
        flush=True,
    )
    if args.prepare_only:
        return 0
    run_name = args.run_name or str(config["output"]["run_name"])
    overrides = {
        "epochs": args.epochs,
        "batch": args.batch,
        "device": args.device,
    }
    train_overfit(
        repo_root,
        subset_dir,
        runs_dir,
        config_path,
        config,
        run_name,
        overrides,
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
