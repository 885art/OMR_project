"""Train and evaluate the reproducible YOLO11n articulation baseline."""

from __future__ import annotations

import argparse
import csv
import hashlib
import json
import os
import shutil
import subprocess
import sys
import time
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

import yaml


def load_json(path: Path) -> dict[str, Any]:
    with path.open("r", encoding="utf-8") as handle:
        return json.load(handle)


def write_json(path: Path, value: Any) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_suffix(path.suffix + ".tmp")
    temporary.write_text(
        json.dumps(value, ensure_ascii=False, indent=2) + "\n", encoding="utf-8"
    )
    temporary.replace(path)


def sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for block in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def git_snapshot(repo_root: Path) -> dict[str, Any]:
    common = ["git", "-c", f"safe.directory={repo_root}", "-C", str(repo_root)]
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


def summarize_csv(path: Path) -> dict[str, Any]:
    with path.open("r", encoding="utf-8-sig", newline="") as handle:
        rows = [
            {key.strip(): value for key, value in row.items()}
            for row in csv.DictReader(handle)
        ]
    if not rows:
        return {"row_count": 0}
    loss_keys = [key for key in rows[0] if key.startswith("train/") and key.endswith("_loss")]
    metric_key = "metrics/mAP50-95(B)"
    best_row = max(rows, key=lambda row: float(row.get(metric_key, 0.0)))
    return {
        "row_count": len(rows),
        "first": rows[0],
        "last": rows[-1],
        "minimum_training_losses": {
            key: min(float(row[key]) for row in rows if row.get(key)) for key in loss_keys
        },
        "best_epoch": int(float(best_row["epoch"])) + 1,
        "best_epoch_metrics": {
            key: float(value)
            for key, value in best_row.items()
            if key.startswith("metrics/") and value
        },
    }


def validate_dataset(
    repo_root: Path, dataset_root: Path, config: dict[str, Any]
) -> dict[str, Any]:
    validator = repo_root / "articulation_experiments" / "dataset" / "validate_yolo_dataset.py"
    report_path = dataset_root / "validation_report.json"
    statistics = load_json(dataset_root / "statistics.json")
    mapping_path = Path(statistics["class_mapping"]).expanduser().resolve()
    command = [
        sys.executable,
        str(validator),
        "--dataset-root",
        str(dataset_root),
        "--class-mapping",
        str(mapping_path),
        "--output",
        str(report_path),
    ]
    completed = subprocess.run(command, cwd=repo_root, check=False)
    if completed.returncode != 0:
        raise RuntimeError(f"Dataset validation failed with exit code {completed.returncode}")
    report = load_json(report_path)
    expected = config["dataset"]
    if not report.get("passed"):
        raise RuntimeError("Dataset validation report did not pass")
    if statistics["splits"]["train"]["tile_count"] != int(expected["expected_train_tiles"]):
        raise RuntimeError("Unexpected train tile count")
    if statistics["splits"]["val"]["tile_count"] != int(expected["expected_val_tiles"]):
        raise RuntimeError("Unexpected validation tile count")
    actual_minimum = float(
        statistics["configuration"].get("minimum_tenuto_bbox_height_pixels", 0.0)
    )
    if actual_minimum != float(expected["minimum_tenuto_bbox_height_pixels"]):
        raise RuntimeError(
            f"Tenuto bbox policy mismatch: {actual_minimum} != "
            f"{expected['minimum_tenuto_bbox_height_pixels']}"
        )
    names = yaml.safe_load((dataset_root / "dataset.yaml").read_text(encoding="utf-8"))["names"]
    if len(names) != int(expected["expected_classes"]):
        raise RuntimeError("Unexpected class count in dataset.yaml")
    return report


def numeric_list(value: Any) -> list[float]:
    if hasattr(value, "tolist"):
        value = value.tolist()
    return [float(item) for item in value]


def collect_metrics(metrics: Any, class_names: dict[int, str]) -> dict[str, Any]:
    aggregate = {
        key: float(value) if hasattr(value, "__float__") else value
        for key, value in metrics.results_dict.items()
    }
    precision = numeric_list(metrics.box.p)
    recall = numeric_list(metrics.box.r)
    ap50 = numeric_list(metrics.box.ap50)
    map50_95 = numeric_list(metrics.box.maps)
    per_class = {}
    for class_id, name in class_names.items():
        p = precision[class_id]
        r = recall[class_id]
        per_class[str(class_id)] = {
            "name": name,
            "precision": p,
            "recall": r,
            "f1": 2.0 * p * r / (p + r) if p + r else 0.0,
            "ap50": ap50[class_id],
            "map50_95": map50_95[class_id],
        }
    return {"aggregate": aggregate, "per_class": per_class}


def snapshot_inputs(
    repo_root: Path,
    dataset_root: Path,
    config_path: Path,
    run_dir: Path,
) -> dict[str, str]:
    statistics = load_json(dataset_root / "statistics.json")
    sources = {
        "baseline_config.yaml": config_path,
        "dataset.yaml": dataset_root / "dataset.yaml",
        "dataset_statistics.json": dataset_root / "statistics.json",
        "dataset_validation_report.json": dataset_root / "validation_report.json",
        "class_mapping.json": Path(statistics["class_mapping"]).expanduser().resolve(),
        "environment_report.json": repo_root / "articulation_experiments" / "outputs" / "environment_report.json",
    }
    hashes = {}
    for destination_name, source in sources.items():
        if source.is_file():
            shutil.copy2(source, run_dir / destination_name)
            hashes[destination_name] = sha256(source)
    return hashes


def train_baseline(
    repo_root: Path,
    config_path: Path,
    config: dict[str, Any],
    dataset_root: Path,
    runs_dir: Path,
    run_name: str,
    overrides: dict[str, Any],
    resume_path: Path | None,
) -> dict[str, Any]:
    os.environ["YOLO_CONFIG_DIR"] = str(
        repo_root / "articulation_experiments" / "outputs" / "ultralytics_config"
    )
    from ultralytics import YOLO

    started_at = datetime.now(timezone.utc)
    started = time.perf_counter()
    training = dict(config["training"])
    training.update({key: value for key, value in overrides.items() if value is not None})
    model_name = str(training.pop("model"))
    initial_weights = training.pop("initial_weights", None)
    training.pop("pretrained", None)

    if resume_path is not None:
        resume_path = resume_path.expanduser().resolve()
        if not resume_path.is_file() or resume_path.parent.name != "weights":
            raise FileNotFoundError(f"Invalid resume checkpoint: {resume_path}")
        run_dir = resume_path.parent.parent
        model = YOLO(str(resume_path))
        model.train(resume=True)
    else:
        run_dir = runs_dir / run_name
        if run_dir.exists():
            raise FileExistsError(
                f"Run already exists: {run_dir}. Use --resume with weights/last.pt."
            )
        pretrained_path = (
            (repo_root / str(initial_weights)).resolve()
            if initial_weights
            else repo_root / "articulation_experiments" / "outputs" / "pretrained" / model_name
        )
        if not pretrained_path.is_file():
            raise FileNotFoundError(f"Missing pretrained weights: {pretrained_path}")
        model = YOLO(str(pretrained_path))
        training.update(config["augmentation"])
        model.train(
            data=str(dataset_root / "dataset.yaml"),
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

    validation = dict(config["validation"])
    best_model = YOLO(str(best_path))
    metrics = best_model.val(
        data=str(dataset_root / "dataset.yaml"),
        imgsz=int(training["imgsz"]),
        batch=int(validation["batch"]),
        device=training["device"],
        workers=int(validation["workers"]),
        split=str(validation["split"]),
        plots=bool(validation["plots"]),
        verbose=bool(validation["verbose"]),
        project=str(run_dir),
        name="final_validation",
        exist_ok=True,
    )
    class_names = {int(key): str(value) for key, value in best_model.names.items()}
    metric_report = collect_metrics(metrics, class_names)
    hashes = snapshot_inputs(repo_root, dataset_root, config_path, run_dir)
    csv_summary = summarize_csv(run_dir / "results.csv")
    summary = {
        "schema_version": 1,
        "generated_at_utc": datetime.now(timezone.utc).isoformat(),
        "started_at_utc": started_at.isoformat(),
        "duration_seconds": time.perf_counter() - started,
        "completed": True,
        "run_name": run_dir.name,
        "model": "YOLO11n Detect",
        "seed": int(config["seed"]),
        "training": training,
        "validation": validation,
        "metrics": metric_report,
        "training_csv": csv_summary,
        "dataset": {
            "root": str(dataset_root),
            "train_tiles": int(config["dataset"]["expected_train_tiles"]),
            "val_tiles": int(config["dataset"]["expected_val_tiles"]),
        },
        "paths": {
            "run_dir": str(run_dir),
            "best_weights": str(best_path),
            "last_weights": str(last_path),
        },
        "input_sha256": hashes,
        "git": git_snapshot(repo_root),
    }
    write_json(run_dir / "baseline_summary.json", summary)
    print(json.dumps(metric_report, ensure_ascii=False, indent=2), flush=True)
    print(f"SUMMARY {run_dir / 'baseline_summary.json'}", flush=True)
    return summary


def parse_args() -> argparse.Namespace:
    script_path = Path(__file__).resolve()
    repo_root = script_path.parents[2]
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--config",
        type=Path,
        default=repo_root / "articulation_experiments" / "configs" / "baseline.yaml",
    )
    parser.add_argument("--dataset-root", type=Path)
    parser.add_argument(
        "--runs-dir",
        type=Path,
        default=repo_root / "articulation_experiments" / "outputs" / "runs",
    )
    parser.add_argument("--run-name")
    parser.add_argument("--epochs", type=int)
    parser.add_argument("--batch", type=int)
    parser.add_argument("--device")
    parser.add_argument("--resume", type=Path)
    parser.add_argument("--preflight-only", action="store_true")
    return parser.parse_args()


def main() -> int:
    args = parse_args()
    script_path = Path(__file__).resolve()
    repo_root = script_path.parents[2]
    config_path = args.config.expanduser().resolve()
    config = yaml.safe_load(config_path.read_text(encoding="utf-8"))
    dataset_root = (
        args.dataset_root.expanduser().resolve()
        if args.dataset_root
        else (repo_root / config["dataset"]["root"]).resolve()
    )
    report = validate_dataset(repo_root, dataset_root, config)
    print(
        f"PREFLIGHT PASSED: dataset validation={report['passed']}, "
        f"train={config['dataset']['expected_train_tiles']}, "
        f"val={config['dataset']['expected_val_tiles']}",
        flush=True,
    )
    if args.preflight_only:
        return 0
    train_baseline(
        repo_root=repo_root,
        config_path=config_path,
        config=config,
        dataset_root=dataset_root,
        runs_dir=args.runs_dir.expanduser().resolve(),
        run_name=args.run_name or str(config["output"]["run_name"]),
        overrides={"epochs": args.epochs, "batch": args.batch, "device": args.device},
        resume_path=args.resume,
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
