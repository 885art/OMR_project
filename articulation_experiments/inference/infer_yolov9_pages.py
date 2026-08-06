"""Run one official YOLOv9 detector over every score page in a directory."""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

from PIL import Image


def main() -> int:
    script = Path(__file__).resolve()
    repo_root = script.parents[2]
    sys.path.insert(0, str(repo_root))

    from articulation_experiments.inference.export_candidates import export_candidates
    from articulation_experiments.inference.infer_tiled_page import (
        draw_predictions,
        tiled_predict,
    )
    from articulation_experiments.inference.merge_tile_predictions import (
        merge_predictions,
    )
    from omr.yolov9_backend import load_yolov9_detector

    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--input-dir", type=Path, required=True)
    parser.add_argument("--output-dir", type=Path, required=True)
    parser.add_argument("--weights", type=Path, required=True)
    parser.add_argument("--data-yaml", type=Path, required=True)
    parser.add_argument("--class-mapping", type=Path, required=True)
    parser.add_argument("--yolov9-root", type=Path, default=repo_root.parent / "yolov9")
    parser.add_argument("--glob", default="*.png")
    parser.add_argument("--tile-size", type=int, default=512)
    parser.add_argument("--overlap", type=int, default=128)
    parser.add_argument("--model-input-size", type=int, default=1024)
    parser.add_argument("--edge-policy", choices=("pad", "shift"), default="shift")
    parser.add_argument("--confidence", type=float, default=0.25)
    parser.add_argument("--nms-iou", type=float, default=0.5)
    parser.add_argument("--batch", type=int, default=4)
    parser.add_argument("--device", default="0")
    parser.add_argument("--resume", action="store_true")
    parser.add_argument(
        "--parenthesis-robust",
        action="store_true",
        help="Run a second view with conservative parenthesis strokes suppressed.",
    )
    parser.add_argument(
        "--validate-hairpins",
        action="store_true",
        help="Keep low-threshold hairpins only when wedge geometry is present.",
    )
    args = parser.parse_args()

    input_dir = args.input_dir.expanduser().resolve()
    output_dir = args.output_dir.expanduser().resolve()
    image_paths = sorted(path for path in input_dir.glob(args.glob) if path.is_file())
    if not image_paths:
        raise FileNotFoundError(f"No pages matching {args.glob!r} in {input_dir}")
    output_dir.mkdir(parents=True, exist_ok=True)

    model = load_yolov9_detector(
        args.weights,
        args.data_yaml,
        yolov9_root=args.yolov9_root,
        device=args.device,
    )
    page_summaries = []
    for index, image_path in enumerate(image_paths, 1):
        prefix = image_path.stem
        summary_path = output_dir / f"{prefix}.summary.json"
        if args.resume and summary_path.is_file():
            summary = json.loads(summary_path.read_text(encoding="utf-8"))
            page_summaries.append(summary)
            print(f"SKIP {index}/{len(image_paths)} {image_path.name}", flush=True)
            continue

        with Image.open(image_path) as opened:
            image = opened.convert("RGB")
        from articulation_experiments.inference.piano_views import tiled_predict_piano_views

        raw, elapsed, view_metadata = tiled_predict_piano_views(
            model, image, prefix, str(image_path), args.tile_size, args.overlap,
            args.confidence, args.device, args.batch, args.model_input_size,
            args.edge_policy, parenthesis_robust=args.parenthesis_robust,
        )
        merged = merge_predictions(raw, args.nms_iou)
        hairpin_validation = None
        if args.validate_hairpins:
            from omr.hairpin import validate_yolo_hairpins

            accepted_hairpins, rejected_hairpins = validate_yolo_hairpins(
                image, merged["predictions"], minimum_model_confidence=args.confidence
            )
            merged["predictions"] = [
                prediction
                for prediction in merged["predictions"]
                if "Hairpin" not in str(prediction.get("raw_class_name", ""))
            ] + accepted_hairpins
            merged["prediction_count"] = len(merged["predictions"])
            hairpin_validation = {
                "accepted_count": len(accepted_hairpins),
                "rejected_count": len(rejected_hairpins),
                "rejected": rejected_hairpins,
            }
        candidates = export_candidates(merged, args.class_mapping.resolve())
        (output_dir / f"{prefix}.raw_tiles.json").write_text(
            json.dumps(raw, indent=2) + "\n", encoding="utf-8"
        )
        (output_dir / f"{prefix}.merged.json").write_text(
            json.dumps(merged, indent=2) + "\n", encoding="utf-8"
        )
        (output_dir / f"{prefix}.candidates.json").write_text(
            json.dumps(candidates, indent=2) + "\n", encoding="utf-8"
        )
        cleaned_page = view_metadata.pop("cleaned_page", None)
        (output_dir / f"{prefix}.parentheses.json").write_text(
            json.dumps(view_metadata, indent=2) + "\n", encoding="utf-8"
        )
        if hairpin_validation is not None:
            (output_dir / f"{prefix}.hairpin_validation.json").write_text(
                json.dumps(hairpin_validation, indent=2) + "\n", encoding="utf-8"
            )
        if cleaned_page is not None:
            cleaned_page.save(output_dir / f"{prefix}.parentheses_suppressed.jpg", quality=92)
        draw_predictions(image, merged["predictions"]).save(
            output_dir / f"{prefix}.annotated.jpg", quality=92
        )
        summary = {
            "page": prefix,
            "tile_count": raw["tile_count"],
            "raw_prediction_count": raw["prediction_count"],
            "merged_prediction_count": merged["prediction_count"],
            "inference_seconds": elapsed,
            "milliseconds_per_tile": elapsed * 1000.0 / raw["tile_count"],
            "detected_parenthesis_count": view_metadata["detected_parenthesis_count"],
            "validated_hairpin_count": (
                hairpin_validation["accepted_count"] if hairpin_validation else None
            ),
        }
        summary_path.write_text(json.dumps(summary, indent=2) + "\n", encoding="utf-8")
        page_summaries.append(summary)
        print(
            f"DONE {index}/{len(image_paths)} {image_path.name}: "
            f"{summary['merged_prediction_count']} detections, {elapsed:.2f}s",
            flush=True,
        )

    aggregate = {
        "page_count": len(page_summaries),
        "merged_prediction_count": sum(
            int(item["merged_prediction_count"]) for item in page_summaries
        ),
        "inference_seconds": sum(float(item["inference_seconds"]) for item in page_summaries),
        "pages": page_summaries,
    }
    (output_dir / "all_pages.summary.json").write_text(
        json.dumps(aggregate, indent=2) + "\n", encoding="utf-8"
    )
    print(json.dumps(aggregate, indent=2), flush=True)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
