"""Run both YOLOv9 detectors on one cached page and export one MusicXML file."""

from __future__ import annotations

import argparse
import json
import pickle
import sys
from collections import Counter
from pathlib import Path

import cv2
import numpy as np
from PIL import Image, ImageDraw


def install_pickle_classes(legacy) -> None:
    module = sys.modules["__main__"]
    for name in ("Bar", "NoteGroup", "Stem", "Rest", "Accidentals", "Clef"):
        setattr(module, name, getattr(legacy, name))


def unique_note_groups(bar_list, note_group_type):
    result, seen = [], set()
    for track in bar_list:
        for bar in track:
            for element in bar.elementList:
                if type(element) is note_group_type and id(element) not in seen:
                    seen.add(id(element))
                    result.append(element)
    return result


def draw_combined_preview(image_path, destination, page_id, symbols, curves):
    with Image.open(image_path) as opened:
        canvas = opened.convert("RGB")
    draw = ImageDraw.Draw(canvas)
    symbol_colors = {
        "accent": "#e41a1c",
        "staccato": "#377eb8",
        "tenuto": "#984ea3",
        "staccatissimo": "#4daf4a",
        "marcato": "#ff7f00",
        "fermata": "#a65628",
        "caesura": "#f781bf",
        "trill": "#00a6a6",
        "turn": "#1f78b4",
        "inverted_turn": "#6a3d9a",
        "mordent": "#b15928",
        "dynamic": "#d62728",
        "crescendo": "#2ca02c",
        "diminuendo": "#17becf",
    }
    for candidate in symbols["candidates"]:
        matched = candidate.get("association_status") == "matched"
        color = symbol_colors.get(candidate["class_name"], "#ff7f00")
        if not matched:
            color = "#9e9e9e"
        bbox = [float(value) for value in candidate["bbox_xyxy"]]
        draw.rectangle(bbox, outline=color, width=4 if matched else 2)
        name = candidate.get("dynamic_text", candidate["class_name"])
        draw.text(
            (bbox[0], max(0, bbox[1] - 13)),
            f"S:{name} {candidate['confidence']:.2f}",
            fill=color,
        )
    curve_colors = {"slur": "#00a000", "tie": "#0055ff"}
    for candidate in curves["candidates"]:
        kind = candidate.get("predicted_type", "curve")
        eligible = bool(candidate.get("xml_eligible"))
        color = curve_colors.get(kind, "#ff9800") if eligible else "#808080"
        bbox = [float(value) for value in candidate["bbox_xyxy"]]
        draw.rectangle(bbox, outline=color, width=4 if eligible else 2)
        confidence = candidate.get(
            "classification_confidence", candidate.get("confidence", 0.0)
        )
        status = "xml" if eligible else "review"
        draw.text(
            (bbox[0], max(0, bbox[1] - 25)),
            f"C:{kind} {status} {confidence:.2f}",
            fill=color,
        )
    preview_path = destination / f"{page_id}.yolov9_all.jpg"
    canvas.save(preview_path, quality=92)
    return preview_path


def main() -> int:
    root = Path(__file__).resolve().parent
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--piece", default="beethoven1")
    parser.add_argument("--page", type=int, default=1)
    parser.add_argument("--device", default="0")
    parser.add_argument("--symbol-confidence", type=float, default=0.05)
    parser.add_argument("--staccato-threshold", type=float, default=0.10)
    parser.add_argument("--dynamic-threshold", type=float, default=0.65)
    parser.add_argument("--symbol-weights", type=Path)
    parser.add_argument("--symbol-data-yaml", type=Path)
    parser.add_argument("--symbol-tile-size", type=int, default=1024)
    parser.add_argument("--symbol-overlap", type=int, default=256)
    parser.add_argument("--symbol-input-size", type=int)
    parser.add_argument(
        "--symbol-edge-policy",
        choices=("pad", "shift"),
        default="pad",
    )
    parser.add_argument("--curve-confidence", type=float, default=0.25)
    parser.add_argument("--curve-xml-confidence", type=float, default=0.30)
    parser.add_argument(
        "--output-dir",
        type=Path,
        default=root / "yolov9_all_outputs",
    )
    args = parser.parse_args()

    sys.path.insert(0, str(root))
    import pdf2musicXML as legacy
    from omr.articulation import process_page_articulations
    from omr.slur_tie import process_page_slurs_ties

    install_pickle_classes(legacy)
    piece_dir = root / "string_dataset" / "pdf_data" / args.piece
    page_id = f"{args.piece}_{args.page}"
    page_dir = piece_dir / "imgs" / page_id
    image_path = page_dir / f"{page_id}.png"
    config = json.loads(
        (piece_dir / f"{args.piece}.json").read_text(encoding="utf-8")
    )
    legacy.NUM_TRACK = int(config["numTrack"])
    legacy.TRACK_SHIFT = config["track_shift"]
    legacy.CLEF_OPTIONS = config["clef_options"]
    legacy.TIME_SIGNATURE = config["tsChange"][0]["time_signature"]
    with (page_dir / f"{page_id}_barlist.pkl").open("rb") as handle:
        bar_list = pickle.load(handle)

    data = np.load(
        page_dir / f"{page_id}.npy", allow_pickle=True
    ).tolist()
    for key, image in data.items():
        data[key] = cv2.resize(
            image, None, fx=2.0, fy=2.0, interpolation=cv2.INTER_NEAREST
        )
    _, staffs = legacy.init_bar_height(data)
    source = cv2.imread(str(image_path), cv2.IMREAD_GRAYSCALE)
    coordinate_scale = (
        data["image"].shape[1] / source.shape[1],
        data["image"].shape[0] / source.shape[0],
    )
    note_groups = unique_note_groups(bar_list, legacy.NoteGroup)
    destination = args.output_dir.resolve() / page_id
    device = None if args.device.lower() == "auto" else args.device

    symbol_options = {
        "coordinate_scale": coordinate_scale,
        "confidence": args.symbol_confidence,
        "class_confidence": {
            "staccato": args.staccato_threshold,
            "dynamic": args.dynamic_threshold,
        },
        "backend": "yolov9",
        "device": device,
        "tile_size": args.symbol_tile_size,
        "overlap": args.symbol_overlap,
        "model_input_size": args.symbol_input_size,
        "edge_policy": args.symbol_edge_policy,
    }
    if args.symbol_weights is not None:
        symbol_options["weights"] = args.symbol_weights
    if args.symbol_data_yaml is not None:
        symbol_options["data_yaml"] = args.symbol_data_yaml
    symbols = process_page_articulations(
        image_path,
        page_id,
        note_groups,
        staffs,
        destination,
        **symbol_options,
    )
    curves = process_page_slurs_ties(
        image_path,
        page_id,
        note_groups,
        staffs,
        destination,
        coordinate_scale=coordinate_scale,
        confidence=args.curve_confidence,
        xml_confidence=args.curve_xml_confidence,
        backend="yolov9",
        device=device,
        visualize=True,
    )

    score, _, _ = legacy.exportXML(bar_list, legacy.NUM_TRACK)
    xml_path = destination / f"{page_id}.yolov9_all.musicxml"
    score.write("musicxml", fp=str(xml_path))
    preview_path = draw_combined_preview(
        image_path, destination, page_id, symbols, curves
    )
    symbol_counts = Counter(
        candidate["class_name"] for candidate in symbols["candidates"]
    )
    curve_counts = Counter(
        relation["predicted_type"] for relation in curves["relations"]
    )
    summary = {
        "page": page_id,
        "symbol_detector": symbols["detector_backend"],
        "symbol_tile_size": args.symbol_tile_size,
        "symbol_model_input_size": (
            args.symbol_input_size or args.symbol_tile_size
        ),
        "symbol_edge_policy": args.symbol_edge_policy,
        "symbols_detected": symbols["candidate_count"],
        "symbols_matched": symbols["association"]["matched_count"],
        "symbol_counts": dict(sorted(symbol_counts.items())),
        "curve_detector": curves["detector"]["backend"],
        "curves_detected": curves["candidate_count"],
        "curves_matched": curves["matched_count"],
        "curves_xml_eligible": curves["xml_eligible_count"],
        "curve_counts": dict(sorted(curve_counts.items())),
        "curve_xml_confidence": args.curve_xml_confidence,
        "combined_preview": str(preview_path),
        "musicxml": str(xml_path),
    }
    summary_path = destination / f"{page_id}.yolov9_all.summary.json"
    summary_path.write_text(
        json.dumps(summary, indent=2) + "\n", encoding="utf-8"
    )
    print(json.dumps(summary, indent=2))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
