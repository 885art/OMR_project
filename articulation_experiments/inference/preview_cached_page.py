"""Apply articulation recognition to one cached 25-omr page and export MusicXML.

This is a quick preview path: it reuses an existing ``*_barlist.pkl`` instead of
rerunning the full legacy segmentation and rhythm pipeline.
"""

from __future__ import annotations

import argparse
import json
import pickle
import sys
from pathlib import Path

import cv2
import numpy as np


def _install_legacy_pickle_classes(legacy) -> None:
    main_module = sys.modules["__main__"]
    for name in ("Bar", "NoteGroup", "Stem", "Rest", "Accidentals", "Clef"):
        setattr(main_module, name, getattr(legacy, name))


def _unique_note_groups(bar_list, note_group_type):
    result = []
    seen = set()
    for track in bar_list:
        for bar in track:
            for element in bar.elementList:
                if type(element) is note_group_type and id(element) not in seen:
                    seen.add(id(element))
                    result.append(element)
    return result


def main() -> int:
    script = Path(__file__).resolve()
    repo_root = script.parents[2]
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--piece", default="beethoven1")
    parser.add_argument("--page", type=int, default=1)
    parser.add_argument("--device", default="0")
    parser.add_argument("--confidence", type=float, default=0.25)
    parser.add_argument(
        "--output-dir",
        type=Path,
        default=repo_root / "articulation_experiments" / "outputs" / "integration_preview",
    )
    args = parser.parse_args()

    sys.path.insert(0, str(repo_root))
    import pdf2musicXML as legacy
    from omr.articulation import process_page_articulations

    _install_legacy_pickle_classes(legacy)
    piece_dir = repo_root / "string_dataset" / "pdf_data" / args.piece
    page_id = f"{args.piece}_{args.page}"
    page_dir = piece_dir / "imgs" / page_id
    image_path = page_dir / f"{page_id}.png"
    barlist_path = page_dir / f"{page_id}_barlist.pkl"
    npy_path = page_dir / f"{page_id}.npy"
    config_path = piece_dir / f"{args.piece}.json"
    for required in (image_path, barlist_path, npy_path, config_path):
        if not required.is_file():
            raise FileNotFoundError(required)

    config = json.loads(config_path.read_text(encoding="utf-8"))
    legacy.NUM_TRACK = int(config["numTrack"])
    legacy.TRACK_SHIFT = config["track_shift"]
    legacy.CLEF_OPTIONS = config["clef_options"]
    legacy.TIME_SIGNATURE = config["tsChange"][0]["time_signature"]
    with barlist_path.open("rb") as handle:
        bar_list = pickle.load(handle)

    data = np.load(npy_path, allow_pickle=True).tolist()
    resize_multiplier = 2.0
    for key, image in data.items():
        data[key] = cv2.resize(
            image,
            None,
            fx=resize_multiplier,
            fy=resize_multiplier,
            interpolation=cv2.INTER_NEAREST,
        )
    _, staffs = legacy.init_bar_height(data)
    note_groups = _unique_note_groups(bar_list, legacy.NoteGroup)
    source_image = cv2.imread(str(image_path), cv2.IMREAD_GRAYSCALE)
    coordinate_scale = (
        data["image"].shape[1] / source_image.shape[1],
        data["image"].shape[0] / source_image.shape[0],
    )
    output_dir = args.output_dir.resolve() / page_id
    document = process_page_articulations(
        image_path,
        page_id,
        note_groups,
        staffs,
        output_dir,
        coordinate_scale=coordinate_scale,
        confidence=args.confidence,
        device=None if args.device.lower() == "auto" else args.device,
    )
    score, _, _ = legacy.exportXML(bar_list, legacy.NUM_TRACK)
    xml_path = output_dir / f"{page_id}.with_articulations.musicxml"
    score.write("musicxml", fp=str(xml_path))
    summary = {
        "page": page_id,
        "detected": document["candidate_count"],
        "matched": document["association"]["matched_count"],
        "musicxml": str(xml_path),
    }
    print(json.dumps(summary, indent=2))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
