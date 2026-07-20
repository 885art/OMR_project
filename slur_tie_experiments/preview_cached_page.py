"""Run slur/tie prototype on one cached 25-omr page."""

from __future__ import annotations

import argparse
import json
import pickle
import sys
from pathlib import Path

import cv2
import numpy as np


def _install_pickle_classes(legacy) -> None:
    module = sys.modules["__main__"]
    for name in ("Bar", "NoteGroup", "Stem", "Rest", "Accidentals", "Clef"):
        setattr(module, name, getattr(legacy, name))


def _groups(bar_list, group_type):
    result, seen = [], set()
    for track in bar_list:
        for bar in track:
            for element in bar.elementList:
                if type(element) is group_type and id(element) not in seen:
                    seen.add(id(element))
                    result.append(element)
    return result


def main() -> int:
    root = Path(__file__).resolve().parents[1]
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--piece", default="beethoven1")
    parser.add_argument("--page", type=int, default=1)
    parser.add_argument("--output-dir", type=Path, default=root / "slur_tie_experiments" / "outputs")
    parser.add_argument("--no-visualize", action="store_true")
    args = parser.parse_args()
    sys.path.insert(0, str(root))
    import pdf2musicXML as legacy
    from omr.slur_tie import process_page_slurs_ties

    _install_pickle_classes(legacy)
    piece_dir = root / "string_dataset" / "pdf_data" / args.piece
    page_id = f"{args.piece}_{args.page}"
    page_dir = piece_dir / "imgs" / page_id
    config = json.loads((piece_dir / f"{args.piece}.json").read_text(encoding="utf-8"))
    legacy.NUM_TRACK = config["numTrack"]
    legacy.TRACK_SHIFT = config["track_shift"]
    legacy.CLEF_OPTIONS = config["clef_options"]
    legacy.TIME_SIGNATURE = config["tsChange"][0]["time_signature"]
    with (page_dir / f"{page_id}_barlist.pkl").open("rb") as handle:
        bar_list = pickle.load(handle)
    data = np.load(page_dir / f"{page_id}.npy", allow_pickle=True).tolist()
    for key, image in data.items():
        data[key] = cv2.resize(image, None, fx=2.0, fy=2.0, interpolation=cv2.INTER_NEAREST)
    _, staffs = legacy.init_bar_height(data)
    image_path = page_dir / f"{page_id}.png"
    source = cv2.imread(str(image_path), cv2.IMREAD_GRAYSCALE)
    scale = (data["image"].shape[1] / source.shape[1], data["image"].shape[0] / source.shape[0])
    groups = _groups(bar_list, legacy.NoteGroup)
    destination = args.output_dir.resolve() / page_id
    document = process_page_slurs_ties(
        image_path, page_id, groups, staffs, destination,
        coordinate_scale=scale, visualize=not args.no_visualize,
    )
    score, _, _ = legacy.exportXML(bar_list, legacy.NUM_TRACK)
    xml_path = destination / f"{page_id}.with_slurs_ties.musicxml"
    score.write("musicxml", fp=str(xml_path))
    print(json.dumps({
        "page": page_id,
        "candidates": document["candidate_count"],
        "matched": document["matched_count"],
        "relations": document["relation_count"],
        "musicxml": str(xml_path),
    }, indent=2))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
