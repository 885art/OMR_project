"""Refresh slur/tie output for a cached piece without rerunning base OMR."""

from __future__ import annotations

import argparse
import json
import pickle
import sys
from collections import Counter
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
    parser.add_argument("--visualize-pages", type=int, nargs="*", default=[1])
    parser.add_argument("--max-tie-span-units", type=float, default=6.5)
    parser.add_argument("--xml-confidence", type=float, default=0.30)
    args = parser.parse_args()
    sys.path.insert(0, str(root))

    import pdf2musicXML as legacy
    from omr.slur_tie import process_page_slurs_ties

    _install_pickle_classes(legacy)
    piece_dir = root / "string_dataset" / "pdf_data" / args.piece
    output_dir = root / "string_dataset" / "output" / args.piece
    curve_dir = output_dir / "slur_tie"
    config = json.loads((piece_dir / f"{args.piece}.json").read_text(encoding="utf-8"))
    legacy.NUM_TRACK = config["numTrack"]
    legacy.TRACK_SHIFT = config["track_shift"]
    legacy.CLEF_OPTIONS = config["clef_options"]
    legacy.TIME_SIGNATURE = config["tsChange"][0]["time_signature"]

    whole = [[] for _ in range(legacy.NUM_TRACK)]
    breakpoints = [0]
    totals = Counter()
    for page in range(1, config["numPage"] + 1):
        page_id = f"{args.piece}_{page}"
        page_dir = piece_dir / "imgs" / page_id
        with (page_dir / f"{page_id}_barlist.pkl").open("rb") as handle:
            bar_list = pickle.load(handle)
        data = np.load(page_dir / f"{page_id}.npy", allow_pickle=True).tolist()
        for key, image in data.items():
            data[key] = cv2.resize(
                image, None, fx=2.0, fy=2.0, interpolation=cv2.INTER_NEAREST
            )
        _, staffs = legacy.init_bar_height(data)
        image_path = page_dir / f"{page_id}.png"
        source = cv2.imread(str(image_path), cv2.IMREAD_GRAYSCALE)
        scale = (
            data["image"].shape[1] / source.shape[1],
            data["image"].shape[0] / source.shape[0],
        )
        document = process_page_slurs_ties(
            image_path,
            page_id,
            _groups(bar_list, legacy.NoteGroup),
            staffs,
            curve_dir,
            coordinate_scale=scale,
            max_tie_span_units=args.max_tie_span_units,
            xml_confidence=args.xml_confidence,
            visualize=page in set(args.visualize_pages),
        )
        totals["candidates"] += document["candidate_count"]
        totals["matched"] += document["matched_count"]
        totals["xml_eligible"] += document["xml_eligible_count"]
        for relation in document["relations"]:
            totals[relation["predicted_type"]] += 1
            if relation["xml_eligible"]:
                totals[f"xml_{relation['predicted_type']}"] += 1

        score, _, _ = legacy.exportXML(
            bar_list, legacy.NUM_TRACK, barsBreakPoints=[0, len(bar_list[0])]
        )
        score.write("musicxml", fp=str(output_dir / f"{page_id}.xml"))
        for track in range(legacy.NUM_TRACK):
            whole[track].extend(bar_list[track])
        breakpoints.append(breakpoints[-1] + len(bar_list[0]))
        print(
            f"{page_id}: {document['xml_eligible_count']} XML / "
            f"{document['candidate_count']} candidates"
        )

    score, _, _ = legacy.exportXML(
        whole, legacy.NUM_TRACK, barsBreakPoints=breakpoints
    )
    combined = output_dir / f"all_{args.piece}_{config['numPage']}.xml"
    score.write("musicxml", fp=str(combined))
    summary = {
        "piece": args.piece,
        "pages": config["numPage"],
        "totals": dict(totals),
        "combined_musicxml": str(combined),
    }
    (curve_dir / f"{args.piece}.summary.json").write_text(
        json.dumps(summary, indent=2) + "\n", encoding="utf-8"
    )
    print(json.dumps(summary, indent=2))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
