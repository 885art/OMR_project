"""Validate a MusicXML trial and summarize its detector/XML contents."""

from __future__ import annotations

import argparse
import json
from collections import Counter
from pathlib import Path


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--musicxml", type=Path, required=True)
    parser.add_argument("--articulation-json", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()

    from music21 import chord, converter, note, stream

    score = converter.parse(str(args.musicxml.resolve()))
    document = json.loads(args.articulation_json.read_text(encoding="utf-8"))
    candidates = list(document.get("candidates", []))
    supported = [item for item in candidates if item.get("association_reason") != "unsupported_class"]
    matched = [item for item in supported if item.get("association_status") == "matched"]
    unsupported = [item for item in candidates if item.get("association_reason") == "unsupported_class"]

    notes = list(score.recurse().getElementsByClass(note.Note))
    chords = list(score.recurse().getElementsByClass(chord.Chord))
    rests = list(score.recurse().getElementsByClass(note.Rest))
    measures = list(score.recurse().getElementsByClass(stream.Measure))
    xml_text = args.musicxml.read_text(encoding="utf-8")
    result = {
        "musicxml": str(args.musicxml.resolve()),
        "music21_parse_ok": True,
        "parts": len(score.parts),
        "measures_across_parts": len(measures),
        "notes": len(notes),
        "chords": len(chords),
        "rests": len(rests),
        "detector": {
            "candidates_after_thresholds": len(candidates),
            "supported_candidates": len(supported),
            "matched_supported_candidates": len(matched),
            "unsupported_structural_candidates": len(unsupported),
            "supported_class_counts": dict(
                Counter(str(item.get("class_name")) for item in supported).most_common()
            ),
            "matched_class_counts": dict(
                Counter(str(item.get("class_name")) for item in matched).most_common()
            ),
        },
        "xml_element_counts": {
            "articulations": xml_text.count("<articulations>"),
            "staccato": xml_text.count("<staccato"),
            "accent": xml_text.count("<accent"),
            "tenuto": xml_text.count("<tenuto"),
            "strong_accent": xml_text.count("<strong-accent"),
            "fermata": xml_text.count("<fermata"),
            "dynamics": xml_text.count("<dynamics"),
            "words": xml_text.count("<words"),
            "wedge": xml_text.count("<wedge"),
            "technical": xml_text.count("<technical"),
            "fingering": xml_text.count("<fingering"),
            "tuplet": xml_text.count("<tuplet"),
            "slur": xml_text.count("<slur"),
            "tie": xml_text.count("<tie "),
        },
        "warning": (
            "This is a one-page execution smoke, not accuracy evidence. The time "
            "signature is inherited from an imported config marked for manual review."
        ),
    }
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(
        json.dumps(result, ensure_ascii=False, indent=2) + "\n", encoding="utf-8"
    )
    print(json.dumps(result, ensure_ascii=False, indent=2))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
