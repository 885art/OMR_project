#!/usr/bin/env python3
"""Generate the canonical 136-class DeepScores-to-YOLO mapping.

The source schema also contains 72 MUSCIMA++ category IDs.  They are not extra
DeepScores classes and several duplicate DeepScores names, so the all-class
detector intentionally uses only annotation_set=deepscores (IDs 1..136).
"""

from __future__ import annotations

import argparse
import json
import os
from pathlib import Path
from typing import Any


EXPECTED_IDS = list(range(1, 137))


def normalized_semantic_class(name: str) -> str:
    """Map all136 names onto semantic classes supported by OMR25.

    Structural classes intentionally keep their original names.  The existing
    articulation/MusicXML stage rejects those as unsupported, preventing the
    all136 detector from duplicating legacy note, rest, stem, beam, staff, and
    pitch reconstruction.
    """

    exact = {
        "caesura": "caesura",
        "ornamentTrill": "trill",
        "ornamentTurn": "turn",
        "ornamentTurnInverted": "inverted_turn",
        "ornamentMordent": "mordent",
        "stringsDownBow": "down_bow",
        "stringsUpBow": "up_bow",
        "arpeggiato": "arpeggio",
        "keyboardPedalPed": "pedal_start",
        "keyboardPedalUp": "pedal_stop",
        "dynamicCrescendoHairpin": "crescendo",
        "dynamicDiminuendoHairpin": "diminuendo",
        "tupletBracket": "tuplet_bracket",
    }
    if name in exact:
        return exact[name]
    side_variants = {
        "articAccent": "accent",
        "articStaccato": "staccato",
        "articTenuto": "tenuto",
        "articStaccatissimo": "staccatissimo",
        "articMarcato": "marcato",
        "fermata": "fermata",
    }
    for prefix, semantic in side_variants.items():
        if name.startswith(prefix):
            return semantic
    prefixes = {
        "dynamicP": "dynamic_letter_p",
        "dynamicM": "dynamic_letter_m",
        "dynamicF": "dynamic_letter_f",
        "dynamicS": "dynamic_letter_s",
        "dynamicZ": "dynamic_letter_z",
        "dynamicR": "dynamic_letter_r",
        "fingering": "fingering_",
        "tremolo": "tremolo_",
        "tuplet": "tuplet_",
    }
    for prefix, semantic_prefix in prefixes.items():
        if name.startswith(prefix):
            suffix = name.removeprefix(prefix)
            return semantic_prefix + suffix.lower()
    return name


def load_json(path: Path) -> dict[str, Any]:
    with path.open("r", encoding="utf-8") as handle:
        return json.load(handle)


def build_mapping(document: dict[str, Any]) -> dict[str, Any]:
    categories = document.get("categories")
    if not isinstance(categories, dict):
        raise ValueError("DeepScores document has no categories object")

    selected = sorted(
        (
            int(raw_id),
            str(record["name"]),
        )
        for raw_id, record in categories.items()
        if record.get("annotation_set") == "deepscores"
    )
    actual_ids = [category_id for category_id, _ in selected]
    if actual_ids != EXPECTED_IDS:
        raise ValueError(
            "Expected canonical DeepScores IDs 1..136, got "
            f"{actual_ids[:5]}...{actual_ids[-5:]} ({len(actual_ids)} classes)"
        )
    names = [name for _, name in selected]
    if len(set(names)) != len(names):
        raise ValueError("Canonical DeepScores class names are not unique")

    classes = []
    for category_id, name in selected:
        if name.endswith("Above"):
            side: str | None = "above"
        elif name.endswith("Below"):
            side = "below"
        else:
            side = None
        classes.append(
            {
                "deepscores_id": category_id,
                "deepscores_name": name,
                "yolo_id": category_id - 1,
                "normalized_semantic_class": normalized_semantic_class(name),
                "side": side,
            }
        )

    return {
        "schema_version": 1,
        "description": (
            "All 136 canonical DeepScores classes in source-ID order. "
            "The 72 MUSCIMA++ compatibility categories are intentionally excluded."
        ),
        "source_annotation_set": "deepscores",
        "invalid_bbox_policy": "drop_nonpositive_and_audit",
        "deepscores_to_yolo": {
            str(category_id): category_id - 1 for category_id, _ in selected
        },
        "yolo_names": names,
        "classes": classes,
    }


def write_json_atomic(path: Path, payload: dict[str, Any]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_suffix(path.suffix + ".tmp")
    with temporary.open("w", encoding="utf-8", newline="\n") as handle:
        json.dump(payload, handle, indent=2, ensure_ascii=False)
        handle.write("\n")
    os.replace(temporary, path)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--source-json", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--overwrite", action="store_true")
    return parser.parse_args()


def main() -> int:
    args = parse_args()
    source = args.source_json.expanduser().resolve()
    output = args.output.expanduser().resolve()
    mapping = build_mapping(load_json(source))
    if output.is_file():
        existing = load_json(output)
        if existing != mapping and not args.overwrite:
            raise ValueError(
                f"Existing mapping differs from canonical source schema: {output}"
            )
        if existing == mapping:
            print(f"MAPPING ALREADY VALID: {output}")
            return 0
    write_json_atomic(output, mapping)
    print(f"WROTE {len(mapping['yolo_names'])}-CLASS MAPPING: {output}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
