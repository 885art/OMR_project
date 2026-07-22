from __future__ import annotations

import unittest

from omr.articulation import associate_candidates, attach_to_music21
from articulation_experiments.inference.export_candidates import export_candidates


class FakeStaff:
    def __init__(self, ys):
        self.left = 0
        self.right = 1000
        self.ys = ys

    def get_yOne_float(self):
        return 10.0


class FakeGroup:
    def __init__(self, bbox):
        self.noteBoxes = [bbox]
        self.boundingBox = bbox
        self.articulations = []


def candidate(class_name, side, bbox, confidence=0.9):
    return {
        "class_name": class_name,
        "side": side,
        "bbox_xyxy": bbox,
        "confidence": confidence,
        "matched_note_id": None,
        "matched_note_group_id": None,
        "association_score": None,
        "association_status": "unmatched",
    }


class RuntimeIntegrationTest(unittest.TestCase):
    def test_expanded_mapping_exports_all_detector_classes(self):
        from pathlib import Path

        merged = {
            "image_id": "synthetic",
            "source_width": 100,
            "source_height": 100,
            "predictions": [
                {"class_id": class_id, "bbox_xyxy": [1, 2, 3, 4], "confidence": 0.9}
                for class_id in range(17)
            ],
        }
        mapping = (
            Path(__file__).resolve().parents[1]
            / "dataset"
            / "class_mapping_expanded.json"
        )
        document = export_candidates(merged, mapping)
        self.assertEqual(document["candidate_count"], 17)
        self.assertEqual(
            {item["class_name"] for item in document["candidates"]},
            {
                "accent", "staccato", "tenuto", "staccatissimo", "marcato",
                "fermata", "caesura", "trill", "turn", "inverted_turn", "mordent",
            },
        )

    def test_associates_by_staff_side_and_horizontal_position(self):
        groups = [FakeGroup((95, 95, 105, 105)), FakeGroup((95, 295, 105, 305))]
        staffs = [FakeStaff((80, 90, 100, 110, 120)), FakeStaff((280, 290, 300, 310, 320))]
        document = {
            "candidates": [
                candidate("accent", "above", (96, 70, 104, 78)),
                candidate("tenuto", "below", (93, 326, 107, 329)),
                candidate("staccato", "above", (400, 70, 406, 76)),
            ]
        }

        associate_candidates(document, groups, staffs)

        self.assertEqual(document["association"]["matched_count"], 2)
        self.assertEqual(document["candidates"][0]["matched_note_group_id"], 0)
        self.assertEqual(document["candidates"][1]["matched_note_group_id"], 1)
        self.assertEqual(document["candidates"][2]["association_status"], "rejected")
        self.assertEqual(groups[0].articulations[0]["class_name"], "accent")
        self.assertEqual(groups[1].articulations[0]["class_name"], "tenuto")

    def test_coordinate_scale_and_duplicate_suppression(self):
        group = FakeGroup((90, 90, 110, 110))
        document = {
            "candidates": [
                candidate("staccato", "above", (48, 30, 52, 35), 0.70),
                candidate("staccato", "above", (48, 30, 52, 35), 0.95),
            ]
        }

        associate_candidates(document, [group], coordinate_scale=2.0, fallback_unit_size=20)

        self.assertEqual(document["association"]["matched_count"], 1)
        self.assertEqual(len(group.articulations), 1)
        self.assertAlmostEqual(group.articulations[0]["confidence"], 0.95)

    def test_musicxml_contains_articulations_and_placement(self):
        from music21 import note, stream

        group = FakeGroup((0, 0, 10, 10))
        group.articulations = [
            {"class_name": "accent", "side": "above"},
            {"class_name": "staccato", "side": "below"},
            {"class_name": "tenuto", "side": "above"},
        ]
        music_note = attach_to_music21(note.Note("C4"), group)
        score = stream.Score([stream.Part([stream.Measure([music_note])])])
        xml = score.write("musicxml")
        with open(xml, "r", encoding="utf-8") as handle:
            text = handle.read()
        self.assertIn("<accent placement=\"above\"", text)
        self.assertIn("<staccato placement=\"below\"", text)
        self.assertIn("<tenuto placement=\"above\"", text)

    def test_musicxml_contains_expanded_symbols(self):
        from music21 import note, stream

        group = FakeGroup((0, 0, 10, 10))
        group.articulations = [
            {"class_name": "staccatissimo", "side": "above"},
            {"class_name": "marcato", "side": "below"},
            {"class_name": "fermata", "side": "below"},
            {"class_name": "caesura", "side": None},
            {"class_name": "trill", "side": "above"},
            {"class_name": "turn", "side": "above"},
            {"class_name": "inverted_turn", "side": "above"},
            {"class_name": "mordent", "side": "above"},
        ]
        music_note = attach_to_music21(note.Note("C4"), group)
        score = stream.Score([stream.Part([stream.Measure([music_note])])])
        xml = score.write("musicxml")
        with open(xml, "r", encoding="utf-8") as handle:
            text = handle.read()
        for token in (
            "<staccatissimo",
            '<strong-accent placement="below" type="down"',
            '<fermata type="inverted"',
            "<caesura",
            "<trill-mark",
            "<turn",
            "<inverted-turn",
            "<mordent",
        ):
            self.assertIn(token, text)


if __name__ == "__main__":
    unittest.main()
